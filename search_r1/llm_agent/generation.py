import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3
    discard_bad_docs: bool = False
    prepend_think_token: bool = False

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        reranker_config,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.reranker_config = reranker_config
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        responses_str = [resp.split('</search>')[0] + '</search>'
                 if '</search>' in resp 
                 else resp.split('</answer>')[0] + '</answer>'
                 if '</answer>' in resp 
                 else resp
                 for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        # if next_obs_ids.shape[1] > self.config.max_obs_length:
        #     print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
        #     next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor, all_questions, all_ground_truth) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        de_facto_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        void_turn_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch
        generate_length_list = []
        stop_flag = False

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            gen_output = self._generate_with_gpu_padding(rollings_active)
            meta_info = gen_output.meta_info
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            generate_mask = responses_ids != self.tokenizer.pad_token_id
            generate_length = generate_mask.sum(dim=1)
            generate_length_list.append(generate_length)

            # Execute in environment and process observations
            next_obs, dones, valid_action, is_search, is_void_turn, is_good_docs = self.execute_predictions(
                responses_str, all_questions, all_ground_truth, self.config.discard_bad_docs, self.config.prepend_think_token, de_facto_search_stats, self.config.max_turns, \
                self.tokenizer.pad_token, active_mask
            )
            void_turn_mask = void_turn_mask * torch.tensor(
                [not v for v in is_void_turn], dtype=torch.bool
            )
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            de_facto_search_stats += torch.tensor(is_good_docs, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
            
        # final LLM rollout
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            generate_mask = responses_ids != self.tokenizer.pad_token_id
            generate_length = generate_mask.sum(dim=1)
            generate_length_list.append(generate_length)

            # # Execute in environment and process observations
            _, dones, valid_action, is_search, is_void_turn, is_good_docs = self.execute_predictions(
                responses_str, None, None, False, self.tokenizer.pad_token, self.config.prepend_think_token, de_facto_search_stats, self.config.max_turns, \
                active_mask, do_search=False
            )
            void_turn_mask = void_turn_mask * torch.tensor(
                [not v for v in is_void_turn], dtype=torch.bool
            )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            de_facto_search_stats += torch.tensor(is_good_docs, dtype=torch.int)

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        if self.config.discard_bad_docs:
            meta_info['de_facto_search_stats'] = de_facto_search_stats.tolist()
        meta_info["void_turn_mask"] = void_turn_mask.tolist()
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        print("VOID_TURN_MASK:", void_turn_mask.sum().item())
        
        return self._compose_final_output(original_left_side, original_right_side, meta_info, generate_length_list, void_turn_mask)

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict,
                            generate_length_list: List,
                            void_turn_mask: torch.Tensor) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        info_mask_right = self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            info_mask_right
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        # final_output['generate_length'] = torch.stack(generate_length_list, dim=-1)
        final_output['void_turn_mask'] = void_turn_mask
        # final_output['query_mask'] = self.get_query_mask(final_output['responses'])
        # final_output['think_latter_part_mask'] = self.get_think_latter_part_mask(final_output['responses'])
        # segment_mask = self.get_segment_mask(info_mask_right)
        # final_output['segment0_mask'] = segment_mask[0]
        # final_output['segment1_mask'] = segment_mask[1]
        # final_output['segment2_mask'] = segment_mask[2]
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def get_segment_mask(self, masks: torch.Tensor) -> List[List[torch.Tensor]]:
        """分段"""
        all_segment_masks = [[torch.zeros_like(mask) for mask in masks] for _ in range(3)]
        for idx, mask in enumerate(masks):
            # 找到mask中连续1的段
            diff = torch.diff(mask, prepend=torch.tensor([0]), append=torch.tensor([0]))
            starts = torch.where(diff == 1)[0]
            ends = torch.where(diff == -1)[0]
            
            # 获取所有连续段
            segments = list(zip(starts.tolist(), ends.tolist()))
            for seg_idx, (start, end) in enumerate(segments[:3]):
                all_segment_masks[seg_idx][idx][start:end] = 1
        all_segment_masks = [torch.stack(segment_masks, dim=0) for segment_masks in all_segment_masks]
        return all_segment_masks

    def get_query_mask(self, responses: torch.Tensor) -> torch.Tensor:
        """Get query mask from responses."""
        all_query_masks = []
        for response_ids in responses:
            tokens = self.tokenizer.convert_ids_to_tokens(response_ids) 
            query_mask = []
            in_query = False
            sentence = ''
            for idx, token in enumerate(tokens):
                sentence += token
                if '<search>' in sentence:
                    in_query = True
                    sentence = ''
                    query_mask.append(0)
                elif '</search>' in sentence:
                    in_query = False
                    sentence = ''
                    query_mask[-2:] = [0, 0]
                    query_mask.append(0)
                else:
                    query_mask.append(1 if in_query else 0)
            query_mask = torch.tensor(query_mask)
            masked_token_ids = response_ids * query_mask
            all_query_masks.append(query_mask)
        return torch.stack(all_query_masks, dim=0)

    def get_think_latter_part_mask(self, responses: torch.Tensor) -> torch.Tensor:
        """Get think latter part mask from responses."""
        all_think_masks = []
        for response_ids in responses:
            tokens = self.tokenizer.convert_ids_to_tokens(response_ids) 
            think_mask = []
            in_think = False
            sentence = ''
            think_length = 0
            for idx, token in enumerate(tokens):
                sentence += token
                if '<think>' in sentence:
                    in_think = True
                    sentence = ''
                    think_mask.append(0)
                elif '</think>' in sentence:
                    in_think = False
                    sentence = ''
                    think_mask[-2:] = [0, 0]
                    think_mask.append(0)
                    think_start_idx = idx - think_length
                    cutoff = think_start_idx + int((think_length-2) * 0.8)
                    think_mask[think_start_idx:cutoff] = [0] * (cutoff - think_start_idx)
                    think_length = 0
                else:
                    think_length+=1
                    think_mask.append(1 if in_think else 0)
            think_mask = torch.tensor(think_mask)
            masked_token_ids = response_ids * think_mask
            all_think_masks.append(think_mask)
        return torch.stack(all_think_masks, dim=0)
            

    def execute_predictions(self, predictions: List[str], the_questions, the_ground_truth, discard_bad_docs, prepend_think_token, de_facto_search_stats, max_turns, pad_token: str, active_mask=None, do_search=True) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """

        def get_qad_scores(queries, documents, answers):
            payload = {
                "queries": queries,
                "documents": documents,
                "rerank_topk": 999,
                "return_scores": True,
                "not_sort": True,
                "answers": answers
            }
            return requests.post("http://127.0.0.1:6980/rerank", json=payload).json()

        def get_ad_scores(documents, answers):
            payload = {
                "documents": documents,
                "rerank_topk": 999,
                "return_scores": True,
                "not_sort": True,
                "answers": answers
            }
            return requests.post("http://127.0.0.1:6980/rerank", json=payload).json()

        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search = [], [], [], []
        is_void_turn = [0] * len(active_mask)
        is_bad_docs = [0] * len(active_mask)
        is_good_docs = [0] * len(active_mask)
        
        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        if do_search and len(search_queries)>0:
            search_results = self.batch_search(search_queries)
            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
            if discard_bad_docs:
                documents = []
                for doc_str in search_results:
                    split_doc = re.split(r'Doc \d+\(Title:', doc_str)
                    split_doc = [{"str": f"(Title:{doc.strip()}"} for doc in split_doc if doc.strip()]
                    documents.append(split_doc)
                questions = [question for action, question in zip(cur_actions, the_questions) if action == 'search']
                ground_truth = [str(gt) for action, gt in zip(cur_actions, the_ground_truth) if action == 'search']
                assert len(questions) == len(documents)
                scores_qad = get_qad_scores(questions, documents, ground_truth)['result']
                scores_ad = get_ad_scores(documents, ground_truth)['result']
        else:
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                elif action == 'search':
                    if discard_bad_docs:
                        this_query_score_qad = scores_qad.pop(0)
                        this_query_score_ad = scores_ad.pop(0)
                        final_score = max(max(this_query_score_ad), max(this_query_score_qad))
                        if final_score < 0.5:
                            next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}\n<warning> The retrieved documents are likely unrelated to the search query; re-do your reasoning inside <think> and </think> and call a search engine using <search> query </search> </warning></information>')
                            # search_results.pop(0)
                            # next_obs.append('\n\n<information><warning> No relevant documents were retrieved, please ask again </warning></information>\n\n')
                            is_bad_docs[i] = 1
                        else:
                            next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                            is_bad_docs[i] = 0
                            is_good_docs[i] = 1
                    else:
                        if prepend_think_token:
                            next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n<think>')
                        else:
                            next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                else:
                    next_obs.append(f'\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                    dones.append(1)
                    valid_action.append(0)
                    is_search.append(0)
                    is_void_turn[i] = 1
            
        assert len(search_results) == 0
        print(
            f"[debug] void turn number: {sum(is_void_turn)} out of {active_mask.sum()} samples"
        )
        if discard_bad_docs:
            print(
                f"[debug] bad docs number: {sum(is_bad_docs)} out of {active_mask.sum()} samples"
            )
            
        return next_obs, dones, valid_action, is_search, is_void_turn, is_good_docs

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = [] 
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(search|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    def batch_search(self, queries: List[str] = None) -> str:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        print("start batch search")
        results = self._batch_search(queries)['result']
        
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        
        if self.reranker_config.enable:
            payload = {
                "queries": queries,
                "topk_retrieval": self.config.topk,
                "topk_rerank": self.reranker_config.select_num,
                "return_scores": True,
                "not_sort": False,
            }
        else:
            payload = {
                "queries": queries,
                "topk_retrieval": self.config.topk,
                "topk_rerank": self.reranker_config.select_num,
                "return_scores": True,
                "not_sort": True,
            }
        
        return requests.post(self.config.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item['document']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
