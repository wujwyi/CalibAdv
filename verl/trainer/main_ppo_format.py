# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em, qa_em_format
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np
import requests
import Levenshtein

def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'web_questions', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle', 'strategyqa']:
        return qa_em_format.compute_score_em
    else:
        raise NotImplementedError

def get_query_sequence(tokenizer, responses):
    responses = tokenizer.decode(responses, skip_special_tokens=True)
    matches = re.findall(r"<search>(.*?)</search>", responses, re.DOTALL)
    return matches if matches else ""

def get_query_doc_sequence(tokenizer, responses):
    responses = tokenizer.decode(responses, skip_special_tokens=True)
    query_matches = re.findall(r"<search>(.*?)</search>", responses, re.DOTALL)
    doc_matches = re.findall(r"<information>(.*?)</information>", responses, re.DOTALL)
    if not responses.startswith('<think>'):
        think_matches = re.findall(r"<think>(.*?)</think>", '<think>'+responses, re.DOTALL)
    else:
        think_matches = re.findall(r"<think>(.*?)</think>", responses, re.DOTALL)
    return query_matches if query_matches else "", doc_matches if doc_matches else "", think_matches if think_matches else ""

def get_query_ids(tokenizer, responses):
    """Get query from responses."""
    tokens = tokenizer.convert_ids_to_tokens(responses) 
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
    masked_token_ids = responses * query_mask
    nonzero = masked_token_ids != 0
    # diff 找出边界
    boundaries = torch.where(nonzero[1:] ^ nonzero[:-1])[0] + 1
    # 加上首尾
    indices = torch.cat([torch.tensor([0]), boundaries, torch.tensor([len(masked_token_ids)])])
    # 根据索引切分
    segments = []
    for i in range(len(indices) - 1):
        seg = masked_token_ids[indices[i]:indices[i+1]]
        if seg.numel() > 0 and seg.any():  # 只要非零段
            segments.append(seg)
    return segments

def edit_distance(t1: torch.Tensor, t2: torch.Tensor) -> int:
    m, n = len(t1), len(t2)
    dp = torch.zeros((m+1, n+1), dtype=torch.int)

    # 初始化边界
    dp[0, :] = torch.arange(n+1)
    dp[:, 0] = torch.arange(m+1)

    # 动态规划
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if t1[i-1] == t2[j-1] else 1
            dp[i, j] = min(
                dp[i-1, j] + 1,      # 删除
                dp[i, j-1] + 1,      # 插入
                dp[i-1, j-1] + cost  # 替换
            )
    return dp[m, n].item()

def normalized_edit_distance(t1: torch.Tensor, t2: torch.Tensor) -> float:
    dist = edit_distance(t1, t2)
    return 2 * dist / (len(t1) + len(t2))

def information_gain_fn(queries):
    if len(queries) <= 1:
        return 0
    normalized_edit_dis = normalized_edit_distance(queries[0], queries[1])
    return min(normalized_edit_dis, 0.8)

def rollout_query_similarity(queries1, queries2):
    if len(queries1) ==0 or len(queries2) == 0:
        return 0
    similarities = []
    for i in range(min(len(queries1), len(queries2))):
        similarities.append(1 - normalized_edit_distance(queries1[i], queries2[i]))
    return sum(similarities) / max(len(queries1), len(queries2))

# def is_garbled(entropy, mask, ratio):
#     diff = torch.diff(mask, prepend=torch.tensor([0]), append=torch.tensor([0]))
#     starts = torch.where(diff == 1)[0]
#     ends = torch.where(diff == -1)[0]

#     if len(starts) == 0 or len(ends) == 0:
#         return True

#     seg_entropy = entropy[starts[0]:ends[0]]
#     seg_entropy = seg_entropy[len(seg_entropy)//4:]
#     mean_entropy_1 = seg_entropy.mean() if len(seg_entropy) > 0 else torch.tensor(0.0)

#     seg_entropy = entropy[starts[-1]:ends[-1]]
#     seg_entropy = seg_entropy[len(seg_entropy)//4:]  # 取后半段
#     mean_entropy_2 = seg_entropy.mean() if len(seg_entropy) > 0 else torch.tensor(0.0)

#     if mean_entropy_2 >= mean_entropy_1 * ratio:
#         return True
#     return False

def segment(mask):
    # 找到mask中连续1的段
    diff = torch.diff(mask, prepend=torch.tensor([0]), append=torch.tensor([0]))
    starts = torch.where(diff == 1)[0]
    ends = torch.where(diff == -1)[0]
    
    # 获取所有连续段
    segments = list(zip(starts.tolist(), ends.tolist()))
    return segments

def get_ppl(log_probs):
    return torch.exp(-log_probs.mean())

def is_garbled(log_prob1, log_prob2, mask, ppl_threshold):
    segments = segment(mask)
    max_seg_ppl = 0
    for start, end in segments:
        seg_ppl1 = get_ppl(log_prob1[start:end]).item()
        seg_ppl2 = get_ppl(log_prob2[start:end]).item()
        max_seg_ppl = max(max_seg_ppl, seg_ppl1, seg_ppl2)
    return max_seg_ppl >= ppl_threshold

def is_garbled_eval(log_prob1, mask, ppl_threshold):
    segments = segment(mask)
    max_seg_ppl = 0
    for start, end in segments:
        seg_ppl1 = get_ppl(log_prob1[start:end]).item()
        max_seg_ppl = max(max_seg_ppl, seg_ppl1)
    return max_seg_ppl >= ppl_threshold

# def is_garbled(entropy, mask, entropy_threshold):
#     segments = segment(mask)
#     max_seg_entrop = 0
#     for start, end in segments:
#         seg_entropy = entropy[start:end].mean().item()
#         max_seg_entrop = max(max_seg_entrop, seg_entropy)
#     return max_seg_entrop >= entropy_threshold

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

def get_silver_docs(question_idx):
    payload = {
        "question_idx": question_idx
    }
    uid2docs, uid2maxscore = requests.post("http://127.0.0.1:7000/get_silver_docs", json=payload).json()
    uid2docs = {int(k): v["docs"] for k, v in uid2docs.items()}
    uid2maxscore = {int(k): v for k, v in uid2maxscore.items()}
    return uid2docs, uid2maxscore

def word_edit_similarity(s1, s2):
    w1 = s1.split()
    w2 = s2.split()
    dist = Levenshtein.distance(w1, w2)
    return 1 - dist / max(len(w1), len(w2)) if max(len(w1), len(w2)) > 0 else 0

class RewardManager():
    """The reward manager.
    """

    def __init__(
        self, 
        tokenizer, 
        num_examine, 
        eval_mode, 
        reward_function, 
        enable_length_score, 
        overlong_buffer_cfg, 
        valid_action_reward, 
        more_turn, 
        information_gain,
        format_reward,
        query_reward,
        query_score_calculator,
        garbled_penalty,
        query_doc_relevance) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.eval_mode = eval_mode
        self.reward_function = reward_function
        self.enable_length_score = enable_length_score
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.valid_action_reward = valid_action_reward
        self.more_turn = more_turn
        self.information_gain = information_gain
        self.format_reward = format_reward
        self.query_reward = query_reward
        self.query_score_calculator = query_score_calculator
        self.garbled_penalty = garbled_penalty
        self.query_doc_relevance = query_doc_relevance
        
    def __call__(self, data: DataProto, max_turns, global_steps=0):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        # ablation_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32) # ablation
        if self.eval_mode:
            reward_2_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
            reward_3_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        format_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        if self.enable_length_score or (self.overlong_buffer_cfg and self.overlong_buffer_cfg.enable):
            length_score_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        if self.valid_action_reward:
            valid_action_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        if self.information_gain:
            information_gain_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        # if self.query_reward:
        if True:
            good_query_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
            max_score_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        # if self.garbled_penalty and self.garbled_penalty.enable:
        if True:
            garbled_penalty_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        if self.query_doc_relevance and self.query_doc_relevance.enable:
            query_doc_relevance_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # all_scores = []

        already_print_data_sources = {}
        garbled_cnt = 0

        format_score_list = []
        valid_response_length_list = []
        answer_score_list = []
        uid_list = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            if not self.eval_mode:
                uid_list.append(data_item.non_tensor_batch['uid'])

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_length_list.append(valid_response_length)
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            # if self.garbled_penalty and self.garbled_penalty.enable and global_steps >= self.garbled_penalty.start_step:
            if not self.eval_mode:
                need_garbled_penalty = is_garbled(data_item.batch['ref_log_prob'], data_item.batch['old_log_probs'], data_item.batch['loss_mask'], self.garbled_penalty.ppl_threshold)
                if 'My previous action is invalid.' in sequences_str:
                    need_garbled_penalty = True
                garbled_penalty_tensor[i, valid_response_length - 1] = -1*self.garbled_penalty.penalty_factor if need_garbled_penalty else 0
            else:
                need_garbled_penalty = is_garbled_eval(data_item.batch['old_log_probs'], data_item.batch['loss_mask'], self.garbled_penalty.ppl_threshold)
                if 'My previous action is invalid.' in sequences_str:
                    need_garbled_penalty = True
                garbled_penalty_tensor[i, valid_response_length - 1] = 1 if need_garbled_penalty else 0
            if need_garbled_penalty:
                garbled_cnt += 1

            answer_score, structure_format_score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, reward_function=self.reward_function)
            format_score_list.append(structure_format_score)
            if isinstance(answer_score, tuple):
                reward_tensor[i, valid_response_length - 1] = answer_score[0]
                reward_2_tensor[i, valid_response_length - 1] = answer_score[1]
                reward_3_tensor[i, valid_response_length - 1] = answer_score[2]
            else:
                # if structure_format_score == 0:
                #     answer_score = 0
                answer_score = answer_score * structure_format_score
                if self.garbled_penalty and self.garbled_penalty.enable and global_steps >= self.garbled_penalty.start_step and need_garbled_penalty:
                    answer_score = 0
                reward_tensor[i, valid_response_length - 1] = answer_score
                answer_score_list.append(answer_score)
            format_reward_tensor[i, valid_response_length - 1] = structure_format_score

            if self.enable_length_score:
                generate_length = data_item.batch['generate_length'].sum()
                if structure_format_score == 1 and answer_score <= 0.6:
                    length_score = min(1, generate_length / (128 * (max_turns+1)))
                elif structure_format_score == 1 and answer_score > 0.6:
                    length_score = 1
                else:
                    length_score = 0

            if self.overlong_buffer_cfg and self.overlong_buffer_cfg.enable:
                generate_length = data_item.batch['generate_length'].tolist()
                penalties = []
                for generate_length_item in generate_length:
                    overlong_buffer_len = self.overlong_buffer_cfg.len
                    expected_len = self.overlong_buffer_cfg.max_res_len - overlong_buffer_len
                    exceed_len = generate_length_item - expected_len
                    penalties.append(max(-1, min(0, -exceed_len / overlong_buffer_len * self.overlong_buffer_cfg.penalty_factor)))
                overlong_penalty = np.array(penalties).min()
            
            if self.enable_length_score or (self.overlong_buffer_cfg and self.overlong_buffer_cfg.enable):
                final_length_score = length_score * 0.2 if overlong_penalty == 0 else overlong_penalty
                length_score_tensor[i, valid_response_length - 1] = final_length_score

            if self.valid_action_reward:
                valid_action_reward_tensor[i, valid_response_length - 1] = data.meta_info['valid_action_stats'][i] / (max_turns + 1)

            if self.information_gain:
                queries = get_query_ids(self.tokenizer, valid_response_ids)
                if structure_format_score == 1:
                    information_gain = information_gain_fn(queries)
                else:
                    information_gain = 0
                information_gain_tensor[i, valid_response_length - 1] = information_gain

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)
                    
        if self.query_doc_relevance and self.query_doc_relevance.enable:
            all_queries = []
            all_split_docs = []
            all_answers = []
            original_index = []
            if self.query_doc_relevance.doc_overlap_penalty == "segment":
                doc_overlap_rate = [[] for _ in range(len(data))]
            elif self.query_doc_relevance.doc_overlap_penalty == "strict":
                doc_overlap_rate = [0 for _ in range(len(data))]
            for i in range(len(data)):
                data_item = data[i]
                format_score = format_score_list[i]
                ground_truth = list(data_item.non_tensor_batch['golden_answers'])
                if ground_truth == ['yes'] or ground_truth == ['no']:
                    continue
                if format_score != 1:
                    continue
                if self.query_doc_relevance.low_score_only and answer_score_list[i] >= 0.7:
                    continue
                if self.query_doc_relevance.high_score_only and answer_score_list[i] < 1.0:
                    continue
                queries, doc_strs = get_query_doc_sequence(self.tokenizer, data_item.batch['responses'])
                queries = [query.strip() for query in queries]

                split_docs = []
                split_docs_copy = []
                for doc_str in doc_strs:
                    split_doc = re.split(r'Doc \d+\(Title:', doc_str)
                    split_doc = [{"str": f"(Title:{doc.strip()}"} for doc in split_doc if doc.strip()]
                    split_doc_copy = [doc["str"] for doc in split_doc]
                    split_docs.append(split_doc)
                    split_docs_copy.append(split_doc_copy)
                
                if self.query_doc_relevance.doc_overlap_penalty == "segment":
                    overlap_rate_this_rollout = []
                    for idx_i in range(len(split_docs_copy)):
                        final_overlap_rate = 0
                        for idx_j in range(idx_i):
                            common_elements = set(split_docs_copy[idx_i]) & set(split_docs_copy[idx_j])
                            overlap_rate = len(common_elements) / len(split_docs_copy[idx_i])
                            if len(common_elements) > 0:
                                final_overlap_rate = 1
                        overlap_rate_this_rollout.append(final_overlap_rate)
                    doc_overlap_rate[i] = overlap_rate_this_rollout
                elif self.query_doc_relevance.doc_overlap_penalty == "strict":
                    final_overlap_rate = 0
                    for idx_i in range(len(split_docs_copy)):
                        for idx_j in range(idx_i):
                            common_elements = set(split_docs_copy[idx_i]) & set(split_docs_copy[idx_j])
                            overlap_rate = len(common_elements) / len(split_docs_copy[idx_i])
                            if len(common_elements) > 0:
                                final_overlap_rate = 1
                    doc_overlap_rate[i] = final_overlap_rate

                if self.query_doc_relevance.use_que_ans:
                    question = data_item.non_tensor_batch['question']
                    queries = []
                    answers = []
                    for split_doc in split_docs:
                        queries.append(question)
                        answers.append(str(ground_truth))
                    
                all_queries.extend(queries)
                all_split_docs.extend(split_docs)
                all_answers.extend(answers)
                original_index.extend([i] * len(queries))
            
            assert len(all_queries) == len(all_split_docs)
            scores_qad = get_qad_scores(all_queries, all_split_docs, all_answers)['result']
            scores_ad = get_ad_scores(all_split_docs, all_answers)['result']
            scores_mean_qad = [[] for _ in range(len(data))]
            scores_mean_ad = [[] for _ in range(len(data))]
            for i in range(len(scores_qad)):
                if self.query_doc_relevance.desc == "mean":
                    score_this_query_qad = sum(scores_qad[i]) / len(scores_qad[i])
                    score_this_query_ad = sum(scores_ad[i]) / len(scores_ad[i])
                elif self.query_doc_relevance.desc == "max":
                    score_this_query_qad = max(scores_qad[i])
                    score_this_query_ad = max(scores_ad[i])
                else:
                    raise ValueError(f"Invalid query_doc_relevance.desc: {self.query_doc_relevance.desc}")
                # if score_this_query < 0.1:
                #     score_this_query = 1e-6
                scores_mean_qad[original_index[i]].append(score_this_query_qad)
                scores_mean_ad[original_index[i]].append(score_this_query_ad)
            for i in range(len(scores_mean_qad)):
                if len(scores_mean_qad[i]) == 0:
                    final_score = 0
                else:
                    # final_score = sum(scores_mean[i]) / len(scores_mean[i])
                    if self.query_doc_relevance.segment_reward:
                        segments = segment(data[i].batch['loss_mask'])
                        for query_idx in range(len(scores_mean_qad[i])):
                            if self.query_doc_relevance.split_mode == "half":
                                the_score = scores_mean_qad[i][query_idx] * 0.5 + scores_mean_ad[i][query_idx] * 0.5
                            elif self.query_doc_relevance.split_mode == "linear":
                                the_score = scores_mean_qad[i][query_idx] * (0.1 + 0.8*(1-query_idx/(max_turns-1))) + scores_mean_ad[i][query_idx] * (0.1 + 0.8*(query_idx/(max_turns-1)))
                            elif self.query_doc_relevance.split_mode == "max":
                                the_score = max(scores_mean_qad[i][query_idx], scores_mean_ad[i][query_idx])
                            elif self.query_doc_relevance.split_mode == "one":
                                the_score = 1
                            else:
                                raise ValueError(f"Invalid query_doc_relevance.split_mode: {self.query_doc_relevance.split_mode}")
                            if self.query_doc_relevance.doc_overlap_penalty == "segment":
                                query_doc_relevance_tensor[i, segments[query_idx][1]] = the_score if doc_overlap_rate[i][query_idx]==0 else 1e-6
                            else:
                                query_doc_relevance_tensor[i, segments[query_idx][1]] = the_score
                            
                    else:
                        if self.query_doc_relevance.split_mode == "half":
                            final_score = (sum(scores_mean_qad[i]) * 0.5 + sum(scores_mean_ad[i]) * 0.5) / max_turns
                        if self.query_doc_relevance.doc_overlap_penalty == "segment":
                            final_score = 0
                            for query_idx in range(len(scores_mean_qad[i])):
                                if doc_overlap_rate[i][query_idx] == 0:
                                    final_score += scores_mean_qad[i][query_idx] * 0.5 + scores_mean_ad[i][query_idx] * 0.5
                                # else:
                                #     final_score -= 0.5
                            final_score = max(final_score / max_turns, 0)
                        elif self.query_doc_relevance.doc_overlap_penalty == "strict":
                            if doc_overlap_rate[i] == 1:
                                final_score = 0
                        # if self.query_doc_relevance.threshold:
                        #     final_score = max((final_score - self.query_doc_relevance.threshold), 0) / (1 - self.query_doc_relevance.threshold)
                        # if self.query_doc_relevance.low_score_only:
                        #     final_score = (-3 / 7 * answer_score_list[i] + 0.3) * final_score
                        query_doc_relevance_tensor[i, valid_response_length_list[i] - 1] = final_score
            
        # if self.query_reward: # ablation中注释掉
        if not self.eval_mode: # ablation
            # uid2docs, uid2maxscore = get_silver_docs(uid_list)
            uid2docs = {}
            uid2maxscore = {}
            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem
                uid = data_item.non_tensor_batch['uid']
                if format_score_list[i] == 1 and answer_score_list[i] >= 0.7:
                # if format_score_list[i] == 1:
                    queries, doc_strs, _ = get_query_doc_sequence(self.tokenizer, data_item.batch['responses'])
                    split_docs = set()
                    for doc_str in doc_strs:
                        split_doc = re.split(r'Doc \d+\(Title:', doc_str)
                        split_doc = [f"(Title:{doc.strip()}" for doc in split_doc if doc.strip()]
                        for doc in split_doc:
                            split_docs.add(doc)
                    if uid not in uid2maxscore:
                        uid2maxscore[uid] = answer_score_list[i]
                        uid2docs[uid] = {k:1 for k in split_docs}
                    else:
                        if answer_score_list[i] > uid2maxscore[uid]:
                            uid2maxscore[uid] = answer_score_list[i]
                            uid2docs[uid] = {k:1 for k in split_docs}
                        elif answer_score_list[i] == uid2maxscore[uid]:
                            for key in split_docs:
                                if key in uid2docs[uid]:
                                    uid2docs[uid][key] += 1
                                else:
                                    uid2docs[uid][key] = 1

            uid2max_doc_score = {}
            for uid in uid2docs:
                uid2max_doc_score[uid] = max(uid2docs[uid].values())

            for i in range(len(data)):
                data_item = data[i]
                uid = data_item.non_tensor_batch['uid']
                if format_score_list[i] == 1 and uid in uid2docs:
                    queries, doc_strs, thinks = get_query_doc_sequence(self.tokenizer, data_item.batch['responses'])
                    segments = segment(data_item.batch['loss_mask'])
                    assert len(segments)-1 == len(queries)
                    doc_set = set()
                    query_set = set()
                    all_split_docs = []
                    all_think = []
                    good_query_cnt = 0
                    for query_idx in range(len(queries)):
                        split_doc = re.split(r'Doc \d+\(Title:', doc_strs[query_idx])
                        split_doc = [f"(Title:{doc.strip()}" for doc in split_doc if doc.strip()]
                        all_split_docs.append(split_doc)
                        is_good_query = 0
                        good_doc_scores = []
                        for doc in split_doc:
                            if doc in uid2docs[uid] and doc not in doc_set:
                                is_good_query += 1
                                good_doc_scores.append(uid2docs[uid][doc]/uid2max_doc_score[uid])
                            doc_set.add(doc)
                        num_docs = len(split_doc)
                        for hist_think in all_think:
                            if word_edit_similarity(thinks[query_idx], hist_think) > 0.5:
                                is_good_query = 0
                        all_think.append(thinks[query_idx])
                        for docs_front in all_split_docs[:query_idx]:
                            common_elements = set(docs_front) & set(all_split_docs[query_idx])
                            if len(common_elements) > 1:
                                is_good_query = 0
                        if is_good_query:
                            query_set.add(queries[query_idx])
                            if self.query_score_calculator == "max":
                                good_query_tensor[i, segments[query_idx][1]] = max(good_doc_scores)
                            elif self.query_score_calculator == "mean":
                                good_query_tensor[i, segments[query_idx][1]] = sum(good_doc_scores) / num_docs
                            elif self.query_score_calculator == "old":
                                good_query_tensor[i, segments[query_idx][1]] = is_good_query / num_docs
                            else:
                                raise ValueError(f"Invalid query_score_calculator: {self.query_score_calculator}")
                        else:
                            good_query_tensor[i, segments[query_idx][1]] = 1e-6
                    max_score_tensor[i, valid_response_length_list[i] - 1] = uid2maxscore[uid]
                
            # for i in range(len(data)):
            #     data_item = data[i]
            #     uid = data_item.non_tensor_batch['uid']
            #     if format_score_list[i] == 1:
            #         queries, doc_strs = get_query_doc_sequence(self.tokenizer, data_item.batch['responses'])
            #         segments = segment(data_item.batch['loss_mask'])
            #         for query_idx in range(len(queries)):
            #             ablation_tensor[i, segments[query_idx][1]] = 1

        print(f'[debug] garbled_cnt: {garbled_cnt} in {len(data)}')
        return {
            'reward_tensor': reward_tensor,
            # 'ablation_tensor': ablation_tensor, # ablation
            **({'format_reward_tensor': format_reward_tensor} if self.format_reward else {}),
            **({'length_score_tensor': length_score_tensor} if (self.overlong_buffer_cfg and self.overlong_buffer_cfg.enable) or self.enable_length_score else {}),
            **({'valid_action_reward_tensor': valid_action_reward_tensor} if self.valid_action_reward else {}),
            **({'reward_2_tensor': reward_2_tensor,
                'reward_3_tensor': reward_3_tensor} if isinstance(answer_score, tuple) else {}),
            **({'information_gain_tensor': information_gain_tensor} if self.information_gain else {}),
            # **({'good_query_tensor': good_query_tensor,
            #     'max_score_tensor': max_score_tensor} if self.query_reward else {}),
            **({'good_query_tensor': good_query_tensor,
                'max_score_tensor': max_score_tensor} if not self.eval_mode else {}),
            # **({'garbled_penalty_tensor': garbled_penalty_tensor} if self.garbled_penalty and self.garbled_penalty.enable else {}),
            **({'garbled_penalty_tensor': garbled_penalty_tensor}),
            **({'query_doc_relevance_tensor': query_doc_relevance_tensor} if self.query_doc_relevance and self.query_doc_relevance.enable else {}),
        }

import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})
    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, eval_mode=False, 
                                reward_function=config.trainer.reward_function, 
                                enable_length_score=config.reward_model.enable_length_score,
                                overlong_buffer_cfg=config.reward_model.overlong_buffer_cfg,
                                valid_action_reward=config.reward_model.valid_action_reward,
                                more_turn=config.reward_model.more_turn,
                                information_gain=config.reward_model.information_gain,
                                format_reward=config.reward_model.format_reward,
                                query_reward=config.reward_model.query_reward,
                                query_score_calculator=config.reward_model.query_score_calculator,
                                garbled_penalty=config.reward_model.garbled_penalty,
                                query_doc_relevance=config.reward_model.query_doc_relevance)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, eval_mode=True, 
                                    reward_function=config.trainer.reward_function,
                                    enable_length_score=False,
                                    overlong_buffer_cfg=None,
                                    valid_action_reward=False,
                                    more_turn=False,
                                    information_gain=False,
                                    format_reward=True,
                                    query_reward=False,
                                    query_score_calculator=config.reward_model.query_score_calculator,
                                    garbled_penalty=config.reward_model.garbled_penalty,
                                    query_doc_relevance=False)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn
                            )
    trainer.init_workers()
    trainer.fit()

def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True):
    """Create a dataset.

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    # Check if a custom dataset class is specified in the data configuration
    # and if the path to the custom class is provided
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        # Dynamically load the custom dataset class
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        # Verify that the custom dataset class inherits from torch.utils.data.Dataset
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(
                f"The custom dataset class '{data_config.custom_cls.name}' from "
                f"'{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset"
            )
    elif "datagen" in data_config and data_config.datagen.get("path", None) is not None and is_train:
        # If a data generation strategy is specified, use the DynamicGenDataset class
        from verl.utils.dataset.dynamicgen_dataset import DynamicGenDataset

        dataset_cls = DynamicGenDataset
        print("Using DynamicGenDataset for data generation.")

    else:
        # Use the default RLHFDataset class if no custom class is specified
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    # Instantiate the dataset using the determined dataset class
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    if data_config.sampler is not None and data_config.sampler.get("class_path", None) is not None:
        curriculum_class = load_extern_type(
            data_config.sampler.class_path,
            data_config.sampler.class_name,
        )
        sampler = curriculum_class(
            data_source=dataset,
            data_config=data_config,
        )
        assert isinstance(sampler, AbstractSampler)
        assert data_config.get("dataloader_num_workers", 8) == 0, (
            "If using curriculum, num_workers must be 0 to prevent data caching. "
            "If the dataloader caches data before the batch is done the "
            "curriculum sampler won't have the opportunity to reorder it. "
        )

    # Use a sampler to facilitate checkpoint resumption.
    # If shuffling is enabled in the data configuration, create a random sampler.
    elif data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        # If shuffling is disabled, use a sequential sampler to iterate through the dataset in order.
        sampler = SequentialSampler(data_source=dataset)

    return sampler

if __name__ == '__main__':
    main()
