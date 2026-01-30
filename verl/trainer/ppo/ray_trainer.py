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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
import shutil
import re
import json
from collections import defaultdict
from tqdm import tqdm
import Levenshtein
import random
import math

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.reward_score.qa_em_format import is_valid_sequence
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
import verl.utils.torch_functional as verl_F

import re
from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['info_mask'] if 'info_mask' in data.batch else data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto,
                    tokenizer,
                    adv_estimator, 
                    neutral, 
                    less_negative, 
                    mask_template_token,
                    model_name,
                    acceptable_mask, 
                    query_reward_percentage, 
                    low_mean, 
                    use_global_std, 
                    norm, 
                    segment_adv_type, 
                    gamma=1.0, 
                    lam=1.0, 
                    segment_adv=False):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        loss_mask = data.batch['loss_mask']
        if segment_adv:
            if 'token_level_query_doc_relevance_scores' in data.batch:
                token_level_query_doc_relevance_scores = data.batch['token_level_query_doc_relevance_scores']
                advantages, returns = core_algos.compute_grpo_outcome_turngroup_advantage(token_level_rewards=token_level_rewards,
                                                                            token_level_query_scores=token_level_query_doc_relevance_scores,
                                                                            eos_mask=loss_mask,
                                                                            index=index)
            elif 'token_level_good_query' in data.batch:
                token_level_good_query = data.batch['token_level_good_query']
                # token_level_good_query = data.batch['ablation_tensor']
                max_score_tensor = data.batch['max_score_tensor']
                if segment_adv_type=='turngroup':
                    advantages, advantages_query, returns = core_algos.compute_grpo_outcome_turngroup_advantage(token_level_rewards=token_level_rewards,
                                                                                token_level_query_scores=token_level_good_query,
                                                                                less_negative=less_negative,
                                                                                eos_mask=loss_mask,
                                                                                index=index)
                    data.batch['advantages_query'] = advantages_query
                elif segment_adv_type=='querygroup':
                    advantages, returns = core_algos.compute_grpo_outcome_querygroup_advantage(token_level_rewards=token_level_rewards,
                                                                                token_level_query_scores=token_level_good_query,
                                                                                eos_mask=loss_mask,
                                                                                index=index)
                elif segment_adv_type=='goodquerypositive':
                    advantages, returns = core_algos.compute_grpo_outcome_goodquerypositive_segment_advantage(token_level_rewards=token_level_rewards,
                                                                                token_level_good_query=token_level_good_query,
                                                                                max_score_tensor=max_score_tensor,
                                                                                less_negative=less_negative,
                                                                                eos_mask=loss_mask,
                                                                                index=index)
                elif segment_adv_type=='softpenalty':
                    advantages, returns = core_algos.compute_grpo_outcome_softpenalty_advantage(token_level_rewards=token_level_rewards,
                                                                                responses=data.batch['responses'],
                                                                                tokenizer=tokenizer,
                                                                                mask_template_token=mask_template_token,
                                                                                model_name=model_name,
                                                                                token_level_good_query=token_level_good_query,
                                                                                less_negative=less_negative,
                                                                                eos_mask=loss_mask,
                                                                                index=index)
                elif segment_adv_type=='softpenalty_turngroup':
                    advantages, returns = core_algos.compute_grpo_outcome_softpenalty_turngroup_advantage(token_level_rewards=token_level_rewards,
                                                                                token_level_query_scores=token_level_good_query,
                                                                                max_score_tensor=max_score_tensor,
                                                                                less_negative=less_negative,
                                                                                eos_mask=loss_mask,
                                                                                index=index)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
                
        else:
            advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                            responses=data.batch['responses'],
                                                                            tokenizer=tokenizer,
                                                                            mask_template_token=mask_template_token,
                                                                            neutral=neutral,
                                                                            less_negative=less_negative,
                                                                            acceptable_mask=acceptable_mask,
                                                                            use_global_std=use_global_std,
                                                                            eos_mask=loss_mask,
                                                                            index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data

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

def get_all_ppl(log_probs, mask):
    return torch.exp(-log_probs[mask.bool()].mean())

def is_garbled(log_prob1, log_prob2, mask, ppl_threshold):
    segments = segment(mask)
    max_seg_ppl = 0
    for start, end in segments:
        seg_ppl1 = get_ppl(log_prob1[start:end]).item()
        seg_ppl2 = get_ppl(log_prob2[start:end]).item()
        max_seg_ppl = max(max_seg_ppl, seg_ppl1, seg_ppl2)
    return max_seg_ppl >= ppl_threshold

def get_rollout(data: DataProto, tokenizer):
    uid2data = {}
    garbled_cnt = 0
    for idx in range(len(data)):
        data_item = data[idx]
        uid = data_item.non_tensor_batch['uid']
        if uid not in uid2data:
            uid2data[uid] = {
                'data_source': data_item.non_tensor_batch['data_source'],
                'question': data_item.non_tensor_batch['question'],
                'ground_truth': list(data_item.non_tensor_batch['golden_answers']),
                'correct_cnt': 0,
                'all_ans_score_same': True,
                'responses': [],
                'largest_ans_score': -1,
                'smallest_ans_score': 99,
                'have_garbled': False,
            }
        response_str = tokenizer.decode(data_item.batch['responses'],skip_special_tokens=True)
        # advantage = data_item.batch['advantages'].max().item()
        reward = data_item.batch['token_level_rewards'].sum(dim=-1).item()
        ans_score = data_item.batch['token_level_scores'].sum(dim=-1).item()
        
        loss_mask = data_item.batch['loss_mask']
        segments = segment(loss_mask)
        seg_responses = []
        seg_old_ppl = []
        seg_ref_ppl = []
        seg_entropy = []
        seg_old_probs = []
        seg_old_log_probs = []
        for start, end in segments:
            seg_responses.append(tokenizer.decode(data_item.batch['responses'][start:end]))
            seg_old_ppl.append(get_ppl(data_item.batch['old_log_probs'][start:end]).item())
            seg_ref_ppl.append(get_ppl(data_item.batch['ref_log_prob'][start:end]).item())
            seg_entropy.append(data_item.batch['entropy'][start:end].mean().item())
            seg_old_log_probs.append(data_item.batch['old_log_probs'][start:end].mean().item())
            seg_old_probs.append(torch.exp(data_item.batch['old_log_probs'][start:end]).mean().item())

        response_info_dict = {
            # 'advantage': advantage,
            'reward': reward,
            'ans_score': ans_score,
            **({'format_score': data_item.batch['token_level_format_scores'].sum(dim=-1).item()} if 'token_level_format_scores' in data_item.batch else {}),
            **({'garbled_penalty': data_item.batch['token_level_garbled_penalty_scores'].sum(dim=-1).item()}),
            'seg_responses': seg_responses,
            'seg_old_ppl': seg_old_ppl,
            'seg_ref_ppl': seg_ref_ppl,
            'seg_entropy': seg_entropy,
            'seg_old_probs': seg_old_probs,
            'seg_old_log_probs': seg_old_log_probs,
            'old_ppl': get_all_ppl(data_item.batch['old_log_probs'], loss_mask).item(),
            'ref_ppl': get_all_ppl(data_item.batch['ref_log_prob'], loss_mask).item(),
            'entropy': data_item.batch['entropy'][loss_mask.bool()].mean().item(),
            'old_log_probs': data_item.batch['old_log_probs'][loss_mask.bool()].mean().item(),
            'prob': torch.exp(data_item.batch['old_log_probs'][loss_mask.bool()]).mean().item(),
            'response': response_str,
        }
        uid2data[uid]['responses'].append(response_info_dict)
        if ans_score != uid2data[uid]['responses'][0]['ans_score']:
            uid2data[uid]['all_ans_score_same'] = False
        if ans_score == 1.0:
        # if ans_score >= 0.7:
            uid2data[uid]['correct_cnt'] += 1
        if ans_score > uid2data[uid]['largest_ans_score']:
            uid2data[uid]['largest_ans_score'] = ans_score
        if ans_score < uid2data[uid]['smallest_ans_score']:
            uid2data[uid]['smallest_ans_score'] = ans_score
        if 'token_level_garbled_penalty_scores' in data_item.batch and \
        data_item.batch['token_level_garbled_penalty_scores'].sum(dim=-1).item() < 0:
            garbled_cnt += 1
            uid2data[uid]['have_garbled'] = True

    least_one_correct = sum([1 if data['correct_cnt'] > 0 else 0 for data in uid2data.values()])
    LOC_correct_rate = (sum([uid2data[uid]['correct_cnt'] for uid in uid2data.keys()]) / least_one_correct) if least_one_correct > 0 else 0
    largest_ans_score = sum([uid2data[uid]['largest_ans_score'] for uid in uid2data.keys()]) / len(uid2data)

    all_incorrect_uid = []
    cnt = 0
    for uid in uid2data.keys():
        if uid2data[uid]['correct_cnt'] == 0:
            all_incorrect_uid.append(uid)
            if uid2data[uid]['largest_ans_score'] < 0.1:
                cnt += 1
    print(f'cnt: {cnt}, len(all_incorrect_uid): {len(all_incorrect_uid)}, {cnt/len(all_incorrect_uid)}')

    return uid2data, least_one_correct/len(uid2data), LOC_correct_rate, garbled_cnt/len(data), largest_ans_score

def get_easy_hard_uid(data: DataProto):
    uid2maxscore = {}
    for idx in range(len(data)):
        data_item = data[idx]
        uid = data_item.non_tensor_batch['uid']
        ans_score = data_item.batch['token_level_scores'].sum(dim=-1).item()
        if uid not in uid2maxscore:
            uid2maxscore[uid] = ans_score
        else:
            uid2maxscore[uid] = max(uid2maxscore[uid], ans_score)
    # uid2correct_num = {}
    # for idx in range(len(data)):
    #     data_item = data[idx]
    #     uid = data_item.non_tensor_batch['uid']
    #     if uid not in uid2correct_num:
    #         uid2correct_num[uid] = 0
    #     ans_score = data_item.batch['token_level_scores'].sum(dim=-1).item()
    #     if ans_score >= 0.7:
    #         uid2correct_num[uid] += 1
    hard_uid = []
    easy_uid = []
    for uid in uid2maxscore.keys():
        if uid2maxscore[uid] < 0.7:
            hard_uid.append(uid)
        else:
            easy_uid.append(uid)
    return hard_uid, easy_uid

def uid_to_data_dict(uid_list, batch_dict, hard_add_times=None):
    data_dict = {}
    index_list = []
    for index, item in enumerate(batch_dict['index']):
        if item in uid_list:
            if hard_add_times!=None and hard_add_times[item] >= 10:
                continue
            index_list.append(index)
    for key in batch_dict.keys():
        data_dict[key] = batch_dict[key][index_list]
    return data_dict

def data_dict_merge(data_dict, new_data_dict):
    if len(data_dict) == 0:
        return new_data_dict
    for key in data_dict.keys():
        if type(data_dict[key]) == torch.Tensor:
            data_dict[key] = torch.cat([data_dict[key], new_data_dict[key]], dim=0)
        elif type(data_dict[key]) == np.ndarray:
            data_dict[key] = np.concatenate([data_dict[key], new_data_dict[key]], axis=0)
        else:
            raise ValueError(f'key {key} is not supported')
    return data_dict

def data_dict_random_select(data_dict, num, last_train_step=None, global_steps=None):
    if last_train_step!=None:
        candidate_uid_list = []
        for item in data_dict['index']:
            if last_train_step[item]==0 or global_steps-last_train_step[item] >= 5:
                candidate_uid_list.append(item)
                last_train_step[item] = global_steps
        candidate_index_list = [i for i in range(len(data_dict['index'])) if data_dict['index'][i] in candidate_uid_list]
        assert len(candidate_index_list) >= num
        select_index_list = random.sample(candidate_index_list, num)
    else:
        select_index_list = random.sample(range(len(data_dict['index'])), num)
    left_index_list = [i for i in range(len(data_dict['index'])) if i not in select_index_list]
    select_data_dict = {}
    left_data_dict = {}
    for key in data_dict.keys():
        select_data_dict[key] = data_dict[key][select_index_list]
        left_data_dict[key] = data_dict[key][left_index_list]
    if last_train_step!=None:
        return select_data_dict, left_data_dict, last_train_step
    return select_data_dict, left_data_dict

def update_times(hard_add_times, hard_uid):
    for uid in hard_uid:
        hard_add_times[uid] += 1
    return hard_add_times

def get_all_equal_mask(data: DataProto, uid2data):
    all_equal_mask = torch.zeros(len(data), dtype=torch.int)
    for idx in range(len(data)):
        data_item = data[idx]
        uid = data_item.non_tensor_batch['uid']
        if uid2data[uid]['correct_cnt'] == 0:
            all_equal_mask[idx] = 0
        else:
            all_equal_mask[idx] = 1
    return all_equal_mask
        

def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        if 'max_pg_loss' in key or 'max_pg_loss_pos_neg_ratio' in key:
            metrics[key] = np.max(val)
        else:
            metrics[key] = np.mean(val)
    return metrics

def move_pth(save_path, global_steps):
    step_save_path = f'{save_path}/step{global_steps}'
    source_folder = f'{save_path}'
    if not os.path.exists(step_save_path):
        os.makedirs(step_save_path)
    for filename in os.listdir(step_save_path):
        file_path = os.path.join(step_save_path, filename)
        os.remove(file_path)
    for filename in os.listdir(source_folder):
        if filename.endswith(".pth"):
            source_path = os.path.join(source_folder, filename)
            target_path = os.path.join(step_save_path, filename)
            shutil.move(source_path, target_path)

def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )

def length_alignment(batchs, tokenizer):
    # Left padding
    only_left_key = 'prompts'
    need_left_token_keys = ['prompts', 'input_ids']
    need_left_bool_keys = ['attention_mask', 'info_mask', 'position_ids']
    max_length = max([batch.batch[only_left_key].shape[-1] for batch in batchs])
    for batch in batchs:
        pad_num = max_length - batch.batch[only_left_key].shape[-1]
        for key in need_left_token_keys:
            batch.batch[key] = torch.nn.functional.pad(batch.batch[key], (pad_num, 0), value=tokenizer.pad_token_id)
        for key in need_left_bool_keys:
            batch.batch[key] = torch.nn.functional.pad(batch.batch[key], (pad_num, 0), value=0)


    # Right padding
    only_right_key = 'responses'
    need_right_token_keys = ['responses', 'responses_with_info_mask', 'input_ids']
    need_right_bool_keys = ['attention_mask', 'info_mask', 'position_ids', 'loss_mask']
    need_right_float_keys = [
        'old_log_probs', 'ref_log_prob', 'entropy', 'token_level_rewards', 
        'token_level_format_scores', 'token_level_scores', 'token_level_good_query', 'max_score_tensor']
    max_length = max([batch.batch[only_right_key].shape[-1] for batch in batchs])
    for batch in batchs:
        pad_num = max_length - batch.batch[only_right_key].shape[-1]
        for key in need_right_token_keys:
            batch.batch[key] = torch.nn.functional.pad(batch.batch[key], (0, pad_num), value=tokenizer.pad_token_id)
        for key in need_right_bool_keys:
            batch.batch[key] = torch.nn.functional.pad(batch.batch[key], (0, pad_num), value=0)
        for key in need_right_float_keys:
            if key in batch.batch:
                batch.batch[key] = torch.nn.functional.pad(batch.batch[key], (0, pad_num), value=0.0)
    return DataProto.concat(batchs)

def get_query_sequence(tokenizer, responses):
    responses = tokenizer.decode(responses, skip_special_tokens=True)
    matches = re.findall(r"<search>(.*?)</search>", responses, re.DOTALL)
    return ".".join(matches) if matches else ""

def edit_distance(s1, s2):
    return Levenshtein.distance(s1.numpy(), s2.numpy())

def select_rollout(data, loss_mask, num_select, selection_method, tokenizer, embedding_model):
    queries = []
    id_map = {}
    remain_idx = []
    cnt = 0
    for idx in range(len(data)):
        data_item = data[idx]

        prompt_ids = data_item['prompts']
        prompt_length = prompt_ids.shape[-1]

        valid_prompt_length = data_item['attention_mask'][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item['responses']
        valid_response_length = data_item['attention_mask'][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        sequences = torch.cat((valid_prompt_ids, valid_response_ids))
        sequences_str = tokenizer.decode(sequences)

        is_valid_format, _ = is_valid_sequence(sequences_str)
        if not is_valid_format:
            remain_idx.append(idx)
            continue

        if selection_method == 'edit_distance':
            segments = segment(loss_mask[idx])
            query_token_ids = []
            for start, end in segments:
                query_token_ids.append(data_item['responses'][start:end])
            queries.append(torch.cat(query_token_ids))
            id_map[cnt] = idx
            cnt += 1
        elif selection_method == 'embedding_distance':
            queries_str = get_query_sequence(tokenizer, valid_response_ids)
            if queries_str=="":
                remain_idx.append(idx)
                continue
            queries.append(queries_str)
            id_map[cnt] = idx
            cnt += 1
        else:
            raise NotImplementedError
    
    # 如果满足format的样本数量小于等于num_select，返回所有满足format的样本，不足的样本随机选择
    if len(queries) <= num_select:
        res_idx = list(id_map.values())
        num_random_idx = num_select - len(queries)
        random_idx = random.sample(remain_idx, num_random_idx)
        res_idx.extend(random_idx)
    else:
        if selection_method == 'edit_distance':
            distance = [[0 for _ in range(len(queries))] for _ in range(len(queries))]
            for i in range(len(queries)):
                for j in range(len(queries)):
                    distance[i][j] = distance[j][i] = edit_distance(queries[i], queries[j])
        elif selection_method == 'embedding_distance':
            query_embeddings = embedding_model.encode(queries)
            distance = cosine_distances(query_embeddings)
        else:
            raise NotImplementedError
        
        # 选择间隔最远的num_select个样本
        selected = [random.randint(0, len(queries) - 1)]
        while len(selected) < num_select:
            candidates = [i for i in range(len(queries)) if i not in selected]
            # 计算每个候选与已选集合的最小距离
            min_dists = [min(distance[i][j] for j in selected) for i in candidates]
            # 选最远的
            best = candidates[np.argmax(min_dists)]
            selected.append(best)
        res_idx = [id_map[i] for i in selected]

    return res_idx

def get_silver_rollouts(question_idx):
    import requests
    payload = {
        "question_idx": question_idx,
        "num_rollouts": 2,
    }
    uid2rollouts, uid2maxscore = requests.post("http://127.0.0.1:6000/get_silver_rollouts", json=payload).json()
    uid2rollouts = {int(k): v for k, v in uid2rollouts.items()}
    uid2maxscore = {int(k): v for k, v in uid2maxscore.items()}
    return uid2rollouts, uid2maxscore

def batch_replacement(batch, uid2rollouts, tokenizer):
    from torch.nn.utils.rnn import pad_sequence
    def create_attention_mask(input_ids: torch.Tensor) -> torch.Tensor:
        """Create attention mask from input ids."""
        return torch.where(input_ids != tokenizer.pad_token_id, 1, 0)
    
    def create_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
        """Create position ids from attention mask."""
        return (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask
        
    uid2rollout_idx = defaultdict(int)
    all_response_ids = []
    all_response_with_info_mask_ids = []
    for idx in range(len(batch.batch)):
        uid = batch[idx].non_tensor_batch['uid']
        silver_rollout = uid2rollouts[uid][uid2rollout_idx[uid]]
        uid2rollout_idx[uid] += 1

        llm_generated_str = silver_rollout['llm_generated_str']
        doc_str = silver_rollout['doc_str']
        
        response_ids = torch.tensor([], dtype=torch.int64)
        response_with_info_mask_ids = torch.tensor([], dtype=torch.int64)
        for query_idx in range(len(doc_str)):
            think_query_ids = tokenizer(llm_generated_str[query_idx], return_tensors="pt")["input_ids"][0]
            doc_ids = tokenizer(doc_str[query_idx], return_tensors="pt")["input_ids"][0]
            info_mask = torch.full(doc_ids.size(), tokenizer.pad_token_id, dtype=doc_ids.dtype)
            response_ids = torch.cat([response_ids, think_query_ids, doc_ids], dim=-1)
            response_with_info_mask_ids = torch.cat([response_with_info_mask_ids, think_query_ids, info_mask], dim=-1)
        think_answer_ids = tokenizer(llm_generated_str[-1], return_tensors="pt")["input_ids"][0]
        response_ids = torch.cat([response_ids, think_answer_ids], dim=-1)
        response_with_info_mask_ids = torch.cat([response_with_info_mask_ids, think_answer_ids], dim=-1)
        all_response_ids.append(response_ids)
        all_response_with_info_mask_ids.append(response_with_info_mask_ids)

    all_response_ids = pad_sequence(all_response_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    all_response_with_info_mask_ids = pad_sequence(all_response_with_info_mask_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    
    batch.batch['input_ids'] = torch.cat([batch.batch['prompts'], all_response_ids], dim=1)
    batch.batch['attention_mask'] = create_attention_mask(batch.batch['input_ids'])
    batch.batch['responses'] = all_response_ids
    batch.batch['responses_with_info_mask'] = all_response_with_info_mask_ids
    batch.batch['info_mask'] = create_attention_mask(torch.cat([batch.batch['prompts'], all_response_with_info_mask_ids], dim=1))
    batch.batch['position_ids'] = create_position_ids(batch.batch['attention_mask'])
    batch.batch['void_turn_mask'] = torch.zeros_like(batch.batch['void_turn_mask'])
    return batch

def adv_metrics(batch):
    max_response_length = batch.batch['responses'].shape[-1]
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()
    advantages = batch.batch['advantages']
    valid_adv = torch.masked_select(advantages, response_mask)
    if 'advantages_query' in batch.batch:
        advantages_query = batch.batch['advantages_query']
        valid_adv_query = torch.masked_select(advantages_query, response_mask)
    return {
        'advantages/mean': torch.mean(valid_adv).detach().item(),
        'advantages/max': torch.max(valid_adv).detach().item(),
        'advantages/min': torch.min(valid_adv).detach().item(),
        **({'advantages_query/mean': torch.mean(valid_adv_query).detach().item(),
            'advantages_query/max': torch.max(valid_adv_query).detach().item(),
            'advantages_query/min': torch.min(valid_adv_query).detach().item(),
        } if 'advantages_query' in batch.batch else {}),
    }

def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    if 'token_level_format_scores' in batch.batch:
        sequence_formate_score = batch.batch['token_level_format_scores'].sum(-1)
    if 'token_level_length_scores' in batch.batch:
        sequence_length_score = batch.batch['token_level_length_scores'].sum(-1)
    if 'token_level_valid_action_scores' in batch.batch:
        sequence_valid_action_score = batch.batch['token_level_valid_action_scores'].sum(-1)
    if 'token_level_information_gain_scores' in batch.batch:
        sequence_information_gain_score = batch.batch['token_level_information_gain_scores'].sum(-1)
    if 'token_level_good_query' in batch.batch:
        sequence_query_score = batch.batch['token_level_good_query'].sum(-1)
    if 'token_level_garbled_penalty_scores' in batch.batch:
        sequence_garbled_penalty_score = batch.batch['token_level_garbled_penalty_scores'].sum(-1)
    if 'token_level_query_doc_relevance_scores' in batch.batch:
        sequence_query_doc_relevance_score = batch.batch['token_level_query_doc_relevance_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    if 'advantages' in batch.batch:
        advantages = batch.batch['advantages']
    if 'returns' in batch.batch:
        returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    if 'advantages' in batch.batch:
        valid_adv = torch.masked_select(advantages, response_mask)
    if 'returns' in batch.batch:
        valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        **({
            # values
            'critic/formate_score/mean': torch.mean(sequence_formate_score).detach().item(),
            'critic/formate_score/max': torch.max(sequence_formate_score).detach().item(),
            'critic/formate_score/min': torch.min(sequence_formate_score).detach().item(),
        } if 'token_level_format_scores' in batch.batch else {}),
        **({
            # values
            'critic/length_score/mean': torch.mean(sequence_length_score).detach().item(),
            'critic/length_score/max': torch.max(sequence_length_score).detach().item(),
            'critic/length_score/min': torch.min(sequence_length_score).detach().item(),
        } if 'token_level_length_scores' in batch.batch else {}),
        **({
            # values
            'critic/sequence_valid_action_score/mean': torch.mean(sequence_valid_action_score).detach().item(),
            'critic/sequence_valid_action_score/max': torch.max(sequence_valid_action_score).detach().item(),
            'critic/sequence_valid_action_score/min': torch.min(sequence_valid_action_score).detach().item(),
        } if 'token_level_valid_action_scores' in batch.batch else {}),
        **({
            # values
            'critic/sequence_information_gain_score/mean': torch.mean(sequence_information_gain_score).detach().item(),
            'critic/sequence_information_gain_score/max': torch.max(sequence_information_gain_score).detach().item(),
            'critic/sequence_information_gain_score/min': torch.min(sequence_information_gain_score).detach().item(),
        } if 'token_level_information_gain_scores' in batch.batch else {}),
        **({
            # values
            'critic/sequence_query_score/mean': torch.mean(sequence_query_score).detach().item(),
            'critic/sequence_query_score/max': torch.max(sequence_query_score).detach().item(),
            'critic/sequence_query_score/min': torch.min(sequence_query_score).detach().item(),
        } if 'token_level_good_query' in batch.batch else {}),
        **({
            # values
            'critic/sequence_garbled_penalty_score/mean': torch.mean(sequence_garbled_penalty_score).detach().item(),
            'critic/sequence_garbled_penalty_score/max': torch.max(sequence_garbled_penalty_score).detach().item(),
            'critic/sequence_garbled_penalty_score/min': torch.min(sequence_garbled_penalty_score).detach().item(),
        } if 'token_level_garbled_penalty_scores' in batch.batch else {}),
        **({
            # values
            'critic/sequence_query_doc_relevance_score/mean': torch.mean(sequence_query_doc_relevance_score).detach().item(),
            'critic/sequence_query_doc_relevance_score/max': torch.max(sequence_query_doc_relevance_score).detach().item(),
            'critic/sequence_query_doc_relevance_score/min': torch.min(sequence_query_doc_relevance_score).detach().item(),
        } if 'token_level_query_doc_relevance_scores' in batch.batch else {}),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        **({'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        } if 'advantages' in batch.batch else {}),
        # returns
        **({'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        } if 'returns' in batch.batch else {}),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    # metrics for actions
    if 'turns_stats' in batch.meta_info:
        metrics['env/number_of_actions/mean'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).mean())
        metrics['env/number_of_actions/max'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).max())
        metrics['env/number_of_actions/min'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).min())
    if 'active_mask' in batch.meta_info:
        metrics['env/finish_ratio'] = 1 - float(np.array(batch.meta_info['active_mask'], dtype=np.int16).mean())
    if 'valid_action_stats' in batch.meta_info:
        metrics['env/number_of_valid_action'] = float(np.array(batch.meta_info['valid_action_stats'], dtype=np.int16).mean())
        metrics['env/ratio_of_valid_action'] = float((np.array(batch.meta_info['valid_action_stats'], dtype=np.int16) / np.array(batch.meta_info['turns_stats'], dtype=np.int16)).mean())
    if 'valid_search_stats' in batch.meta_info:
        metrics['env/number_of_valid_search'] = float(np.array(batch.meta_info['valid_search_stats'], dtype=np.int16).mean())
    if 'de_facto_search_stats' in batch.meta_info:
        metrics['env/number_of_de_facto_search'] = float(np.array(batch.meta_info['de_facto_search_stats'], dtype=np.int16).mean())
    if "void_turn_mask" in batch.meta_info:
        metrics["env/void_turn_ratio"] = 1 - float(
            np.array(batch.meta_info["void_turn_mask"], dtype=np.int16).mean()
        )


    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor', 'rollout']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
    
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()
        self._init_logger()
    
    def _init_logger(self):
        from verl.utils.tracking import Tracking
        self.logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

    def _create_dataloader(self):
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error',
                                         prepend_think_token=self.config.prepend_think_token)
        if self.config.data.second_train_files is not None:
            self.train_dataset_second = RLHFDataset(parquet_files=self.config.data.second_train_files,
                                                   tokenizer=self.tokenizer,
                                                   prompt_key=self.config.data.prompt_key,
                                                   max_prompt_length=self.config.data.max_prompt_length,
                                                   filter_prompts=True,
                                                   return_raw_chat=self.config.data.get('return_raw_chat', False),
                                                   truncation='error',
                                                   prepend_think_token=self.config.prepend_think_token)
        if self.config.data.train_data_num is not None:
            if self.config.data.train_data_num > len(self.train_dataset.dataframe):
                print(f"[WARNING] training dataset size is smaller than desired size. Using the dataset as the original size {len(self.train_dataset.dataframe)}")
            else:
                self.train_dataset.dataframe = self.train_dataset.dataframe.sample(self.config.data.train_data_num, random_state=42)
        print(f"filtered training dataset size: {len(self.train_dataset.dataframe)}")

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           shuffle=self.config.data.shuffle_train_dataloader,
                                           drop_last=True,
                                           collate_fn=collate_fn)
        if self.config.data.second_train_files is not None:
            second_batch_size = math.floor(self.config.data.train_batch_size * self.config.hard_replay.hard_percentage)
            self.train_dataloader_second = DataLoader(dataset=self.train_dataset_second,
                                                      batch_size=second_batch_size,
                                                      shuffle=self.config.data.shuffle_train_dataloader,
                                                      drop_last=True,
                                                      collate_fn=collate_fn)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error',
                                       prepend_think_token=self.config.prepend_think_token)
        if self.config.data.val_data_num is not None:
            if self.config.data.val_data_num > len(self.val_dataset.dataframe):
                print(f"[WARNING] validation dataset size is smaller than desired size. Using the dataset as the original size {len(self.val_dataset.dataframe)}")
            else:
                self.val_dataset.dataframe = self.val_dataset.dataframe.sample(self.config.data.val_data_num, random_state=42)
        print(f"filtered validation dataset size: {len(self.val_dataset.dataframe)}")

        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=self.config.data.val_batch_size,
                                         shuffle=False,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')
        
        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _validate(self):
        """
        The training loop of PPO with global metric computation.
        Accumulates metrics across all batches before computing final statistics.
        """
        import torch
        reward_tensor_lst = []
        data_source_lst = []

        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url = self.config.retriever.url,
            topk = self.config.retriever.topk,
            discard_bad_docs = self.config.discard_bad_docs,
            prepend_think_token = self.config.prepend_think_token,
        )

        # Agent config preparation
        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            is_validation = True,
            reranker_config=self.config.reranker,
        )

        if not self.config.do_search:
            for test_data in self.val_dataloader:
                test_batch = DataProto.from_single_dict(test_data)

                # we only do validation on rule-based rm
                if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                    return {}

                test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': False,
                    'validate': True,
                }

                # pad to be divisible by dp_size
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                # unpad
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
                print('validation generation end')

                test_batch = test_batch.union(test_output_gen_batch)

                # evaluate using reward_function
                # for certain reward function (e.g. sandbox), the generation can overlap with reward
                reward_tensor = self.val_reward_fn(test_batch)

                reward_tensor_lst.append(reward_tensor)
                data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor['reward_tensor'].shape[0]))
                # if isinstance(reward_tensor, tuple):
                #     data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor[0].shape[0]))
                # else:
                #     data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
        else:
            for batch_dict in self.val_dataloader:
                timing_raw = {}
                test_batch: DataProto = DataProto.from_single_dict(batch_dict)
                # test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)
                
                test_gen_batch = test_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': False,
                    'validate': True,
                }
                with _timer('step', timing_raw):
                    first_input_ids = test_gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone()
                    with _timer('gen', timing_raw):
                        generation_manager.timing_raw = timing_raw
                        all_questions = test_batch.non_tensor_batch['question']
                        all_ground_truth = test_batch.non_tensor_batch['golden_answers']
                        final_gen_batch_output = generation_manager.run_llm_loop(
                            gen_batch=test_gen_batch,
                            initial_input_ids=first_input_ids,
                            all_questions=all_questions,
                            all_ground_truth=all_ground_truth
                        )

                    if 'input_ids' in final_gen_batch_output.batch:
                        final_gen_batch_output.batch['input_ids'] = final_gen_batch_output.batch['input_ids'].long()
                    if 'responses' in final_gen_batch_output.batch:
                        final_gen_batch_output.batch['responses'] = final_gen_batch_output.batch['responses'].long()
                    if 'prompts' in final_gen_batch_output.batch:
                        final_gen_batch_output.batch['prompts'] = final_gen_batch_output.batch['prompts'].long()

                    output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                    final_gen_batch_output = final_gen_batch_output.union(output)
                    
                    test_batch = test_batch.union(final_gen_batch_output)
                    test_batch, _ = self._create_loss_mask(test_batch, {}, 0)
                    
                    for key in test_batch.batch.keys():
                        if key not in ['old_log_probs', 'entropy']:
                            test_batch.batch[key] = test_batch.batch[key].long()
                    
                    # evaluate using reward_function
                    # for certain reward function (e.g. sandbox), the generation can overlap with reward
                    reward_tensor = self.val_reward_fn(test_batch, self.config.max_turns)

                    reward_tensor_lst.append(reward_tensor)
                    data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor['reward_tensor'].shape[0]))
                    # if isinstance(reward_tensor, tuple):
                    #     data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor[0].shape[0]))
                    # else:
                    #     data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        reward_tensor = torch.cat([rw['reward_tensor'].sum(-1) for rw in reward_tensor_lst], dim=0).cpu()
        if 'reward_2_tensor' in reward_tensor_lst[0]:
            reward_2_tensor = torch.cat([rw['reward_2_tensor'].sum(-1) for rw in reward_tensor_lst], dim=0).cpu()
            reward_3_tensor = torch.cat([rw['reward_3_tensor'].sum(-1) for rw in reward_tensor_lst], dim=0).cpu()
        if 'format_reward_tensor' in reward_tensor_lst[0]:
            format_reward_tensor = torch.cat([rw['format_reward_tensor'].sum(-1) for rw in reward_tensor_lst], dim=0).cpu()
        if 'garbled_penalty_tensor' in reward_tensor_lst[0]:
            garbled_penalty_tensor = torch.cat([rw['garbled_penalty_tensor'].sum(-1) for rw in reward_tensor_lst], dim=0).cpu()
        # if isinstance(reward_tensor_lst[0], tuple):
        #     reward_tensor = torch.cat([rw[0].sum(-1) for rw in reward_tensor_lst], dim=0).cpu()
        #     if len(reward_tensor_lst[0]) == 4:
        #         reward_2_tensor = torch.cat([rw[1].sum(-1) for rw in reward_tensor_lst], dim=0).cpu()
        #         reward_3_tensor = torch.cat([rw[2].sum(-1) for rw in reward_tensor_lst], dim=0).cpu()
        #     format_reward_tensor = torch.cat([rw[-1].sum(-1) for rw in reward_tensor_lst], dim=0).cpu()
        # else:
        #     reward_tensor = torch.cat([rw.sum(-1) for rw in reward_tensor_lst], dim=0).cpu()  # (batch_size,)

        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}
        data_source_reward_2 = {}
        data_source_reward_3 = {}
        data_source_format_reward = {}
        data_source_garbled_penalty = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
                data_source_reward_2[data_source] = []
                data_source_reward_3[data_source] = []
                data_source_format_reward[data_source] = []
                data_source_garbled_penalty[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())
            if 'reward_2_tensor' in reward_tensor_lst[0]:
                data_source_reward_2[data_source].append(reward_2_tensor[i].item())
                data_source_reward_3[data_source].append(reward_3_tensor[i].item())
            if 'format_reward_tensor' in reward_tensor_lst[0]:
                data_source_format_reward[data_source].append(format_reward_tensor[i].item())
            else:
                data_source_format_reward[data_source] = []
            if 'garbled_penalty_tensor' in reward_tensor_lst[0]:
                data_source_garbled_penalty[data_source].append(garbled_penalty_tensor[i].item())
            else:
                data_source_garbled_penalty[data_source] = []

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            if 'reward_2_tensor' in reward_tensor_lst[0]:
                metric_dict[f'val/test_em/{data_source}'] = np.mean(rewards)
                metric_dict[f'val/test_cem/{data_source}'] = np.mean(data_source_reward_2[data_source])
                metric_dict[f'val/test_f1/{data_source}'] = np.mean(data_source_reward_3[data_source])
            else:
                metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)
        if 'format_reward_tensor' in reward_tensor_lst[0]:
            for data_source, rewards in data_source_format_reward.items():
                metric_dict[f'val/test_format_score/{data_source}'] = np.mean(rewards)
        if 'garbled_penalty_tensor' in reward_tensor_lst[0]:
            for data_source, rewards in data_source_garbled_penalty.items():
                metric_dict[f'val/test_garbled_penalty/{data_source}'] = np.mean(rewards)

        return metric_dict


    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
            
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        f'global_step_{self.global_steps}')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                             f'global_step_{self.global_steps}')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        logger = self.logger
        self.global_steps = self.config.start_global_step
        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1

        # Agent config preparation
        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url = self.config.retriever.url,
            topk = self.config.retriever.topk,
            discard_bad_docs = self.config.discard_bad_docs,
            prepend_think_token = self.config.prepend_think_token,
        )

        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            reranker_config=self.config.reranker,
        )
        with open('rollout/easy_question_idx.json','r') as f:
            easy_question_idx = json.load(f)
        with open('rollout/hard_solvable_question_idx.json','r') as f:
            hard_solvable_question_idx = json.load(f)
        with open('rollout/hard_unsolvable_question_idx.json','r') as f:
            hard_unsolvable_question_idx = json.load(f)

        # add tqdm
        progress_bar = tqdm(total=min(self.config.trainer.total_epochs * len(self.train_dataloader), self.total_training_steps), initial=1, desc="Training Progress")
        
        now_batchs_for_loss = []
        batchs_original = []
        generate_turns = 0

        # start training loop
        hard_data_dict = {}
        easy_data_dict = {}
        hard_add_times = defaultdict(int)
        # last_train_step = defaultdict(int)
        for epoch in range(self.config.trainer.total_epochs):
            data_iter = iter(self.train_dataloader)
            if self.config.data.second_train_files is not None:
                data_iter_second = iter(self.train_dataloader_second)
            epoch_end = False
            while(not epoch_end):
                print(f'epoch {epoch}, step {self.global_steps}')
                metrics = {}
                timing_raw = {}

                if self.config.hard_replay.enable:
                    hard_data_enough = True
                    easy_data_enough = True

                    hard_data_demand = math.floor(self.config.data.train_batch_size * self.config.hard_replay.hard_percentage)
                    easy_data_demand = math.ceil(self.config.data.train_batch_size * (1 - self.config.hard_replay.hard_percentage))
                    num_hard_data = len(hard_data_dict['index']) if len(hard_data_dict)!=0 else 0
                    num_easy_data = len(easy_data_dict['index']) if len(easy_data_dict)!=0 else 0


                    if num_hard_data < hard_data_demand:
                        hard_data_enough = False
                        print(f"hard data not enough, sampling, num of hard data: {num_hard_data}, demand: {hard_data_demand}")
                    else:
                        if self.config.hard_replay.use_older:
                            num_hard_data_older = 0
                            for item in hard_data_dict['index']:
                                if last_train_step[item]==0 or self.global_steps-last_train_step[item] >= 5:
                                    num_hard_data_older += 1
                            if num_hard_data_older < hard_data_demand:
                                hard_data_enough = False
                                print(f"older hard data not enough, sampling, num of hard data: {num_hard_data}, older: {num_hard_data_older}, demand: {hard_data_demand}")
                    if num_easy_data < easy_data_demand:
                        easy_data_enough = False
                        print(f"easy data not enough, sampling, num of easy data: {num_easy_data}, demand: {easy_data_demand}")

                    if hard_data_enough and easy_data_enough:
                        print("hard and easy data enough, using hard and easy data")
                        if self.config.hard_replay.use_older:
                            train_hard_data, hard_data_dict, last_train_step = data_dict_random_select(hard_data_dict, hard_data_demand, last_train_step, self.global_steps)
                        else:
                            train_hard_data, hard_data_dict = data_dict_random_select(hard_data_dict, hard_data_demand)
                        easy_cnt = 0
                        hard_solvable_cnt = 0
                        hard_unsolvable_cnt = 0
                        for item in train_hard_data['index']:
                            if item in easy_question_idx:
                                easy_cnt += 1
                            elif item in hard_solvable_question_idx:
                                hard_solvable_cnt += 1
                            elif item in hard_unsolvable_question_idx:
                                hard_unsolvable_cnt += 1
                        # print(f"Hard: easy_cnt: {easy_cnt},{easy_cnt/hard_data_demand}; hard_solvable_cnt: {hard_solvable_cnt},{hard_solvable_cnt/hard_data_demand}; hard_unsolvable_cnt: {hard_unsolvable_cnt},{hard_unsolvable_cnt/hard_data_demand}")
                        train_easy_data, easy_data_dict = data_dict_random_select(easy_data_dict, easy_data_demand)
                        easy_cnt_1 = 0
                        hard_solvable_cnt_1 = 0
                        hard_unsolvable_cnt_1 = 0
                        for item in train_easy_data['index']:
                            if item in easy_question_idx:
                                easy_cnt_1 += 1
                            elif item in hard_solvable_question_idx:
                                hard_solvable_cnt_1 += 1
                            elif item in hard_unsolvable_question_idx:
                                hard_unsolvable_cnt_1 += 1
                        # print(f"Easy: easy_cnt: {easy_cnt_1},{easy_cnt_1/easy_data_demand}; hard_solvable_cnt: {hard_solvable_cnt_1},{hard_solvable_cnt_1/easy_data_demand}; hard_unsolvable_cnt: {hard_unsolvable_cnt_1},{hard_unsolvable_cnt_1/easy_data_demand}")
                        print(f"All: easy_cnt: {easy_cnt+easy_cnt_1},{(easy_cnt+easy_cnt_1)/(hard_data_demand+easy_data_demand)}; hard_solvable_cnt: {hard_solvable_cnt+hard_solvable_cnt_1},{(hard_solvable_cnt+hard_solvable_cnt_1)/(hard_data_demand+easy_data_demand)}; hard_unsolvable_cnt: {hard_unsolvable_cnt+hard_unsolvable_cnt_1},{(hard_unsolvable_cnt+hard_unsolvable_cnt_1)/(hard_data_demand+easy_data_demand)}")
                        metrics.update({
                            'easy_rate': (easy_cnt+easy_cnt_1)/(hard_data_demand+easy_data_demand),
                            'hard_solvable_rate': (hard_solvable_cnt+hard_solvable_cnt_1)/(hard_data_demand+easy_data_demand),
                            'hard_unsolvable_rate': (hard_unsolvable_cnt+hard_unsolvable_cnt_1)/(hard_data_demand+easy_data_demand),
                        })
                        print(f"Remain: Num of hard data: {len(hard_data_dict['index'])}, Num of easy data: {len(easy_data_dict['index'])}")
                    # if easy_data_enough:
                    #     try:
                    #         train_hard_data = next(data_iter_second)
                    #     except StopIteration:
                    #         data_iter_second = iter(self.train_dataloader_second)
                    #         train_hard_data = next(data_iter_second)
                    #     last_train_step = defaultdict(int)
                    #     for item in train_hard_data['index']:
                    #         last_train_step[item] = self.global_steps
                    #     train_easy_data, easy_data_dict, _ = data_dict_random_select(easy_data_dict, easy_data_demand, last_train_step, self.global_steps)
                        batch_dict = data_dict_merge(train_hard_data, train_easy_data)
                    else:
                        try:
                            batch_dict = next(data_iter)
                        except StopIteration:
                            epoch_end = True
                            break
                else:
                    try:
                        batch_dict = next(data_iter)
                    except StopIteration:
                        epoch_end = True
                        break
                
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                    


                if self.config.actor_rollout_ref.rollout.generate_and_select.enable and \
                self.global_steps >= self.config.actor_rollout_ref.rollout.generate_and_select.start_step:
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.generate_and_select.candidates, interleave=True)
                else:
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)

                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                ####################
                # original code here

                with _timer('step', timing_raw):
                    if not self.config.do_search:
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                        # batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                                # dtype=object)
                        batch.non_tensor_batch['uid'] = batch.non_tensor_batch['index'].copy()
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)

                ####################
                # Below is aLL about agents - the "LLM + forloop"
                ####################
                # with _timer('step', timing_raw):
                    else:
                        enough_rollout = False
                        # generate_turns = 0
                        done_gen_batch_output = []
                        last_gen_bacth_len = len(gen_batch.batch)
                        remain = np.array(range(last_gen_bacth_len))
                        old_idx = []
                        all_questions = batch.non_tensor_batch['question']
                        all_ground_truth = batch.non_tensor_batch['golden_answers']
                        while not enough_rollout:
                            generate_turns += 1
                            print(f'generate_turns: {generate_turns}')
                            first_input_ids = gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone().long()
                            with _timer('gen', timing_raw):
                                generation_manager.timing_raw = timing_raw
                                final_gen_batch_output = generation_manager.run_llm_loop(
                                    gen_batch=gen_batch,
                                    initial_input_ids=first_input_ids,
                                    all_questions=all_questions,
                                    all_ground_truth=all_ground_truth,
                                )

                            # final_gen_batch_output.batch.apply(lambda x: x.long(), inplace=True)
                            for key in final_gen_batch_output.batch.keys():
                                final_gen_batch_output.batch[key] = final_gen_batch_output.batch[key].long()

                            with torch.no_grad():
                                output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                                final_gen_batch_output = final_gen_batch_output.union(output)

                            if self.config.actor_rollout_ref.actor.garbled_regenerate.enable and \
                                self.config.actor_rollout_ref.actor.garbled_regenerate.start_step <= self.global_steps:
                                with torch.no_grad():
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(final_gen_batch_output)
                                
                                final_gen_batch_output.batch = final_gen_batch_output.batch[:last_gen_bacth_len]

                                if generate_turns >= self.config.actor_rollout_ref.actor.garbled_regenerate.max_generate_turns:
                                    done_idx = list(range(len(final_gen_batch_output.batch)))
                                    old_idx = old_idx + remain[done_idx].tolist()
                                    print("max_generate_turns reached, stop regenerate.")
                                    done_gen_batch_output.append(final_gen_batch_output[done_idx])
                                    break
                                
                                response_length = final_gen_batch_output.batch['responses'].shape[-1]
                                response_mask = final_gen_batch_output.batch['attention_mask'][:, -response_length:]
                                loss_mask = final_gen_batch_output.batch['info_mask'][:, -response_length:]
                                done_idx = []
                                regenerate_idx = []
                                for idx in range(len(final_gen_batch_output.batch)):
                                    if is_garbled(ref_log_prob.batch[idx]['ref_log_prob'],
                                                  output.batch[idx]['old_log_probs'],
                                                  loss_mask[idx],
                                                  self.config.reward_model.garbled_penalty.ppl_threshold):
                                        regenerate_idx.append(idx)       
                                    else:
                                        done_idx.append(idx)
                                        old_idx.append(remain[idx])
                                remain = remain[regenerate_idx]
                                print(f"num_done: {len(done_idx)}, num_regenerate: {len(regenerate_idx)}")
                                
                                gen_batch.batch = gen_batch.batch[regenerate_idx]
                                all_questions = all_questions[regenerate_idx]
                                all_ground_truth = all_ground_truth[regenerate_idx]
                                last_gen_batch_len = len(regenerate_idx)
                                if last_gen_batch_len:
                                    gen_batch, _ = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_wg.world_size)
                                done_gen_batch_output.append(final_gen_batch_output[done_idx])
                                if len(regenerate_idx)==0:
                                    enough_rollout = True
                            else:
                                enough_rollout = True

                        
                        if self.config.actor_rollout_ref.actor.garbled_regenerate.enable and \
                        self.config.actor_rollout_ref.actor.garbled_regenerate.start_step <= self.global_steps:
                            final_gen_batch_output = length_alignment(done_gen_batch_output, self.tokenizer)
                            old_idx = np.array(old_idx)
                            indices = torch.tensor(np.argsort(old_idx).tolist())
                            DataProto.reorder(final_gen_batch_output, indices)

                        if self.config.actor_rollout_ref.rollout.generate_and_select.enable:
                            response_length = final_gen_batch_output.batch['responses'].shape[-1]
                            response_mask = final_gen_batch_output.batch['attention_mask'][:, -response_length:]
                            loss_mask = final_gen_batch_output.batch['info_mask'][:, -response_length:]
                            num_candidates = self.config.actor_rollout_ref.rollout.generate_and_select.candidates
                            selected_rollouts_idx = []
                            for idx in range(0, len(final_gen_batch_output.batch), num_candidates):
                                candidates = final_gen_batch_output.batch[idx: idx + num_candidates]
                                selection_method = self.config.actor_rollout_ref.rollout.generate_and_select.selection_method
                                group_selected_rollouts_idx = select_rollout(candidates,
                                                                   loss_mask[idx: idx + num_candidates],
                                                                   self.config.actor_rollout_ref.rollout.n_agent, 
                                                                   selection_method,
                                                                   self.tokenizer,
                                                                   self.embedding_model if selection_method == 'embedding_distance' else None)
                                group_selected_rollouts_idx = [idx + i for i in group_selected_rollouts_idx]
                                selected_rollouts_idx.extend(group_selected_rollouts_idx)
                        
                            final_gen_batch_output.batch = final_gen_batch_output.batch[selected_rollouts_idx]
                            batch.batch = batch.batch[selected_rollouts_idx]
                            for key in batch.non_tensor_batch.keys():
                                batch.non_tensor_batch[key] = batch.non_tensor_batch[key][selected_rollouts_idx]

                        # batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                        #                                         dtype=object)
                        batch.non_tensor_batch['uid'] = batch.non_tensor_batch['index'].copy()
                                            
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(final_gen_batch_output)

                    ####################
                    ####################

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # batch.batch.apply(lambda x, key: x.long() if key != "old_log_probs" else x, inplace=True, key=True)
                    for key in batch.batch.keys():
                        if key not in ['old_log_probs', 'entropy']:
                            batch.batch[key] = batch.batch[key].long()

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    if self.config.do_search and self.config.actor_rollout_ref.actor.state_masking:
                        batch, metrics = self._create_loss_mask(batch, metrics, self.global_steps)

                    loss_mask = batch.batch['loss_mask']
                    old_log_probs = batch.batch['old_log_probs']
                    old_log_prob_mean = verl_F.masked_mean(old_log_probs, loss_mask)
                    old_probs = torch.exp(old_log_probs)
                    old_probs_mean = verl_F.masked_mean(old_probs, loss_mask)
                    metrics['actor/old_log_prob_mean'] = old_log_prob_mean.item()
                    metrics['actor/old_probs_mean'] = old_probs_mean.item()

                    # torch.save(batch.batch['responses'], f'response_ids/{self.global_steps}.pt')
                    # print(f'Saved response ids to response_ids/{self.global_steps}.pt')

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        reward_tensor = self.reward_fn(batch, self.config.max_turns, self.global_steps)

                        batch.batch['token_level_scores'] = reward_tensor['reward_tensor']
                        if 'format_reward_tensor' in reward_tensor:
                            batch.batch['token_level_format_scores'] = reward_tensor['format_reward_tensor']
                        if 'length_score_tensor' in reward_tensor:
                            batch.batch['token_level_length_scores'] = reward_tensor['length_score_tensor']
                        if 'valid_action_reward_tensor' in reward_tensor:
                            batch.batch['token_level_valid_action_scores'] = reward_tensor['valid_action_reward_tensor']
                        if 'information_gain_tensor' in reward_tensor:
                            batch.batch['token_level_information_gain_scores'] = reward_tensor['information_gain_tensor']
                        if 'good_query_tensor' in reward_tensor:
                            batch.batch['token_level_good_query'] = reward_tensor['good_query_tensor']
                            batch.batch['max_score_tensor'] = reward_tensor['max_score_tensor']
                        if 'garbled_penalty_tensor' in reward_tensor:
                            batch.batch['token_level_garbled_penalty_scores'] = reward_tensor['garbled_penalty_tensor']
                        if 'query_doc_relevance_tensor' in reward_tensor:
                            batch.batch['token_level_query_doc_relevance_scores'] = reward_tensor['query_doc_relevance_tensor']
                        if 'ablation_tensor' in reward_tensor:
                            batch.batch['ablation_tensor'] = reward_tensor['ablation_tensor']

                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # compute rewards. apply_kl_penalty if available
                        # if not self.config.actor_rollout_ref.actor.use_kl_loss:
                        #     batch, kl_metrics = apply_kl_penalty(batch,
                        #                                          kl_ctrl=self.kl_ctrl,
                        #                                          kl_penalty=self.config.algorithm.kl_penalty)
                        #     metrics.update(kl_metrics)
                        # else:
                        if 'query_doc_relevance_tensor' in reward_tensor and not self.config.reward_model.query_doc_relevance.segment_reward:
                            # batch.batch['token_level_rewards'] = (batch.batch['token_level_scores'] + \
                            # batch.batch['token_level_query_doc_relevance_scores'])
                            batch.batch['token_level_rewards'] = (batch.batch['token_level_scores'] * 0.7 + \
                            batch.batch['token_level_query_doc_relevance_scores'] * 0.3)
                            # batch.batch['token_level_rewards'] = batch.batch['token_level_query_doc_relevance_scores']
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores'] # ablation注释掉
                            # batch.batch['token_level_rewards'] = reward_tensor['ablation_tensor'] # ablation
                            # batch.batch['token_level_rewards'] = (batch.batch['token_level_scores'] * 0.7 + \
                            # reward_tensor['ablation_tensor'] * 0.3) # ablation
                            # batch.batch['token_level_rewards'] = batch.batch['token_level_format_scores']


                        if self.config.use_prelabed_rollout:
                            batchs_original=batch
                            uid2score = defaultdict(list)
                            uid2idx = defaultdict(list)
                            for idx in range(len(batch)):
                                uid = batch.non_tensor_batch['uid'][idx]
                                uid2score[uid].append(batch[idx].batch['token_level_scores'].sum(dim=-1).item())
                                uid2idx[uid].append(idx)
                            all_wrong_uid = []
                            for uid in uid2score:
                                correct_num = sum([1 if score >= 0.7 else 0 for score in uid2score[uid]])
                                if correct_num == 0:
                                    all_wrong_uid.append(uid)
                            uid2rollouts, uid2maxscore = get_silver_rollouts(all_wrong_uid)
                            have_prelabed_rollout_uid = uid2rollouts.keys()

                            replace_idx = []
                            for uid in have_prelabed_rollout_uid:
                                replace_idx.extend(uid2idx[uid][:len(uid2rollouts[uid])])
                            remain_idx = list(set(range(len(batch))) - set(replace_idx))
                            print("Number of replace_idx: ", len(replace_idx))
                            print("Number of remain_idx: ", len(remain_idx))
                            if len(replace_idx):
                                replace_batch = batch[replace_idx]
                                replace_batch = DataProto.concat([replace_batch])
                                if 'token_level_good_query' in replace_batch.batch:
                                    replace_batch.pop(batch_keys=['token_level_scores','token_level_format_scores', 'token_level_good_query', 'max_score_tensor', 'token_level_rewards'])
                                else:
                                    replace_batch.pop(batch_keys=['token_level_scores','token_level_format_scores', 'token_level_rewards'])
                                replace_batch = batch_replacement(replace_batch, uid2rollouts, self.tokenizer)
                                replace_batch, pad_size = pad_dataproto_to_divisor(replace_batch, self.actor_rollout_wg.world_size)
                                output = self.actor_rollout_wg.compute_log_prob(replace_batch)
                                replace_batch = replace_batch.union(output, True)
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(replace_batch)
                                replace_batch = replace_batch.union(ref_log_prob, True)
                                replace_batch = DataProto.concat([unpad_dataproto(replace_batch, pad_size)])
                                replace_batch, _ = self._create_loss_mask(replace_batch, {}, self.global_steps)
                                remain_batch = batch[remain_idx]
                                remain_batch = DataProto.concat([remain_batch])
                                if 'token_level_good_query' in remain_batch.batch:
                                    remain_batch.pop(batch_keys=['token_level_scores','token_level_format_scores', 'token_level_good_query', 'max_score_tensor', 'token_level_rewards'])
                                else:
                                    remain_batch.pop(batch_keys=['token_level_scores','token_level_format_scores', 'token_level_rewards'])
                                batch = length_alignment([remain_batch, replace_batch], self.tokenizer)

                                reward_tensor = self.reward_fn(batch, self.config.max_turns, self.global_steps)
                                batch.batch['token_level_scores'] = reward_tensor['reward_tensor']
                                if 'format_reward_tensor' in reward_tensor:
                                    batch.batch['token_level_format_scores'] = reward_tensor['format_reward_tensor']
                                if 'good_query_tensor' in reward_tensor:
                                    batch.batch['token_level_good_query'] = reward_tensor['good_query_tensor']
                                    batch.batch['max_score_tensor'] = reward_tensor['max_score_tensor']
                                batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        elif self.config.difficulty_filter.enable and self.global_steps >= self.config.difficulty_filter.start_step:
                        # if self.config.difficulty_filter.enable:
                            batchs_original.append(batch)
                            uid2score = defaultdict(list)
                            uid2idx = defaultdict(list)
                            for idx in range(len(batch)):
                                uid = batch.non_tensor_batch['uid'][idx]
                                uid2score[uid].append(batch[idx].batch['token_level_scores'].sum(dim=-1).item())
                                uid2idx[uid].append(idx)
                            keep_idx = []
                            if self.config.difficulty_filter.dynamic_threshold:
                                difficulty_threshold = self.config.difficulty_filter.threshold * (epoch + 1)
                            else:
                                difficulty_threshold = self.config.difficulty_filter.threshold
                            for uid in uid2score:
                                correct_num = sum([1 if score >= 0.7 else 0 for score in uid2score[uid]])
                                correct_rate = correct_num / len(uid2score[uid])
                                if correct_rate > 0 and correct_rate <= difficulty_threshold:
                                    keep_idx.extend(uid2idx[uid])
                            batch = batch[keep_idx]
                            now_batchs_for_loss.append(batch)
                            
                            target_bs = self.config.data.train_batch_size
                            now_bs = sum([len(batch_for_loss.batch) for batch_for_loss in now_batchs_for_loss]) / self.config.actor_rollout_ref.rollout.n_agent
                            if now_bs < target_bs:
                                print(f'{now_bs=} < {target_bs=}, keep generating')
                                continue
                            else:
                                print(f'{now_bs=} >= {target_bs=}, stop generating')
                                batchs_original = length_alignment(batchs_original, self.tokenizer)
                                now_batchs_for_loss = length_alignment(now_batchs_for_loss, self.tokenizer)
                                train_bs = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n_agent
                                now_batchs_for_loss.batch = now_batchs_for_loss.batch[:train_bs]
                                for key in now_batchs_for_loss.non_tensor_batch.keys():
                                    now_batchs_for_loss.non_tensor_batch[key] = now_batchs_for_loss.non_tensor_batch[key][:train_bs]
                                batch = now_batchs_for_loss
                        elif self.config.hard_replay.enable:
                            if not hard_data_enough or not easy_data_enough:
                                batchs_original.append(batch)
                                hard_uid, easy_uid = get_easy_hard_uid(batch)
                                easy_cnt = 0
                                hard_solvable_cnt = 0
                                hard_unsolvable_cnt = 0
                                for uid in hard_uid:
                                    if uid in easy_question_idx:
                                        easy_cnt += 1
                                    elif uid in hard_solvable_question_idx:
                                        hard_solvable_cnt += 1
                                    elif uid in hard_unsolvable_question_idx:
                                        hard_unsolvable_cnt += 1
                                print(f"Hard: Easy cnt: {easy_cnt},{easy_cnt/len(hard_uid)}; Hard solvable cnt: {hard_solvable_cnt},{hard_solvable_cnt/len(hard_uid)}; Hard unsolvable cnt: {hard_unsolvable_cnt},{hard_unsolvable_cnt/len(hard_uid)}")
                                easy_cnt = 0
                                hard_solvable_cnt = 0
                                hard_unsolvable_cnt = 0
                                for uid in easy_uid:
                                    if uid in easy_question_idx:
                                        easy_cnt += 1
                                    elif uid in hard_solvable_question_idx:
                                        hard_solvable_cnt += 1
                                    elif uid in hard_unsolvable_question_idx:
                                        hard_unsolvable_cnt += 1
                                print(f"Easy: Easy cnt: {easy_cnt},{easy_cnt/len(easy_uid)}; Hard solvable cnt: {hard_solvable_cnt},{hard_solvable_cnt/len(easy_uid)}; Hard unsolvable cnt: {hard_unsolvable_cnt},{hard_unsolvable_cnt/len(easy_uid)}")
                                # import pdb; pdb.set_trace()
                                hard_add_times = update_times(hard_add_times, hard_uid)
                                new_hard_data_dict = uid_to_data_dict(hard_uid, batch_dict, hard_add_times)
                                new_easy_data_dict = uid_to_data_dict(easy_uid, batch_dict)
                                easy_data_dict = data_dict_merge(easy_data_dict, new_easy_data_dict)
                                hard_data_dict = data_dict_merge(hard_data_dict, new_hard_data_dict)
                                print(f"Add {len(hard_uid)} hard data and {len(easy_uid)} easy data")
                                print(f"Have {len(hard_data_dict['index'])} hard data and {len(easy_data_dict['index'])} easy data")
                                continue
                            else:
                                if len(batchs_original) > 0:
                                    batchs_original = length_alignment(batchs_original, self.tokenizer)
                                hard_uid, _ = get_easy_hard_uid(batch)
                                hard_add_times = update_times(hard_add_times, hard_uid)
                                new_hard_data_dict = uid_to_data_dict(hard_uid, batch_dict, hard_add_times)
                                hard_data_dict = data_dict_merge(hard_data_dict, new_hard_data_dict)
                                print(f"Add {len(hard_uid)} hard data")
                                print(f"Have {len(hard_data_dict['index'])} hard data")
                                print(f"Reset easy data")
                                easy_data_dict = {}
                        else:
                            batchs_original = batch

   

                        if len(batchs_original) > 0: 
                            uid2data, least_one_correct_rate, LOC_correct_rate, garbled_cnt, largest_ans_score = get_rollout(batchs_original, self.tokenizer)
                            metrics.update({'correct_rate/largest_ans_score': largest_ans_score,
                                            'correct_rate/least_one_correct': least_one_correct_rate,
                                            'correct_rate/LOC_correct': LOC_correct_rate})
                        if self.config.hard_replay.enable:
                            _, least_one_correct_rate_forloss, LOC_correct_rate_forloss, _, largest_ans_score_forloss = get_rollout(batch, self.tokenizer)
                            metrics.update({'correct_rate/largest_ans_score_forloss': largest_ans_score_forloss,
                                            'correct_rate/least_one_correct_forloss': least_one_correct_rate_forloss,
                                            'correct_rate/LOC_correct_forloss': LOC_correct_rate_forloss})
                        if self.config.use_prelabed_rollout:
                            _, least_one_correct_rate, _, _, largest_ans_score = get_rollout(batch, self.tokenizer)
                            metrics.update({'correct_rate/least_one_correct_w_prelabeled': least_one_correct_rate,
                                          'correct_rate/largest_ans_score_w_prelabeled': largest_ans_score})
                        # import pdb; pdb.set_trace()
                        if self.config.reward_model.garbled_penalty.enable:
                            metrics.update({'correct_rate/garbled_cnt': garbled_cnt})
                        if self.config.actor_rollout_ref.actor.save_rollout.enable:
                            save_path = self.config.actor_rollout_ref.actor.save_rollout.save_path
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            with open(f'{save_path}/rollout_{self.global_steps}.json', 'w') as f:
                                json.dump(uid2data, f, indent=4)
                            # if self.config.actor_rollout_ref.actor.save_rollout.save_all_incorrect:
                                
                        if len(batchs_original) > 0: 
                            metrics.update(compute_data_metrics(batch=batchs_original, use_critic=self.use_critic))

                        if self.config.actor_rollout_ref.actor.remove_all_same.enable and \
                        self.global_steps >= self.config.actor_rollout_ref.actor.remove_all_same.start_step:
                            keep_idx = []
                            for idx in range(len(batch)):
                                uid = batch.non_tensor_batch['uid'][idx]
                                # if uid2data[uid]['correct_cnt']>0:
                                # if not uid2data[uid]['all_ans_score_same']:
                                if uid2data[uid]['largest_ans_score'] - uid2data[uid]['smallest_ans_score'] >= \
                                self.config.actor_rollout_ref.actor.remove_all_same.threshold:
                                    keep_idx.append(idx)
                            metrics.update({'correct_rate/after_filter': len(keep_idx)/len(batch)})
                            batch.batch = batch.batch[keep_idx]
                            for key in batch.non_tensor_batch.keys():
                                batch.non_tensor_batch[key] = batch.non_tensor_batch[key][keep_idx]
                            batch, pad_size = pad_dataproto_to_divisor(batch, self.actor_rollout_wg.world_size)
                        
                        if self.config.actor_rollout_ref.actor.remove_garbled:
                            keep_idx = []
                            for idx in range(len(batch)):
                                # data_item = batch[idx]
                                # if 'token_level_garbled_penalty_scores' in data_item.batch and \
                                # data_item.batch['token_level_garbled_penalty_scores'].sum(dim=-1).item() >= 0:
                                #     keep_idx.append(idx)
                                uid = batch.non_tensor_batch['uid'][idx]
                                if not uid2data[uid]['have_garbled']:
                                    keep_idx.append(idx)
                            metrics.update({'correct_rate/after_filter': len(keep_idx)/len(batch)})
                            print(f"remove garbled {len(batch)-len(keep_idx)} / {len(batch)}")
                            batch.batch = batch.batch[keep_idx]
                            for key in batch.non_tensor_batch.keys():
                                batch.non_tensor_batch[key] = batch.non_tensor_batch[key][keep_idx]
                            batch, pad_size = pad_dataproto_to_divisor(batch, self.actor_rollout_wg.world_size)


                        if self.config.actor_rollout_ref.actor.mask_void_turns.enable and \
                        self.global_steps >= self.config.actor_rollout_ref.actor.mask_void_turns.start_step:
                            keep_idx = []
                            for idx in range(len(batch)):
                                if batch.meta_info['void_turn_mask'][idx]:
                                    keep_idx.append(idx)
                            metrics.update({'correct_rate/after_mask': len(keep_idx)/len(batch)})
                            batch.batch = batch.batch[keep_idx]
                            for key in batch.non_tensor_batch.keys():
                                batch.non_tensor_batch[key] = batch.non_tensor_batch[key][keep_idx]
                            batch, pad_size = pad_dataproto_to_divisor(batch, self.actor_rollout_wg.world_size)

                        # compute advantages, executed on the driver process
                        if self.config.fast_test == False:
                            segment_adv = self.config.reward_model.query_doc_relevance.enable and self.config.reward_model.query_doc_relevance.segment_reward
                            segment_adv = segment_adv or self.config.reward_model.query_reward
                            mask_template_token = True if self.config.mask_template_token.enable and self.global_steps >= self.config.mask_template_token.start_step else False
                            batch = compute_advantage(batch,
                                                    tokenizer=self.tokenizer,
                                                    adv_estimator=self.config.algorithm.adv_estimator,
                                                    neutral = self.config.algorithm.neutral,
                                                    less_negative = self.config.algorithm.less_negative,
                                                    mask_template_token = mask_template_token,
                                                    model_name = self.config.mask_template_token.model_name,
                                                    acceptable_mask = self.config.algorithm.acceptable_mask,
                                                    query_reward_percentage=self.config.algorithm.query_reward_percentage,
                                                    low_mean=self.config.algorithm.low_mean,
                                                    use_global_std=self.config.algorithm.use_global_std,
                                                    norm=self.config.algorithm.norm,
                                                    segment_adv_type=self.config.algorithm.segment_adv_type,
                                                    gamma=self.config.algorithm.gamma,
                                                    lam=self.config.algorithm.lam,
                                                    segment_adv=segment_adv,
                                                )
                            # if self.config.mask_template_token.enable and self.global_steps >= self.config.mask_template_token.start_step:
                            #     responses = batch.batch['responses']
                            #     think_ids = torch.tensor([13708, 766, 26865, 29, 1339, 397, 10370], device=responses.device)
                            #     think_token_mask = torch.isin(responses, think_ids)
                            #     advantages = batch.batch['advantages']
                            #     advantages[think_token_mask] = 0
                            #     batch.batch['advantages'] = advantages
                            metrics.update(adv_metrics(batch))

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps and self.config.fast_test == False:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            if self.config.trainer.save_index.enable:
                                if not os.path.exists(f'{self.config.trainer.save_index.save_path}'):
                                    os.makedirs(f'{self.config.trainer.save_index.save_path}')
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        if self.config.trainer.save_index.enable:
                            move_pth(self.config.trainer.save_index.save_path, self.global_steps)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and self.global_steps >= self.config.trainer.start_save and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            try:
                                self._save_checkpoint()
                            except Exception as e:
                                logger.error(f"Failed to save checkpoint: {e}")

                # collect metrics
                if len(batchs_original) > 0: 
                    metrics.update(compute_timing_metrics(batch=batchs_original, timing_raw=timing_raw))

                if self.config.difficulty_filter.enable:
                    metrics.update({'correct_rate/generate_turns': generate_turns})
                generate_turns = 0
                batchs_original = []
                now_batchs_for_loss = []
                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if self.global_steps >= self.total_training_steps and self.config.fast_test == False:

                    # perform validation after training
                    # if self.val_reward_fn is not None:
                    #     val_metrics = self._validate()
                    #     pprint(f'Final validation metrics: {val_metrics}')
                    #     logger.log(data=val_metrics, step=self.global_steps)
                    self._save_checkpoint()
                    return
            if self.config.trainer.save_each_epoch:
                try:
                    self._save_checkpoint()
                except Exception as e:
                    logger.error(f"Failed to save checkpoint: {e}")
    
    def _create_loss_mask(self, batch, metrics, step):
        """Create loss mask for state tokens."""
        response_length = batch.batch['responses'].shape[-1]
        response_mask = batch.batch['attention_mask'][:, -response_length:]
        
        loss_mask = batch.batch['info_mask'][:, -response_length:]
        batch.batch['loss_mask'] = loss_mask

        if self.config.actor_rollout_ref.actor.mask_void_turns.enable \
        and step >= self.config.actor_rollout_ref.actor.mask_void_turns.start_step:
            batch.batch["loss_mask"] = batch.batch[
                "loss_mask"
            ] * batch.batch["void_turn_mask"].reshape(-1, 1)

        metrics.update({
            'state_tokens/total': loss_mask.sum().item(),
            'state_tokens/coverage': (loss_mask.sum() / response_mask.sum()).item(),
        })
        
        return batch, metrics
