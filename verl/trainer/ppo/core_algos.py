# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

print("*"*100)
import numpy as np
import torch
from collections import defaultdict
import math

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config): # seems never used?
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   responses: torch.Tensor,
                                   tokenizer,
                                   mask_template_token,
                                   neutral,
                                   less_negative,
                                   acceptable_mask,
                                   use_global_std: bool,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    id2index = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2index[index[i]].append(i)
        global_std = torch.std(scores)
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if neutral.enable and scores[i] - id2mean[index[i]] < 0 and scores[i] >= neutral.threshold:
                scores[i] = 0
            if acceptable_mask.enable:
                if scores[i] >= acceptable_mask.threshold and scores[i] < id2mean[index[i]]:
                    scores[i] = id2mean[index[i]]
            if use_global_std:
                scores[i] = (scores[i] - id2mean[index[i]]) / (global_std + epsilon)
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

        # torch.save({"advantage": scores.detach().cpu(), "eos": eos_mask.detach().cpu()}, f"advantage/{savename}")

        
        positive_indices = torch.where(scores > 0)
        negative_indices = torch.where(scores < 0)
        positive_sum = scores[positive_indices].sum()
        positive_mean = scores[positive_indices].mean()
        negative_sum = scores[negative_indices].sum()
        ratio = positive_sum / (-negative_sum)
        print(f"Original advantage positive/negative ratio: {ratio}")
        print(f"Original advantage negative/positive ratio: {-negative_sum / positive_sum}")
        print("-"*50)
        print(f"positive numel / negative numel: {positive_indices[0].numel() / negative_indices[0].numel() if negative_indices[0].numel() > 0 else 0.0}")
        print("-"*50)
        # import pdb; pdb.set_trace()

        if mask_template_token:
            # template_ids = torch.tensor([13708, 766, 26865, 29, 1339, 397, 10370], device=responses.device)
            template_ids = torch.tensor([29, 1339, 397, 10370], device=responses.device)
            template_token_mask = torch.isin(responses, template_ids)
            scores[template_token_mask] = 0

        if less_negative.enable:
            if less_negative.clip:
                eps = 1e-8
                clip_threshold = -positive_mean * less_negative.target_ratio
                factor = torch.where(scores < clip_threshold, clip_threshold / (scores + eps), torch.ones_like(scores))
                scores = scores * factor
            else:
                if less_negative.only_answer:
                    if less_negative.in_group:
                        for uid in id2index:
                            x_indices = []
                            y_indices = []
                            for idx in id2index[uid]:
                                diff = torch.diff(eos_mask[idx], prepend=torch.tensor([0]), append=torch.tensor([0]))
                                starts = torch.where(diff == 1)[0]
                                ends = torch.where(diff == -1)[0]
                                x = torch.full((ends[-1] - starts[-1],), idx)
                                y = torch.arange(starts[-1], ends[-1])
                                x_indices.append(x)
                                y_indices.append(y)
                            x_indices = torch.cat(x_indices)
                            y_indices = torch.cat(y_indices)
                            ans_indices = tuple([x_indices, y_indices])
                            ans_advantage = scores[ans_indices]
                            ans_positive_mask = ans_advantage > 0
                            ans_negative_mask = ans_advantage < 0
                            ans_positive_indices = (x_indices[ans_positive_mask], y_indices[ans_positive_mask])
                            ans_negative_indices = (x_indices[ans_negative_mask], y_indices[ans_negative_mask])
                            ans_positive_sum = scores[ans_positive_indices].sum()
                            ans_negative_sum = scores[ans_negative_indices].sum()
                            ans_ratio = ans_positive_sum / (-ans_negative_sum) if ans_negative_sum != 0 else 1
                            if less_negative.adjust_positive:
                                scores[ans_positive_indices] *= 1 / (ans_ratio * less_negative.target_ratio)
                            else:
                                scores[ans_negative_indices] *= ans_ratio * less_negative.target_ratio
                    else:
                        x_indices = []
                        y_indices = []
                        for idx in range(len(eos_mask)):
                            diff = torch.diff(eos_mask[idx], prepend=torch.tensor([0]), append=torch.tensor([0]))
                            starts = torch.where(diff == 1)[0]
                            ends = torch.where(diff == -1)[0]
                            x = torch.full((ends[-1] - starts[-1],), idx)
                            y = torch.arange(starts[-1], ends[-1])
                            x_indices.append(x)
                            y_indices.append(y)
                        x_indices = torch.cat(x_indices)
                        y_indices = torch.cat(y_indices)
                        ans_indices = tuple([x_indices, y_indices])
                        ans_advantage = scores[ans_indices]
                        ans_positive_mask = ans_advantage > 0
                        ans_negative_mask = ans_advantage < 0
                        ans_positive_indices = (x_indices[ans_positive_mask], y_indices[ans_positive_mask])
                        ans_negative_indices = (x_indices[ans_negative_mask], y_indices[ans_negative_mask])
                        ans_positive_sum = scores[ans_positive_indices].sum()
                        ans_negative_sum = scores[ans_negative_indices].sum()
                        ans_ratio = ans_positive_sum / (-ans_negative_sum) if ans_negative_sum != 0 else 1
                        print(f'ans_positive_sum / (-ans_negative_sum): {ans_ratio}')
                        if less_negative.adjust_positive:
                            scores[ans_positive_indices] *= 1 / (ans_ratio * less_negative.target_ratio)
                        else:
                            scores[ans_negative_indices] *= ans_ratio * less_negative.target_ratio
                elif less_negative.turn_level:
                    if less_negative.in_group:
                        for uid in id2index:
                            query_x_indices = [[] for _ in range(4)]
                            query_y_indices = [[] for _ in range(4)]
                            ans_x_indices = []
                            ans_y_indices = []
                            max_query_turn = 0
                            for idx in id2index[uid]:
                                diff = torch.diff(eos_mask[idx], prepend=torch.tensor([0]), append=torch.tensor([0]))
                                starts = torch.where(diff == 1)[0]
                                ends = torch.where(diff == -1)[0]

                                if len(starts) == 0 or len(ends) == 0:
                                    continue

                                for query_idx in range(len(starts)-1):
                                    x = torch.full((ends[query_idx] - starts[query_idx],), idx)
                                    y = torch.arange(starts[query_idx], ends[query_idx])
                                    query_x_indices[query_idx].append(x)
                                    query_y_indices[query_idx].append(y)
                                max_query_turn = max(max_query_turn, len(starts)-1)

                                ans_x = torch.full((ends[-1] - starts[-1],), idx)
                                ans_y = torch.arange(starts[-1], ends[-1])
                                ans_x_indices.append(ans_x)
                                ans_y_indices.append(ans_y)

                            turn_indices = []
                            for i in range(max_query_turn):
                                query_x_indices[i] = torch.cat(query_x_indices[i])
                                query_y_indices[i] = torch.cat(query_y_indices[i])
                                turn_indices.append(tuple([query_x_indices[i], query_y_indices[i]]))
                            ans_x_indices = torch.cat(ans_x_indices)
                            ans_y_indices = torch.cat(ans_y_indices)
                            ans_indices = tuple([ans_x_indices, ans_y_indices])
                            turn_indices.append(ans_indices)

                            for turn_idx, this_turn_indices in enumerate(turn_indices):
                                this_turn_advantage = scores[this_turn_indices]
                                this_turn_positive_mask = this_turn_advantage > 0
                                this_turn_negative_mask = this_turn_advantage < 0
                                this_turn_positive_indices = (this_turn_indices[0][this_turn_positive_mask], this_turn_indices[1][this_turn_positive_mask])
                                this_turn_negative_indices = (this_turn_indices[0][this_turn_negative_mask], this_turn_indices[1][this_turn_negative_mask])
                                this_turn_positive_sum = scores[this_turn_positive_indices].sum()
                                this_turn_negative_sum = scores[this_turn_negative_indices].sum()

                                adv_value_ratio = this_turn_positive_sum / (-this_turn_negative_sum) if this_turn_negative_sum != 0 else 1
                                numel_ratio = this_turn_positive_indices[0].numel() / (this_turn_negative_indices[0].numel()) if this_turn_negative_indices[0].numel() != 0 else 1
                                this_turn_ratio = numel_ratio if less_negative.use_numel_ratio else adv_value_ratio
                                if less_negative.adjust_positive:
                                    if this_turn_positive_sum == 0:
                                        scores[this_turn_negative_indices] = 0
                                    else:
                                        scores[this_turn_positive_indices] *= 1 / (this_turn_ratio) * less_negative.target_ratio
                                else:
                                    scores[this_turn_negative_indices] *= this_turn_ratio * less_negative.target_ratio
                                    if this_turn_negative_sum==0:
                                        scores[this_turn_positive_indices] = 0
                    else:
                        query_x_indices = [[] for _ in range(4)]
                        query_y_indices = [[] for _ in range(4)]
                        ans_x_indices = []
                        ans_y_indices = []
                        max_query_turn = 0
                        for idx in range(len(eos_mask)):
                            diff = torch.diff(eos_mask[idx], prepend=torch.tensor([0]), append=torch.tensor([0]))
                            starts = torch.where(diff == 1)[0]
                            ends = torch.where(diff == -1)[0]

                            if len(starts) == 0 or len(ends) == 0:
                                continue

                            for query_idx in range(len(starts)):
                                res_str = tokenizer.decode(responses[idx, starts[query_idx]:ends[query_idx]])
                                x = torch.full((ends[query_idx] - starts[query_idx],), idx)
                                y = torch.arange(starts[query_idx], ends[query_idx])
                                if '<answer>' in res_str or query_idx == len(starts) - 1:
                                    ans_x_indices.append(x)
                                    ans_y_indices.append(y)
                                else:
                                    query_x_indices[query_idx].append(x)
                                    query_y_indices[query_idx].append(y)
                            max_query_turn = max(max_query_turn, len(starts)-1)

                            
                            # for query_idx in range(len(starts)-1):
                            #     x = torch.full((ends[query_idx] - starts[query_idx],), idx)
                            #     y = torch.arange(starts[query_idx], ends[query_idx])
                            #     query_x_indices[query_idx].append(x)
                            #     query_y_indices[query_idx].append(y)
                            # max_query_turn = max(max_query_turn, len(starts)-1)

                            # ans_x = torch.full((ends[-1] - starts[-1],), idx)
                            # ans_y = torch.arange(starts[-1], ends[-1])
                            # ans_x_indices.append(ans_x)
                            # ans_y_indices.append(ans_y)

                        turn_indices = []
                        for i in range(max_query_turn):
                            query_x_indices[i] = torch.cat(query_x_indices[i])
                            query_y_indices[i] = torch.cat(query_y_indices[i])
                            turn_indices.append(tuple([query_x_indices[i], query_y_indices[i]]))
                        ans_x_indices = torch.cat(ans_x_indices)
                        ans_y_indices = torch.cat(ans_y_indices)
                        ans_indices = tuple([ans_x_indices, ans_y_indices])
                        turn_indices.append(ans_indices)

                        for turn_idx, this_turn_indices in enumerate(turn_indices):
                            this_turn_advantage = scores[this_turn_indices]
                            this_turn_positive_mask = this_turn_advantage > 0
                            this_turn_negative_mask = this_turn_advantage < 0
                            this_turn_positive_indices = (this_turn_indices[0][this_turn_positive_mask], this_turn_indices[1][this_turn_positive_mask])
                            this_turn_negative_indices = (this_turn_indices[0][this_turn_negative_mask], this_turn_indices[1][this_turn_negative_mask])
                            this_turn_positive_sum = scores[this_turn_positive_indices].sum()
                            this_turn_negative_sum = scores[this_turn_negative_indices].sum()
                            adv_value_ratio = this_turn_positive_sum / (-this_turn_negative_sum) if this_turn_negative_sum != 0 else 1
                            numel_ratio = this_turn_positive_indices[0].numel() / (this_turn_negative_indices[0].numel()) if this_turn_negative_indices[0].numel() != 0 else 1
                            this_turn_ratio = numel_ratio if less_negative.use_numel_ratio else adv_value_ratio
                            print(f'Turn {turn_idx}: positive_numel / negative_numel: {numel_ratio}')
                            print(f'Turn {turn_idx}: positive_numel_mean {this_turn_positive_indices[0].numel()/torch.unique(this_turn_positive_indices[0]).numel() if torch.unique(this_turn_positive_indices[0]).numel() else 0}, negative_numel_mean {this_turn_negative_indices[0].numel() / torch.unique(this_turn_negative_indices[0]).numel() if torch.unique(this_turn_negative_indices[0]).numel() else 0}')
                            print(f'Turn {turn_idx}: positive_sum / (-negative_sum): {adv_value_ratio}')
                            # if less_negative.adjust_positive:
                            #     if this_turn_positive_sum == 0:
                            #         scores[this_turn_negative_indices] = 0
                            #     else:
                            #         scores[this_turn_positive_indices] *= 1 / (this_turn_ratio * less_negative.target_ratio)
                            # else:
                            #     scores[this_turn_negative_indices] *= this_turn_ratio * less_negative.target_ratio
                elif less_negative.query_ans_level:
                    query_indices = []
                    query_indices = []
                    ans_x_indices = []
                    ans_y_indices = []
                    max_query_turn = 0
                    for idx in range(len(eos_mask)):
                        diff = torch.diff(eos_mask[idx], prepend=torch.tensor([0]), append=torch.tensor([0]))
                        starts = torch.where(diff == 1)[0]
                        ends = torch.where(diff == -1)[0]

                        if len(starts) == 0 or len(ends) == 0:
                            continue

                        for query_idx in range(len(starts)-1):
                            x = torch.full((ends[query_idx] - starts[query_idx],), idx)
                            y = torch.arange(starts[query_idx], ends[query_idx])
                            query_x_indices.append(x)
                            query_y_indices.append(y)

                        ans_x = torch.full((ends[-1] - starts[-1],), idx)
                        ans_y = torch.arange(starts[-1], ends[-1])
                        ans_x_indices.append(ans_x)
                        ans_y_indices.append(ans_y)

                    turn_indices = []
                    query_x_indices = torch.cat(query_x_indices)
                    query_y_indices = torch.cat(query_y_indices)
                    turn_indices.append(tuple([query_x_indices, query_y_indices]))
                    ans_x_indices = torch.cat(ans_x_indices)
                    ans_y_indices = torch.cat(ans_y_indices)
                    ans_indices = tuple([ans_x_indices, ans_y_indices])
                    turn_indices.append(ans_indices)

                    for turn_idx, this_turn_indices in enumerate(turn_indices):
                        this_turn_advantage = scores[this_turn_indices]
                        this_turn_positive_mask = this_turn_advantage > 0
                        this_turn_negative_mask = this_turn_advantage < 0
                        this_turn_positive_indices = (this_turn_indices[0][this_turn_positive_mask], this_turn_indices[1][this_turn_positive_mask])
                        this_turn_negative_indices = (this_turn_indices[0][this_turn_negative_mask], this_turn_indices[1][this_turn_negative_mask])
                        this_turn_positive_sum = scores[this_turn_positive_indices].sum()
                        this_turn_negative_sum = scores[this_turn_negative_indices].sum()
                        adv_value_ratio = this_turn_positive_sum / (-this_turn_negative_sum) if this_turn_negative_sum != 0 else 1
                        numel_ratio = this_turn_positive_indices[0].numel() / (this_turn_negative_indices[0].numel()) if this_turn_negative_indices[0].numel() != 0 else 1
                        this_turn_ratio = numel_ratio if less_negative.use_numel_ratio else adv_value_ratio
                        print(f'Turn {turn_idx}: positive_numel / negative_numel: {numel_ratio}')
                        print(f'Turn {turn_idx}: positive_numel_mean {this_turn_positive_indices[0].numel()/torch.unique(this_turn_positive_indices[0]).numel() if torch.unique(this_turn_positive_indices[0]).numel() else 0}, negative_numel_mean {this_turn_negative_indices[0].numel() / torch.unique(this_turn_negative_indices[0]).numel() if torch.unique(this_turn_negative_indices[0]).numel() else 0}')
                        print(f'Turn {turn_idx}: positive_sum / (-negative_sum): {adv_value_ratio}')
                        if turn_idx == 0:
                            if this_turn_positive_sum == 0:
                                scores[this_turn_negative_indices] = 0
                            else:
                                scores[this_turn_positive_indices] *= 1 / (this_turn_ratio * 1.0)
                        else:
                            scores[this_turn_negative_indices] *= this_turn_ratio * 0.7
                else:
                    if less_negative.in_group:
                        print("Ingroup")
                        for uid in id2index:
                            this_group_scores = scores[id2index[uid]]
                            this_group_positive_indices = this_group_scores > 0
                            this_group_negative_indices = this_group_scores < 0
                            this_group_positive_sum = this_group_scores[this_group_positive_indices].sum()
                            this_group_negative_sum = this_group_scores[this_group_negative_indices].sum()
                            this_group_adv_value_ratio = this_group_positive_sum / (-this_group_negative_sum) if this_group_negative_sum != 0 else 1
                            if less_negative.adjust_positive:
                                if this_group_positive_sum == 0:
                                    this_group_scores[this_group_negative_indices] = 0
                                else:
                                    this_group_scores[this_group_positive_indices] *= 1 / (this_group_adv_value_ratio * less_negative.target_ratio)
                            else:
                                this_group_scores[this_group_negative_indices] *= this_group_adv_value_ratio * less_negative.target_ratio
                            scores[id2index[uid]] = this_group_scores
                    else:
                        if less_negative.adjust_positive:
                            if ratio == 0:
                                scores[negative_indices] = 0
                            else:
                                scores[positive_indices] *= 1 / (ratio * less_negative.target_ratio)
                        else:
                            scores[negative_indices] *= ratio * less_negative.target_ratio
        
        if less_negative.enable or mask_template_token:
            positive_sum = scores[positive_indices].sum()
            negative_sum = scores[negative_indices].sum()
            print(f"Adjusted positive/negative ratio: {positive_sum / (-negative_sum)}")
            print(f"Adjusted negative/positive ratio: {-negative_sum / positive_sum}")

    return scores, scores

def get_ratio(all_advantage, advantage_name):
    positive_indices = torch.where(all_advantage > 0)
    negative_indices = torch.where(all_advantage < 0)
    positive_sum = all_advantage[positive_indices].sum()
    positive_mean = all_advantage[positive_indices].mean()
    negative_sum = all_advantage[negative_indices].sum()
    ratio = positive_sum / (-negative_sum)
    print(f"Original {advantage_name} positive/negative ratio: {ratio}")
    print(f"Original {advantage_name} negative/positive ratio: {-negative_sum / positive_sum}")
    print("-"*50)
    print(f"positive {advantage_name} numel / negative numel: {positive_indices[0].numel() / negative_indices[0].numel()}")
    print("-"*50)
    return positive_indices, negative_indices, positive_mean, ratio

def advantage_scale(scores, advantage_name, less_negative, eos_mask, id2index, responses, tokenizer):
    positive_indices, negative_indices, positive_mean, ratio = get_ratio(scores, advantage_name)
    if less_negative.enable:
        if less_negative.clip:
            eps = 1e-8
            clip_threshold = -positive_mean * less_negative.target_ratio
            factor = torch.where(scores < clip_threshold, clip_threshold / (scores + eps), torch.ones_like(scores))
            scores = scores * factor
        else:
            if less_negative.only_answer:
                if less_negative.in_group:
                    risky_cnt = 0
                    for uid in id2index:
                        x_indices = []
                        y_indices = []
                        for idx in id2index[uid]:
                            diff = torch.diff(eos_mask[idx], prepend=torch.tensor([0]), append=torch.tensor([0]))
                            starts = torch.where(diff == 1)[0]
                            ends = torch.where(diff == -1)[0]
                            # res_str = tokenizer.decode(responses[idx, starts[-1]:ends[-1]])
                            # if len(starts) == 0 or len(ends) == 0 or '<answer>' not in res_str:
                            #     continue
                            if len(starts) == 0 or len(ends) == 0:
                                continue
                            x = torch.full((ends[-1] - starts[-1],), idx)
                            y = torch.arange(starts[-1], ends[-1])
                            x_indices.append(x)
                            y_indices.append(y)
                        if len(x_indices) == 0:
                            continue
                        x_indices = torch.cat(x_indices)
                        y_indices = torch.cat(y_indices)
                        ans_indices = tuple([x_indices, y_indices])
                        ans_advantage = scores[ans_indices]
                        ans_positive_mask = ans_advantage > 0
                        ans_negative_mask = ans_advantage < 0
                        ans_positive_indices = (x_indices[ans_positive_mask], y_indices[ans_positive_mask])
                        ans_negative_indices = (x_indices[ans_negative_mask], y_indices[ans_negative_mask])
                        ans_positive_sum = scores[ans_positive_indices].sum()
                        ans_negative_sum = scores[ans_negative_indices].sum()
                        ans_ratio = ans_positive_sum / (-ans_negative_sum) if ans_negative_sum != 0 else 0
                        if less_negative.only_apply_risky:
                            if ans_positive_sum!=0 and (-ans_negative_sum / ans_positive_sum) <= less_negative.target_ratio:
                                continue
                        risky_cnt += 1
                        if less_negative.adjust_positive:
                            if ans_positive_sum == 0 or ans_negative_sum == 0:
                                scores[ans_negative_indices] = 0
                                scores[ans_positive_indices] = 0
                            else:
                                scores[ans_positive_indices] *= 1 / (ans_ratio * less_negative.target_ratio)
                        else:
                            scores[ans_negative_indices] *= ans_ratio * less_negative.target_ratio
                            if ans_positive_sum == 0 or ans_negative_sum == 0:
                                scores[ans_negative_indices] = 0
                                scores[ans_positive_indices] = 0
                    print(f"Risky count: {risky_cnt}, Group count: {len(id2index)}, Risky ratio: {risky_cnt / len(id2index)}")
                else:
                    x_indices = []
                    y_indices = []
                    for idx in range(len(eos_mask)):
                        diff = torch.diff(eos_mask[idx], prepend=torch.tensor([0]), append=torch.tensor([0]))
                        starts = torch.where(diff == 1)[0]
                        ends = torch.where(diff == -1)[0]
                        if len(starts) == 0 or len(ends) == 0:
                            continue
                        x = torch.full((ends[-1] - starts[-1],), idx)
                        y = torch.arange(starts[-1], ends[-1])
                        x_indices.append(x)
                        y_indices.append(y)
                    x_indices = torch.cat(x_indices)
                    y_indices = torch.cat(y_indices)
                    ans_indices = tuple([x_indices, y_indices])
                    ans_advantage = scores[ans_indices]
                    ans_positive_mask = ans_advantage > 0
                    ans_negative_mask = ans_advantage < 0
                    ans_positive_indices = (x_indices[ans_positive_mask], y_indices[ans_positive_mask])
                    ans_negative_indices = (x_indices[ans_negative_mask], y_indices[ans_negative_mask])
                    ans_positive_sum = scores[ans_positive_indices].sum()
                    ans_negative_sum = scores[ans_negative_indices].sum()
                    ans_ratio = ans_positive_sum / (-ans_negative_sum) if ans_negative_sum != 0 else 0
                    print(f'ans_positive_sum / (-ans_negative_sum): {ans_ratio}')
                    if less_negative.adjust_positive:
                        scores[ans_positive_indices] *= 1 / (ans_ratio * less_negative.target_ratio)
                    else:
                        scores[ans_negative_indices] *= ans_ratio * less_negative.target_ratio
            elif less_negative.turn_level:
                if less_negative.in_group:
                    for uid in id2index:
                        query_x_indices = [[] for _ in range(4)]
                        query_y_indices = [[] for _ in range(4)]
                        ans_x_indices = []
                        ans_y_indices = []
                        max_query_turn = 0
                        for idx in id2index[uid]:
                            diff = torch.diff(eos_mask[idx], prepend=torch.tensor([0]), append=torch.tensor([0]))
                            starts = torch.where(diff == 1)[0]
                            ends = torch.where(diff == -1)[0]

                            if len(starts) == 0 or len(ends) == 0:
                                continue

                            for query_idx in range(len(starts)-1):
                                res_str = tokenizer.decode(responses[idx, starts[query_idx]:ends[query_idx]])
                                x = torch.full((ends[query_idx] - starts[query_idx],), idx)
                                y = torch.arange(starts[query_idx], ends[query_idx])
                                if '<answer>' in res_str or query_idx == len(starts) - 1:
                                    ans_x_indices.append(x)
                                    ans_y_indices.append(y)
                                else:
                                    query_x_indices[query_idx].append(x)
                                    query_y_indices[query_idx].append(y)
                            max_query_turn = max(max_query_turn, len(starts)-1)

                            # ans_x = torch.full((ends[-1] - starts[-1],), idx)
                            # ans_y = torch.arange(starts[-1], ends[-1])
                            # ans_x_indices.append(ans_x)
                            # ans_y_indices.append(ans_y)

                        turn_indices = []
                        for i in range(max_query_turn):
                            query_x_indices[i] = torch.cat(query_x_indices[i])
                            query_y_indices[i] = torch.cat(query_y_indices[i])
                            turn_indices.append(tuple([query_x_indices[i], query_y_indices[i]]))
                        if len(ans_x_indices) > 0:
                            ans_x_indices = torch.cat(ans_x_indices)
                            ans_y_indices = torch.cat(ans_y_indices)
                            ans_indices = tuple([ans_x_indices, ans_y_indices])
                            turn_indices.append(ans_indices)

                        risky_cnt = 0
                        for turn_idx, this_turn_indices in enumerate(turn_indices):
                            this_turn_advantage = scores[this_turn_indices]
                            this_turn_positive_mask = this_turn_advantage > 0
                            this_turn_negative_mask = this_turn_advantage < 0
                            this_turn_positive_indices = (this_turn_indices[0][this_turn_positive_mask], this_turn_indices[1][this_turn_positive_mask])
                            this_turn_negative_indices = (this_turn_indices[0][this_turn_negative_mask], this_turn_indices[1][this_turn_negative_mask])
                            this_turn_positive_sum = scores[this_turn_positive_indices].sum()
                            this_turn_negative_sum = scores[this_turn_negative_indices].sum()

                            adv_value_ratio = this_turn_positive_sum / (-this_turn_negative_sum) if this_turn_negative_sum != 0 else 0
                            numel_ratio = this_turn_positive_indices[0].numel() / (this_turn_negative_indices[0].numel()) if this_turn_negative_indices[0].numel() != 0 else 1
                            this_turn_ratio = numel_ratio if less_negative.use_numel_ratio else adv_value_ratio
                            if less_negative.only_apply_risky:
                                if this_turn_positive_sum!=0 and (-this_turn_negative_sum / this_turn_positive_sum) <= less_negative.target_ratio:
                                    continue
                            risky_cnt += 1
                            if less_negative.adjust_positive:
                                if this_turn_positive_sum == 0 or this_turn_negative_sum == 0:
                                    scores[this_turn_negative_indices] = 0
                                    scores[this_turn_positive_indices] = 0
                                else:
                                    scores[this_turn_positive_indices] *= 1 / (this_turn_ratio) * less_negative.target_ratio
                            else:
                                scores[this_turn_negative_indices] *= this_turn_ratio * less_negative.target_ratio
                                if this_turn_negative_sum == 0:
                                    scores[this_turn_positive_indices] = 0
                else:
                    query_x_indices = [[] for _ in range(4)]
                    query_y_indices = [[] for _ in range(4)]
                    ans_x_indices = []
                    ans_y_indices = []
                    max_query_turn = 0
                    for idx in range(len(eos_mask)):
                        diff = torch.diff(eos_mask[idx], prepend=torch.tensor([0]), append=torch.tensor([0]))
                        starts = torch.where(diff == 1)[0]
                        ends = torch.where(diff == -1)[0]

                        if len(starts) == 0 or len(ends) == 0:
                            continue

                        for query_idx in range(len(starts)-1):
                            x = torch.full((ends[query_idx] - starts[query_idx],), idx)
                            y = torch.arange(starts[query_idx], ends[query_idx])
                            query_x_indices[query_idx].append(x)
                            query_y_indices[query_idx].append(y)
                        max_query_turn = max(max_query_turn, len(starts)-1)

                        ans_x = torch.full((ends[-1] - starts[-1],), idx)
                        ans_y = torch.arange(starts[-1], ends[-1])
                        ans_x_indices.append(ans_x)
                        ans_y_indices.append(ans_y)

                    turn_indices = []
                    for i in range(max_query_turn):
                        query_x_indices[i] = torch.cat(query_x_indices[i])
                        query_y_indices[i] = torch.cat(query_y_indices[i])
                        turn_indices.append(tuple([query_x_indices[i], query_y_indices[i]]))
                    ans_x_indices = torch.cat(ans_x_indices)
                    ans_y_indices = torch.cat(ans_y_indices)
                    ans_indices = tuple([ans_x_indices, ans_y_indices])
                    turn_indices.append(ans_indices)

                    for turn_idx, this_turn_indices in enumerate(turn_indices):
                        this_turn_advantage = scores[this_turn_indices]
                        this_turn_positive_mask = this_turn_advantage > 0
                        this_turn_negative_mask = this_turn_advantage < 0
                        this_turn_positive_indices = (this_turn_indices[0][this_turn_positive_mask], this_turn_indices[1][this_turn_positive_mask])
                        this_turn_negative_indices = (this_turn_indices[0][this_turn_negative_mask], this_turn_indices[1][this_turn_negative_mask])
                        this_turn_positive_sum = scores[this_turn_positive_indices].sum()
                        this_turn_negative_sum = scores[this_turn_negative_indices].sum()
                        adv_value_ratio = this_turn_positive_sum / (-this_turn_negative_sum) if this_turn_negative_sum != 0 else 0
                        numel_ratio = this_turn_positive_indices[0].numel() / (this_turn_negative_indices[0].numel()) if this_turn_negative_indices[0].numel() != 0 else 1
                        this_turn_ratio = numel_ratio if less_negative.use_numel_ratio else adv_value_ratio
                        print(f'Turn {turn_idx}: positive_numel / negative_numel: {numel_ratio}')
                        print(f'Turn {turn_idx}: positive_numel_mean {this_turn_positive_indices[0].numel()/torch.unique(this_turn_positive_indices[0]).numel() if torch.unique(this_turn_positive_indices[0]).numel() else 0}, negative_numel_mean {this_turn_negative_indices[0].numel() / torch.unique(this_turn_negative_indices[0]).numel() if torch.unique(this_turn_negative_indices[0]).numel() else 0}')
                        print(f'Turn {turn_idx}: positive_sum / (-negative_sum): {adv_value_ratio}')
                        if less_negative.adjust_positive:
                            if this_turn_positive_sum == 0 or this_turn_negative_sum == 0:
                                scores[this_turn_negative_indices] = 0
                                scores[this_turn_positive_indices] = 0
                            else:
                                scores[this_turn_positive_indices] *= 1 / (this_turn_ratio * less_negative.target_ratio)
                        else:
                            scores[this_turn_negative_indices] *= this_turn_ratio * less_negative.target_ratio
            else:
                if less_negative.in_group:
                    print("Ingroup")
                    for uid in id2index:
                        this_group_scores = scores[id2index[uid]]
                        this_group_positive_indices = this_group_scores > 0
                        this_group_negative_indices = this_group_scores < 0
                        this_group_positive_sum = this_group_scores[this_group_positive_indices].sum()
                        this_group_negative_sum = this_group_scores[this_group_negative_indices].sum()
                        this_group_adv_value_ratio = this_group_positive_sum / (-this_group_negative_sum) if this_group_negative_sum != 0 else 0
                        if less_negative.adjust_positive:
                            if this_group_positive_sum == 0 or this_group_negative_sum == 0:
                                this_group_scores[this_group_negative_indices] = 0
                                this_group_scores[this_group_positive_indices] = 0
                            else:
                                this_group_scores[this_group_positive_indices] *= 1 / (this_group_adv_value_ratio * less_negative.target_ratio)
                        else:
                            this_group_scores[this_group_negative_indices] *= this_group_adv_value_ratio * less_negative.target_ratio
                        scores[id2index[uid]] = this_group_scores
                else:
                    if less_negative.adjust_positive:
                        if ratio == 0:
                            scores[negative_indices] = 0
                        else:
                            scores[positive_indices] *= 1 / (ratio * less_negative.target_ratio)
                    else:
                        scores[negative_indices] *= ratio * less_negative.target_ratio
    
    positive_sum = scores[positive_indices].sum()
    negative_sum = scores[negative_indices].sum()
    print(f"Adjusted positive/negative ratio: {positive_sum / (-negative_sum)}")
    print(f"Adjusted negative/positive ratio: {-negative_sum / positive_sum}")
    return scores

def compute_grpo_outcome_turngroup_advantage(token_level_rewards: torch.Tensor,
                                   token_level_query_scores: torch.Tensor,
                                   less_negative,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    print("compute_grpo_outcome_turngroup_advantage")
    response_length = token_level_rewards.shape[-1]
    ans_scores = token_level_rewards.sum(dim=-1)
    query_scores = []
    query_scores_indices = []
    bsz = ans_scores.shape[0]
    max_turn = 0
    for i in range(bsz):
        relevance_scores_indices = torch.nonzero(token_level_query_scores[i]).squeeze()
        relevance_scores = token_level_query_scores[i][relevance_scores_indices]
        relevance_scores = relevance_scores.tolist()
        if type(relevance_scores) != list:
            relevance_scores = [relevance_scores]
        relevance_scores_indices = relevance_scores_indices.tolist()
        if type(relevance_scores_indices) != list:
            relevance_scores_indices = [relevance_scores_indices]
        query_scores.append(relevance_scores)
        query_scores_indices.append(relevance_scores_indices)
        max_turn = max(max_turn, len(relevance_scores_indices))

    id2query_score = [defaultdict(list) for _ in range(max_turn)]
    id2query_mean = [{} for _ in range(max_turn)]
    id2query_std = [{} for _ in range(max_turn)]

    id2ans_score = defaultdict(list)
    id2ans_mean = {}
    id2ans_std = {}
    id2maxans_score = defaultdict(int)

    with torch.no_grad():
        for i in range(bsz):
            for turn_idx in range(len(query_scores[i])):
                id2query_score[turn_idx][index[i]].append(query_scores[i][turn_idx])
            id2ans_score[index[i]].append(ans_scores[i])
            id2maxans_score[index[i]] = max(id2maxans_score[index[i]], ans_scores[i])

        for idx in id2ans_score:
            for turn_idx in range(max_turn):
                if len(id2query_score[turn_idx][idx]) <= 1:
                    id2query_mean[turn_idx][idx] = torch.tensor(0.0)
                    id2query_std[turn_idx][idx] = torch.tensor(1.0)
                elif len(id2query_score[turn_idx][idx]) > 1:
                    id2query_mean[turn_idx][idx] = torch.mean(torch.tensor(id2query_score[turn_idx][idx]))
                    id2query_std[turn_idx][idx] = torch.std(torch.tensor([id2query_score[turn_idx][idx]]))

            if len(id2ans_score[idx]) == 1:
                id2ans_mean[idx] = torch.tensor(0.0)
                id2ans_std[idx] = torch.tensor(1.0)
            elif len(id2ans_score[idx]) > 1:
                id2ans_mean[idx] = torch.mean(torch.tensor(id2ans_score[idx]))
                id2ans_std[idx] = torch.std(torch.tensor([id2ans_score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")

        all_advantage_ans = []
        all_advantage_query = []
        for i in range(bsz):
            last_query_adv = None
            for turn_idx in range(len(query_scores[i])):   
                this_query_advscores = (query_scores[i][turn_idx] - id2query_mean[turn_idx][index[i]]) / (id2query_std[turn_idx][index[i]] + epsilon)
                adv_this_query = torch.full((query_scores_indices[i][turn_idx],), this_query_advscores)
                if last_query_adv is not None:
                    adv_this_query[:last_query_adv.shape[0]] = last_query_adv
                last_query_adv = adv_this_query
            if last_query_adv is not None:
                zero_adv = torch.zeros((response_length,))
                zero_adv[:last_query_adv.shape[0]] = last_query_adv
                last_query_adv = zero_adv
            else:
                last_query_adv = torch.zeros((response_length,))
            all_advantage_query.append(last_query_adv)
            
            ans_scores[i] = (ans_scores[i] - id2ans_mean[index[i]]) / (id2ans_std[index[i]] + epsilon)
            adv_ans = torch.full((response_length,), ans_scores[i])
            all_advantage_ans.append(adv_ans)

        all_advantage_ans = torch.stack(all_advantage_ans) * eos_mask
        all_advantage_query = torch.stack(all_advantage_query) * eos_mask
        
        all_advantage_ans = advantage_scale(all_advantage_ans, "ans_advantage", less_negative, eos_mask)
        all_advantage_query = advantage_scale(all_advantage_query, "query_advantage", less_negative, eos_mask)
    
    return all_advantage_query, all_advantage_ans, all_advantage_ans

def compute_grpo_outcome_softpenalty_turngroup_advantage(token_level_rewards: torch.Tensor,
                                   token_level_query_scores: torch.Tensor,
                                   max_score_tensor: torch.Tensor,
                                   less_negative,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    print("compute_grpo_outcome_softpenalty_turngroup_advantage")
    response_length = token_level_rewards.shape[-1]
    ans_scores = token_level_rewards.sum(dim=-1)
    group_max_score = max_score_tensor.sum(dim=-1)
    query_scores = []
    query_scores_indices = []
    bsz = ans_scores.shape[0]
    max_turn = 0
    for i in range(bsz):
        relevance_scores_indices = torch.nonzero(token_level_query_scores[i]).squeeze()
        relevance_scores = token_level_query_scores[i][relevance_scores_indices]
        relevance_scores = relevance_scores.tolist()
        if type(relevance_scores) != list:
            relevance_scores = [relevance_scores]
        relevance_scores_indices = relevance_scores_indices.tolist()
        if type(relevance_scores_indices) != list:
            relevance_scores_indices = [relevance_scores_indices]
        query_scores.append(relevance_scores)
        query_scores_indices.append(relevance_scores_indices)
        max_turn = max(max_turn, len(relevance_scores_indices))

    id2query_score = [defaultdict(list) for _ in range(max_turn)]
    id2query_mean = [{} for _ in range(max_turn)]
    id2query_std = [{} for _ in range(max_turn)]

    id2ans_score = defaultdict(list)
    id2ans_mean = {}
    id2ans_std = {}
    id2maxans_score = defaultdict(int)
    id2index = defaultdict(list)

    with torch.no_grad():
        for i in range(bsz):
            id2ans_score[index[i]].append(ans_scores[i])
            id2maxans_score[index[i]] = max(id2maxans_score[index[i]], ans_scores[i])
            id2index[index[i]].append(i)

        for i in range(bsz):
            for turn_idx in range(len(query_scores[i])):
                if query_scores[i][turn_idx] > 1e-6:
                    adjusted_query_score = ans_scores[i] + (group_max_score[i] - ans_scores[i]) * query_scores[i][turn_idx]
                    query_scores[i][turn_idx] = adjusted_query_score
                    id2query_score[turn_idx][index[i]].append(adjusted_query_score)
                else:
                    query_scores[i][turn_idx] = ans_scores[i]
                    id2query_score[turn_idx][index[i]].append(ans_scores[i])

        for idx in id2ans_score:
            for turn_idx in range(max_turn):
                if len(id2query_score[turn_idx][idx]) <= 1:
                    id2query_mean[turn_idx][idx] = torch.tensor(0.0)
                    id2query_std[turn_idx][idx] = torch.tensor(1.0)
                elif len(id2query_score[turn_idx][idx]) > 1:
                    id2query_mean[turn_idx][idx] = torch.mean(torch.tensor(id2query_score[turn_idx][idx]))
                    id2query_std[turn_idx][idx] = torch.std(torch.tensor([id2query_score[turn_idx][idx]]))

            if len(id2ans_score[idx]) == 1:
                id2ans_mean[idx] = torch.tensor(0.0)
                id2ans_std[idx] = torch.tensor(1.0)
            elif len(id2ans_score[idx]) > 1:
                id2ans_mean[idx] = torch.mean(torch.tensor(id2ans_score[idx]))
                id2ans_std[idx] = torch.std(torch.tensor([id2ans_score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")

        all_advantage = []
        for i in range(bsz):
            last_query_adv = None
            for turn_idx in range(len(query_scores[i])):   
                this_query_advscores = (query_scores[i][turn_idx] - id2query_mean[turn_idx][index[i]]) / (id2query_std[turn_idx][index[i]] + epsilon)
                adv_this_query = torch.full((query_scores_indices[i][turn_idx],), this_query_advscores)
                if last_query_adv is not None:
                    adv_this_query[:last_query_adv.shape[0]] = last_query_adv
                last_query_adv = adv_this_query
            
            ans_scores[i] = (ans_scores[i] - id2ans_mean[index[i]]) / (id2ans_std[index[i]] + epsilon)
            adv_ans = torch.full((response_length,), ans_scores[i])
            if last_query_adv is not None:
                adv_ans[:last_query_adv.shape[0]] = last_query_adv
            all_advantage.append(adv_ans)

        all_advantage = torch.stack(all_advantage) * eos_mask
        all_advantage = advantage_scale(all_advantage, "ans_advantage", less_negative, eos_mask, id2index)
    
    return all_advantage, all_advantage
    

def compute_grpo_outcome_querygroup_advantage(token_level_rewards: torch.Tensor,
                                   token_level_query_scores: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    print("compute_grpo_outcome_querygroup_advantage")
    response_length = token_level_rewards.shape[-1]
    ans_scores = token_level_rewards.sum(dim=-1)
    query_scores = []
    query_scores_indices = []
    bsz = ans_scores.shape[0]
    max_turn = 0
    for i in range(bsz):
        relevance_scores_indices = torch.nonzero(token_level_query_scores[i]).squeeze()
        relevance_scores = token_level_query_scores[i][relevance_scores_indices]
        relevance_scores = relevance_scores.tolist()
        if type(relevance_scores) != list:
            relevance_scores = [relevance_scores]
        relevance_scores_indices = relevance_scores_indices.tolist()
        if type(relevance_scores_indices) != list:
            relevance_scores_indices = [relevance_scores_indices]
        query_scores.append(relevance_scores)
        query_scores_indices.append(relevance_scores_indices)
        max_turn = max(max_turn, len(relevance_scores_indices))

    id2query_score = defaultdict(list)
    id2query_mean = {}
    id2query_std = {}

    id2ans_score = defaultdict(list)
    id2ans_mean = {}
    id2ans_std = {}

    with torch.no_grad():
        for i in range(bsz):
            id2query_score[index[i]].extend(query_scores[i])
            id2ans_score[index[i]].append(ans_scores[i])
        for idx in id2ans_score:
            if len(id2query_score[idx]) <= 1:
                id2query_mean[idx] = torch.tensor(0.0)
                id2query_std[idx] = torch.tensor(1.0)
            elif len(id2query_score[idx]) > 1:
                id2query_mean[idx] = torch.mean(torch.tensor(id2query_score[idx]))
                id2query_std[idx] = torch.std(torch.tensor(id2query_score[idx]))

            if len(id2ans_score[idx]) == 1:
                id2ans_mean[idx] = torch.tensor(0.0)
                id2ans_std[idx] = torch.tensor(1.0)
            elif len(id2ans_score[idx]) > 1:
                id2ans_mean[idx] = torch.mean(torch.tensor(id2ans_score[idx]))
                id2ans_std[idx] = torch.std(torch.tensor([id2ans_score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")

        all_advantage = []
        for i in range(bsz):
            last_adv = None
            for turn_idx in range(len(query_scores[i])):
                this_query_advscores = (query_scores[i][turn_idx] - id2query_mean[index[i]]) / (id2query_std[index[i]] + epsilon)
                adv_this_query = torch.full((query_scores_indices[i][turn_idx],), this_query_advscores)
                if last_adv is not None:
                    adv_this_query[:last_adv.shape[0]] = last_adv
                last_adv = adv_this_query
            
            ans_scores[i] = (ans_scores[i] - id2ans_mean[index[i]]) / (id2ans_std[index[i]] + epsilon)
            adv_ans = torch.full((response_length,), ans_scores[i])
            
            if last_adv is not None:
                adv_ans[:last_adv.shape[0]] = last_adv * 0.7 + adv_ans[:last_adv.shape[0]] * 0.3
                # adv_ans[:last_adv.shape[0]] = last_adv
            all_advantage.append(adv_ans)
        all_advantage = torch.stack(all_advantage) * eos_mask
    
    return all_advantage, all_advantage


def compute_grpo_outcome_goodquerypositive_segment_advantage(token_level_rewards: torch.Tensor,
                                   token_level_good_query: torch.Tensor,
                                   max_score_tensor: torch.Tensor,
                                   less_negative,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    print("compute_grpo_outcome_goodquerypositive_segment_advantage")
    response_length = token_level_rewards.shape[-1]
    ans_scores = token_level_rewards.sum(dim=-1)
    group_max_score = max_score_tensor.sum(dim=-1)
    query_scores = []
    query_scores_indices = []
    bsz = ans_scores.shape[0]
    max_turn = 0
    for i in range(bsz):
        good_scores_indices = torch.nonzero(token_level_good_query[i]).squeeze()
        good_scores = token_level_good_query[i][good_scores_indices]
        good_scores = good_scores.tolist()
        if type(good_scores) != list:
            good_scores = [good_scores]
        good_scores_indices = good_scores_indices.tolist()
        if type(good_scores_indices) != list:
            good_scores_indices = [good_scores_indices]
        query_scores.append(good_scores)
        query_scores_indices.append(good_scores_indices)
        max_turn = max(max_turn, len(good_scores_indices))

    id2ans_score = defaultdict(list)
    id2ans_mean = {}
    id2ans_std = {}
    id2max_score = {}

    with torch.no_grad():
        for i in range(bsz):
            id2ans_score[index[i]].append(ans_scores[i])
            if index[i] not in id2max_score:
                id2max_score[index[i]] = ans_scores[i]
            else:
                id2max_score[index[i]] = max(id2max_score[index[i]], ans_scores[i])
        global_std = torch.std(ans_scores)
        for idx in id2ans_score:
            if len(id2ans_score[idx]) == 1:
                id2ans_mean[idx] = torch.tensor(0.0)
                id2ans_std[idx] = torch.tensor(1.0)
            elif len(id2ans_score[idx]) > 1:
                id2ans_mean[idx] = torch.mean(torch.tensor(id2ans_score[idx]))
                id2ans_std[idx] = torch.std(torch.tensor([id2ans_score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        
        all_advantage = []
        id2adv_mean = {}
        id2adv_std = {}
        for i in range(bsz):
            ans_adv = (ans_scores[i] - id2ans_mean[index[i]]) / (id2ans_std[index[i]] + epsilon)
            adv_ans = torch.full((response_length,), ans_adv)

            last_adv = None
            for turn_idx in range(len(query_scores[i])):
                if query_scores[i][turn_idx] > 0.1:
                    this_query_score = (group_max_score[i] - id2ans_mean[index[i]]) * query_scores[i][turn_idx] + id2ans_mean[index[i]]
                    this_query_score = max(this_query_score, ans_scores[i])
                    # assert id2max_score[index[i]] == group_max_score[i]
                    if id2max_score[index[i]] < group_max_score[i]:
                        this_query_score = (this_query_score - id2ans_mean[index[i]]) / (global_std + epsilon)
                    else:
                        this_query_score = (this_query_score - id2ans_mean[index[i]]) / (id2ans_std[index[i]] + epsilon)
                else:
                    this_query_score = ans_scores[i]
                    this_query_score = (this_query_score - id2ans_mean[index[i]]) / (id2ans_std[index[i]] + epsilon)
                adv_this_query = torch.full((query_scores_indices[i][turn_idx],), this_query_score)
                if last_adv is not None:
                    adv_this_query[:last_adv.shape[0]] = last_adv
                last_adv = adv_this_query
            
            if last_adv is not None:
                adv_ans[:last_adv.shape[0]] = last_adv
            all_advantage.append(adv_ans)

        all_advantage = torch.stack(all_advantage) * eos_mask

        all_advantage = advantage_scale(all_advantage, "advantage", less_negative, eos_mask)
    
    return all_advantage, all_advantage

def compute_grpo_outcome_softpenalty_advantage(token_level_rewards: torch.Tensor,
                                   responses: torch.Tensor,
                                   tokenizer,
                                   mask_template_token,
                                   model_name,
                                   token_level_good_query: torch.Tensor,
                                   less_negative,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    print("compute_grpo_outcome_softpenalty_advantage")
    response_length = token_level_rewards.shape[-1]
    ans_scores = token_level_rewards.sum(dim=-1)
    query_scores = []
    query_scores_indices = []
    bsz = ans_scores.shape[0]
    max_turn = 0
    for i in range(bsz):
        good_scores_indices = torch.nonzero(token_level_good_query[i]).squeeze()
        good_scores = token_level_good_query[i][good_scores_indices]
        good_scores = good_scores.tolist()
        if type(good_scores) != list:
            good_scores = [good_scores]
        good_scores_indices = good_scores_indices.tolist()
        if type(good_scores_indices) != list:
            good_scores_indices = [good_scores_indices]
        query_scores.append(good_scores)
        query_scores_indices.append(good_scores_indices)
        max_turn = max(max_turn, len(good_scores_indices))

    id2ans_score = defaultdict(list)
    id2ans_mean = {}
    id2ans_std = {}
    id2index = defaultdict(list)

    id2query_score = [defaultdict(list) for _ in range(max_turn)]
    id2query_mean = [{} for _ in range(max_turn)]

    with torch.no_grad():
        for i in range(bsz):
            for turn_idx in range(len(query_scores[i])):
                id2query_score[turn_idx][index[i]].append(query_scores[i][turn_idx])
            id2index[index[i]].append(i)
            id2ans_score[index[i]].append(ans_scores[i])

        global_std = torch.std(ans_scores)
        for idx in id2ans_score:
            for turn_idx in range(max_turn):
                id2query_mean[turn_idx][idx] = torch.mean(torch.tensor(id2query_score[turn_idx][idx]))
            if len(id2ans_score[idx]) == 1:
                id2ans_mean[idx] = torch.tensor(0.0)
                id2ans_std[idx] = torch.tensor(1.0)
            elif len(id2ans_score[idx]) > 1:
                id2ans_mean[idx] = torch.mean(torch.tensor(id2ans_score[idx]))
                id2ans_std[idx] = torch.std(torch.tensor([id2ans_score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        
        all_advantage = []
        for i in range(bsz):
            ans_score = (ans_scores[i] - id2ans_mean[index[i]]) / (id2ans_std[index[i]] + epsilon)
            adv_ans = torch.full((response_length,), ans_score)
            all_advantage.append(adv_ans)
        all_advantage = torch.stack(all_advantage) * eos_mask

        if not less_negative.softpenalty_first:
            all_advantage = advantage_scale(all_advantage, "advantage", less_negative, eos_mask, id2index)

        negative_rollout_cnt = 0
        negative_query_cnt = 0
        soft_penalty_rollout_cnt = 0
        soft_penalty_query_cnt = 0
        soft_penalty_query_scores = []
        pos_query_cnt = 0
        pos_reset2zero_cnt = 0

        id2maxadv = defaultdict(float)
        for i in range(bsz):
            id2maxadv[index[i]] = max(id2maxadv[index[i]], all_advantage[i][0])

        for i in range(bsz):
            ans_score = all_advantage[i][0]
            assert not (sum(all_advantage[i]) != 0 and ans_score == 0)
            if ans_score > 0 and less_negative.reset2zero:
                pos_query_cnt += len(query_scores[i])
                adv_ans = all_advantage[i]
                last_adv = None
                for turn_idx in range(len(query_scores[i])):
                    this_query_advscore = ans_score
                    if id2query_mean[turn_idx][index[i]] == 1:
                        this_query_advscore = 0
                        pos_reset2zero_cnt += 1
                    adv_this_query = torch.full((query_scores_indices[i][turn_idx],), this_query_advscore)
                    if last_adv is not None:
                        adv_this_query[:last_adv.shape[0]] = last_adv
                    last_adv = adv_this_query
                
                if last_adv is not None:
                    adv_ans[:last_adv.shape[0]] = last_adv
                all_advantage[i] = adv_ans
            elif ans_score < 0:
                negative_rollout_cnt += 1
                negative_query_cnt += len(query_scores[i])
                is_soft_penalty_rollout = 0

                adv_ans = all_advantage[i]
                last_adv = None
                for turn_idx in range(len(query_scores[i])):
                    this_query_advscore = ans_score
                    if query_scores[i][turn_idx] > 1e-6:
                        if less_negative.aggressive_softpenalty and query_scores[i][turn_idx] >= 1/3:
                            this_query_advscore = 0
                        else:
                            if less_negative.softpenalty_positive_upper_limit:
                                this_query_advscore = ans_score + (id2maxadv[index[i]] - ans_score) * query_scores[i][turn_idx] 
                            else:
                                this_query_advscore = ans_score * (1 - query_scores[i][turn_idx])
                        soft_penalty_query_scores.append(query_scores[i][turn_idx])
                        soft_penalty_query_cnt += 1
                        is_soft_penalty_rollout = 1
                    adv_this_query = torch.full((query_scores_indices[i][turn_idx],), this_query_advscore)
                    if last_adv is not None:
                        adv_this_query[:last_adv.shape[0]] = last_adv
                    last_adv = adv_this_query
                soft_penalty_rollout_cnt += is_soft_penalty_rollout
                
                if last_adv is not None:
                    adv_ans[:last_adv.shape[0]] = last_adv
                all_advantage[i] = adv_ans
        if negative_rollout_cnt != 0 and negative_query_cnt != 0:
            print(f"negative_rollout_cnt: {negative_rollout_cnt}, soft_penalty_rollout_cnt: {soft_penalty_rollout_cnt}, {soft_penalty_rollout_cnt / negative_rollout_cnt}")
            print(f"negative_query_cnt: {negative_query_cnt}, soft_penalty_query_cnt: {soft_penalty_query_cnt}, {soft_penalty_query_cnt / negative_query_cnt}")
            hist, bins = np.histogram(soft_penalty_query_scores, bins=10)
            for i in range(len(hist)):
                print(f"[{bins[i]:.2f}, {bins[i+1]:.2f}): {hist[i]}")
            if pos_query_cnt != 0:
                print(f"pos_query_cnt:{pos_query_cnt}, pos_reset2zero_cnt: {pos_reset2zero_cnt}, {pos_reset2zero_cnt / pos_query_cnt}")
        all_advantage = all_advantage * eos_mask

        if mask_template_token:
            if model_name == 'llama':
                template_ids = torch.tensor([524, 27, 694, 366, 29, 1363, 397], device=responses.device)
            elif model_name == 'qwen':
                template_ids = torch.tensor([690, 366, 27, 522, 29, 1339], device=responses.device)
            else:
                raise NotImplementedError
            template_token_mask = torch.isin(responses, template_ids)
            print(f"scale {template_token_mask.sum()} template tokens")

            template_scores = all_advantage * template_token_mask
            positive_indices = torch.where(template_scores > 0)
            negative_indices = torch.where(template_scores < 0)

            if positive_indices[0].numel() > 0 and negative_indices[0].numel() > 0:
                positive_sum = template_scores[positive_indices].sum()
                negative_sum = template_scores[negative_indices].sum()
                ratio = positive_sum / (-negative_sum)
                print(f"Template tokens original positive/negative ratio: {ratio}")

                scale = 1.0 / (ratio + 1e-8)
                all_advantage[positive_indices] *= scale

                template_scores = all_advantage * template_token_mask
                positive_sum = template_scores[positive_indices].sum()
                negative_sum = template_scores[negative_indices].sum()
                print(f"Template tokens adjusted positive/negative ratio: {positive_sum / (-negative_sum)}")
            elif positive_indices[0].numel() > 0 and negative_indices[0].numel() == 0:
                print("Template tokens contain only positive advantage, zeroing them.")
                all_advantage[positive_indices] = 0

        if less_negative.softpenalty_first:
            all_advantage = advantage_scale(all_advantage, "advantage", less_negative, eos_mask, id2index, responses, tokenizer)
        else:
            get_ratio(all_advantage, "advantage")
    
    return all_advantage, all_advantage

def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio

def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.
    Args:
        loss_mat: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_agg_mode: (str) choices: "token-mean" /
                                      "seq-mean-token-sum" /
                                      "seq-mean-token-mean" /
                                      "seq-mean-token-sum-norm" /
            "token-mean" is the default behavior
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss

def count_overflow_underflow(t: torch.Tensor, eos_mask):
    assert t.is_floating_point(), "仅适用于浮点类型"

    info = torch.finfo(t.dtype)
    tensor_num = t.numel()

    # --- 上溢 ---
    # pos_overflow = torch.sum(torch.isposinf(t) * eos_mask)   # +inf
    neg_overflow = torch.sum(torch.isneginf(t) * eos_mask)   # -inf

    # --- 下溢 ---
    underflow_mask = (t == 0) * eos_mask
    underflow = underflow_mask.sum()

    return {
        # "pos_overflow": pos_overflow.item() / tensor_num,
        "neg_overflow": neg_overflow.item() / tensor_num,
        "underflow": underflow.item() / tensor_num
    }

def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange, loss_agg_mode):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    pg_loss = torch.max(pg_losses, pg_losses2)
    pg_loss = agg_loss(pg_loss, eos_mask, loss_agg_mode)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl

def compute_policy_loss_lld(old_log_prob, log_prob, advantages, eos_mask, cliprange, loss_agg_mode, gating):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    lld_reduce = -negative_approx_kl * eos_mask
    lld_reduce = torch.clamp(lld_reduce, min=0)
    preserves_token_mask = advantages > 0
    lld_reduce = lld_reduce * preserves_token_mask
    if gating:
        rollout_sum = lld_reduce.sum(dim=1)
        gate_mask = rollout_sum > 0
        lld_reduce = lld_reduce * gate_mask.unsqueeze(-1)
    preserves_token_num = (preserves_token_mask * eos_mask).sum()
    lld_loss = lld_reduce.sum() / preserves_token_num if preserves_token_num > 0 else torch.tensor(0).to(lld_reduce.device)
    

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    pg_loss = torch.max(pg_losses, pg_losses2)
    pg_loss = agg_loss(pg_loss, eos_mask, loss_agg_mode)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl, lld_loss

def compute_policy_loss_balance(old_log_prob, log_prob, advantages, eos_mask, cliprange, pg_loss_ratio, pos_neg_both):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    eps = 1e-8
    pg_loss = torch.max(pg_losses, pg_losses2) * eos_mask
    # factor = torch.where(pg_loss > pg_loss_range, pg_loss_range / (pg_loss + eps), torch.ones_like(pg_loss))
    # pg_loss = pg_loss * factor
    # if pos_neg_both:
    #     factor = torch.where(pg_loss < -pg_loss_range, -pg_loss_range / (pg_loss - eps), torch.ones_like(pg_loss))
    #     pg_loss = pg_loss * factor
    positive_indices = torch.where(pg_loss > 0)
    negative_indices = torch.where(pg_loss < 0)
    positive_sum = pg_loss[positive_indices].sum()
    negative_sum = pg_loss[negative_indices].sum()
    pg_loss_neg_pos_ratio = negative_sum / (-positive_sum) if positive_sum != 0 else 1
    # print("pg_loss_neg_pos_ratio: ", pg_loss_neg_pos_ratio)
    # print(f"pg_loss: {verl_F.masked_mean(pg_loss, eos_mask)}")
    pg_loss[positive_indices] *= pg_loss_neg_pos_ratio * pg_loss_ratio
    if positive_sum == 0:
        pg_loss[negative_indices] *= 0
    # positive_indices = torch.where(pg_loss > 0)
    # negative_indices = torch.where(pg_loss < 0)
    # positive_sum = pg_loss[positive_indices].sum()
    # negative_sum = pg_loss[negative_indices].sum()
    # pg_loss_neg_pos_ratio = negative_sum / (-positive_sum) if positive_sum != 0 else 1
    # print("Adjusted pg_loss_neg_pos_ratio: ", pg_loss_neg_pos_ratio)


    pg_loss = verl_F.masked_mean(pg_loss, eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)

    return pg_loss, pg_clipfrac, ppo_kl

def compute_policy_loss_cispo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the CISPO policy objective and related metrics.
    CISPO (Clipped Importance Sampling Policy Optimization) clips importance sampling weights
    instead of dropping tokens, which is beneficial for training on sparse but critical tokens
    and long-context reasoning in RL.
    Reference: https://www.arxiv.org/pdf/2506.13585
    Args:
        old_log_prob (torch.Tensor): Log-probabilities under old policy, shape (batch_size, response_length)
        log_prob (torch.Tensor): Log-probabilities under current policy, shape (batch_size, response_length)
        advantages (torch.Tensor): Advantage estimates, shape (batch_size, response_length)
        response_mask (torch.Tensor): Mask for valid tokens, shape (batch_size, response_length)
        loss_agg_mode (str): Aggregation mode for loss computation
        config (AlgoConfig): Algorithm configuration containing CISPO parameters
    Returns:
        tuple: (pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower)
    """
    cliprange = 0.2
    cliprange_low = 100
    cliprange_high = 100
    cispo_clip_ratio_low = 0.2
    cispo_clip_ratio_high = 0.2

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cispo_clip_ratio_low, 1 + cispo_clip_ratio_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    # clip_pg_losses1 = torch.maximum(
    #     pg_losses1, pg_losses2
    # )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    ratio = ratio.detach()
    importance_sampling_weight = torch.clamp(ratio, max=1 + cispo_clip_ratio_high, min=1 - cispo_clip_ratio_low)
    pos_adv_mask = (advantages > 0) & (ratio > 1 + cliprange_high)
    neg_adv_mask = (advantages < 0) & (ratio < 1 - cliprange_low)
    adv_mask = ~(pos_adv_mask | neg_adv_mask)
    pg_losses = -advantages * log_prob * importance_sampling_weight * adv_mask

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl

def compute_policy_loss_clip_higher(old_log_prob, log_prob, advantages, eos_mask, cliprange, cliprange_low=None,
    cliprange_high=None):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange_low, 1.0 + cliprange_high)

    pg_loss = verl_F.masked_mean(torch.max(pg_losses1, pg_losses2), eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl

def compute_policy_loss_part_clip_change(old_log_prob, log_prob, advantages, format_reward, eos_mask, change_part_mask, cliprange, cliprange_low=None,
    cliprange_high=None):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses1 = -advantages * ratio
    pg_losses2_aggressive = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    pg_losses2_normal = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    format_scores = format_reward.sum(dim=-1)
    change_part_mask[format_scores<=1e-6]=0
    change_part_mask=change_part_mask.bool()
    pg_losses2 = torch.where(change_part_mask, pg_losses2_aggressive, pg_losses2_normal)

    pg_loss = verl_F.masked_mean(torch.max(pg_losses1, pg_losses2), eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl
    

def compute_policy_loss_kl_cov(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    loss_agg_mode="token-mean",
    k_percent=0.2,
    ppo_kl_coef=1,
):
    negative_approx_kl = log_prob - old_log_prob

    abs_kl = negative_approx_kl.abs()

    ratio = torch.exp(negative_approx_kl)

    ppo_kl = verl_F.masked_mean(negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio

    pg_losses_kl = - advantages * ratio + ppo_kl_coef * abs_kl

    pg_losses = pg_losses1

    all_valid = (response_mask > 0)
    all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0] 
    all_valid_adv = advantages[all_valid].detach().reshape(-1).cpu()
    all_valid_logp = log_prob[all_valid].detach().reshape(-1).cpu()

    k = min(k_percent, len(all_valid_adv))

    if k != 0:
        cov_lst_all = (all_valid_adv - all_valid_adv.mean()) * (all_valid_logp - all_valid_logp.mean())
        k_percent_nums = max(1, int(len(cov_lst_all) * k / 100))
        large_cov_idxs = torch.topk(cov_lst_all, k_percent_nums, largest=True).indices
        
        if len(large_cov_idxs) != 0:
            large_cov_idxs = all_valid_idx[large_cov_idxs]
            pg_losses[large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]] = pg_losses_kl[large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]]
        cov_idxs = torch.stack((large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]), dim=1)
    else:
        cov_idxs = torch.empty((0, 2), device=all_valid.device, dtype=torch.long)

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, torch.tensor(0.), ppo_kl, cov_idxs

def compute_policy_loss_clip_cov(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    loss_agg_mode="token-mean",
    clip_ratio=0.0002,
    clip_cov_lb=1.0,
    clip_cov_ub=5.0,
):
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    
    corr = torch.ones_like(advantages)
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_by_origin = (pg_losses2 > pg_losses1) & (response_mask > 0)
    
    cov_all = (advantages- verl_F.masked_mean(advantages, response_mask)) * (log_prob- verl_F.masked_mean(log_prob.detach(), response_mask))
    cov_all[response_mask == 0] = -torch.inf
    cov_all[clip_by_origin] = -torch.inf
    
    clip_num = max(int(clip_ratio * response_mask.sum().item()), 1)
    top_k_idx = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb) & (response_mask > 0)
    top_k_idx = torch.nonzero(top_k_idx)
    
    if len(top_k_idx) > 0:
        perm = torch.randperm(len(top_k_idx))
        top_k_idx = top_k_idx[perm[:min(clip_num, len(top_k_idx))]]
    else:
        top_k_idx = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)
    
    corr[top_k_idx[:, 0], top_k_idx[:, 1]] = 0
    
    pg_clipfrac = verl_F.masked_mean((corr==0).float(), response_mask)

    pg_losses = torch.maximum(pg_losses1, pg_losses2) * corr
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, top_k_idx

def compute_policy_loss_TopCovIdx(old_log_prob, log_prob, advantages, eos_mask, cliprange):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)
    
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    
    # Clip-cov index
    clip_ratio=0.0002
    clip_cov_lb=1.0
    clip_cov_ub=5.0
    clip_by_origin = (pg_losses2 > pg_losses1) & (eos_mask > 0)

    cov_all = (advantages- verl_F.masked_mean(advantages, eos_mask)) * (log_prob- verl_F.masked_mean(log_prob.detach(), eos_mask))
    cov_all[eos_mask == 0] = -torch.inf
    cov_all[clip_by_origin] = -torch.inf

    clip_num = max(int(clip_ratio * eos_mask.sum().item()), 1)
    top_k_idx = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb) & (eos_mask > 0)
    top_k_idx = torch.nonzero(top_k_idx)

    if len(top_k_idx) > 0:
        perm = torch.randperm(len(top_k_idx))
        top_k_idx = top_k_idx[perm[:min(clip_num, len(top_k_idx))]]
    else:
        top_k_idx = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)

    # kl-cov index
    k_percent=0.2
    ppo_kl_coef=1
    abs_kl = negative_approx_kl.abs()
    ppo_kl_abs = verl_F.masked_mean(negative_approx_kl.abs(), eos_mask)
    pg_losses_kl = pg_losses1 + ppo_kl_coef * abs_kl

    all_valid = (eos_mask > 0)
    all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0] 
    all_valid_adv = advantages[all_valid].detach().reshape(-1).cpu()
    all_valid_logp = log_prob[all_valid].detach().reshape(-1).cpu()

    k = min(k_percent, len(all_valid_adv))
    
    if k != 0:
        cov_lst_all = (all_valid_adv - all_valid_adv.mean()) * (all_valid_logp - all_valid_logp.mean())
        k_percent_nums = max(1, int(len(cov_lst_all) * k / 100))
        large_cov_idxs = torch.topk(cov_lst_all, k_percent_nums, largest=True).indices
        cov_idxs = torch.stack((large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]), dim=1)
    else:
        cov_idxs = torch.empty((0,), device=all_valid.device, dtype=torch.long)

    pg_loss = verl_F.masked_mean(torch.max(pg_losses1, pg_losses2), eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl, top_k_idx, cov_idxs

def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
