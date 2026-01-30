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
Preprocess the nq dataset to parquet format
"""

import re
import os
import json
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


def make_prefix(dp, template_type):
    question = dp['question']
    context = dp['context']

    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""Answer the given question with some potentially useful context. \
You should analyze the question carefully, evaluate the given context (which may or may not be useful), and then generate an accurate and well-reasoned response. \
You should first have a reasoning process in mind and then provides the answer. \
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. \
Question: {question} Context: {context} \n"""
    else:
        raise NotImplementedError    
    return prefix


def format_reference(retrieval_result):

    format_reference = ''
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item['contents']
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

    return format_reference


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/nq_rag')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--train_retrieval_cache', type=str, default='')
    parser.add_argument('--test_retrieval_cache', type=str, default='')

    args = parser.parse_args()

    data_source = 'nq'

    dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'nq')

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    # read retrieval cache
    print('reading retrieval cache...')
    retrieval_cache = json.load(open(args.train_retrieval_cache))
    retrieval_cache.update(json.load(open(args.test_retrieval_cache)))
    import pdb; pdb.set_trace()

    # add a column for the retrieval context
    def add_context(example):
        example['context'] = format_reference(retrieval_cache[example['question']][:args.topk])
        return example
    
    train_dataset = train_dataset.map(function=add_context)
    test_dataset = test_dataset.map(function=add_context)

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            example['question'] = example['question'].strip()
            if example['question'][-1] != '?':
                example['question'] += '?'
            question = make_prefix(example, template_type=args.template_type)
            solution = {
                "target": example['golden_answers'],
            }

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "fact-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    train_output = []
    test_output = []
    for item in train_dataset:
        train_output.append(item)
    for item in test_dataset:
        test_output.append(item)
    with open(os.path.join(local_dir, 'train_10.json'), 'w') as f:
        json.dump(train_output[:10], f)
    with open(os.path.join(local_dir, 'test_10.json'), 'w') as f:
        json.dump(test_output[:10], f)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
