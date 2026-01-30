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
Preprocess the QA dataset to parquet format
"""

import re
import os
import json
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse

def format_reference(retrieval_result):

    format_reference = ''
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item['contents']
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

    return format_reference

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/qa_rag')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--data_sources', default='nq')
    parser.add_argument('--retrieval_caches_dir', type=str)


    args = parser.parse_args()

    # data_source = 'nq'
    data_sources = args.data_sources.split(',')
    retrieval_caches = [args.retrieval_caches_dir+f'/{data_source}_train_top10.json' for data_source in data_sources]

    all_dataset = []

    for index in range(len(data_sources)):
        print(f'Processing {data_sources[index]}')
        dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', data_sources[index])

        train_dataset = dataset['train']
        retrieval_cache = json.load(open(retrieval_caches[index], 'r'))

        def add_context(example):
            example['context'] = format_reference(retrieval_cache[example['question']][:args.topk])
            return example
        
        train_dataset = train_dataset.map(function=add_context)

        # add a row to each data item that represents a unique id
        def make_map_fn(split):

            def process_fn(example, idx):
                # import pdb; pdb.set_trace()
                example['question'] = example['question'].strip()
                if example['question'][-1] != '?':
                    example['question'] += '?'
                question = make_prefix(example, template_type=args.template_type)
                solution = {
                    "target": example['golden_answers'],
                }

                data = {
                    "data_source": data_sources[index],
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
        all_dataset.append(train_dataset)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    all_train_dataset = datasets.concatenate_datasets(all_dataset)
    all_train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
