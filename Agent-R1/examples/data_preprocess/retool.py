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
Preprocess the DAPO-Math-17k dataset to parquet format
"""

import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/retool")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # 加载 DAPO-Math-17k 数据集
    dapo_dataset = datasets.load_dataset(
        "haizhongzheng/DAPO-Math-17K-cleaned"
    )

    # 获取 DAPO-Math-17k 的前5k条作为训练集
    dataset = dapo_dataset["train"]

    #   随机划分出验证集
    test_dataset = dataset.shuffle(seed=42).select(range(100))
    train_dataset = dataset.shuffle(seed=42).select(range(100, len(dataset)))

    instruction = "Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox, and the output (wrapped in `<interpreter>output_str</interpreter>`) can be returned to aid your reasoning and help you arrive at the final answer. The Python code should be complete scripts, including necessary imports. \nEach code snippet is wrapped with `<code>\n```python\ncode snippet\n```\n</code>`.\nThe last part of your response should be in the following format:\n<answer>\n\\boxed{{'The final answer goes here.'}}\n</answer>\n\n*user question:*\n{question}\n\nRemember to place the final answer in the last part using the format: \n<answer>\n\\boxed{{'The final answer goes here.'}}\n</answer>"

    # 处理 DAPO-Math-17k 训练集
    def process_dapo(example, idx):
        # 从 prompt 中获取 content (question_raw)
        prompt = example.get("prompt", "")
        question = instruction.format(question=prompt)

        # 获取正确答案
        ground_truth = example.get("target", "")

        # 获取 data_source 和 ability
        data_source = example.get("data_source", "BytedTsinghua-SIA/DAPO-Math-17k")
        ability = example.get("ability", "")

        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "ability": ability,
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth,
            },
            "extra_info": {
                "split": "train",
                "index": idx,
            },
        }
        return data

    processed_train = train_dataset.map(function=process_dapo, with_indices=True)
    processed_test = test_dataset.map(function=process_dapo, with_indices=True)
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # 确保本地目录存在
    os.makedirs(local_dir, exist_ok=True)

    processed_train.to_parquet(os.path.join(local_dir, "train.parquet"))
    processed_test.to_parquet(os.path.join(local_dir, "test.parquet"))
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
