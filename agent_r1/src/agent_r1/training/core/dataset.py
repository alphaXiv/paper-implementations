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

from typing import List, Union, Optional
import re
from omegaconf import DictConfig

from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset

from agent_r1.tool.base import BaseToolEnv

class ToolRLDataset(RLHFDataset):
    """
    Dataset for tool use in RLHF
    """
    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        env: Optional[BaseToolEnv] = None,
    ):
        self.env = env
        self.use_default_tool_template = config.get("use_default_tool_template", True)
        if self.use_default_tool_template and self.env is not None:
            self.tools = [tool.tool_description for tool in self.env.tools]
        self.use_custom_system_prompt = config.get("use_custom_system_prompt", False)
        super().__init__(data_files, tokenizer, config)

    def _build_messages(self, example: dict):
        messages = example.pop(self.prompt_key)
        
        # Apply custom system prompt if needed
        if self.use_custom_system_prompt and self.env is not None:
            if isinstance(messages, list):
                if messages[0]["role"] == "system":
                    messages[0]["content"] = messages[0]["content"] + self.env.system_prompt
                else:
                    system_msg = [{"role": "system", "content": self.env.system_prompt}]
                    messages = system_msg + messages
        
        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        # Apply the appropriate chat template based on settings
        if self.use_default_tool_template and hasattr(self, 'tools'):
            raw_prompt = self.tokenizer.apply_chat_template(messages, tools=self.tools, add_generation_prompt=True, tokenize=False)
        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict