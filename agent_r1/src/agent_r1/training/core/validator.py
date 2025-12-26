# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Validation logic for agent training.
"""

from collections import defaultdict
from typing import Optional

import numpy as np

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.tracking import ValidationGenerationsLogger

from agent_r1.tool.agent_loop import ToolGenerationManager, ToolGenerationConfig
from agent_r1.training.utils.metric_utils import (
    dump_generations,
    log_val_generations,
)
from verl.trainer.ppo.metric_utils import process_validation_metrics
from agent_r1.tool.base import BaseToolEnv


class AgentValidator:
    """Handles validation logic for agent training"""

    def __init__(
        self,
        tokenizer,
        config,
        val_reward_fn,
        val_dataloader,
        val_env: Optional[BaseToolEnv],
        validation_generations_logger: ValidationGenerationsLogger,
    ):
        """
        Initialize the AgentValidator.

        Args:
            tokenizer: Tokenizer for encoding/decoding
            config: Configuration object
            val_reward_fn: Reward function for validation
            val_dataloader: DataLoader for validation data
            val_env: Tool environment for validation
            validation_generations_logger: Logger for validation generations
        """
        self.tokenizer = tokenizer
        self.config = config
        self.val_reward_fn = val_reward_fn
        self.val_dataloader = val_dataloader
        self.val_env = val_env
        self.validation_generations_logger = validation_generations_logger

    def _process_batch(self, test_batch: DataProto, generation_manager: ToolGenerationManager, actor_rollout_wg) -> tuple:
        """
        Process a single validation batch.

        Args:
            test_batch: Batch to process
            generation_manager: Manager for generation
            actor_rollout_wg: Actor rollout worker group

        Returns:
            Tuple of (input_texts, output_texts, scores, reward_extra_info, data_sources)
        """
        # Repeat batch
        test_batch = test_batch.repeat(
            repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
            interleave=True
        )

        # Store original inputs
        input_ids = test_batch.batch["input_ids"]
        input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]

        # Prepare batch for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "raw_prompt" in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        
        test_gen_batch = test_batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )

        # Set generation meta info
        test_gen_batch.meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
            "validate": True,
        }
        print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

        # Pad to be divisible by dp_size
        test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
            test_gen_batch,
            actor_rollout_wg.world_size
        )
        
        # Run generation
        test_output_gen_batch_padded = generation_manager.run_llm_loop(
            test_gen_batch_padded,
            env=self.val_env,
        )

        # Unpad
        test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
        print("validation generation end")

        # Store generated outputs
        output_ids = test_output_gen_batch.batch["responses"]
        output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]

        # Union batches
        test_batch = test_batch.union(test_output_gen_batch)

        # Evaluate using reward function
        result = self.val_reward_fn(test_batch, return_dict=True)
        reward_tensor = result["reward_tensor"]
        scores = reward_tensor.sum(-1).cpu().tolist()

        # Collect reward extra info
        reward_extra_info = defaultdict(list)
        reward_extra_info["reward"].extend(scores)
        reward_extra_info["turns"].extend(test_batch.batch["turns"].tolist())
        if "reward_extra_info" in result:
            for key, lst in result["reward_extra_info"].items():
                reward_extra_info[key].extend(lst)

        # Get data sources
        data_sources = test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])

        return input_texts, output_texts, scores, dict(reward_extra_info), data_sources

    def _compute_metrics(self, data_sources: np.ndarray, sample_inputs: list, reward_extra_infos_dict: dict) -> dict:
        """
        Process and aggregate validation metrics.

        Args:
            data_sources: Array of data sources
            sample_inputs: List of input texts
            reward_extra_infos_dict: Dictionary of reward extra info

        Returns:
            Dictionary of validation metrics
        """
        data_src2var2metric2val = process_validation_metrics(
            data_sources,
            sample_inputs,
            reward_extra_infos_dict
        )
        
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(
                        metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]
                    ) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict

    def run(self, actor_rollout_wg, global_steps: int) -> dict:
        """
        Run validation and return metrics.

        Args:
            actor_rollout_wg: Actor rollout worker group
            global_steps: Current global training step

        Returns:
            Dictionary of validation metrics
        """
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        # Prepare generation config and manager
        gen_config = ToolGenerationConfig(
            max_turns=self.config.tool.val_kwargs.max_turns,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_response_length_single_turn=self.config.data.max_response_length_single_turn,
            use_batch_tool_calls=self.config.tool.val_kwargs.use_batch_tool_calls
        )

        generation_manager = ToolGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=actor_rollout_wg,
            config=gen_config,
            is_validation=True,
        )

        # Process each validation batch
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            
            input_texts, output_texts, scores, reward_extra_info, data_sources = self._process_batch(
                test_batch,
                generation_manager,
                actor_rollout_wg
            )

            # Collect samples
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_scores.extend(scores)

            # Collect reward extra info
            for key, lst in reward_extra_info.items():
                reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(data_sources)

        # Log validation generations
        log_val_generations(
            inputs=sample_inputs,
            outputs=sample_outputs,
            scores=sample_scores,
            logger_config=self.config.trainer.logger,
            validation_generations_logger=self.validation_generations_logger,
            global_steps=global_steps,
            num_generations_to_log=self.config.trainer.log_val_generations,
        )

        # Dump generations if configured
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
                global_steps=global_steps,
            )

        # Validate reward extra info lengths
        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), \
                f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        # Concatenate data sources
        data_sources = np.concatenate(data_source_lst, axis=0)

        # Compute and return metrics
        return self._compute_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
