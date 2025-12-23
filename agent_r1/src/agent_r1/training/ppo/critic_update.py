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
Implement a multiprocess PPOCritic for Agent R1.
Extends verl's DataParallelPPOCritic with action_mask support.
"""

import logging
import os

import torch
from verl import DataProto
from verl.utils.debug import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import rearrange_micro_batches
from verl.utils.torch_functional import masked_mean
from verl.workers.critic.dp_critic import DataParallelPPOCritic

from agent_r1.training.ppo import algorithms

__all__ = ["DataParallelR1PPOCritic"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelR1PPOCritic(DataParallelPPOCritic):
    """
    Extended version of verl's DataParallelPPOCritic that uses action_mask
    to distinguish model-generated tokens from external interaction tokens
    (user responses, tool outputs, etc.).
    
    Only overrides update_critic to use action_mask instead of response_mask.
    All other functionality (forward, compute_values, optimizer_step) is inherited.
    """
    
    @GPUMemoryLogger(role="dp critic", logger=logger)
    def update_critic(self, data: DataProto):
        # make sure we are in training mode
        self.critic_module.train()
        metrics = {}

        select_keys = ['input_ids', 'responses', 'attention_mask', 'position_ids', 'values', 'returns', 'action_mask']
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu

                self.critic_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all devices
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(torch.cuda.current_device())  # critic device is cpu when using offload
                    responses = data["responses"]
                    attention_mask = data["attention_mask"]
                    values = data["values"]
                    returns = data["returns"]
                    response_length = responses.size(1)

                    # state_mask = attention_mask[:, -response_length - 1:-1]
                    # state_mask[:, 1:] = data['action_mask'][:, :-1]
                    state_mask = data['action_mask']
                    vpreds = self._forward_micro_batch(data)

                    # assert not torch.any(torch.isnan(vpreds)).item()

                    vf_loss, vf_clipfrac = algorithms.compute_value_loss(vpreds=vpreds,
                                                                         values=values,
                                                                         returns=returns,
                                                                         state_mask=state_mask,
                                                                         cliprange_value=self.config.cliprange_value)
                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = vf_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = vf_loss / self.gradient_accumulation

                    loss.backward()

                    data = {
                        'critic/vf_loss': vf_loss.detach().item(),
                        'critic/vf_clipfrac': vf_clipfrac.detach().item(),
                        'critic/vpred_mean': masked_mean(vpreds, state_mask).detach().item(),
                    }

                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"critic/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.critic_optimizer.zero_grad()
        return metrics
