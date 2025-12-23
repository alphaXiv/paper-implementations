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
from agent_r1.training.core.trainer import RayAgentTrainer
from agent_r1.tool.envs import _default_env
from agent_r1.tool.tools import _default_tool

import os

import hydra
import ray

@hydra.main(config_path="config", config_name="agent_trainer", version_base=None)
def main(config):
    run_agent(config)


def run_agent(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
        )

    runner = TrainingOrchestrator.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # Lightweight coordinator - reserves minimal resources for orchestration
class TrainingOrchestrator:
    def run(self, config):
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local
        from verl.utils import hf_tokenizer

        from .workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        from agent_r1.training.utils.resource_pool import ResourcePoolManager, Role

        from .rewards.reward_scorer import RewardScorer
        from verl.utils.dataset.rl_dataset import collate_fn
        from .dataset import ToolRLDataset

        import torch
        from torch.utils.data import RandomSampler, SequentialSampler

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
            Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            "global_pool": [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: "global_pool",
            Role.Critic: "global_pool",
            Role.RefPolicy: "global_pool",
        }
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        reward_fn = RewardScorer(
            tokenizer=tokenizer,
            num_examine=num_examine,
            reward_fn_key=config.data.reward_fn_key,
        )

        val_reward_fn = RewardScorer(
            tokenizer=tokenizer,
            num_examine=num_examine,
            reward_fn_key=config.data.reward_fn_key,
        )

        tools = [_default_tool(name) for name in config.tool.tools]
        env = _default_env(config.tool.env)(tools=tools, max_tool_response_length=config.tool.max_tool_response_length)
        
        train_dataset = ToolRLDataset(
            data_files=config.data.train_files,
            tokenizer=tokenizer,
            config=config.data,
            env=env,
        )
        val_dataset = ToolRLDataset(
            data_files=config.data.val_files,
            tokenizer=tokenizer,
            config=config.data,
            env=env,
        )
        
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(config.data.get("seed", 1))
        train_sampler = RandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
        
        trainer = RayAgentTrainer(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            env=env,
            val_env=val_env,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
