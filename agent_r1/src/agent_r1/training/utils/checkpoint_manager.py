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
Checkpoint management for training state including models and dataloaders.
"""

import os
from typing import Optional, Tuple

import torch

from verl.single_controller.ray import RayWorkerGroup
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path


class CheckpointManager:
    def __init__(
        self,
        local_dir: str,
        max_actor_ckpt_to_keep: Optional[int] = None,
        max_critic_ckpt_to_keep: Optional[int] = None,
        del_local_after_load: bool = False,
    ):
        self.local_dir = local_dir
        self.max_actor_ckpt_to_keep = max_actor_ckpt_to_keep
        self.max_critic_ckpt_to_keep = max_critic_ckpt_to_keep
        self.del_local_after_load = del_local_after_load

    def _get_checkpoint_path(self, global_step: int) -> str:
        """Get the checkpoint directory path for a given global step."""
        return os.path.join(self.local_dir, f"global_step_{global_step}")

    def _update_latest_iteration_tracker(self, global_step: int):
        """Update the latest checkpointed iteration tracker file."""
        tracker_path = os.path.join(self.local_dir, "latest_checkpointed_iteration.txt")
        with open(tracker_path, "w") as f:
            f.write(str(global_step))

    def _save_dataloader_state(self, checkpoint_dir: str, state_dict: dict):
        """Save dataloader state to checkpoint directory."""
        dataloader_path = os.path.join(checkpoint_dir, "data.pt")
        torch.save(state_dict, dataloader_path)

    def _load_dataloader_state(self, checkpoint_dir: str) -> Optional[dict]:
        dataloader_path = os.path.join(checkpoint_dir, "data.pt")
        if os.path.exists(dataloader_path):
            return torch.load(dataloader_path, weights_only=False)
        else:
            print(f"Warning: No dataloader state found at {dataloader_path}, will start from scratch")
            return None

    def save_checkpoint(
        self,
        global_step: int,
        actor_wg: RayWorkerGroup,
        critic_wg: Optional[RayWorkerGroup] = None,
        dataloader_state: Optional[dict] = None,
    ) -> str:
        checkpoint_dir = self._get_checkpoint_path(global_step)
        print(f"Saving checkpoint to: {checkpoint_dir}")

        # Save actor model
        actor_local_path = os.path.join(checkpoint_dir, "actor")
        actor_wg.save_checkpoint(
            actor_local_path,
            None,  # No remote path (HDFS not supported)
            global_step,
            max_ckpt_to_keep=self.max_actor_ckpt_to_keep,
        )

        # Save critic model if available
        if critic_wg is not None:
            critic_local_path = os.path.join(checkpoint_dir, "critic")
            critic_wg.save_checkpoint(
                critic_local_path,
                None,  # No remote path (HDFS not supported)
                global_step,
                max_ckpt_to_keep=self.max_critic_ckpt_to_keep,
            )

        # Save dataloader state if provided
        if dataloader_state is not None:
            self._save_dataloader_state(checkpoint_dir, dataloader_state)

        # Update latest iteration tracker
        self._update_latest_iteration_tracker(global_step)

        return checkpoint_dir

    def load_checkpoint(
        self,
        actor_wg: RayWorkerGroup,
        critic_wg: Optional[RayWorkerGroup] = None,
        resume_mode: str = "auto",
        resume_from_path: Optional[str] = None,
    ) -> Tuple[int, Optional[dict]]:
        if resume_mode == "disable":
            print("Resume disabled, starting from scratch")
            return 0, None

        # Determine checkpoint folder
        checkpoint_folder = None

        if resume_mode == "auto":
            # Find latest checkpoint automatically
            checkpoint_folder = self.local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)

            if global_step_folder is None:
                print("No checkpoint found, starting from scratch")
                return 0, None

        elif resume_mode == "resume_path":
            # Use specific path
            assert isinstance(resume_from_path, str), "resume_from_path must be a string"
            assert "global_step_" in resume_from_path, "resume_from_path must contain 'global_step_'"
            
            global_step_folder = resume_from_path
            if not os.path.isabs(global_step_folder):
                working_dir = os.getcwd()
                global_step_folder = os.path.join(working_dir, global_step_folder)

        else:
            raise ValueError(f"Invalid resume_mode: {resume_mode}. Must be 'disable', 'auto', or 'resume_path'")

        print(f"Loading checkpoint from: {global_step_folder}")

        # Extract global step from folder name
        global_step = int(global_step_folder.split("global_step_")[-1])
        print(f"Resuming from global step {global_step}")

        # Load actor model
        actor_path = os.path.join(global_step_folder, "actor")
        actor_wg.load_checkpoint(actor_path, del_local_after_load=self.del_local_after_load)

        # Load critic model if available
        if critic_wg is not None:
            critic_path = os.path.join(global_step_folder, "critic")
            critic_wg.load_checkpoint(critic_path, del_local_after_load=self.del_local_after_load)

        # Load dataloader state
        dataloader_state = self._load_dataloader_state(global_step_folder)

        return global_step, dataloader_state
