import subprocess
import os
from pathlib import Path

DATA_PATH = Path("data/processed/")

cmd = [
    "python", "-m", "verl.trainer.main_ppo",
    f"data.train_files={DATA_PATH / 'dapo_math_17k.parquet'}", 
    "actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B",
    "algorithm.adv_estimator=grpo",
    "actor_rollout_ref.actor.use_kl_loss=False",
    "actor_rollout_ref.actor.entropy_coeff=0",
    "data.train_batch_size=256",
    "data.max_prompt_length=1024",
    "data.max_response_length=15360",
    "actor_rollout_ref.actor.ppo_mini_batch_size=64",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
    "++actor_rollout_ref.actor.clip_range_low=0.2",
    "++actor_rollout_ref.actor.clip_range_high=0.28",
    "actor_rollout_ref.actor.optim.lr=1e-6",
    "actor_rollout_ref.rollout.temperature=1.0",
    "actor_rollout_ref.rollout.n=8",
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
    # "actor_rollout_ref.rollout.tensor_model_parallel_size=1", # uncomment if you are on 1 GPU
    "++actor_rollout_ref.rollout.max_num_batched_tokens=16384",  # Must be >= max_model_len (16384)
    "reward_model.reward_manager=dapo",
    "++trainer.seed=42",
    "++trainer.save_steps=250",
    "++trainer.max_checkpoints=5",
    "++trainer.logger=wandb",
    "++trainer.project_name=verl-grpo-dapo",
    f"++data.val_files={DATA_PATH / 'dapo_math_17k.parquet'}",
    f"++data.test_files={DATA_PATH / 'dapo_math_17k.parquet'}",
    # "++trainer.n_gpus_per_node=1",  # default is 8, change if you are using different number of GPUs 

]

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

subprocess.run(cmd, env=env, check=True)