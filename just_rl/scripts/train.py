import subprocess
import os
from pathlib import Path

DATA_PATH = Path("data/processed/")

cmd = [
    "python", "-m", "verl.trainer.main_ppo",
    f"data.train_files={DATA_PATH / 'dapo_math_17k_processed.parquet'}", 
    "actor_rollout_ref.model.path=HuggingFaceTB/SmolLM-360M",
    "algorithm.adv_estimator=grpo",
    "actor_rollout_ref.actor.use_kl_loss=False",
    "actor_rollout_ref.actor.entropy_coeff=0",
    "data.train_batch_size=256",
    "data.max_prompt_length=1024",
    "data.max_response_length=15360",
    "actor_rollout_ref.actor.ppo_mini_batch_size=64",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
    "actor_rollout_ref.actor.clip_range_low=0.2",
    "actor_rollout_ref.actor.clip_range_high=0.28",
    "actor_rollout_ref.actor.optim.lr=1e-6",
    "actor_rollout_ref.rollout.temperature=1.0",
    "actor_rollout_ref.rollout.n=8",
    "reward_model.reward_manager=dapo",
]

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

subprocess.run(cmd, env=env, check=True)