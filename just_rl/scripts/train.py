import subprocess
import os
from pathlib import Path

# Use absolute path so Ray workers can find the data regardless of their working directory
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = (PROJECT_ROOT / "data" / "processed").resolve()

cmd = [
    "uv", "run", "--active", "python", "-m", "verl.trainer.main_ppo",
    f"data.train_files={DATA_PATH / 'dapo_math_17k.parquet'}", 
    "actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B",
    "algorithm.adv_estimator=grpo",
    "actor_rollout_ref.actor.use_kl_loss=False",
    "actor_rollout_ref.actor.entropy_coeff=0",
    "data.train_batch_size=256",
    "data.max_prompt_length=1024",
    "data.max_response_length=15360",
    "actor_rollout_ref.actor.ppo_mini_batch_size=64",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8",
    "++actor_rollout_ref.actor.clip_range_low=0.2",
    "++actor_rollout_ref.actor.clip_range_high=0.28",
    "actor_rollout_ref.actor.optim.lr=1e-6",
    "actor_rollout_ref.rollout.temperature=1.0",
    "actor_rollout_ref.rollout.n=8",
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16",
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1", # uncomment if you are on 1 GPU
    "++actor_rollout_ref.rollout.max_num_batched_tokens=16384",  # Must be >= max_model_len (16384)
    "reward_model.reward_manager=dapo",
    "++trainer.seed=42",
    "++trainer.save_steps=250",
    "++trainer.max_checkpoints=5",
    "++trainer.logger=['wandb', 'console']",
    "++trainer.project_name=verl-grpo-dapo",
    "++trainer.experiment_name=verl-justrl-grpo-1",
    "++trainer.log_freq=1",
    f"++data.val_files={DATA_PATH / 'dapo_math_17k.parquet'}",
    f"++data.test_files={DATA_PATH / 'dapo_math_17k.parquet'}",
    "++trainer.n_gpus_per_node=1",  # default is 8, change if you are using different number of GPUs 
    "++actor_rollout_ref.model.torch_dtype=bfloat16",   # or float16
    "++actor_rollout_ref.model.attn_implementation=flash_attention_2",
    "++actor_rollout_ref.model.load_in_4bit=False",
    "++actor_rollout_ref.model.load_in_8bit=False",
    "data.truncation=right",
]

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

# Configure Ray/uv to use the existing virtual environment
# Set UV_PROJECT_ENVIRONMENT to match VIRTUAL_ENV so Ray recognizes it
venv_path = Path(".venv").resolve()
if venv_path.exists():
    python_exe = venv_path / "bin" / "python"
    if python_exe.exists():
        python_path = str(python_exe.resolve())
        venv_abs_path = str(venv_path)
        # Set UV_PROJECT_ENVIRONMENT to tell uv where the project venv is
        env["UV_PROJECT_ENVIRONMENT"] = venv_abs_path
        # Set VIRTUAL_ENV to absolute path
        env["VIRTUAL_ENV"] = venv_abs_path
        # Set PYTHON so Ray workers use the same interpreter
        env["PYTHON"] = python_path

subprocess.run(cmd, env=env, check=True)