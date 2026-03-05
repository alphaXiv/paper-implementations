set -x

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set"
    echo "Please run: export HF_TOKEN=your_huggingface_token"
    exit 1
fi

# Login to HuggingFace to access models
huggingface-cli login --token "$HF_TOKEN"

# Alternative approach using custom_reward_function config
# This version uses an external reward function file instead of built-in VERL reward scoring

#bsz was changed to 128 from 256 because the verl script dropped off data sample if it is not under the batch size we had set

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    trainer.val_before_train=False \
    data.train_files=./src/data/countdown-0.4/train.parquet \
    data.val_files=./src/data/countdown-0.4/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=256 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path=meta-llama/Llama-3.2-3B-Instruct \
    +actor_rollout_ref.model.lora_rank=64 \
    +actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_countdown' \
    trainer.experiment_name='qwen2.5_3b_grpo_countdown_lora' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.total_epochs=100 \
    custom_reward_function.path=./countdown_reward.py \
    custom_reward_function.name=countdown_reward_function
  
    # actor_rollout_ref.actor.ppo_mini_batch_size=256 \  
    # data.train_batch_size=1024 \  
    # trainer.n_gpus_per_node=8 \  
    # actor_rollout_ref.model.use_shm=True \

# After training completes, evaluate the saved model on the test set:
# python3 evaluate_model.py \
#     --model_path <path_to_saved_checkpoint> \
#     --base_model meta-llama/Llama-3.2-3B-Instruct \
#     --test_file ./src/data/countdown-0.1/test.parquet \
#     --task_type countdown \
#     --output_file ./eval_results_countdown.json
