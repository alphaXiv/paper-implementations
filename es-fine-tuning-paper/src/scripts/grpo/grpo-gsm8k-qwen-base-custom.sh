#!/bin/bash
set -x

# GRPO training for GSM8K using BASE Qwen model with custom chat template tokenizer
# Model: Qwen/Qwen2.5-3B (base, not instruct)
# Tokenizer: Custom tokenizer with "Question: {input} Answer: Let's think step by step." template

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set"
    echo "Please run: export HF_TOKEN=your_huggingface_token"
    exit 1
fi

# Login to HuggingFace to access models
huggingface-cli login --token "$HF_TOKEN"

# Prepare custom tokenizer if not already created
if [ ! -d "./tokenizers/qwen2.5-3b-base-chat" ]; then
    echo "Creating custom tokenizer for Qwen base model..."
    python3 base_model_tokenizer.py \
        --model_path Qwen/Qwen2.5-3B \
        --save_path ./tokenizers/qwen2.5-3b-base-chat
fi

# Train base Qwen model with custom tokenizer
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    trainer.val_before_train=False \
    data.train_files=./src/data/gsm8k-0.4/train.parquet \
    data.val_files=./src/data/gsm8k-0.4/validation.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B \
    +actor_rollout_ref.model.tokenizer_path=./tokenizers/qwen2.5-3b-base-chat \
    +actor_rollout_ref.model.lora_rank=64 \
    +actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
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
    actor_rollout_ref.rollout.max_num_batched_tokens=65535 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_gsm8k_base_custom' \
    trainer.experiment_name='qwen2.5_3b_base_custom_template_lora' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=23 \
    trainer.test_freq=1 \
    trainer.total_epochs=1

# After training completes, evaluate the saved model on the reserved test set:
# bash evaluate_gsm8k.sh --base_model Qwen/Qwen2.5-3B \
#     --project_name verl_grpo_gsm8k_base_custom \
#     --experiment_name qwen2.5_3b_base_custom_template_lora