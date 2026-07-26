set -x

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set"
    echo "Please run: export HF_TOKEN=your_huggingface_token"
    exit 1
fi

# Login to HuggingFace to access models
huggingface-cli login --token "$HF_TOKEN"

# First, prepare the data with test set reserved for final evaluation
# python3 grpo_data_gsm8k.py --local_dir ./src/data/gsm8k-0.1 --train_split 0.1 --test_samples 200

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    trainer.val_before_train=False \
    data.train_files=./src/data/gsm8k-0.1/train.parquet \
    data.val_files=./src/data/gsm8k-0.1/validation.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
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
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='qwen2.5_3b_ins_grpo_lora_0.1' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=23 \
    trainer.test_freq=23 \
    trainer.total_epochs=1
  
    # actor_rollout_ref.actor.ppo_mini_batch_size=256 \  
    # data.train_batch_size=1024 \  
    # trainer.n_gpus_per_node=8 \  
    # actor_rollout_ref.model.use_shm=True \

# After training completes, evaluate the saved model on the reserved test set:
# python3 evaluate_model.py \
#     --model_path <path_to_saved_checkpoint> \
#     --base_model qwen/Qwen2.5-3B-Instruct \
#     --test_file ./src/data/gsm8k-0.1/test.parquet \
#     --task_type gsm8k \
#     --output_file ./eval_results_gsm8k.json
