export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export N_GPUS=8
export ROLLOUT_TP_SIZE=1

export HYDRA_FULL_ERROR=1

export DATA_DIR="data/processed"
export BASE_MODEL="Qwen/Qwen2.5-0.5B"
export EXPERIMENT_NAME="verl-justrl-grpo-yes-strict-box-verify"
export CKPT_DIR="checkpoints/$EXPERIMENT_NAME"
export TRAIN_DATA_FILE="train.parquet"
export VAL_DATA_FILE="val.parquet"


python -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/$TRAIN_DATA_FILE \
    data.val_files=$DATA_DIR/$VAL_DATA_FILE \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    data.truncation=left \
    data.max_prompt_length=1024 \
    data.max_response_length=7168 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_num_batched_tokens=262144 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    ++actor_rollout_ref.actor.clip_range_low=0.2 \
    ++actor_rollout_ref.actor.clip_range_high=0.28 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    reward_model.reward_manager=dapo \
    ++trainer.seed=42 \
    trainer.save_freq=5 \
    trainer.test_freq=10 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=verl-grpo-dapo \
    trainer.experiment_name=$EXPERIMENT_NAME \
    ++trainer.log_freq=1 \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.total_training_steps=2000 \
    trainer.default_local_dir=$CKPT_DIR \

    
