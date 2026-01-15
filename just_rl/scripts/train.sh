export CUDA_VISIBLE_DEVICES=0
export N_GPUS=1
export ROLLOUT_TP_SIZE=1

export HYDRA_FULL_ERROR=1

export DATA_DIR="data/processed"
export BASE_MODEL="Qwen/Qwen2.5-0.5B"
export EXPERIMENT_NAME="verl-justrl-grpo-1"
export CKPT_DIR="checkpoints/$EXPERIMENT_NAME"

python -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/dapo_math_17k.parquet \
    data.val_files=$DATA_DIR/dapo_math_17k.parquet \
    data.train_batch_size=256 \
    data.val_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=15360 \
    data.truncation=right \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    ++actor_rollout_ref.actor.clip_range_low=0.2 \
    ++actor_rollout_ref.actor.clip_range_high=0.28 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    reward_model.reward_manager=dapo \
    ++trainer.seed=42 \
    ++trainer.save_steps=250 \
    ++trainer.max_checkpoints=5 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=verl-grpo-dapo \
    trainer.experiment_name=$EXPERIMENT_NAME \
    ++trainer.log_freq=1 \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.total_training_steps=2000 \
    trainer.default_local_dir=$CKPT_DIR \

    
