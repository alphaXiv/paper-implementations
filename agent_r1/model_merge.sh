export CHECKPOINT_DIR=checkpoints/hotpotqa/ppo-qwen2.5-1.5b-instruct/global_step_1/actor
export HF_MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct
export TARGET_DIR=./converted_model

python3 src/verl/scripts/model_merger.py --backend fsdp --hf_model_path $HF_MODEL_PATH --local_dir $CHECKPOINT_DIR --target_dir $TARGET_DIR