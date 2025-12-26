export CHECKPOINT_DIR=<your_checkpoint_dir>
export HF_MODEL_PATH=<your_hf_model_path>
export TARGET_DIR=<your_target_dir>

python3 verl/scripts/model_merger.py --backend fsdp --hf_model_path $HF_MODEL_PATH --local_dir $CHECKPOINT_DIR --target_dir $TARGET_DIR