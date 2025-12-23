export CUDA_VISIBLE_DEVICES=0
export MODEL_NAME=./converted_model

vllm serve $MODEL_NAME --enable-auto-tool-choice --tool-call-parser hermes --served-model-name agent --port 8000