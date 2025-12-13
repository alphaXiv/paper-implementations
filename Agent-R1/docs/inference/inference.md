### Inference Guide

#### Step 1: Convert Training Checkpoints to HF Format

Before running inference, you need to convert your training checkpoints to the Hugging Face format compatible with vLLM.

**Check Your Checkpoints**

First, ensure your training script saved checkpoints properly:
- Make sure your training configuration had `trainer.save_freq > 0` (default is `-1`, which doesn't save checkpoints)
- Verify that checkpoint files exist in your specified output directory

**Convert Checkpoints**

Use the model merge script to convert your checkpoints:

```bash
# Copy the merge script to your working directory (if needed)
cp scripts/model_merge.sh ./

# Modify the script to use your checkpoint directory
# Edit the script and replace <your_checkpoint_dir> with your actual checkpoint path, e.g. ./checkpoints/<your_project_name>/<your_experiment_name>/global_step_<your_global_step>/actor
# Edit the script and replace <your_hf_model_path> with your actual HF model path, e.g. Qwen/Qwen2.5-3B-Instruct
# Edit the script and replace <your_target_dir> with your actual target directory

# Run the conversion
bash model_merge.sh
```

#### Step 2: Deploy the vLLM Service

After converting your model, the next step is to start the vLLM service that will load the model and expose an API endpoint.

```bash
# Copy the serve script (if needed)
cp scripts/vllm_serve.sh ./

# Edit vllm_serve.sh and replace <your_model_name> with the path to your 
# converted HF model from Step 1

# Start the vLLM service
bash vllm_serve.sh
```

#### Step 3: Running Inference

Once the service is running, you can interact with your model in two ways:

**Interactive Chat Mode**

For interactive conversations with the model:

```bash
# Start the interactive chat interface
python3 -m agent_r1.vllm_infer.chat
```

This launches an interactive terminal where you can chat with the model directly, including support for tools if your model was trained with tool-use capabilities.

**Single Query Mode**

For automation or one-off queries:

```bash
# Run a single inference query
python3 -m agent_r1.vllm_infer.run --question "What is the capital of France?"
```

#### Advanced Usage

**Custom Configuration**

You can create a custom configuration file to override defaults:

```python
# custom_config.py
TOOLS = ["CUSTOM_TOOL"]
OPENAI_API_KEY = "EMPTY"
OPENAI_API_BASE = "http://localhost:8000/v1"
MODEL_NAME = "agent"
TEMPERATURE = 0.8
TOP_P = 0.9
MAX_TOKENS = 1024
REPETITION_PENALTY = 1.1
```

Then pass it to your inference script:

```bash
python3 -m agent_r1.vllm_infer.chat --config path/to/custom_config.py
```