#!/bin/bash
# Single script to setup and run post-training Qwen3-1.7B-Base on HumanEval with GPT 5.2
# Usage: bash run.sh

set -e

MODEL="Qwen/Qwen3-1.7B-Base"
HOURS=10
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
RESULTS="results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS"

echo "=========================================="
echo "AgentPostTrain: Complete Setup & Training"
echo "=========================================="
echo ""

# Check API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY is not set"
    echo "Please set it with: export OPENAI_API_KEY='your-key-here'"
    exit 1
fi

# Step 1: Install dependencies
echo "Step 1: Checking system dependencies..."
if ! command -v apptainer &> /dev/null; then
    echo "Installing Apptainer..."
    sudo apt-get update
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:apptainer/ppa
    sudo apt-get update
    sudo apt-get install -y apptainer
fi
echo "✓ Dependencies ready"

# Step 2: Build container
echo ""
echo "Step 2: Checking container..."
if [ ! -f "containers/standard_minimal.sif" ]; then
    echo "Building optimized container (this takes 30+ minutes)..."
    sudo apptainer build src/agent_post_train/containers/standard_minimal.sif src/agent_post_train/containers/standard_minimal.def
fi
echo "✓ Container ready"

# Step 3: Download model
echo ""
echo "Step 3: Checking model cache..."
mkdir -p "${HF_HOME}"
MODEL_CACHE="${HF_HOME}/hub/models--Qwen--Qwen3-1.7B-Base"
if [ ! -d "$MODEL_CACHE" ] || [ -z "$(ls -A "$MODEL_CACHE" 2>/dev/null)" ]; then
    echo "Downloading Qwen3-1.7B-Base model..."
    sudo apptainer exec --nv --bind "${HF_HOME}:${HF_HOME}" --env HF_HOME="${HF_HOME}" \
        src/agent_post_train/containers/standard_minimal.sif python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Downloading Qwen3-1.7B-Base...')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B-Base', cache_dir='${HF_HOME}')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-1.7B-Base', cache_dir='${HF_HOME}', torch_dtype='auto', device_map='auto')
print('✓ Model downloaded')
"
fi
echo "✓ Model ready"

# Step 4: Prepare training
echo ""
echo "Step 4: Preparing training environment..."

# Load and substitute variables in prompt
PROMPT=$(sed "s|\${MODEL}|${MODEL}|g; s|\${HOURS}|${HOURS}|g" src/agent_post_train/prompt.txt)

# Create timer script
cat > /tmp/timer.sh <<EOF
#!/bin/bash
DEADLINE=\$(( $(date +%s) + ${HOURS} * 3600 ))
REMAINING=\$((DEADLINE - \$(date +%s)))
[ \$REMAINING -le 0 ] && echo "Timer expired!" || printf "Time left: %d:%02d\n" \$((REMAINING/3600)) \$(((REMAINING%3600)/60))
EOF
chmod +x /tmp/timer.sh

# Prepare job directory
JOB_DIR="/tmp/agentposttrain_$$"
mkdir -p "$JOB_DIR/task"
cp src/agent_post_train/eval/evaluate.py "$JOB_DIR/task/"
cp /tmp/timer.sh "$JOB_DIR/task/timer.sh"
echo "$PROMPT" > "$JOB_DIR/task/prompt.txt"
cp src/agent_post_train/agents/codex/solve.sh "$JOB_DIR/agent_solve.sh"
chmod +x "$JOB_DIR/agent_solve.sh"

echo "✓ Training environment ready"

# Step 5: Run training
echo ""
echo "=========================================="
echo "Step 5: Starting Training (${HOURS} hours)"
echo "=========================================="
echo "Results will be saved to: $RESULTS"
echo "Logs: $RESULTS/output.log"
echo ""

timeout --signal=TERM --kill-after=30s "$((HOURS * 60 + 5))m" \
sudo apptainer exec --nv -c \
    --env PATH="/home/ben/.local/bin:$PATH" \
    --env HF_HOME="${HF_HOME}" \
    --env CODEX_API_KEY="${OPENAI_API_KEY}" \
    --env VLLM_API_KEY="inspectai" \
    --env PYTHONNOUSERSITE="1" \
    --env PROMPT="${PROMPT}" \
    --env MODEL_TO_TRAIN="${MODEL}" \
    --env AGENT_CONFIG="gpt-5.2" \
    --bind "${HF_HOME}:${HF_HOME}" \
    --home "${JOB_DIR}:/home/ben" \
    --pwd "/home/ben/task" \
    --writable-tmpfs \
    src/agent_post_train/containers/standard_minimal.sif \
    bash /home/ben/agent_solve.sh > "$RESULTS/output.log" 2>&1

# Step 6: Copy results
echo ""
echo "Step 6: Collecting results..."
[ -d "$JOB_DIR/task/final_model" ] && cp -r "$JOB_DIR/task/final_model" "$RESULTS/" || echo "⚠ No final_model found"
cp -r "$JOB_DIR/task" "$RESULTS/task" 2>/dev/null || true
echo "✓ Results collected"

# Step 7: Final evaluation
echo ""
echo "=========================================="
echo "Step 7: Running Final Evaluation"
echo "=========================================="
echo ""

if [ -d "$RESULTS/final_model" ]; then
    sudo apptainer exec --nv \
        --env HF_HOME="${HF_HOME}" \
        --env OPENAI_API_KEY="${OPENAI_API_KEY}" \
        --env VLLM_API_KEY="inspectai" \
        --env PYTHONNOUSERSITE="1" \
        --writable-tmpfs \
        --bind "$(pwd):$(pwd)" \
        --bind "${HF_HOME}:${HF_HOME}" \
        --pwd "$(pwd)/eval" \
        src/agent_post_train/containers/standard_minimal.sif \
        python3 evaluate.py --model-path "$RESULTS/final_model" --limit -1 \
        --json-output-file "$RESULTS/metrics.json" > "$RESULTS/eval.txt" 2>&1
    
    echo "Evaluation Results:"
    echo "-------------------"
    cat "$RESULTS/eval.txt"
    echo ""
    
    if [ -f "$RESULTS/metrics.json" ]; then
        echo "Metrics saved to: $RESULTS/metrics.json"
    fi
else
    echo "⚠ Skipping evaluation: no final_model found"
fi

echo ""
echo "=========================================="
echo "Complete!"
echo "=========================================="
echo "Results directory: $RESULTS"
echo "  - output.log: Training logs"
echo "  - eval.txt: Final evaluation"
echo "  - metrics.json: Evaluation metrics"
echo "  - final_model/: Trained model (if available)"
echo ""
