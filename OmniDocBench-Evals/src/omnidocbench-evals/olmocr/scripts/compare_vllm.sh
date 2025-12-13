#!/bin/bash

# Compares VLM inference between vLLM and HuggingFace checkpoints
# Usage: ./scripts/compare_vllm.sh <model_path> [--max-tokens N] [--num-prompts N] [--prob-threshold F] [--seed N]

set -e

# Default values
DEFAULT_MAX_TOKENS=1000
DEFAULT_NUM_PROMPTS=100
DEFAULT_PROB_THRESHOLD=0.20
DEFAULT_SEED=42

# Parse arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_path> [--max-tokens N] [--num-prompts N] [--prob-threshold F] [--seed N]"
    echo "Example: $0 Qwen/Qwen2.5-VL-7B-Instruct"
    echo "Example: $0 s3://ai2-oe-data/jakep/olmocr/model --max-tokens 50 --num-prompts 200"
    exit 1
fi

MODEL_PATH="$1"
MAX_TOKENS="$DEFAULT_MAX_TOKENS"
NUM_PROMPTS="$DEFAULT_NUM_PROMPTS"
PROB_THRESHOLD="$DEFAULT_PROB_THRESHOLD"
SEED="$DEFAULT_SEED"

# Parse optional arguments
shift 1
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --num-prompts)
            NUM_PROMPTS="$2"
            shift 2
            ;;
        --prob-threshold)
            PROB_THRESHOLD="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "Error: There are uncommitted changes in the repository."
    echo "Please commit or stash your changes before running the comparison."
    echo ""
    echo "Uncommitted changes:"
    git status --short
    exit 1
fi

# Use conda environment Python if available, otherwise use system Python
if [ -n "$CONDA_PREFIX" ]; then
    PYTHON="$CONDA_PREFIX/bin/python"
    echo "Using conda Python from: $CONDA_PREFIX"
else
    PYTHON="python"
    echo "Warning: No conda environment detected, using system Python"
fi

# Get version from version.py
VERSION=$($PYTHON -c 'import olmocr.version; print(olmocr.version.VERSION)')
echo "OlmOCR version: $VERSION"

# Get first 10 characters of git hash
GIT_HASH=$(git rev-parse HEAD | cut -c1-10)
echo "Git hash: $GIT_HASH"

# Get current git branch name
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Git branch: $GIT_BRANCH"

# Create full image tag
IMAGE_TAG="olmocr-compare-vllm-${VERSION}-${GIT_HASH}"
echo "Building Docker image with tag: $IMAGE_TAG"

# Build the Docker image
echo "Building Docker image..."
docker build --platform linux/amd64 -f ./Dockerfile -t $IMAGE_TAG .

# Get Beaker username
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
echo "Beaker user: $BEAKER_USER"

# Push image to beaker
echo "Trying to push image to Beaker..."
if ! beaker image create --workspace ai2/oe-data-pdf --name $IMAGE_TAG $IMAGE_TAG 2>/dev/null; then
    echo "Warning: Beaker image with tag $IMAGE_TAG already exists. Using existing image."
fi

# Create Python script to run beaker experiment
cat << 'EOF' > /tmp/run_compare_vllm_experiment.py
import sys
from beaker import Beaker, ExperimentSpec, TaskSpec, TaskContext, ResultSpec, TaskResources, ImageSource, Priority, Constraints, EnvVar, DataMount

# Get parameters from command line
image_tag = sys.argv[1]
beaker_user = sys.argv[2]
git_branch = sys.argv[3]
git_hash = sys.argv[4]
model_path = sys.argv[5]
max_tokens = sys.argv[6]
num_prompts = sys.argv[7]
prob_threshold = sys.argv[8]
seed = sys.argv[9]

# Initialize Beaker client
b = Beaker.from_env(default_workspace="ai2/olmocr")

# Check if AWS credentials secret exists
aws_creds_secret = f"{beaker_user}-AWS_CREDENTIALS_FILE"
try:
    # Try to get the secret to see if it exists
    b.secret.get(aws_creds_secret, workspace="ai2/olmocr")
    has_aws_creds = True
    print(f"Found AWS credentials secret: {aws_creds_secret}")
except:
    has_aws_creds = False
    print(f"AWS credentials secret not found: {aws_creds_secret}")

# Build commands for comparison job
commands = []
if has_aws_creds:
    commands.extend([
        "mkdir -p ~/.aws",
        'echo "$AWS_CREDENTIALS_FILE" > ~/.aws/credentials'
    ])

commands.extend([
    # Install accelerate
    "pip install accelerate",
    # Run comparison
    f'python -m olmocr.train.compare_vllm_checkpoint --model {model_path} --max-tokens {max_tokens} --num-prompts {num_prompts} --prob-threshold {prob_threshold} --seed {seed}'
])

# Build task spec with optional env vars
task_spec_args = {
    "name": "olmocr-compare-vllm",
    "image": ImageSource(beaker=f"{beaker_user}/{image_tag}"),
    "command": [
        "bash", "-c",
        " && ".join(commands)
    ],
    "context": TaskContext(
        priority=Priority.normal,
        preemptible=True,
    ),
    "resources": TaskResources(gpu_count=1),
    "constraints": Constraints(cluster=["ai2/ceres-cirrascale", "ai2/jupiter-cirrascale-2"]),
    "result": ResultSpec(path="/noop-results"),
    "datasets": [
        DataMount.new(mount_path="/weka/oe-data-default", weka="oe-data-default"),
        DataMount.new(mount_path="/weka/oe-training-default", weka="oe-training-default"),
    ]
}

# Add env vars if AWS credentials exist
if has_aws_creds:
    task_spec_args["env_vars"] = [
        EnvVar(name="AWS_CREDENTIALS_FILE", secret=aws_creds_secret)
    ]

# Create experiment spec
experiment_spec = ExperimentSpec(
    description=f"OlmOCR vLLM vs HF Comparison - Branch: {git_branch}, Commit: {git_hash}, Model: {model_path}",
    budget="ai2/oe-base",
    tasks=[TaskSpec(**task_spec_args)],
)

# Create the experiment
experiment = b.experiment.create(spec=experiment_spec, workspace="ai2/olmocr")
print(f"Created comparison experiment: {experiment.id}")
print(f"View at: https://beaker.org/ex/{experiment.id}")
EOF

# Run the Python script to create the experiment
echo "Creating Beaker experiment..."
echo "Comparing model: $MODEL_PATH"
echo "Max tokens: $MAX_TOKENS"
echo "Number of prompts: $NUM_PROMPTS"
echo "Probability threshold: $PROB_THRESHOLD"
echo "Random seed: $SEED"
$PYTHON /tmp/run_compare_vllm_experiment.py $IMAGE_TAG $BEAKER_USER $GIT_BRANCH $GIT_HASH "$MODEL_PATH" "$MAX_TOKENS" "$NUM_PROMPTS" "$PROB_THRESHOLD" "$SEED"

# Clean up temporary file
rm /tmp/run_compare_vllm_experiment.py

echo "Comparison experiment submitted successfully!"