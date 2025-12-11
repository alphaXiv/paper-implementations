#!/bin/bash

# Compresses an OlmOCR model using quantization
# Usage: ./scripts/compress_model.sh <recipe_path> <input_model_path> <output_model_path> [--calibration-pdfs PATTERN]

set -e

# Default calibration PDFs pattern
DEFAULT_CALIBRATION_PDFS="/weka/oe-data-default/jakep/olmOCR-mix-0225-benchmark_set/*.pdf"

# Parse arguments
if [ $# -lt 3 ]; then
    echo "Usage: $0 <recipe_path> <input_model_path> <output_model_path> [--calibration-pdfs PATTERN]"
    echo "Example: $0 olmocr/train/quantization_configs/qwen2_5vl_w8a8_int8.yaml ./olmocrv2-base/ s3://ai2-oe-data/jakep/olmocr/compressed-model"
    echo "Example with custom PDFs: $0 recipe.yaml ./model/ s3://output/ --calibration-pdfs '/path/to/pdfs/*.pdf'"
    exit 1
fi

RECIPE="$1"
INPUT_MODEL="$2"
OUTPUT_MODEL="$3"
CALIBRATION_PDFS="$DEFAULT_CALIBRATION_PDFS"

# Check for optional calibration-pdfs argument
shift 3
while [[ $# -gt 0 ]]; do
    case $1 in
        --calibration-pdfs)
            CALIBRATION_PDFS="$2"
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
    echo "Please commit or stash your changes before running the compression."
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
IMAGE_TAG="olmocr-compress-${VERSION}-${GIT_HASH}"
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
cat << 'EOF' > /tmp/run_compress_experiment.py
import sys
from beaker import Beaker, ExperimentSpec, TaskSpec, TaskContext, ResultSpec, TaskResources, ImageSource, Priority, Constraints, EnvVar, DataMount

# Get parameters from command line
image_tag = sys.argv[1]
beaker_user = sys.argv[2]
git_branch = sys.argv[3]
git_hash = sys.argv[4]
recipe = sys.argv[5]
input_model = sys.argv[6]
output_model = sys.argv[7]
calibration_pdfs = sys.argv[8]

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

# Build commands for compression job
commands = []
if has_aws_creds:
    commands.extend([
        "mkdir -p ~/.aws",
        'echo "$AWS_CREDENTIALS_FILE" > ~/.aws/credentials'
    ])

commands.extend([
    # Install llmcompressor
    "pip install llmcompressor==0.6.0",
    # Run compression
    f'python -m olmocr.train.compress_checkpoint --recipe {recipe} {input_model} {output_model} --calibration-pdfs "{calibration_pdfs}"'
])

# Build task spec with optional env vars
task_spec_args = {
    "name": "olmocr-compress",
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
    description=f"OlmOCR Model Compression - Branch: {git_branch}, Commit: {git_hash}, Recipe: {recipe}",
    budget="ai2/oe-base",
    tasks=[TaskSpec(**task_spec_args)],
)

# Create the experiment
experiment = b.experiment.create(spec=experiment_spec, workspace="ai2/olmocr")
print(f"Created compression experiment: {experiment.id}")
print(f"View at: https://beaker.org/ex/{experiment.id}")
EOF

# Run the Python script to create the experiment
echo "Creating Beaker experiment..."
echo "Compressing model from: $INPUT_MODEL to: $OUTPUT_MODEL"
echo "Using recipe: $RECIPE"
echo "Using calibration PDFs: $CALIBRATION_PDFS"
$PYTHON /tmp/run_compress_experiment.py $IMAGE_TAG $BEAKER_USER $GIT_BRANCH $GIT_HASH "$RECIPE" "$INPUT_MODEL" "$OUTPUT_MODEL" "$CALIBRATION_PDFS"

# Clean up temporary file
rm /tmp/run_compress_experiment.py

echo "Compression experiment submitted successfully!"