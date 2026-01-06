## Instructions

The repo assumes you are running on an instance with 8 GPUs. Lamdba Labs 8xA100 (80GB SXM4) is what I was using. 

All scripts/commands should be run in the root of the `just_rl` project unless otherwise specified. 

Setup `uv` and install dependencies with `uv sync`.

Activate virtual env with `source .venv/bin/activate`.

Prepare data for training with `python scripts/prepare_data.py`. The script will throw an error at the end, you can ignore it. Rename `train.parquet` inside `/data/processed/` to `dapo_math_17k.parquet`. 

Run training script with `python scripts/train.py`. 