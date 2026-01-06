#!/usr/bin/env python3

import os
from huggingface_hub import HfApi, login

# Login to Hugging Face
login()

# Initialize the API
api = HfApi()

# Create the repo if it doesn't exist
repo_id = "alphaXiv/spurious-rewards-data"
try:
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False)
    print(f"Created repo {repo_id}")
except Exception as e:
    print(f"Repo {repo_id} already exists or error: {e}")

# Base directory for data
data_dir = "src/spurious_rewards/code/data"

# Upload the entire data folder
if os.path.exists(data_dir):
    api.upload_folder(
        folder_path=data_dir,
        repo_id=repo_id,
        repo_type="dataset"
    )
    print(f"Uploaded data folder to {repo_id}")
else:
    print(f"Data directory {data_dir} does not exist")

print("Data upload completed.")