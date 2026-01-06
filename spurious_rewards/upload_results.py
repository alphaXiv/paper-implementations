#!/usr/bin/env python3

import os
from huggingface_hub import HfApi, login

# Login to Hugging Face
login()

# Initialize the API
api = HfApi()

# Create the repo if it doesn't exist
repo_id = "alphaXiv/spurious-rewards-reasoning-traces"
try:
    api.create_repo(repo_id=repo_id, repo_type="model", private=False)
    print(f"Created repo {repo_id}")
except Exception as e:
    print(f"Repo {repo_id} already exists or error: {e}")

# Base directory for results
results_dir = "src/spurious_rewards/code/results"

# Upload each JSON file to the repo
for folder in ["base", "step50", "step200", "step400", "step1000"]:
    folder_path = os.path.join(results_dir, folder)
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.json'):
                file_path = os.path.join(folder_path, file_name)
                # Upload to the repo, in a folder structure
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=f"{folder}/{file_name}",
                    repo_id=repo_id,
                    repo_type="model"
                )
                print(f"Uploaded {folder}/{file_name} to {repo_id}")
    else:
        print(f"Folder {folder_path} does not exist")

print("All uploads completed.")