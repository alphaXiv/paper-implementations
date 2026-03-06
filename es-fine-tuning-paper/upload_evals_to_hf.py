import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, login

# Login to HuggingFace
token = os.getenv("HF_TOKEN")
if not token:
    print("HF_TOKEN not found in environment variables.")
    print("Please set it with: export HF_TOKEN=your_token_here")
    print("Or login with: huggingface-cli login")
    exit(1)

login(token=token)

api = HfApi()
repo = "alphaXiv/es-grpo-results"
evals = "./src/evals"

print(f"Uploading to: {repo}")
create_repo(repo_id=repo, repo_type="dataset", exist_ok=True, private=False)

for subdir in ["es-evals", "grpo-evals"]:
    path = Path(evals) / subdir
    if not path.exists():
        continue
    print(f"\n{subdir}/")
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith("."):
                continue
            local = Path(root) / file
            rel = local.relative_to(evals)
            try:
                api.upload_file(
                    path_or_fileobj=str(local),
                    path_in_repo=str(rel),
                    repo_id=repo,
                    repo_type="dataset"
                )
                print(f"  ✓ {rel}")
            except Exception as e:
                print(f"  ✗ {rel}, error: {e}")
print(f"\n✅ Done! https://huggingface.co/datasets/{repo}")
