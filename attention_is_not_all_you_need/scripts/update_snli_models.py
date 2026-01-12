#!/usr/bin/env python3
"""
Update SNLI model checkpoints on HuggingFace with latest trained models.
"""

from pathlib import Path
from huggingface_hub import HfApi

# Configuration
HF_REPO = "alphaXiv/attention-is-not-all-you-need-models"
LATEST_OUTPUT = "/home/ubuntu/central-texas-storage/paper-implementations/attention_is_not_all_you_need/outputs/2026-01-12_12-38-19"

def upload_snli_models():
    """Upload updated SNLI model checkpoints to HuggingFace."""
    
    api = HfApi()
    
    # Check if logged in
    try:
        user = api.whoami()
        print(f"✅ Logged in as: {user['name']}")
    except Exception as e:
        print("❌ Not logged in to Hugging Face!")
        print("Please run: huggingface-cli login")
        return
    
    print(f"\nUploading SNLI models to: {HF_REPO}")
    print(f"Source directory: {LATEST_OUTPUT}\n")
    
    models = ["grassmann_snli", "transformer_snli"]
    
    for model in models:
        checkpoint_path = Path(LATEST_OUTPUT) / model / "checkpoints" / "best.pt"
        
        if not checkpoint_path.exists():
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            continue
        
        # Get file size
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"📤 Uploading {model}/checkpoints/best.pt ({size_mb:.1f} MB)...")
        
        try:
            api.upload_file(
                path_or_fileobj=str(checkpoint_path),
                path_in_repo=f"{model}/checkpoints/best.pt",
                repo_id=HF_REPO,
                repo_type="model",
            )
            print(f"   ✅ Successfully uploaded {model}")
        except Exception as e:
            print(f"   ❌ Failed to upload {model}: {e}")
        
        # Also upload result JSON files
        json_files = [
            "snli_validation_results.json",
            "snli_test_results.json",
            "snli_results.json",
        ]
        
        for json_file in json_files:
            json_path = Path(LATEST_OUTPUT) / model / json_file
            if json_path.exists():
                try:
                    api.upload_file(
                        path_or_fileobj=str(json_path),
                        path_in_repo=f"{model}/{json_file}",
                        repo_id=HF_REPO,
                        repo_type="model",
                    )
                    print(f"   ✅ Uploaded {json_file}")
                except Exception as e:
                    print(f"   ⚠️  Failed to upload {json_file}: {e}")
    
    print("\n" + "="*60)
    print("✅ SNLI models updated successfully!")
    print(f"View at: https://huggingface.co/{HF_REPO}")
    print("="*60)

if __name__ == "__main__":
    upload_snli_models()
