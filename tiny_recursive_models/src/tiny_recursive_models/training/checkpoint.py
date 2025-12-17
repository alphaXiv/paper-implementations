"""Checkpoint saving and loading utilities."""

import os
import torch
from torch import nn

from tiny_recursive_models.training.config import PretrainConfig, TrainState


def save_train_state(config: PretrainConfig, train_state: TrainState):
    """Save model checkpoint to disk.
    
    Args:
        config: Training configuration
        train_state: Current training state
    """
    # FIXME: Only saved model.
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, f"step_{train_state.step}"))


def load_checkpoint(model: nn.Module, config: PretrainConfig):
    """Load model checkpoint from disk or HuggingFace.
    
    Args:
        model: Model to load checkpoint into
        config: Training configuration with load_checkpoint path
    """
    if config.load_checkpoint is not None:
        print(f"Loading checkpoint {config.load_checkpoint}")

        checkpoint_path = config.load_checkpoint
        
        # Check if this is a HuggingFace repo path (format: "username/repo/filename")
        if "/" in checkpoint_path and not os.path.exists(checkpoint_path):
            try:
                from huggingface_hub import hf_hub_download
                
                # Parse HuggingFace path: "alphaXiv/trm-model-maze/maze_hard_step_32550"
                parts = checkpoint_path.split("/", 2)
                if len(parts) < 3:
                    raise ValueError(
                        f"HuggingFace path must be in format 'username/repo/filename'. Got: {checkpoint_path}"
                    )
                
                repo_id = f"{parts[0]}/{parts[1]}"
                filename = parts[2]
                
                print(f"Downloading from HuggingFace: repo={repo_id}, file={filename}")
                checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename)
                print(f"Downloaded to: {checkpoint_path}")
                
            except ImportError:
                raise ImportError(
                    "huggingface_hub is required to load checkpoints from HuggingFace. "
                    "Install it with: pip install huggingface_hub"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to download checkpoint from HuggingFace: {e}")

        # Load state dict
        state_dict = torch.load(checkpoint_path, map_location="cuda")

        # Always strip compile/DataParallel style prefixes so keys match the
        # non-compiled module. We won't be using torch.compile in eval.
        def _strip_prefixes(sd: dict) -> dict:
            out: dict[str, torch.Tensor] = {}
            for k, v in sd.items():
                key = k
                if isinstance(key, str):
                    # remove a leading '.' if present
                    if key.startswith('.'):
                        key = key[1:]
                    # known wrapper prefixes to drop
                    for pref in ("_orig_mod.", "_orig._mod.", "module."):
                        if key.startswith(pref):
                            key = key[len(pref):]
                            break
                out[key] = v
            return out

        state_dict = _strip_prefixes(state_dict)

        # Resize and reset puzzle emb if needed
        try:
            expected_shape: torch.Size = model.model.puzzle_emb.weights.shape  # type: ignore
            puzzle_emb_name = "model.inner.puzzle_emb.weights"
            if puzzle_emb_name in state_dict:
                puzzle_emb = state_dict[puzzle_emb_name]
                if getattr(puzzle_emb, 'shape', None) != expected_shape:
                    print(f"Resetting puzzle embedding as shape is different. Found {getattr(puzzle_emb, 'shape', None)}, Expected {expected_shape}")
                    state_dict[puzzle_emb_name] = (
                        torch.mean(puzzle_emb, dim=0, keepdim=True).expand(expected_shape).contiguous()
                    )
        except Exception:
            pass
        
        model.load_state_dict(state_dict, assign=True)
