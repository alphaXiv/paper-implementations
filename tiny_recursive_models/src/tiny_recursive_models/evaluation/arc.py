"""ARC-AGI evaluator with pass@k voting."""

from typing import Dict, Sequence, Optional
import os
import json

import torch
import numpy as np
import torch.distributed as dist

from tiny_recursive_models.data.build_arc_dataset import inverse_aug, grid_hash, arc_grid_to_np
from tiny_recursive_models.data.common import PuzzleDatasetMetadata


def _crop_np(grid: np.ndarray) -> np.ndarray:
    """Find maximum-sized rectangle without any EOS token (values outside [2, 11]).
    Returns the cropped grid with values shifted to [0..9] and dtype=uint8.
    Pure NumPy implementation to avoid Numba runtime issues.
    """
    g = np.asarray(grid, dtype=grid.dtype).reshape(30, 30)
    nr, nc = g.shape
    max_area = 0
    max_r = 0
    max_c = 0
    # Track current usable width (num_c) as rows grow
    num_c = nc
    for num_r in range(1, nr + 1):
        # shrink num_c if an EOS token is encountered in this new row
        for c in range(1, num_c + 1):
            x = g[num_r - 1, c - 1]
            if (x < 2) or (x > 11):
                num_c = c - 1
                break
        area = num_r * num_c
        if area > max_area:
            max_area = area
            max_r = num_r
            max_c = num_c
    out = g[:max_r, :max_c] - 2
    return out.astype(np.uint8, copy=False)


class ARC:
    """ARC-AGI task evaluator with pass@k metrics and submission generation."""
    
    required_outputs = {"inputs", "puzzle_identifiers", "q_halt_logits", "preds"}
    
    def __init__(self, data_path: str, 
        eval_metadata: PuzzleDatasetMetadata, 
        submission_K: int = 2, 
        pass_Ks: Sequence[int] = (1, 2, 5, 10, 100, 1000), 
        aggregated_voting: bool = True):
        super().__init__()
        self.pass_Ks = pass_Ks
        self.submission_K = submission_K
        self.aggregated_voting = aggregated_voting
        self.blank_identifier_id = eval_metadata.blank_identifier_id

        # Load identifiers and test puzzles
        with open(os.path.join(data_path, "identifiers.json"), "r") as f:
            self.identifier_map = json.load(f)
        with open(os.path.join(data_path, "test_puzzles.json"), "r") as f:
            self.test_puzzles = json.load(f)
            
        # States
        self._local_hmap = {}
        self._local_preds = {}
        
    def begin_eval(self):
        """Reset evaluator state for a new evaluation run."""
        if not self.aggregated_voting:
            # Clear previous predictions
            self._local_hmap = {}
            self._local_preds = {}
    
    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        """Process one batch of predictions.
        
        Args:
            batch: Input batch (inputs, labels, puzzle_identifiers)
            preds: Model predictions (preds, q_halt_logits)
        """
        # Collect required outputs to CPU
        outputs = {}
        q_values = None

        for collection in (batch, preds):
            for k, v in collection.items():
                if k in self.required_outputs:
                    if k == "q_halt_logits":
                        q_values = v.to(torch.float64).sigmoid().cpu()
                    else:
                        outputs[k] = v.cpu()
                        
        assert q_values is not None

        # Remove padding from outputs
        mask = outputs["puzzle_identifiers"] != self.blank_identifier_id
        outputs = {k: v[mask] for k, v in outputs.items()}

        # Get predictions
        for identifier, input, pred, q in zip(outputs["puzzle_identifiers"].numpy(), outputs["inputs"].numpy(), outputs["preds"].numpy(), q_values.numpy()):
            name = self.identifier_map[identifier]
            orig_name, _inverse_fn = inverse_aug(name)
            
            input_hash = grid_hash(_inverse_fn(_crop_np(input)))
            
            pred = _inverse_fn(_crop_np(pred))
            assert np.all((pred >= 0) & (pred <= 9)), f"Puzzle {name}'s prediction out of 0-9 range."  # Sanity check

            # Store into local state
            pred_hash = grid_hash(pred)

            self._local_hmap[pred_hash] = pred
            
            self._local_preds.setdefault(orig_name, {})
            self._local_preds[orig_name].setdefault(input_hash, [])
            self._local_preds[orig_name][input_hash].append((pred_hash, float(q)))
    
    def result(self, save_path: Optional[str], rank: int, world_size: int, group: Optional[torch.distributed.ProcessGroup] = None) -> Optional[Dict[str, float]]:
        """Compute final metrics after all batches processed.
        
        Args:
            save_path: Directory to save submission file
            rank: Current process rank
            world_size: Total number of processes
            group: Process group for distributed communication
            
        Returns:
            Dictionary of pass@k metrics (only on rank 0)
        """
        # Gather predictions to rank 0 for voting
        global_hmap_preds = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object((self._local_hmap, self._local_preds), global_hmap_preds, dst=0, group=group)
        
        # Rank 0 logic
        if rank != 0:
            return

        submission = {}
        # Puzzle-averaged correctness (as before)
        correct = [0.0 for _ in range(len(self.pass_Ks))]
        # Also track pooled per-example totals to enable robust CI downstream
        total_examples = 0
        total_correct_per_k = [0 for _ in range(len(self.pass_Ks))]

        for name, puzzle in self.test_puzzles.items():
            # Process test examples in this puzzle
            submission[name] = []
            num_test_correct = [0 for _ in range(len(self.pass_Ks))]
            for pair in puzzle["test"]:
                input_hash = grid_hash(arc_grid_to_np(pair["input"]))
                label_hash = grid_hash(arc_grid_to_np(pair["output"]))
                
                p_map = {}
                for hmap, preds in global_hmap_preds:  # type: ignore
                    for h, q in preds.get(name, {}).get(input_hash, []):
                        p_map.setdefault(h, [0, 0])
                        p_map[h][0] += 1
                        p_map[h][1] += q
                        
                if not len(p_map):
                    print(f"Puzzle {name} has no predictions.")
                    continue

                for h, stats in p_map.items():
                    stats[1] /= stats[0]
                    
                p_map = sorted(p_map.items(), key=lambda kv: kv[1], reverse=True)

                # vote for different Ks
                for i, k in enumerate(self.pass_Ks):
                    ok = False
                    for h, stats in p_map[:k]:
                        ok |= h == label_hash
                        
                    num_test_correct[i] += ok
                    
                # Query grids
                pred_grids = []
                for h, stats in p_map[:self.submission_K]:
                    for hmap, preds in global_hmap_preds:  # type: ignore
                        if h in hmap:
                            pred_grids.append(hmap[h])
                            break
                        
                # Pad to K
                while len(pred_grids) < self.submission_K:
                    pred_grids.append(pred_grids[0])
                
                submission[name].append({f"attempt_{i + 1}": grid.tolist() for i, grid in enumerate(pred_grids)})

            # Total correctness (puzzle-averaged)
            for i in range(len(self.pass_Ks)):
                correct[i] += num_test_correct[i] / len(puzzle["test"])

            # Accumulate pooled per-example statistics
            total_examples += len(puzzle["test"])
            for i in range(len(self.pass_Ks)):
                total_correct_per_k[i] += num_test_correct[i]

        # Save submission
        if save_path is not None:
            with open(os.path.join(save_path, "submission.json"), "w") as f:
                json.dump(submission, f)

        # Final results
        all_results = {f"ARC/pass@{k}": correct[i] / len(self.test_puzzles) for i, k in enumerate(self.pass_Ks)}
        # Add per-example pooled pass@k (useful for confidence intervals with clear N)
        if total_examples > 0:
            all_results["ARC/example_N"] = float(total_examples)
            for i, k in enumerate(self.pass_Ks):
                all_results[f"ARC/example_pass@{k}"] = float(total_correct_per_k[i]) / float(total_examples)

        return all_results
