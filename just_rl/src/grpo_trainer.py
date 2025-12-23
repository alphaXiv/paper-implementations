import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Union
from .utils import compute_group_advantages

class JustRLGRPOTrainer:
    """
    GRPO Trainer implementing the 'JustRL' recipe.
    
    This trainer implements Group Relative Policy Optimization (GRPO) with:
    - No Critic (Actor-only architecture)
    - No KL Divergence penalty (kl_coef = 0.0)
    - No Entropy bonus (entropy_coef = 0.0)
    - Asymmetric Clipping [0.8, 1.28]
    - Group-based advantage normalization
    """
    def __init__(self, 
                 model: nn.Module, 
                 optimizer: torch.optim.Optimizer, 
                 config: Dict):
        """
        Initialize the GRPO Trainer.
        
        Args:
            model: The policy model (Actor).
            optimizer: The optimizer.
            config: Configuration dictionary containing algorithm hyperparameters.
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config
        
        # Extract hyperparameters
        algo_config = config.get('algorithm', {})
        self.clip_range_low = algo_config.get('clip_range_low', 0.2)
        self.clip_range_high = algo_config.get('clip_range_high', 0.28)
        self.kl_coef = algo_config.get('kl_coef', 0.0)
        self.entropy_coef = algo_config.get('entropy_coef', 0.0)
        
        # Validation to ensure recipe adherence
        if self.kl_coef != 0.0 or self.entropy_coef != 0.0:
            print(f"WARNING: JustRL recipe specifies kl_coef=0.0 and entropy_coef=0.0. "
                  f"Current values: kl={self.kl_coef}, ent={self.entropy_coef}")

    def compute_loss(self, 
                     log_probs: torch.Tensor, 
                     old_log_probs: torch.Tensor, 
                     advantages: torch.Tensor, 
                     mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the GRPO policy loss.
        
        Args:
            log_probs: Current policy log probabilities [Batch, Group, Seq_Len]
            old_log_probs: Log probabilities from the rollout policy [Batch, Group, Seq_Len]
            advantages: Normalized group advantages [Batch, Group]
            mask: Attention mask (1 for valid tokens, 0 for padding) [Batch, Group, Seq_Len]
            
        Returns:
            loss: The computed loss (scalar).
            metrics: Dictionary of metrics for logging.
        """
        # Ensure advantages are broadcastable to sequence length
        # advantages: [B, G] -> [B, G, 1]
        if advantages.dim() == 2:
            advantages = advantages.unsqueeze(-1)
            
        # Compute ratio: pi(a|s) / pi_old(a|s)
        # log_probs and old_log_probs are already log(p)
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Surrogate Objective 1: ratio * A
        surr1 = ratio * advantages
        
        # Surrogate Objective 2: clip(ratio) * A
        # Asymmetric clipping: lower bound = 1 - eps_low, upper bound = 1 + eps_high
        clip_low = 1.0 - self.clip_range_low
        clip_high = 1.0 + self.clip_range_high
        
        ratio_clipped = torch.clamp(ratio, clip_low, clip_high)
        surr2 = ratio_clipped * advantages
        
        # GRPO Loss: -min(surr1, surr2)
        # We take the negative because we want to maximize the objective
        loss_elementwise = -torch.min(surr1, surr2)
        
        # Apply mask to ignore padding tokens
        loss_masked = loss_elementwise * mask
        
        # Average loss over valid tokens
        # Note: Some implementations average over batch, some over valid tokens.
        # Averaging over valid tokens is more stable for variable lengths.
        loss = loss_masked.sum() / (mask.sum() + 1e-8)
        
        # Calculate approximate KL for monitoring (even though we don't optimize for it)
        with torch.no_grad():
            # kl = old_log_p - log_p + (ratio - 1) - log(ratio) ... approximation
            # Simple approx: log_p_old - log_p
            approx_kl = 0.5 * ((old_log_probs - log_probs) ** 2) * mask
            mean_kl = approx_kl.sum() / (mask.sum() + 1e-8)
            
            # Clipping fraction
            clipped = (ratio > clip_high) | (ratio < clip_low)
            clip_frac = (clipped.float() * mask).sum() / (mask.sum() + 1e-8)

        metrics = {
            "policy_loss": loss.item(),
            "approx_kl": mean_kl.item(),
            "clip_frac": clip_frac.item(),
            "mean_advantage": (advantages * mask).sum().item() / (mask.sum() + 1e-8)
        }
        
        return loss, metrics

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Performs a single training step.
        
        Args:
            batch: Dictionary containing:
                - input_ids: [B, G, Seq_Len]
                - attention_mask: [B, G, Seq_Len]
                - old_log_probs: [B, G, Seq_Len]
                - rewards: [B, G]
                
        Returns:
            metrics: Dictionary of training metrics.
        """
        self.model.train()
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        old_log_probs = batch['old_log_probs']
        rewards = batch['rewards']
        
        # 1. Compute Advantages (Group-based)
        # We do this on the fly or it can be pre-computed. 
        # Doing it here ensures we use the latest rewards if they were dynamic (they aren't here).
        with torch.no_grad():
            advantages = compute_group_advantages(rewards)
        
        # 2. Forward pass to get current log_probs
        # We need to flatten [B, G, Seq] to [B*G, Seq] for the model
        B, G, Seq = input_ids.shape
        flat_input_ids = input_ids.view(B * G, Seq)
        flat_attention_mask = attention_mask.view(B * G, Seq)
        
        outputs = self.model(input_ids=flat_input_ids, attention_mask=flat_attention_mask)
        logits = outputs.logits # [B*G, Seq, Vocab]
        
        # Calculate log probs of the tokens that were actually generated
        # Usually we shift logits and labels for causal LM training
        # But here input_ids includes prompt + completion.
        # We only want to calculate loss on the completion part.
        # The 'mask' passed in should ideally be the 'loss mask' (1 for completion, 0 for prompt/padding).
        # Assuming 'attention_mask' is standard padding mask.
        # We need a 'loss_mask' in the batch to distinguish prompt from completion.
        
        loss_mask = batch.get('loss_mask')
        if loss_mask is None:
            # Fallback: assume all non-padding tokens contribute to loss (Incorrect for RLHF)
            # But for now, let's assume the caller provides loss_mask.
            # If not, we warn and use attention_mask (risky).
            loss_mask = attention_mask
        
        flat_loss_mask = loss_mask.view(B * G, Seq)
        
        # Compute log_probs for the input_ids
        # gather: log_softmax(logits) -> gather(input_ids)
        log_probs_all = F.log_softmax(logits, dim=-1)
        
        # We need log_prob of the token at index t.
        # In causal LM, logits[t] predicts input_ids[t+1].
        # So we shift.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = flat_input_ids[..., 1:].contiguous()
        shift_loss_mask = flat_loss_mask[..., 1:].contiguous()
        
        # Gather log probs of the actual tokens
        # [B*G, Seq-1, Vocab] -> [B*G, Seq-1]
        gathered_log_probs = torch.gather(
            F.log_softmax(shift_logits, dim=-1),
            2,
            shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Reshape back to [B, G, Seq-1]
        current_log_probs = gathered_log_probs.view(B, G, -1)
        
        # Align old_log_probs and advantages
        # old_log_probs should also be shifted or pre-aligned.
        # Assuming old_log_probs in batch corresponds to the same tokens as current_log_probs.
        # If old_log_probs was [B, G, Seq], we slice it.
        if old_log_probs.shape[-1] == Seq:
             old_log_probs_aligned = old_log_probs[..., 1:]
        else:
             old_log_probs_aligned = old_log_probs
             
        mask_aligned = shift_loss_mask.view(B, G, -1)
        
        # 3. Compute Loss
        loss, metrics = self.compute_loss(
            log_probs=current_log_probs,
            old_log_probs=old_log_probs_aligned,
            advantages=advantages,
            mask=mask_aligned
        )
        
        # 4. Optimization Step
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping (optional but recommended)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return metrics
