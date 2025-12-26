# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence

import verl.utils.torch_functional as verl_F
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


def update_optimizer_training_steps(config, total_training_steps: int) -> None:
    from omegaconf import OmegaConf, open_dict
    
    try:
        OmegaConf.set_struct(config, True)
        with open_dict(config):
            if OmegaConf.select(config, "actor_rollout_ref.actor.optim"):
                config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            if OmegaConf.select(config, "critic.optim"):
                config.critic.optim.total_training_steps = total_training_steps
    except Exception as e:
        print(f"Warning: Could not set total_training_steps in config. Error: {e}")


def extract_and_pad_by_mask(tensor: torch.Tensor, mask: torch.Tensor, padding_value=0.0):
    """
    Extract elements from tensor according to mask and pad extracted sequences to same length.
    
    Args:
        tensor: Tensor of shape (batch_size, seq_len, ...) or (batch_size, seq_len)
        mask: Binary mask of shape (batch_size, seq_len) where 1 indicates positions to keep
        padding_value: Value to use for padding
        
    Returns:
        tuple containing:
            - padded_tensor: Tensor with masked elements extracted and padded to same length
            - lengths: Original lengths of each sequence before padding
            - indices: List of indices where mask=1 for each item in batch
    """
    batch_size = tensor.shape[0]
    device = tensor.device
    
    # Store the extracted sequences and their original indices
    extracted_tensors = []
    original_indices = []
    lengths = []
    
    # For each item in the batch
    for i in range(batch_size):
        # Find indices where mask is 1
        indices = torch.where(mask[i] > 0)[0]
        
        if len(indices) == 0:
            # Handle empty mask case by creating a dummy tensor with correct shape
            extracted = torch.zeros(0, device=device)
        else:
            # Extract the values at those indices
            extracted = tensor[i, indices]
        
        extracted_tensors.append(extracted)
        original_indices.append(indices)
        lengths.append(len(indices))
    
    # Pad sequences to the same length
    if any(lengths):  # Check if we have at least one non-empty sequence
        padded_tensor = pad_sequence(extracted_tensors, batch_first=True, padding_value=padding_value)
    else:
        padded_tensor = torch.zeros((batch_size, 0), device=device)
    
    # Convert lengths to tensor
    lengths = torch.tensor(lengths, device=device)
    
    return padded_tensor, lengths, original_indices


def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    action_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        action_mask: `(torch.Tensor)`
            shape: (bs, response_length). Action mask where model-generated tokens have mask 1, 
            and tokens from external interactions (user responses, tool outputs) have mask 0.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []

        extracted_rewards, lengths, indices = extract_and_pad_by_mask(token_level_rewards, action_mask)
        extracted_values, _, _ = extract_and_pad_by_mask(values, action_mask)

        max_length = max(lengths)

        for t in reversed(range(max_length)):
            nextvalues = extracted_values[:, t + 1] if t < max_length - 1 else 0.0
            delta = extracted_rewards[:, t] + gamma * nextvalues - extracted_values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        extracted_advantages = torch.stack(advantages_reversed[::-1], dim=1)

        advantages = torch.zeros_like(token_level_rewards)
        for i, length in enumerate(lengths):
            advantages[i, indices[i]] = extracted_advantages[i][:length]

        returns = advantages + values
        # Agent-R1 Unique Contribution: Advantage Masks
        # In tool-using agents, not all tokens contribute equally to the reward.
        # Only tool calls and their immediate context should affect learning.
        # Action masks zero out advantages for non-action tokens, preventing
        # the model from being penalized for "thinking out loud".
        advantages = verl_F.masked_whiten(advantages, action_mask)
    return advantages, returns

# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    action_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        action_mask: `(torch.Tensor)`
            shape: (bs, response_length). Action mask where model-generated tokens have mask 1, 
            and tokens from external interactions (user responses, tool outputs) have mask 0.
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    # Sum rewards only for action tokens
    scores = (token_level_rewards * action_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1).tile([1, response_length]) * action_mask

    return scores, scores



def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.
    Args:
        loss_mat: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_agg_mode: (str) choices: "token-mean" /
                                      "seq-mean-token-sum" /
                                      "seq-mean-token-mean" /
                                      "seq-mean-token-sum-norm" /
            "token-mean" is the default behavior
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    action_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode="token-mean",
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122
    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        action_mask: `(torch.Tensor)`
            shape: (bs, response_length). Action mask where model-generated tokens have mask 1, 
            and tokens from external interactions (user responses, tool outputs) have mask 0.
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        cliprange_low: (float)
            The lower clip range used in PPO.
        cliprange_high: (float)
            The higher clip range used in PPO.
        clip_ratio_c: (float) default: 3.0
            The lower bound of the ratio for dual-clip PPO, See https://arxiv.org/pdf/1912.09729
        loss_agg_mode: (str) choices: "token-mean" /
                                      "seq-mean-token-sum" /
                                      "seq-mean-token-mean" /
                                      "seq-mean-token-sum-norm" /
            "token-mean" is the default behavior

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            the fraction of policy gradient loss being clipped
        ppo_kl: (float)
            the estimated KL divergence between the latest updating policy and the old sampling policy
        pg_clipfrac_lower: (float)
            the fraction of policy gradient loss being clipped when the advantage is negative
    """
    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, action_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), action_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), action_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=action_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_entropy_loss(logits, action_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        action_mask: `(torch.Tensor)`
            shape: (bs, response_length). Action mask where model-generated tokens have mask 1, 
            and tokens from external interactions (user responses, tool outputs) have mask 0.

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=action_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, state_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)
        state_mask: `(torch.Tensor)`
            shape: (bs, response_length). State mask where model-generated tokens have mask 1, 
            and tokens from external interactions (user responses, tool outputs) have mask 0.

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), state_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), state_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == "low_var_kl":
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def apply_kl_penalty(data, kl_ctrl: AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to token-level scores to compute token-level rewards.
    
    Args:
        data: DataProto containing batch data
        kl_ctrl: KL controller for managing KL coefficient
        kl_penalty: Type of KL penalty to apply
        multi_turn: Whether using multi-turn mode
        
    Returns:
        tuple: (data with updated token_level_rewards, metrics dict)
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    if "action_mask" in data.batch:
        action_mask = data.batch["action_mask"]
    else:
        action_mask = response_mask

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * action_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = verl_F.masked_mean(kld, mask=action_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data):
    """Compute response mask from attention mask.
    
    Args:
        data: DataProto containing batch data
        
    Returns:
        torch.Tensor: Response mask of shape (batch_size, response_length)
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def balance_batch_by_seqlen(batch, world_size: int, metrics: dict, logging_prefix: str = "global_seqlen"):
    attention_mask = batch.batch["attention_mask"]
    batch_size = attention_mask.shape[0]
    
    global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1).tolist()
    
    global_partition_lst = get_seqlen_balanced_partitions(
        global_seqlen_lst, 
        k_partitions=world_size, 
        equal_size=True
    )
    
    global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
    batch.reorder(global_idx)
    
    global_balance_stats = log_seqlen_unbalance(
        seqlen_list=global_seqlen_lst, 
        partitions=global_partition_lst, 
        prefix=logging_prefix
    )
    metrics.update(global_balance_stats)


def create_action_mask(batch, metrics):
    """
    Create an action mask for advantage calculation.
    
    This function identifies which tokens are generated by the model (action_mask=1)
    and which tokens are from external interactions (action_mask=0) like user messages and tool responses.
    External interaction tokens should be excluded from policy gradient updates.
    
    In multi-turn conversations, the entire user message including tags like <|im_start|>user
    up to the <|im_start|>assistant and the starting <think> are considered external interactions.
    
    Args:
        batch: DataProto containing batch data
        metrics: Dictionary to update with action mask metrics
        
    Returns:
        tuple: (batch with action_mask, updated metrics dict)
    """
    response_length = batch.batch["responses"].shape[-1]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    if "action_mask" not in batch.batch.keys():
        action_mask = torch.ones_like(response_mask)
        print("[WARNING] No action mask found in batch, using all ones")
    else:
        action_mask = batch.batch["action_mask"]
    
    # Log what percentage of tokens are actions vs external interactions
    action_ratio = action_mask.sum().item() / (response_mask.sum().item() + 1e-8)
    metrics["action/ratio"] = action_ratio
    metrics["action/length/max"] = action_mask.sum(dim=-1).max().item()
    metrics["action/length/min"] = action_mask.sum(dim=-1).min().item()
    metrics["action/length/mean"] = action_mask.sum(dim=-1).float().mean().item()
    
    return batch, metrics


def compute_advantage(data, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True):
    """Compute advantages and returns using the specified estimator.
    
    Args:
        data: DataProto containing batch data
        adv_estimator: Type of advantage estimator (GAE or GRPO)
        gamma: Discount factor
        lam: Lambda parameter for GAE
        num_repeat: Number of repeats for GRPO
        multi_turn: Whether using multi-turn mode
        norm_adv_by_std_in_grpo: Whether to normalize advantages by std in GRPO
        
    Returns:
        DataProto: Updated data with advantages and returns
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    
    if adv_estimator == "gae":
        advantages, returns = compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            action_mask=data.batch["action_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == "grpo":
        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            action_mask=data.batch["action_mask"],
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data
