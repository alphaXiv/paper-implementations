import os
import logging
import torch
import hydra
import wandb
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

# vLLM import
try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

from justrl_reproduction.src.grpo_trainer import JustRLGRPOTrainer
from justrl_reproduction.verifiers import MathRuleVerifier
from justrl_reproduction.src.utils import compute_group_advantages

logger = logging.getLogger(__name__)

class ParquetDataset(Dataset):
    def __init__(self, file_path: str):
        self.data = pd.read_parquet(file_path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data.iloc[idx].to_dict()

def collate_fn(batch):
    return batch

@hydra.main(version_base=None, config_path="../config", config_name="justrl_deepseek_1.5b")
def main(cfg: DictConfig):
    # 1. Setup
    set_seed(42)
    wandb.init(
        project="justrl-reproduction",
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.get("use_wandb", True) else "disabled"
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 2. Load Data
    data_path = os.path.join(hydra.utils.get_original_cwd(), cfg.data.train_file)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found at {data_path}. Run prepare_data.sh first.")
    
    dataset = ParquetDataset(data_path)
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.data.global_batch_size // cfg.trainer.n_rollouts, # This is tricky. 
        # Global batch size usually means number of prompts or number of optimization samples.
        # Paper: "Global Batch Size: 256". "Micro Batch Size: 1".
        # If Global Batch Size is 256 prompts, and we do G=8 rollouts, that's 2048 sequences.
        # Let's assume Global Batch Size = 256 prompts.
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # 3. Initialize Verifier
    verifier = MathRuleVerifier()
    
    # 4. Initialize Model & Tokenizer (Training Model)
    logger.info(f"Loading model: {cfg.model.name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype=torch.bfloat16 if cfg.model.bf16 else torch.float32,
        attn_implementation="flash_attention_2" if cfg.model.flash_attn else "eager",
        trust_remote_code=True,
        device_map="auto"
    )
    model.train()
    
    # 5. Initialize Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.trainer.lr)
    
    # 6. Initialize Trainer Helper
    grpo_trainer = JustRLGRPOTrainer(model, optimizer, OmegaConf.to_container(cfg, resolve=True))
    
    # 7. Initialize vLLM (Inference Engine)
    # Note: In a real single-GPU setup, running vLLM and Training Model simultaneously is hard due to VRAM.
    # We assume this script runs on a node with sufficient VRAM or multiple GPUs where vLLM and PyTorch can coexist,
    # or that vLLM is used carefully.
    # For reproduction logic, we instantiate it.
    if LLM is not None:
        logger.info("Initializing vLLM...")
        # We might need to limit vLLM GPU memory utilization to leave room for training
        llm = LLM(
            model=cfg.model.name,
            trust_remote_code=True,
            gpu_memory_utilization=0.4, # Reserve memory for training model
            max_model_len=cfg.data.max_prompt_length + cfg.data.max_response_length
        )
        sampling_params = SamplingParams(
            temperature=cfg.trainer.temperature,
            max_tokens=cfg.data.max_response_length,
            n=cfg.trainer.n_rollouts
        )
    else:
        logger.warning("vLLM not installed. Rollouts will be simulated or fail.")
        llm = None

    # 8. Training Loop
    global_step = 0
    
    # We iterate through the dataloader. 
    # Each item is a prompt.
    # We need to accumulate gradients to reach global_batch_size.
    # Let's assume the dataloader provides the micro-batches.
    
    # Actually, let's simplify: 
    # We process `batch_size` prompts at a time.
    # For each prompt, we generate `n_rollouts`.
    # Total sequences = batch_size * n_rollouts.
    
    # Config says: Global Batch Size 256.
    # If we process 1 prompt per GPU (Micro Batch 1), we need accumulation.
    # For this script, let's implement a loop that processes `micro_batch_size` prompts,
    # generates rollouts, computes loss, and accumulates.
    
    micro_batch_size = cfg.trainer.ppo_micro_batch_size # Use this as the prompt batch size per step
    if micro_batch_size is None:
        micro_batch_size = 1
        
    train_loader = DataLoader(dataset, batch_size=micro_batch_size, shuffle=True, collate_fn=collate_fn)
    
    logger.info("Starting training...")
    
    for epoch in range(int(cfg.trainer.epochs)):
        for batch_idx, batch_prompts_data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            
            prompts = [item['prompt'] for item in batch_prompts_data]
            ground_truths = [item['answer'] for item in batch_prompts_data] # Assuming 'answer' key
            
            # --- Step 1: Rollout ---
            if llm:
                # Generate outputs
                outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
                
                # Collect generations
                all_generations = []
                all_prompts_text = []
                all_ground_truths = []
                
                for i, output in enumerate(outputs):
                    prompt = prompts[i]
                    gt = ground_truths[i]
                    for generated_seq in output.outputs:
                        all_prompts_text.append(prompt)
                        all_generations.append(generated_seq.text)
                        all_ground_truths.append(gt)
            else:
                # Fallback or Mock
                logger.warning("Skipping generation (No vLLM).")
                continue
                
            # --- Step 2: Reward Computation ---
            rewards = []
            for gen, gt in zip(all_generations, all_ground_truths):
                reward = verifier.verify(gen, gt)
                rewards.append(reward)
            
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
            
            # Reshape rewards to [Batch, Group]
            # batch_size = len(prompts)
            # group_size = n_rollouts
            # rewards_tensor = rewards_tensor.view(len(prompts), cfg.trainer.n_rollouts)
            
            # Compute Advantages
            # We need to group them by prompt
            n_rollouts = cfg.trainer.n_rollouts
            rewards_grouped = rewards_tensor.view(-1, n_rollouts)
            advantages_grouped = compute_group_advantages(rewards_grouped)
            advantages = advantages_grouped.view(-1) # Flatten back
            
            # --- Step 3: Prepare Training Batch ---
            # We need to tokenize (Prompt + Generation)
            # And create labels/masks
            
            full_sequences = [p + g for p, g in zip(all_prompts_text, all_generations)]
            
            inputs = tokenizer(
                full_sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.data.max_prompt_length + cfg.data.max_response_length
            ).to(device)
            
            # Create Loss Mask (Mask out prompt tokens)
            # We need to know length of prompt tokens
            prompt_inputs = tokenizer(
                all_prompts_text,
                return_tensors="pt",
                padding=True, # Note: Padding might mess up length calculation if not careful
                truncation=True,
                max_length=cfg.data.max_prompt_length
            )
            
            # A simple way to mask: 
            # Find where the generation starts. 
            # Since we just concatenated strings, we can re-tokenize prompts to find their lengths.
            # But tokenization isn't always additive (subword merges).
            # For exactness, usually we tokenize prompt, tokenize generation, and concat token ids.
            # Let's do that for correctness.
            
            input_ids_list = []
            attention_mask_list = []
            loss_mask_list = []
            
            for prompt, gen in zip(all_prompts_text, all_generations):
                p_ids = tokenizer.encode(prompt, add_special_tokens=False)
                g_ids = tokenizer.encode(gen, add_special_tokens=False)
                
                # Combine
                full_ids = p_ids + g_ids
                # Truncate
                max_len = cfg.data.max_prompt_length + cfg.data.max_response_length
                if len(full_ids) > max_len:
                    full_ids = full_ids[:max_len]
                
                # Create masks
                # 1 for attention, 1 for loss (on generation only)
                att_mask = [1] * len(full_ids)
                # Loss mask: 0 for prompt, 1 for generation
                # Note: The model predicts the NEXT token. 
                # So label at index i is input_ids[i+1].
                # We usually mask the positions corresponding to prompt tokens.
                # If we use standard CausalLM loss, we set labels to -100 for prompt.
                
                l_mask = [0] * len(p_ids) + [1] * (len(full_ids) - len(p_ids))
                
                input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
                attention_mask_list.append(torch.tensor(att_mask, dtype=torch.long))
                loss_mask_list.append(torch.tensor(l_mask, dtype=torch.long))
                
            # Pad batch
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
            attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0).to(device)
            loss_mask = torch.nn.utils.rnn.pad_sequence(loss_mask_list, batch_first=True, padding_value=0).to(device)
            
            # --- Step 4: Compute Old Log Probs ---
            # We need the log probs of the generated tokens under the current policy (before update).
            # Since we just generated them, the current model IS the policy.
            # We run a forward pass in no_grad mode.
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits # [B, Seq, Vocab]
                
                # Shift logits and labels
                # Logits at t predict t+1
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                shift_loss_mask = loss_mask[..., 1:].contiguous()
                
                # Compute log probs
                # gather log probs of the actual tokens
                log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                # gather
                old_log_probs = torch.gather(log_probs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
                
                # Apply mask (zero out prompt log probs)
                old_log_probs = old_log_probs * shift_loss_mask
            
            # --- Step 5: Train Step ---
            # Now we call the trainer to do the update
            # The trainer expects:
            # batch = {
            #   'input_ids': ...,
            #   'attention_mask': ...,
            #   'loss_mask': ..., (shifted or unshifted? Trainer usually handles shifting or expects aligned)
            #   'old_log_probs': ...,
            #   'advantages': ...
            # }
            
            # Let's check grpo_trainer.py implementation.
            # It calculates new log probs and compares with old.
            # It expects 'loss_mask' to be aligned with logits/labels.
            
            # Advantages need to be expanded to sequence length?
            # GRPO: Advantage is per-sample (per rollout).
            # So A_i is scalar for the whole sequence i.
            # We broadcast it.
            
            # advantages is [Batch_Size * Group_Size]
            # We need [Batch_Size * Group_Size, Seq_Len]
            # Broadcast
            advantages_expanded = advantages.unsqueeze(1).expand_as(old_log_probs)
            
            batch = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'loss_mask': shift_loss_mask, # Pass shifted mask
                'old_log_probs': old_log_probs,
                'advantages': advantages_expanded
            }
            
            metrics = grpo_trainer.train_step(batch)
            
            # --- Logging ---
            # Calculate mean response length
            avg_len = np.mean([len(g) for g in all_generations]) # Char length, approx
            # Better: token length
            avg_token_len = np.mean([len(ids) - len(p_ids) for ids, p_ids in zip(input_ids_list, [tokenizer.encode(p, add_special_tokens=False) for p in all_prompts_text])])
            
            metrics['response_length'] = avg_token_len
            metrics['avg_reward'] = rewards_tensor.mean().item()
            
            wandb.log(metrics, step=global_step)
            global_step += 1
            
            # Save checkpoint
            if global_step % 500 == 0:
                save_path = os.path.join(os.getcwd(), f"checkpoints/step_{global_step}")
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                logger.info(f"Saved checkpoint to {save_path}")

    # Final Save
    final_path = os.path.join(os.getcwd(), "final_model")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info("Training complete.")

if __name__ == "__main__":
    main()
