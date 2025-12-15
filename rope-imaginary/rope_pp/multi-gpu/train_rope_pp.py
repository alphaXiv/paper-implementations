import os
import sys
import json
import random
from datetime import datetime

import torch
import numpy as np

import datasets
import wandb

import deepspeed

from transformers import AutoTokenizer

from llama_variants.configuration_llama import LlamaConfig
from llama_variants.modeling_llama_rope_pp import LlamaForCausalLM

from utils.dataset_utils import StreamingTrainingJsonlZSD, StreamingTrainingHuggingFace, EvaluatingDataset
from utils.training_engine import train_with_accelerate

root = os.getcwd()
tokenizer_path = 'meta-llama/Meta-Llama-3-8B'

cache_dir = ''  # set a cache_dir

train_dataset_hf_id = 'mlfoundations/dclm-baseline-1.0'  # Hugging Face dataset ID
train_dataset_label = 'text'

valid_dataset_hf_id = 'wikitext'  # Hugging Face dataset ID
valid_dataset_name = 'wikitext-2-raw-v1'  # Subset name
valid_dataset_split = 'validation'
valid_dataset_abbr = 'wikitext'
valid_dataset_label = 'text'

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.set_default_dtype(torch.bfloat16)

import argparse

parser = argparse.ArgumentParser(description='define fp config')
parser.add_argument('--imag', action='store_true', default=False)
parser.add_argument('--imag_mode', choices=['imag1', 'imag2', ], default='imag1')

# imag1 stands for rope_pp_eh, and imag2 stands for rope_pp_ec, 

parser.add_argument('--config_abbr', type=str, default='376m')
parser.add_argument('--exp_name', type=str, default='rope_pp-376m')
parser.add_argument('--load_ckpt', type=int, default=None, help='Checkpoint to load from')

parser.add_argument('--max_length', type=int, default=4096, help='Maximum sequence length')
parser.add_argument('--rope_theta', type=float, default=None, help='RoPE theta base (default: model config default)')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size per device')
parser.add_argument('--max_steps', type=int, default=100000, help='Maximum training steps')
parser.add_argument('--warmup_steps', type=int, default=500, help='Warmup steps (set to 0 for decay phase)')

parser.add_argument('--local_rank', type=int, default=-1)

args = parser.parse_args()

rope_config = {
    'imag': args.imag, 
    'imag_mode': args.imag_mode, 
}

config_abbr = args.config_abbr
config_path = f'{root}/configs/rope-{config_abbr}-config.json'

exp_name = args.exp_name
load_ckpt = args.load_ckpt

gradient_accumulation_steps = 1

batch_size = args.batch_size
max_length = args.max_length
valid_size = 4096  # Reduced to avoid OOM during evaluation

max_steps = args.max_steps
eval_steps = 500
warmup_steps = args.warmup_steps

save_steps = 10000
steps_to_save = [100, max_steps]

# ref: https://www.deepspeed.ai/docs/config-json/

ds_config = {
    "bf16": {
        "enabled": True
    },
    "zero_allow_untested_optimizer": True,
    "zero_force_ds_cpu_optimizer": False,
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e7,
        "stage3_max_live_parameters": 1e7,
        "stage3_max_reuse_distance": 1e7,
        "stage3_gather_16bit_weights_on_model_save": True,
        "stage3_prefetch_bucket_size": 5e7,
        "stage3_param_persistence_threshold": 1e5,
    },
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "steps_per_print": 100,
    "train_batch_size": batch_size * gradient_accumulation_steps,
    "wall_clock_breakdown": False, 
}

# ref: https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/zero/partition_parameters.py#L603

with deepspeed.zero.Init(dtype=torch.bfloat16, config_dict_or_path=ds_config):

    config = LlamaConfig.from_pretrained(config_path)
    config.gradient_checkpointing = True
    config.use_cache = False
    config._attn_implementation = "flash_attention_2"
    config.torch_dtype = torch.bfloat16
    if args.rope_theta is not None:
        config.rope_theta = args.rope_theta
    config.rope_config = rope_config
    config.ignore_index = config.eos_token_id

    if load_ckpt is not None:
        load_path = f'{root}/checkpoints/{exp_name}/checkpoint-{load_ckpt}'
        model = LlamaForCausalLM.from_pretrained(load_path, config=config)
        if rank == 0:
            print(f'Model loaded from checkpoint: {load_path}', '\n')
    else:
        model = LlamaForCausalLM(config=config)
        if rank == 0:
            print('Model initialized from scratch', '\n')

# Get rank for logging (Accelerate will handle distributed init)
rank = 0
if torch.distributed.is_initialized():
    rank = torch.distributed.get_rank()

# Training configuration
training_config = {
    'output_dir': f'{root}/checkpoints/{exp_name}',
    'max_steps': max_steps,
    'batch_size': batch_size,
    'gradient_accumulation_steps': gradient_accumulation_steps,
    'learning_rate': 5e-4,
    'weight_decay': 0.1,
    'adam_beta1': 0.95,
    'adam_beta2': 0.99,
    'warmup_steps': warmup_steps,
    'max_grad_norm': 1.0,
    'eval_steps': eval_steps,
    'save_steps': save_steps,
    'steps_to_save': steps_to_save,
    'max_length': max_length,
    'valid_dataset_abbr': valid_dataset_abbr,
    'logging_steps': 1,
    'resume_from_checkpoint': None,
}

if rank == 0:
    print(f'{config = }', '\n')
    print('ds_config = ', json.dumps(ds_config, indent=2), '\n')
    print('training_config = ', json.dumps(training_config, indent=2), '\n')

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# Load validation dataset from Hugging Face Hub
if rank == 0:
    print(f'Loading validation dataset from Hugging Face: {valid_dataset_hf_id}/{valid_dataset_name}', '\n')

valid_dataset = datasets.load_dataset(valid_dataset_hf_id, valid_dataset_name, split=valid_dataset_split, 
                                      cache_dir=cache_dir)
# wikitext has a lot of empty lines -> causes NaNs                                      
valid_dataset = valid_dataset.filter(lambda x: len(x[valid_dataset_label].strip()) > 50)
valid_dataset = valid_dataset.select(range(min(valid_size, len(valid_dataset))))

if rank == 0:
    print(valid_dataset, '\n')

# Load training dataset from Hugging Face Hub
if rank == 0:
    print(f'Loading training dataset from Hugging Face: {train_dataset_hf_id}', '\n')

train_dataset = StreamingTrainingHuggingFace(
    dataset_id=train_dataset_hf_id, 
    tokenizer=tokenizer, 
    label_name=train_dataset_label, 
    train_length=max_length, 
    num_data=max_steps * batch_size * gradient_accumulation_steps, 
    seed=seed,
    split='train',
    streaming=True,
    cache_dir=cache_dir
)

if rank == 0:
    print('dataset is ready !', '\n')

# Initialize WandB (only on rank 0)
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_PROJECT"] = "rope_pp"
os.environ["WANDB_DIR"] = f"{root}/wandb"

if rank == 0:
    wandb.init(
        project="rope_pp",
        name=f'{exp_name}-{datetime.now().strftime("%Y%m%d-%H%M%S")}',
        config={**training_config, **ds_config},
        dir=f'{root}/wandb',
    )
    print('checkpoints and model will be saved in', training_config['output_dir'], '\n')

# Train!
train_with_accelerate(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    config=training_config,
    deepspeed_config=ds_config,
)

if rank == 0:
    wandb.finish()
