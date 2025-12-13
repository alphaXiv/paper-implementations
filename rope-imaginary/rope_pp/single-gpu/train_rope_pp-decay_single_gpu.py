import os
import sys
import json
import random
import logging
from datetime import datetime

# Suppress transformers default logging output to console
logging.getLogger("transformers.trainer").setLevel(logging.WARNING)

# Add the parent directory to sys.path to allow imports from sibling directories
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

import datasets

from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments

from llama_variants.configuration_llama import LlamaConfig
from llama_variants.modeling_llama_rope_pp import LlamaForCausalLM

from utils.dataset_utils import StreamingTrainingJsonlZSD, StreamingTrainingHuggingFace, EvaluatingDataset
from utils.callback_utils import CheckpointingCallback, CustomLoggingCallback
from utils.trainer_utils import TrainerWithDatasetCheckpointing

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
parser.add_argument('--save_abbr', type=str, default='376m')
parser.add_argument('--load_ckpt', type=int)
parser.add_argument('--decay_step', type=int)

args = parser.parse_args()

rope_config = {
    'imag': args.imag, 
    'imag_mode': args.imag_mode, 
}

config_abbr = args.config_abbr
config_path = f'{root}/configs/rope-{config_abbr}-config.json'

save_abbr = args.save_abbr
load_ckpt = args.load_ckpt
load_path = f'{root}/checkpoints/{save_abbr}/checkpoint-{load_ckpt}'
save_path = f'{root}/checkpoints/{save_abbr}-ckpt{load_ckpt}-decay'

# MODIFIED FOR SINGLE GPU - Optimized for 40GB A100
batch_size = 2  # Very small batch to fit in memory
gradient_accumulation_steps = 64  # Simulate effective batch size of 128

max_length = 4096  # If still OOM, try 2048 or 3072
valid_size = 4096  # Reduced validation size

max_steps = args.decay_step
eval_steps = 500
warmup_steps = 0

save_steps = 2000
steps_to_save = [2, max_steps]

# Load config and create model (NO DEEPSPEED)
config = LlamaConfig.from_pretrained(config_path)
config.gradient_checkpointing = True  # CRITICAL for memory
config.use_cache = False  # Required for gradient checkpointing
config._attn_implementation = "flash_attention_2"
config.torch_dtype = torch.bfloat16
config.rope_config = rope_config
config.ignore_index = config.eos_token_id

# Load model from checkpoint
model = LlamaForCausalLM.from_pretrained(load_path, config=config)

# Move to GPU and enable gradient checkpointing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.gradient_checkpointing_enable()  # Explicitly enable gradient checkpointing

train_args = {
    'per_device_train_batch_size': batch_size, 
    'per_device_eval_batch_size': 1,  # Use batch size of 1 for eval to save memory
    'do_train': True, 'do_eval': True, 'bf16': True, 'bf16_full_eval': True, 
    'optim': 'adamw_torch', 'learning_rate': 5e-4, 'weight_decay': 0.1, 
    'adam_beta1': 0.95, 'adam_beta2': 0.99, 'num_train_epochs': 1, 
    'lr_scheduler_type': 'warmup_stable_decay', 'warmup_steps': 0,
    'lr_scheduler_kwargs': {'num_decay_steps': max_steps - 1, }, 

    'label_names': [], 'output_dir': save_path, 
    'eval_strategy': 'steps', 'eval_steps': eval_steps, 
    'logging_strategy': 'steps', 'logging_steps': 1,
    'save_strategy': 'steps', 'save_steps': save_steps, 
    'gradient_accumulation_steps': gradient_accumulation_steps, 
    'max_steps': max_steps, 'disable_tqdm': True, 'save_only_model': True,
    'logging_first_step': False,  # Don't log the first step
    'dataloader_num_workers': 2,  # Reduced workers to save memory
    'dataloader_pin_memory': False,  # Disable pin memory to save RAM
    'max_grad_norm': 1.0,  # Gradient clipping
}

print(f'{config = }', '\n')
print('train_args = ', json.dumps(train_args, indent=2), '\n')
print(f'model is loaded from {load_path} !', '\n')
print('save ckpt at', sorted(list(range(0, max_steps, save_steps))[1:] + steps_to_save),'\n')

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

print('tokenizer is ready !', '\n')

# Load validation dataset from Hugging Face Hub
print(f'Loading validation dataset from Hugging Face: {valid_dataset_hf_id}/{valid_dataset_name}', '\n')

valid_dataset = datasets.load_dataset(valid_dataset_hf_id, valid_dataset_name, split=valid_dataset_split, 
                                      cache_dir=cache_dir)
valid_dataset = valid_dataset.filter(lambda x: len(x[valid_dataset_label].strip()) > 50)
valid_dataset = valid_dataset.select(range(min(valid_size, len(valid_dataset))))

print(valid_dataset, '\n')

# Load training dataset from Hugging Face Hub
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

valid_dataset = EvaluatingDataset(dataset=valid_dataset, tokenizer=tokenizer, 
                                  label_name=valid_dataset_label, valid_length=max_length)

print('dataset is ready !', '\n')

os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_PROJECT"] = "rope_pp"
os.environ["WANDB_DIR"] = f"{root}/wandb"

training_args = TrainingArguments(
    report_to='wandb',  # Keep WandB logging
    logging_dir=f'{root}/wandb',
    run_name=f'{save_abbr}-ckpt{load_ckpt}-decay-single-gpu-{datetime.now().strftime("%Y%m%d-%H%M%S")}',
    include_for_metrics='loss', **train_args
)

print('checkpoints and model will be saved in', train_args['output_dir'], '\n')

start_time = datetime.now()

trainer = TrainerWithDatasetCheckpointing(
    model=model, tokenizer=tokenizer, args=training_args,
    train_dataset=train_dataset, eval_dataset={valid_dataset_abbr: valid_dataset}, 
    callbacks=[CheckpointingCallback(steps_to_save=steps_to_save), 
               CustomLoggingCallback(max_steps=max_steps, 
                                     batch_size=batch_size * gradient_accumulation_steps, 
                                     max_length=max_length, 
                                     world_size=1, 
                                     valid_dataset_abbr=valid_dataset_abbr, 
                                     logging_steps=20)
                ], 
)

# Remove default progress callback that prints JSON logs
from transformers.trainer_callback import PrinterCallback
trainer.remove_callback(PrinterCallback)

trainer.can_return_loss = True

trainer.train()

end_time = datetime.now()
total_time = end_time - start_time
print(f"[{str(end_time)}] 100.00% {max_steps} / {max_steps} [{str(total_time)} / {str(total_time)}]")
print('training is over !')
