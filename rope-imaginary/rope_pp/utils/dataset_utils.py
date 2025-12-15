import os
import io
import json

import zstandard
import pyarrow.parquet as pq

import torch
import random

import numpy as np

from datasets import Dataset
from transformers import AutoTokenizer


class StreamingTrainingJsonlZSD(torch.utils.data.Dataset):
    
    def __init__(self, data_root, tokenizer, label_name, train_length=4096, min_length=512, num_data=-1, seed=42, dataset_ckpt_path=None):
                
        self.data_root = data_root

        self.data_path = sorted([f'{data_root}/{path}' for path in os.listdir(data_root) if os.path.isdir(f'{data_root}/{path}') and 'git' not in path])
        self.data_path = sorted(sum([[f'{data_root}/{path}' for path in os.listdir(data_root)] for data_root in self.data_path], []))
        self.data_path = sorted(sum([[f'{data_root}/{path}' for path in os.listdir(data_root)] for data_root in self.data_path], []))

        random.shuffle(self.data_path)
        self.dctx = zstandard.ZstdDecompressor()
        
        self.tokenizer = tokenizer
        self.label_name = label_name
        
        self.len = num_data
        self.train_length = train_length 
        self.min_length = min_length

        self.pivot = torch.distributed.get_rank()
        self.size = torch.distributed.get_world_size()
        
        self.token_buffer, self.file_buffer = [], None

        self.file_buffer = open(self.data_path[self.pivot], 'rb')
        self.file_buffer = self.dctx.stream_reader(self.file_buffer)
        self.file_buffer = io.TextIOWrapper(self.file_buffer, encoding='utf-8')
        self.sample_idx = 0

        if dataset_ckpt_path is not None:
            dataset_ckpt_path = f"{dataset_ckpt_path}/dataset_ckpt-{self.pivot:{len(str(self.size))}d}-{self.size}.pt"
            dataset_ckpt = torch.load(dataset_ckpt_path, weights_only=False)
            self.data_path = dataset_ckpt['data_path']
            self.label_name = dataset_ckpt['label_name']
            self.pivot = dataset_ckpt['pivot']
            self.size = dataset_ckpt['size']
            self.file_buffer = open(self.data_path[self.pivot], 'rb')
            self.file_buffer = self.dctx.stream_reader(self.file_buffer)
            self.file_buffer = io.TextIOWrapper(self.file_buffer, encoding='utf-8')
            self.sample_idx = dataset_ckpt['sample_idx']
            for _ in range(self.sample_idx):
                sample = self.file_buffer.readline()
            self.token_buffer = dataset_ckpt['token_buffer']

    def __len__(self):
        return self.len
    
    def __getitem__(self, _):

        if len(self.token_buffer) > self.train_length:
            input_ids = torch.tensor(self.token_buffer[:self.train_length]).long()
            position_ids = torch.tensor(list(range(self.train_length))).long()
            self.token_buffer = self.token_buffer[self.train_length:]

        else:
            input_ids = self.token_buffer if len(self.token_buffer) < self.min_length else []
            position_ids = list(range(self.train_length)) if len(self.token_buffer) < self.min_length else []

            while len(input_ids) < self.train_length:
                sample = self.file_buffer.readline()
                while sample == '':
                    self.pivot = (self.pivot + self.size) % len(self.data_path)
                    self.file_buffer = open(self.data_path[self.pivot], 'rb')
                    self.file_buffer = self.dctx.stream_reader(self.file_buffer)
                    self.file_buffer = io.TextIOWrapper(self.file_buffer, encoding='utf-8')
                    self.sample_idx = 0
                    sample = self.file_buffer.readline()
                self.sample_idx += 1
                
                sample = json.loads(sample)[self.label_name]
                extended_input_ids = self.tokenizer(sample, truncation=False)['input_ids']
                extended_position_ids = list(range(len(extended_input_ids)))

                if len(extended_input_ids) < self.min_length:
                    continue

                input_ids = input_ids + extended_input_ids
                position_ids = position_ids + extended_position_ids
            
            self.token_buffer = input_ids[self.train_length:]
            input_ids = torch.tensor(input_ids[:self.train_length]).long()
            position_ids = torch.tensor(position_ids[:self.train_length]).long()
                
        return {'input_ids': input_ids, 'labels': input_ids, 'position_ids': position_ids}  


class StreamingTrainingParquet(torch.utils.data.Dataset):
    
    def __init__(self, data_root, tokenizer, label_name, train_length=4096, min_length=512, num_data=-1, seed=42, dataset_ckpt_path=None, file_depth=1):
                
        self.data_root = data_root

        self.data_path = sorted([f'{data_root}/{path}' for path in os.listdir(data_root) if not (os.path.isdir(f'{data_root}/{path}') and 'git' in path)])

        for _ in range(file_depth):
            self.data_path = sorted(sum([[f'{data_root}/{path}' for path in os.listdir(data_root)] for data_root in self.data_path], []))

        random.shuffle(self.data_path)
        
        self.tokenizer = tokenizer
        self.label_name = label_name
        
        self.len = num_data
        self.train_length = train_length 
        self.min_length = min_length

        self.pivot = torch.distributed.get_rank()
        self.size = torch.distributed.get_world_size()
        
        self.token_buffer, self.file_buffer = [], None

        self.file_buffer = pq.ParquetFile(self.data_path[self.pivot])
        self.table_idx, self.table_num = 0, self.file_buffer.num_row_groups
        self.table_buffer = self.file_buffer.read_row_group(self.table_idx)
        self.sample_idx, self.sample_num = 0, len(self.table_buffer[self.label_name])
        
        if dataset_ckpt_path is not None:
            dataset_ckpt_path = f"{dataset_ckpt_path}/dataset_ckpt-{self.pivot:{len(str(self.size))}d}-{self.size}.pt"
            dataset_ckpt = torch.load(dataset_ckpt_path, weights_only=False)
            self.data_path = dataset_ckpt['data_path']
            self.label_name = dataset_ckpt['label_name']
            self.pivot = dataset_ckpt['pivot']
            self.size = dataset_ckpt['size']
            self.file_buffer = pq.ParquetFile(self.data_path[self.pivot])
            self.table_idx = dataset_ckpt['table_idx']
            self.table_num = dataset_ckpt['table_num']
            self.table_buffer = dataset_ckpt['table_buffer']
            self.sample_idx = dataset_ckpt['sample_idx']
            self.sample_num = dataset_ckpt['sample_num']
            self.token_buffer = dataset_ckpt['token_buffer']
       
    def __len__(self):
        return self.len
    
    def __getitem__(self, _):

        if len(self.token_buffer) > self.train_length:
            input_ids = torch.tensor(self.token_buffer[:self.train_length]).long()
            position_ids = torch.tensor(list(range(self.train_length))).long()
            self.token_buffer = self.token_buffer[self.train_length:]

        else:
            input_ids = self.token_buffer
            position_ids = list(range(self.train_length))
            while len(input_ids) < self.train_length:
                
                while self.sample_idx >= self.sample_num:
                    self.table_idx += 1
                    while self.table_idx >= self.table_num:
                        self.pivot = (self.pivot + self.size) % len(self.data_path)
                        self.file_buffer = pq.ParquetFile(self.data_path[self.pivot])
                        self.table_idx, self.table_num = 0, self.file_buffer.num_row_groups
                    self.table_buffer = self.file_buffer.read_row_group(self.table_idx)
                    self.sample_idx, self.sample_num = 0, len(self.table_buffer[self.label_name])
                
                sample = str(self.table_buffer[self.label_name][self.sample_idx])
                extended_input_ids = self.tokenizer(sample, truncation=False)['input_ids']
                extended_position_ids = list(range(len(extended_input_ids)))

                if len(extended_input_ids) < self.min_length:
                    continue

                input_ids = input_ids + extended_input_ids
                position_ids = position_ids + extended_position_ids
                self.sample_idx += 1
            
            self.token_buffer = input_ids[self.train_length:]
            input_ids = torch.tensor(input_ids[:self.train_length]).long()
            position_ids = torch.tensor(position_ids[:self.train_length]).long()
        
        return {'input_ids': input_ids, 'labels': input_ids, 'position_ids': position_ids}  


class EvaluatingDataset(torch.utils.data.Dataset):
    
    def __init__(self, dataset, tokenizer, label_name, valid_length=4096):
                
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.label_name = label_name
        
        self.len = len(dataset)
        self.valid_length = valid_length 
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):

        sample = self.dataset[idx]
        inputs = self.tokenizer(sample[self.label_name], truncation=True, 
                                   max_length=self.valid_length, padding='max_length')
        inputs.update({'labels': inputs['input_ids']})

        return inputs


class StreamingTrainingHuggingFace(torch.utils.data.Dataset):
    """
    Streaming dataset that loads from Hugging Face Hub and processes
    samples similar to StreamingTrainingJsonlZSD.
    """
    
    def __init__(self, dataset_id, tokenizer, label_name, train_length=4096, 
                 min_length=512, num_data=-1, seed=42, split='train', 
                 streaming=True, cache_dir=None):
        
        from datasets import load_dataset
        
        self.tokenizer = tokenizer
        self.label_name = label_name
        
        self.len = num_data
        self.train_length = train_length 
        self.min_length = min_length
        
        # Load dataset from Hugging Face
        self.dataset = load_dataset(
            dataset_id, 
            split=split,
            streaming=streaming,
            cache_dir=cache_dir
        )
        
        # Shuffle if streaming
        if streaming:
            self.dataset = self.dataset.shuffle(seed=seed, buffer_size=10000)
        
        self.dataset_iter = iter(self.dataset)
        self.token_buffer = []
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, _):
        
        if len(self.token_buffer) > self.train_length:
            input_ids = torch.tensor(self.token_buffer[:self.train_length]).long()
            position_ids = torch.tensor(list(range(self.train_length))).long()
            self.token_buffer = self.token_buffer[self.train_length:]
        
        else:
            input_ids = self.token_buffer if len(self.token_buffer) < self.min_length else []
            position_ids = list(range(self.train_length)) if len(self.token_buffer) < self.min_length else []
            
            while len(input_ids) < self.train_length:
                try:
                    sample = next(self.dataset_iter)
                except StopIteration:
                    # Reset iterator if we reach the end
                    self.dataset_iter = iter(self.dataset)
                    sample = next(self.dataset_iter)
                
                # Extract text from sample
                text = sample[self.label_name]
                
                # Tokenize
                extended_input_ids = self.tokenizer(text, truncation=False)['input_ids']
                extended_position_ids = list(range(len(extended_input_ids)))
                
                # Skip if too short
                if len(extended_input_ids) < self.min_length:
                    continue
                
                input_ids = input_ids + extended_input_ids
                position_ids = position_ids + extended_position_ids
            
            self.token_buffer = input_ids[self.train_length:]
            input_ids = torch.tensor(input_ids[:self.train_length]).long()
            position_ids = torch.tensor(position_ids[:self.train_length]).long()
        
        return {'input_ids': input_ids, 'labels': input_ids, 'position_ids': position_ids}

