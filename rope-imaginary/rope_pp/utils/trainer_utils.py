import os
import torch

import torch.nn.functional as F

from transformers import Trainer, TrainingArguments
from transformers import DefaultDataCollator

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from utils.dataset_utils import StreamingTrainingParquet, StreamingTrainingJsonlZSD, StreamingTrainingHuggingFace


class TrainerWithDatasetCheckpointing(Trainer):

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)

        self.accelerator.wait_for_everyone()

        # Handle both distributed and single GPU training
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            size = torch.distributed.get_world_size()
        else:
            rank = 0
            size = 1

        model_ckpt_path = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        model_ckpt_path = os.path.join(run_dir, model_ckpt_path)

        dataset_ckpt_path = f"{model_ckpt_path}/dataset_ckpt-{rank:{len(str(size))}d}-{size}.pt"
        dataset_ckpt_path = os.path.join(model_ckpt_path, dataset_ckpt_path)

        if isinstance(self.train_dataset, StreamingTrainingParquet):
            
            dataset_ckpt = {
                'data_path': self.train_dataset.data_path, 
                'label_name': self.train_dataset.label_name, 
                'pivot': self.train_dataset.pivot, 'size': self.train_dataset.size, 
                'table_idx': self.train_dataset.table_idx, 
                'table_num': self.train_dataset.table_num, 
                'table_buffer': self.train_dataset.table_buffer, 
                'sample_idx': self.train_dataset.sample_idx, 
                'sample_num': self.train_dataset.sample_num, 
                'token_buffer': self.train_dataset.token_buffer, 
            }

            torch.save(dataset_ckpt, dataset_ckpt_path)
        
        elif isinstance(self.train_dataset, StreamingTrainingJsonlZSD):
            
            dataset_ckpt = {
                'data_path': self.train_dataset.data_path, 
                'label_name': self.train_dataset.label_name, 
                'pivot': self.train_dataset.pivot, 'size': self.train_dataset.size, 
                'sample_idx': self.train_dataset.sample_idx, 
                'token_buffer': self.train_dataset.token_buffer, 
            }

            torch.save(dataset_ckpt, dataset_ckpt_path)
        
        elif isinstance(self.train_dataset, StreamingTrainingHuggingFace):
            # For HuggingFace streaming dataset, save the token buffer state
            dataset_ckpt = {
                'token_buffer': self.train_dataset.token_buffer, 
            }

            torch.save(dataset_ckpt, dataset_ckpt_path)
