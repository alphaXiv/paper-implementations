import os
import sys
import json
import random
from datetime import datetime

import torch

from transformers import Trainer, TrainingArguments
from transformers import TrainerCallback, TrainerState, TrainerControl


class CustomLoggingCallback(TrainerCallback):

    def __init__(self, max_steps, batch_size, max_length, world_size, valid_dataset_abbr, logging_steps=10):
        self.cur_step, self.max_steps = 0, max_steps
        self.start_time, self.valid_total_time = datetime.now(), 0
        self.last_step_time = None
        self.batch_token = batch_size * max_length
        self.world_size = world_size
        self.valid_dataset_abbr = valid_dataset_abbr
        self.logging_steps = logging_steps

        # Handle both distributed and single GPU training
        if torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = 0

    def _format_time(self, seconds):
        """Format seconds into human-readable time (e.g., 1.23m, 45.67s, 2.34h)"""
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.2f}m"
        else:
            return f"{seconds / 3600:.2f}h"

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        now = datetime.now()
        
        if logs is not None and 'grad_norm' in logs:
            # Training step logging
            if self.cur_step == 0:
                self.start_time = now
                self.last_step_time = now
                
            else:
                # Calculate time metrics
                total_time = (now - self.start_time).total_seconds()
                train_total_time = total_time - self.valid_total_time
                step_time_ms = (now - self.last_step_time).total_seconds() * 1000 if self.last_step_time else 0
                
                # Calculate token metrics
                num_consume_token = self.cur_step * self.batch_token
                tokens_per_sec = (self.batch_token / (step_time_ms / 1000)) if step_time_ms > 0 else 0
                avg_tgs = num_consume_token / train_total_time / self.world_size if train_total_time > 0 else 0
                
                # Calculate progress
                cur_percent = self.cur_step / self.max_steps * 100
                
                # Get loss and grad norm from logs
                loss = logs.get('loss', 0.0)
                grad_norm = logs.get('grad_norm', 0.0)
                learning_rate = logs.get('learning_rate', 0.0)
                
                # Calculate learning rate multiplier (relative to base lr)
                lr_multiplier = learning_rate / args.learning_rate if args.learning_rate > 0 else 1.0
                
                # Format step string with leading zeros
                step_str = f"{self.cur_step:0{len(str(self.max_steps))}d}"
                
                if self.rank == 0 and self.cur_step % self.logging_steps == 0:
                    # Nanochat-style log format
                    log_str = (
                        f"step {step_str}/{self.max_steps} ({cur_percent:.2f}%) | "
                        f"loss: {loss:.6f} | "
                        f"grad norm: {grad_norm:.4f} | "
                        f"lr: {learning_rate:.2e} | "
                        f"lrm: {lr_multiplier:.2f} | "
                        f"dt: {step_time_ms:.2f}ms | "
                        f"tok/sec: {tokens_per_sec:,.0f} | "
                        f"avg tok/sec: {avg_tgs:,.0f} | "
                        f"total time: {self._format_time(total_time)}"
                    )
                    print(log_str)
            
            self.last_step_time = now
            self.cur_step += 1
            
        else:
            # Evaluation logging
            total_time = (now - self.start_time).total_seconds()
            cur_percent = self.cur_step / self.max_steps * 100

            if self.cur_step == 0:
                self.start_time = now
                total_time = 0
                
            if self.rank == 0:
                step_str = f"{self.cur_step:0{len(str(self.max_steps))}d}"
                eval_str = (
                    f"step {step_str}/{self.max_steps} ({cur_percent:.2f}%) | "
                    f"EVALUATION | "
                    f"total time: {self._format_time(total_time)}"
                )
                print(eval_str, end='')
    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if f'eval_{self.valid_dataset_abbr}_runtime' in metrics:
            self.valid_total_time += metrics[f'eval_{self.valid_dataset_abbr}_runtime']
            
            # Print evaluation metrics
            if self.rank == 0 and metrics:
                eval_loss = metrics.get(f'eval_{self.valid_dataset_abbr}_loss', None)
                if eval_loss is not None:
                    print(f" | eval loss: {eval_loss:.6f}")
                else:
                    print()


class CheckpointingCallback(TrainerCallback):
    def __init__(self, steps_to_save):
        self.steps_to_save = steps_to_save

        # Handle both distributed and single GPU training
        if torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = 0

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step in self.steps_to_save:
            control.should_save = True
            control.should_evaluate = True
            if self.rank == 0:
                print(f"Saving checkpoint at step {state.global_step}")
        return control
