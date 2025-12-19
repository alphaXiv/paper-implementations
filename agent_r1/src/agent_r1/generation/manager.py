"""
Tool generation manager for LLM agents
"""

import torch
import re
from typing import List, Dict, Any, Tuple, Union
from dataclasses import dataclass
from PIL import Image
import numpy as np

from .tensor_utils import TensorHelper, TensorConfig
from agent_r1.tool.base import BaseToolEnv, BaseImageToolEnv

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

@dataclass
class ToolGenerationConfig:
    """Configuration for tool-based generation"""
    max_turns: int
    max_prompt_length: int 
    max_response_length: int
    max_response_length_single_turn: int
    use_batch_tool_calls: bool = False
    
class ToolGenerationManager:
    """Manager for handling LLM tool-based generation and interaction"""
    
    def __init__(
        self,
        tokenizer,
        processor,
        actor_rollout_wg,
        config: ToolGenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _example_level_pad(self, data: Union[List[Any], np.ndarray, torch.Tensor],
                           active_mask: torch.Tensor,
                           pad_value: Any = None) -> Union[List[Any], np.ndarray, torch.Tensor]:
        """Pad data according to active mask.
        
        Args:
            data: Data to be padded. Can be list of any type (str, list, dict, etc.)
            active_mask: Boolean tensor indicating which positions are active
            pad_value: Value to use for padding. If None, will use:
                - Empty string "" for strings
                - Empty list [] for lists
                - Empty dict {} for dicts
                - None for other types
                
        Returns:
            Padded data with same type as input
        """
        # Get batch size from active mask
        batch_size = active_mask.shape[0]
        
        # Determine pad value if not provided
        if pad_value is None:
            if len(data) > 0:
                first_elem = data[0]
                if isinstance(first_elem, str):
                    pad_value = ""
                elif isinstance(first_elem, list):
                    pad_value = []
                elif isinstance(first_elem, torch.Tensor):
                    pad_value = torch.full_like(first_elem, fill_value=self.tokenizer.pad_token_id, dtype=first_elem.dtype, device=first_elem.device)
                else:
                    raise NotImplementedError(f"[WARNING] Unsupported data type: {type(first_elem)}")
                
        # Create padded output
        padded_data = [pad_value] * batch_size
        
        # Fill active positions
        s = 0
        for i, is_active in enumerate(active_mask):
            if is_active:
                padded_data[i] = data[s]
                s += 1
        
        # Convert to numpy array if input was numpy array
        if isinstance(data, np.ndarray):
            padded_data = np.array(padded_data, dtype=object)
        elif isinstance(data, torch.Tensor):
            padded_data = torch.stack(padded_data, dim=0)
            
        return padded_data
    
    def _create_response_action_mask(self, responses_ids_list: List[List[int]], tool_responses_ids_list: List[List[int]]) -> List[List[int]]:
        """
        Create action masks for responses identifying which tokens are from the model vs external.
        
        Args:
            responses_ids_list: List of lists containing model's response token IDs
            tool_responses_ids_list: List of lists containing tool response token IDs
            
        Returns:
            action_masks: List of lists with 1s for model-generated tokens and 0s for external tokens
        """
        action_masks = []
        
        for model_ids, tool_ids in zip(responses_ids_list, tool_responses_ids_list):
            # Create mask: 1 for model tokens, 0 for tool tokens
            action_mask = [1] * len(model_ids) + [0] * len(tool_ids)
            action_masks.append(action_mask)

        return action_masks
 
    def _update_rolling_state(self, rollings, responses_ids: torch.Tensor, 
                            tool_responses: List[str], tool_responses_images: List[List[Image.Image]]):
        is_multi_modal = "multi_modal_data" in rollings.non_tensor_batch.keys()

        row_dict_list = []
        formatted_tool_responses = []
        raw_tool_responses = []
        action_masks = []
        
        for i, (tool_response, tool_responses_image) in enumerate(zip(tool_responses, tool_responses_images)):
            row_dict={}
            if is_multi_modal and '<image>' in tool_response and tool_responses_image is not None:
                assert len(tool_responses_image) == tool_response.count('<image>'), f"[WARNING] TOOL RESPONSE IMAGE NUMBER NOT MATCH, {len(tool_responses_image)} != {tool_response.count('<image>')} for {tool_response}"
                raw_tool_responses.append(tool_response.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>'))
                row_dict['multi_modal_data'] = {'image': tool_responses_image}
                image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
                row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
                image_grid_thw = image_inputs['image_grid_thw']
                if image_grid_thw is not None:
                    merge_length = self.processor.image_processor.merge_size**2
                    index = 0
                    while '<image>' in tool_response:
                        tool_response = tool_response.replace(
                            '<image>',
                            '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                            '<|vision_end|>',
                            1,
                        )
                        index += 1

                    tool_response = tool_response.replace('<|placeholder|>', self.processor.image_token)

            else:
                raw_tool_responses.append(tool_response)
            formatted_tool_responses.append(tool_response)
            row_dict_list.append(row_dict)

        tool_responses_ids = self._batch_tokenize(formatted_tool_responses)

        if "responses" not in rollings.batch.keys():
            rollings.batch['responses'] = self.tensor_fn.concatenate_with_padding([
                responses_ids,
                tool_responses_ids
            ], pad_to_left=False)
        else:
            rollings.batch['responses'] = self.tensor_fn.concatenate_with_padding([
                rollings.batch['responses'],
                responses_ids,
                tool_responses_ids
            ], pad_to_left=False)

        rollings.batch['responses'] = rollings.batch['responses'][:, :self.config.max_response_length]

        responses_ids_list = []
        tool_responses_ids_list = []

        for i, (responses_ids_, tool_responses_ids_) in enumerate(zip(responses_ids, tool_responses_ids)):
            responses_ids_ = responses_ids_[responses_ids_ != self.tokenizer.pad_token_id].tolist()
            tool_responses_ids_ = tool_responses_ids_[tool_responses_ids_ != self.tokenizer.pad_token_id].tolist()
            responses_ids_list.append(responses_ids_)
            tool_responses_ids_list.append(tool_responses_ids_)

        action_masks = self._create_response_action_mask(responses_ids_list, tool_responses_ids_list)

        if "action_mask" not in rollings.non_tensor_batch.keys():
            rollings.non_tensor_batch['action_mask'] = np.array(action_masks, dtype=object)
        else:
            new_action_masks = []
            for i, action_mask in enumerate(rollings.non_tensor_batch['action_mask']):
                new_action_masks.append(action_mask + action_masks[i])
            rollings.non_tensor_batch['action_mask'] = np.array(new_action_masks, dtype=object)

        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            responses_ids,
            tool_responses_ids
        ])

        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        
        if is_multi_modal:
            multi_modal_data = rollings.non_tensor_batch['multi_modal_data']
            multi_modal_inputs = rollings.non_tensor_batch['multi_modal_inputs']

            new_multi_modal_data = []
            new_multi_modal_inputs = []

            for row_dict, multi_modal_data_, multi_modal_inputs_ in zip(row_dict_list, multi_modal_data, multi_modal_inputs):
                if 'multi_modal_data' in row_dict.keys():
                    new_multi_modal_data.append({"image":multi_modal_data_['image'] + row_dict['multi_modal_data']['image']})
                else:
                    new_multi_modal_data.append({"image":multi_modal_data_['image']})
                if 'multi_modal_inputs' in row_dict.keys():
                    new_multi_modal_inputs.append({key: torch.cat((val,row_dict['multi_modal_inputs'][key]),dim=0) for key, val in multi_modal_inputs_.items()})
                else:
                    new_multi_modal_inputs.append({key: val for key, val in multi_modal_inputs_.items()})

            rollings.non_tensor_batch['multi_modal_data'] = np.array(new_multi_modal_data, dtype=object)
            rollings.non_tensor_batch['multi_modal_inputs'] = np.array(new_multi_modal_inputs, dtype=object)

            from verl.models.transformers.qwen2_vl import get_rope_index
            new_postion_ids = []
            for i in range(len(new_multi_modal_data)):
                new_postion_ids.append(get_rope_index(
                    processor=self.processor,
                    input_ids=new_input_ids[i],
                    image_grid_thw=new_multi_modal_inputs[i]['image_grid_thw'],
                    attention_mask=new_attention_mask[i],
                ))

            new_position_ids = torch.stack(new_postion_ids, dim=0)
        else:
            new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        rollings.batch['input_ids'] = new_input_ids
        rollings.batch['position_ids'] = new_position_ids
        rollings.batch['attention_mask'] = new_attention_mask

        raw_prompt_ids = rollings.non_tensor_batch['raw_prompt_ids'].tolist()
        new_raw_prompt_ids = []

        for raw_prompt_id, responses_ids_, raw_tool_response in zip(raw_prompt_ids, responses_ids_list, raw_tool_responses):
            if len(responses_ids_) > 0 or len(raw_tool_response) > 0:
                tool_response_ids = self.tokenizer.encode(raw_tool_response, add_special_tokens=False)
                new_raw_prompt_ids.append(raw_prompt_id + responses_ids_ + tool_response_ids)
            else:
                new_raw_prompt_ids.append(raw_prompt_id)

        rollings.non_tensor_batch['raw_prompt_ids'] = np.array(new_raw_prompt_ids, dtype=object)

        return rollings
    
    def run_llm_loop(self, gen_batch, env: Union[BaseToolEnv, BaseImageToolEnv]) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""

        batch_size = gen_batch.batch['input_ids'].shape[0]
        
        active_mask = torch.ones(batch_size, dtype=torch.bool)
        turns = torch.zeros(batch_size, dtype=torch.int32)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch
        meta_info = gen_batch.meta_info
        prompts = gen_batch.batch['input_ids'][:, -self.config.max_prompt_length:].clone()

        # Main generation loop
        for _ in range(self.config.max_turns):
            if not active_mask.sum():
                break

            # Check if any sequence exceeds max length
            effective_len = rollings.batch['attention_mask'].sum(dim=1)
            length_exceeded = effective_len > self.config.max_prompt_length

            if length_exceeded.sum() > 0:
                
                active_mask[length_exceeded] = 0

            raw_prompt_ids = rollings.non_tensor_batch['raw_prompt_ids']
            length_exceeded = [len(prompt_id) > self.config.max_prompt_length for prompt_id in raw_prompt_ids]
            if any(length_exceeded):
                  
                active_mask[length_exceeded] = 0
            
            if not active_mask.sum():
               
                break
            
            if hasattr(rollings, 'non_tensor_batch') and rollings.non_tensor_batch:
                # Create active batch with tensor data
                rollings_active = DataProto.from_dict(
                    tensors={
                        k: v[active_mask] for k, v in rollings.batch.items()
                    },
                    non_tensors={
                        k: v[active_mask.numpy()] for k, v in rollings.non_tensor_batch.items()
                    },
                    meta_info=meta_info
                )
            else:
                rollings_active = DataProto.from_dict(
                    tensors={
                        k: v[active_mask] for k, v in rollings.batch.items()
                    },
                    meta_info=meta_info
                )

            rollings_active, pad_size = pad_dataproto_to_divisor(rollings_active, self.actor_rollout_wg.world_size)
            gen_output = self.actor_rollout_wg.generate_sequences(rollings_active)
            gen_output = unpad_dataproto(gen_output, pad_size=pad_size)

            raw_responses_ids = gen_output.batch['responses']
            responses_ids = env.process_responses_ids(self.tokenizer, raw_responses_ids)
            raw_responses = self.tokenizer.batch_decode(responses_ids, skip_special_tokens=True)
            if isinstance(env, BaseToolEnv):
                if self.config.use_batch_tool_calls:
                    tool_responses, _, new_active_masks = env.batch_step(raw_responses)
                else:
                    tool_responses = []
                    new_active_masks = []
                    for raw_response in raw_responses:
                        tool_response, _, active = env.step(raw_response)
                        tool_responses.append(tool_response)
                        new_active_masks.append(active)
                tool_images = [[]] * len(raw_responses)
            elif isinstance(env, BaseImageToolEnv):
                if self.config.use_batch_tool_calls:
                    tool_responses, tool_images, _, new_active_masks = env.batch_step(raw_responses)
                else:
                    tool_responses = []
                    tool_images = []
                    new_active_masks = []
                    for raw_response in raw_responses:
                        assistant_message, tool_message, tool_image, success, stop = env.step(raw_response)
                        tool_responses.append(tool_message)
                        tool_images.append(tool_image)
                        new_active_masks.append(stop)

            responses_ids = self._example_level_pad(responses_ids, active_mask)
            tool_responses = self._example_level_pad(tool_responses, active_mask, pad_value="")
            tool_images = self._example_level_pad(tool_images, active_mask, pad_value=[])

            active_mask[active_mask.clone()] = torch.tensor(new_active_masks, dtype=torch.bool)

            turns[active_mask] += 1

            active_num_list.append(active_mask.sum().item())

            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                tool_responses,
                tool_images
            )
 
        

        # Compose final output
        final_output = {}
        final_output['turns'] = turns
        final_output['prompts'] = prompts
        final_output['responses'] = rollings.batch['responses'].long()
        final_output['input_ids'] = torch.cat([
            prompts,
            rollings.batch['responses'].long()
        ], dim=1)
        final_output['attention_mask'] = self.tensor_fn.create_attention_mask(final_output['input_ids'])
        if "multi_modal_data" in rollings.non_tensor_batch.keys():
            from verl.models.transformers.qwen2_vl import get_rope_index
            position_ids = []
            for i in range(len(rollings.non_tensor_batch['multi_modal_data'])):
                position_ids.append(get_rope_index(
                    processor=self.processor,
                    input_ids=final_output['input_ids'][i],
                    image_grid_thw=rollings.non_tensor_batch['multi_modal_inputs'][i]['image_grid_thw'],
                    attention_mask=final_output['attention_mask'][i],
                ))

            position_ids = torch.stack(position_ids, dim=0)
            final_output['position_ids'] = position_ids
        else:
            final_output['position_ids'] = self.tensor_fn.create_position_ids(final_output['attention_mask'])

        response_length = final_output['responses'].shape[-1]
        response_mask = final_output['attention_mask'][:, -response_length:]

        final_output['action_mask'] = response_mask.clone()

        for i, action_mask in enumerate(rollings.non_tensor_batch['action_mask']):
            mask_len = min(len(action_mask), response_mask.shape[1])
            final_output['action_mask'][i, :mask_len] = torch.tensor(action_mask[:mask_len]) * response_mask[i, :mask_len]
        
        final_output = DataProto.from_dict(final_output)
        final_output.non_tensor_batch = rollings.non_tensor_batch
        
        return final_output