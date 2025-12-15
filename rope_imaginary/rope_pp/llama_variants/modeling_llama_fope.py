# copy from https://github.com/huggingface/transformers/blob/v4.51.0/src/transformers/models/llama/modeling_llama.py

from functools import partial
from typing import Callable, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    LossKwargs,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    can_return_tuple,
    is_torch_flex_attn_available,
    logging,
    replace_return_docstrings,
)
from transformers.utils.deprecation import deprecate_kwarg
from .configuration_llama import LlamaConfig

import math


if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask

    from transformers.integrations.flex_attention import make_flex_block_causal_mask


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "meta-llama/Llama-2-7b-hf"
_CONFIG_FOR_DOC = "LlamaConfig"


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


# class LlamaRotaryEmbedding(nn.Module):
#     def __init__(self, config: LlamaConfig, device=None):
#         super().__init__()
#         # # BC: "rope_type" was originally "type"
#         # if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
#         #     self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
#         # else:
#         #     self.rope_type = "default"
#         self.max_seq_len_cached = config.max_position_embeddings
#         self.original_max_seq_len = config.max_position_embeddings

#         self.config = config
#         # self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

#         base = config.rope_theta
#         partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        
#         head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
#         dim = int(head_dim * partial_rotary_factor)

#         step = 1 if (self.config.rope_config['1d'] and (self.config.rope_config['1d_mode'] in ['1d2', '1dl'])) else 2

#         if self.config.rope_config['imp'] is None:
#             inv_freq = 1.0 / (base ** (torch.arange(0, dim, step, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
#             inv_freq, scaling = inv_freq, torch.ones_like(inv_freq)
#         else:
#             u = self.config.rope_config['imp']['u']
#             beta = (1 - 0.0001) / math.log(10000)  # 0.10861
            
#             a = (u+1) * beta
#             inv_freq = a * (1 - torch.arange(0, dim, step, dtype=torch.int64).to(device=device, dtype=torch.float) / dim) ** u
            
#             p = 1/(a*u) * (inv_freq/a) ** (1/u-1)
#             rho = 1 / (math.log(10000) * inv_freq)
#             rho[inv_freq < 1e-4] = 0

#             scaling = torch.sqrt(rho / p)
#             if self.config.rope_config['imp_mode'] == 'pad':
#                 scaling[inv_freq < 1e-4] = 0
#             elif self.config.rope_config['imp_mode'] == 'partial':
#                 scaling[inv_freq < 1e-4] = 1
#                 inv_freq[inv_freq < 1e-4] = 0
#             elif self.config.rope_config['imp_mode'] == 'none':
#                 scaling[inv_freq < 1e-4] = 1
#             else:
#                 raise KeyError(f"imp mode '{self.config.rope_config['imp_mode']}' not supported.")

#         self.register_buffer("inv_freq", inv_freq, persistent=False)
#         self.register_buffer("scaling", scaling, persistent=False)
#         self.original_inv_freq = self.inv_freq

#     @torch.no_grad()
#     # @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
#     def forward(self, x, position_ids):
#         # print(f'{position_ids.shape[1]=}', flush=True)

#         inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
#         position_ids_expanded = position_ids[:, None, :].float()

#         device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
#         with torch.autocast(device_type=device_type, enabled=False):  # Force float32
#             freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
#             if self.config.rope_config['1d'] and self.config.rope_config['1d_mode'] in ['1d1', '1d2']:
#                 emb = freqs
#                 scaling = self.scaling
#             else:  # 1dl
#                 emb = torch.cat((freqs, freqs), dim=-1)
#                 scaling = torch.cat((self.scaling, self.scaling), dim=-1)
#             cos = emb.cos() * scaling[None, None, :]
#             sin = emb.sin() * scaling[None, None, :]

#         return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

# def rotate_half(x):
#     """Rotates half the hidden dims of the input."""
#     x1 = x[..., : x.shape[-1] // 2]
#     x2 = x[..., x.shape[-1] // 2 :]
#     return torch.cat((-x2, x1), dim=-1)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def _non_meta_init_device(config: LlamaConfig) -> torch.device:
    if config.init_device is not None and config.init_device != "meta":
        return torch.device(config.init_device)
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RotaryEmbedding(nn.Module):
    """
    [Rotary positional embeddings (RoPE)](https://arxiv.org/abs/2104.09864).
    """

    def __init__(
        self, 
        config: LlamaConfig, 
        cache: int = None, 
        dim: int = None,
        n_channels: int = None,
        prefix: str = "attn",
        freq_distribution: str = None,
        freq_learnable: bool = None,
        # use_rope_cache: bool = True,
        use_rope_cache: bool = None,
        include_neg_freq: bool = None,
        floor_freq_ratio: float = None,
        clamp_floor_freq: bool = None,
        clamp_floor_to_zero: bool = None,
        upper_freq_ratio: float = None,
        clamp_upper_freq: float = None,
        clamp_upper_to_zero: bool = None,
        zero_freq_ratio: float = None,
        clamp_to_linear: bool = None,
        clamp_to_linear_mode: str = None,
        init_upper_freq: float = None,
        init_floor_freq: float = None,
    ):
        super().__init__()
        self.config = config
        self.__cache = cache
        
        self.prefix = prefix
        self.suffix = "rope"
        self.use_rope_cache = use_rope_cache
        
        self.freq_distribution = freq_distribution if freq_distribution is not None else self.config.rope_init_distribution
        if self.freq_distribution not in ["constant", "linear", "uniform", "gaussian", "exponential"]:
            raise ValueError(f"Unsupported frequency distribution: {self.freq_distribution}")
        
        self.freq_learnable = freq_learnable if freq_learnable is not None else self.config.rope_learnable
        self.include_neg_freq = include_neg_freq if include_neg_freq is not None else self.config.rope_include_neg_freq
        
        if dim is not None:
            self.dim = dim
        elif self.prefix == "attn":
            self.dim = self.config.hidden_size // self.config.num_attention_heads
        else:
            self.dim = self.config.hidden_size
        
        if n_channels is not None:
            self.n_channels = n_channels
        elif self.prefix == "attn" and self.config.rope_learnable and self.config.rope_no_repetition:
            self.n_channels = self.config.num_attention_heads
        else:
            self.n_channels = 1
            
        self.init_floor_freq = init_floor_freq if init_floor_freq is not None else self.config.rope_init_floor_freq
        self.init_upper_freq = init_upper_freq if init_upper_freq is not None else self.config.rope_init_upper_freq
        
        self.clamp_floor_freq = clamp_floor_freq if clamp_floor_freq is not None else self.config.rope_clamp_floor_freq
        if self.clamp_floor_freq:
            self.floor_freq_ratio = floor_freq_ratio if floor_freq_ratio is not None else self.config.rope_floor_freq_ratio
            self.floor_freq = 2*torch.pi/self.config.max_position_embeddings * self.floor_freq_ratio
            
            self.clamp_floor_to_zero = clamp_floor_to_zero if clamp_floor_to_zero is not None else self.config.rope_clamp_floor_to_zero
            self.clamp_floor_value = 0.0 if self.clamp_floor_to_zero else 2*torch.pi/self.config.max_position_embeddings*self.clamp_floor_ratio
        else:
            self.floor_freq = 0.0
        
        self.clamp_upper_freq = clamp_upper_freq if clamp_upper_freq is not None else self.config.rope_clamp_upper_freq
        if self.clamp_upper_freq:
            self.upper_freq_ratio = upper_freq_ratio if upper_freq_ratio is not None else self.config.rope_upper_freq_ratio
            self.upper_freq = torch.pi * self.upper_freq_ratio
            
            self.clamp_upper_to_zero = clamp_upper_to_zero if clamp_upper_to_zero is not None else self.config.rope_clamp_upper_to_zero
            self.clamp_upper_value = 0.0 if self.clamp_upper_to_zero else torch.pi * self.clamp_upper_ratio
        else:
            self.upper_freq = 1.0
            
        if self.clamp_floor_freq or self.clamp_upper_freq:
            assert self.upper_freq >= self.floor_freq
        
        self.zero_freq_ratio = zero_freq_ratio if zero_freq_ratio is not None else self.config.rope_zero_freq_ratio
        
        self.clamp_to_linear = clamp_to_linear if clamp_to_linear is not None else self.config.rope_clamp_to_linear
        self.clamp_to_linear_mode = clamp_to_linear_mode if clamp_to_linear_mode is not None else self.config.rope_clamp_to_linear_mode
        
        if self.freq_learnable:
            self.inv_freq = nn.Parameter(
                self.get_inv_freq(self.dim, _non_meta_init_device(config)),
                requires_grad=True
            )
        else:
            self.inv_freq = self.get_inv_freq(self.dim, _non_meta_init_device(config))
            
            if self.use_rope_cache:
                # Warm up cache.
                self.get_rotary_embedding(config.max_position_embeddings, _non_meta_init_device(config))
    
        
    def extra_yarn(self, inv_freq = None) -> torch.Tensor:
        assert self.config.len_extra_orig_length != 0
        
        def find_correction_dim(num_rotations, dim, base=self.config.rope_theta, orig_max_position_embeddings=self.config.len_extra_orig_length):
            return (dim * math.log(orig_max_position_embeddings/(num_rotations * 2 * math.pi)))/(2 * math.log(base))

        def find_correction_range(low_rot, high_rot, dim, base=self.config.rope_theta, orig_max_position_embeddings=self.config.len_extra_orig_length):
            low = math.floor(find_correction_dim(
                low_rot, dim, base, orig_max_position_embeddings))
            high = math.ceil(find_correction_dim(
                high_rot, dim, base, orig_max_position_embeddings))
            return max(low, 0), min(high, dim-1)  # Clamp values just in case

        def linear_ramp_mask(low, fast, dim):
            if low == fast:
                fast += 0.001  # Prevent singularity
            linear_func = (torch.arange(dim, dtype=torch.float32) - low) / (fast - low)
            ramp_func = torch.clamp(linear_func, 0, 1)
            return ramp_func
        
        if inv_freq is None:
            pos_freqs = self.config.rope_theta ** (torch.arange(0, self.dim, 2, device=_non_meta_init_device(self.config), dtype=torch.float) / self.dim)
            inv_freq_extra = 1.0 / pos_freqs
            inv_freq_inter = 1.0 / (self.config.len_extra_yarn_scale * pos_freqs)
        else:
            inv_freq_extra = inv_freq
            inv_freq_inter = inv_freq / self.config.len_extra_yarn_scale
       
        low, high = find_correction_range(self.config.len_extra_yarn_beta_fast, self.config.len_extra_yarn_beta_slow, self.dim)
        inv_freq_mask = (1 - linear_ramp_mask(low, high, self.dim // 2).float().to(_non_meta_init_device(self.config))) * self.config.len_extra_yarn_factor
        inv_freq = inv_freq_inter * (1 - inv_freq_mask) + inv_freq_extra * inv_freq_mask
        
        return inv_freq
    
    def extra_pi(self, inv_freq = None) -> torch.Tensor:
        assert self.config.len_extra_orig_length != 0
            
        self.pi_ratio = self.config.len_extra_orig_length / self.config.max_position_embeddings
        
        inv_freq = inv_freq * self.pi_ratio
        
        return inv_freq
    
    def get_inv_freq(self, dim: int, device: torch.device) -> torch.Tensor:
        if self.freq_distribution == "constant": # self.config.rope_no_rotary:
            torch.ones_like(torch.arange(0, dim, 2, device=device, dtype=torch.float))
        elif self.freq_distribution == "linear": # self.config.rope_linear:
            inv_freq = 2*torch.pi/self.config.max_position_embeddings * torch.arange(0, dim, 2, device=device, dtype=torch.float) 
            
            inv_freq[inv_freq > self.clamp_upper_ratio * torch.pi] = self.clamp_upper_value
            
            # 翻转，保持频率递减
            inv_freq = inv_freq.flip(0)
            
        elif self.freq_distribution == "uniform": # self.config.rope_uniform:
            inv_freq = 1.0 * torch.rand(dim//2, device=device, dtype=torch.float)
            
        elif self.freq_distribution == "gaussian": # self.config.rope_gaussian:
            inv_freq = torch.randn(dim//2, device=device, dtype=torch.float).abs()
            inv_freq = inv_freq / inv_freq.max()
        else:
            inv_freq = 1.0 / (
                self.config.rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim)
            )
        
        inv_freq = self.init_floor_freq + inv_freq * (self.init_upper_freq - self.init_floor_freq)
        
        if self.config.len_extra and self.config.len_extra_before_clamp:
            if self.config.len_extra_type == "PI":
                inv_freq = self.extra_pi(inv_freq)
        
            elif self.config.len_extra_type == "YARN":
                inv_freq = self.extra_yarn(inv_freq)
        
        if self.clamp_floor_freq:
            inv_freq[inv_freq < self.floor_freq] = self.clamp_floor_value
        if self.clamp_upper_freq:
            inv_freq[inv_freq > self.upper_freq] = self.clamp_upper_value
            
        if self.clamp_to_linear:
            freq_ratio = inv_freq / (2*torch.pi/self.config.max_position_embeddings)
            if self.clamp_to_linear_mode == "ceil":
                inv_freq = 2*torch.pi/self.config.max_position_embeddings * freq_ratio.ceil()
            elif self.clamp_to_linear_mode == "floor":
                inv_freq = 2*torch.pi/self.config.max_position_embeddings * freq_ratio.floor()
            elif self.clamp_to_linear_mode == "half":
                _half = (freq_ratio != 0).float() * 0.5
                inv_freq = 2*torch.pi/self.config.max_position_embeddings * (freq_ratio.floor() + _half)
            elif self.clamp_to_linear_mode == "arange":
                _linear = (freq_ratio != 0).float()
                _linear[_linear!=0] = torch.arange(0, 1, 1/_linear.sum(), device=device, dtype=torch.float)
                
                inv_freq = 2*torch.pi/self.config.max_position_embeddings * (freq_ratio.floor() + _linear)    
            elif self.clamp_to_linear_mode == "flip_arange":
                _linear = (freq_ratio != 0).float()
                _linear[_linear!=0] = (torch.arange(0, 1, 1/_linear.sum(), device=device, dtype=torch.float)).flip(0)
                
                inv_freq = 2*torch.pi/self.config.max_position_embeddings * (freq_ratio.floor() + _linear)  
            else:
                raise ValueError(f"Unsupported clamp_to_linear_mode: {self.clamp_to_linear_mode}")
        
        if self.zero_freq_ratio >= 0.0:
            zero_freq_num = (inv_freq == 0.0).sum()
            zero_freq_num_ceil = math.ceil(dim//2 * self.zero_freq_ratio)
            
            if zero_freq_num > zero_freq_num_ceil:
                zero_freq_indeces = torch.nonzero(inv_freq == 0.0, as_tuple=False).squeeze()
                reset_freq_num = zero_freq_num - zero_freq_num_ceil

                inv_freq[zero_freq_indeces[:reset_freq_num]] = \
                    2*torch.pi/self.config.max_position_embeddings*torch.arange(1, reset_freq_num+1, device=device, dtype=torch.float)
            else:
                non_zero_freq_indeces = torch.nonzero(inv_freq != 0.0, as_tuple=False).squeeze()
                reset_freq_num = zero_freq_num_ceil - zero_freq_num
                
                if self.clamp_upper_to_zero:
                    inv_freq[non_zero_freq_indeces[:reset_freq_num]] = 0.0
                else:
                    inv_freq[non_zero_freq_indeces[-reset_freq_num:]] = 0.0
                    
        if self.config.len_extra and not self.config.len_extra_before_clamp:
            if self.config.len_extra_type == "PI":
                inv_freq = self.extra_pi(inv_freq)
        
            elif self.config.len_extra_type == "YARN":
                inv_freq = self.extra_yarn(inv_freq)
        
        if self.include_neg_freq:
            inv_freq *= (-1) ** torch.arange(0, dim//2, device=device, dtype=torch.float)
        
        if self.config.fourier and self.config.fourier_ignore_zero:
            inv_freq = inv_freq[inv_freq != 0.0]
        
        if self.prefix == "embed":
            inv_freq = inv_freq # shape: (dim//2, )
        elif self.prefix == "attn":
            if self.config.rope_learnable and self.config.rope_no_repetition:
                inv_freq = inv_freq.repeat(self.n_channels, 1) # shape: (num_attention_heads, dim//2)
            else:
                inv_freq = inv_freq[None, :] # shape: (1, dim//2)
        else:
            raise ValueError(f"Unsupported prefix: {self.prefix}")
        
        return inv_freq
        
    def get_rotary_embedding(
        self, 
        seq_len: int, 
        device: torch.device, 
        layer_idx: Optional[int] = None,
        use_rope_cache: bool = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        use_rope_cache = use_rope_cache or ((not self.freq_learnable) and self.use_rope_cache)
        
        if use_rope_cache:
            if (
                (pos_sin := self.__cache.get(f"{self.prefix}_{self.suffix}_pos_sin")) is not None
                and (pos_cos := self.__cache.get(f"{self.prefix}_{self.suffix}_pos_cos")) is not None
                and pos_sin.shape[-2] >= seq_len
                and pos_cos.shape[-2] >= seq_len
            ):
                if pos_sin.device != device:
                    pos_sin = pos_sin.to(device)
                    self.__cache[f"{self.prefix}_{self.suffix}_pos_sin"] = pos_sin
                if pos_cos.device != device:
                    pos_cos = pos_cos.to(device)
                    self.__cache[f"{self.prefix}_{self.suffix}_pos_cos"] = pos_cos
                    
                if self.prefix == "embed":
                    return pos_sin[:, :seq_len, :], pos_cos[:, :seq_len, :]
                elif self.prefix == "attn":
                    return pos_sin[:, :, :seq_len, :], pos_cos[:, :, :seq_len, :]
                else:
                    raise ValueError(f"Unsupported prefix: {self.prefix}")

        with torch.autocast(device.type, enabled=False):
            
            if self.freq_learnable:
                if self.clamp_floor_freq or self.clamp_upper_freq:
                    sign = self.inv_freq.data.sign()
                    self.inv_freq.data.abs_().clamp_(
                        2*torch.pi/self.config.max_position_embeddings*self.clamp_floor_ratio, torch.pi*self.clamp_upper_ratio
                    ).mul_(sign)
                else:
                    self.inv_freq.data.clamp_(-torch.pi, torch.pi)
            
            if self.config.rope_no_pos:
                seq = torch.ones(seq_len, device=device, dtype=torch.float) # ablation
            else:
                seq = torch.arange(seq_len, device=device, dtype=torch.float) # original
            
            if self.prefix == "embed":
                freqs = torch.einsum("t, d -> td", seq, self.inv_freq) # shape: (seq_len, dim//2)
            elif self.prefix == "attn":
                freqs = torch.einsum("t, hd -> htd", seq, self.inv_freq) # shape: (1 or num_attention_heads, seq_len, dim//2)
            else:
                raise ValueError(f"Unsupported prefix: {self.prefix}")
            
            if self.suffix == "fourier": 
                positions = freqs.unsqueeze(0) # shape: (1, 1 or num_attention_heads, seq_len, dim//2) or (1, seq_len, dim//2)
            else:
                positions = torch.cat((freqs, freqs), dim=-1).unsqueeze(0) # shape: (1, 1 or num_attention_heads, seq_len, dim) or (1, seq_len, dim)
                
            pos_sin, pos_cos = positions.sin(), positions.cos()

        if (not self.freq_learnable) and self.use_rope_cache:
            self.__cache[f"{self.prefix}_{self.suffix}_pos_sin"] = pos_sin
            self.__cache[f"{self.prefix}_{self.suffix}_pos_cos"] = pos_cos
            
        return pos_sin, pos_cos

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        if self.prefix == "embed":
            B, T, hs = x.size()
            x = x.view(B, T, 2, hs // 2)
            
        elif self.prefix == "attn":
            B, nh, T, hs = x.size()
            x = x.view(B, nh, T, 2, hs // 2)
            
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        if not inverse:
            return ((t * pos_cos) + (self.rotate_half(t) * pos_sin)).to(t.dtype)
        else:
            return ((t * pos_cos) - (self.rotate_half(t) * pos_sin)).to(t.dtype)
    
    def forward(
        self, 
        x: torch.Tensor, 
        all_len: int, 
        layer_idx: Optional[int] = None, 
        inverse: bool = False,
        use_rope_cache: bool = None
    ) -> torch.Tensor:
        
        if self.config.rope_full_precision:
            x_ = x.float()
        else:
            x_ = x
        
        with torch.autocast(x.device.type, enabled=False):
            x_len = x_.shape[-2]
            pos_sin, pos_cos = self.get_rotary_embedding(all_len, x_.device, layer_idx=layer_idx, use_rope_cache=use_rope_cache)
            pos_sin = pos_sin.type_as(x_)
            pos_cos = pos_cos.type_as(x_)
            
            if self.prefix == "embed":
                x_ = self.apply_rotary_pos_emb(
                    pos_sin[:, all_len - x_len : x_len, :], 
                    pos_cos[:, all_len - x_len : x_len, :], 
                    x_,
                    inverse
                )
            elif self.prefix == "attn":
                x_ = self.apply_rotary_pos_emb(
                    pos_sin[:, :, all_len - x_len : all_len, :], 
                    pos_cos[:, :, all_len - x_len : all_len, :], 
                    x_,
                    inverse
                )
            else:
                raise ValueError(f"Unsupported prefix: {self.prefix}")
            
        return x_.type_as(x)

class FourierEmbedding(RotaryEmbedding):
    def __init__(self, config, *args, **kwargs):
        self.config = config
        
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        dim = self.config.fourier_dim if self.config.fourier_dim > self.head_dim else self.head_dim
        
        super().__init__(config, dim=dim, *args, **kwargs)
        
        self.suffix = "fourier"
        
        if self.config.fourier_ignore_zero:
            self.input_dim = self.inv_freq.size(-1)
            self.output_dim = min(self.input_dim, self.head_dim//4) # TODO: self.head_dim//8
        else:
            self.input_dim = self.dim // 2
            self.output_dim = self.head_dim // 2
        
        if self.prefix == "embed":
            self.input_shape = "btD"
            self.output_shape = "btd"
        elif self.prefix == "attn":
            self.input_shape = "bhtD"
            self.output_shape = "bhtd"
            
        # if self.prefix == "attn" and self.config.fourier_separate_head:
        #     size = (self.config.num_attention_heads, self.input_dim, self.output_dim)
        #     self.coef_shape = "hDd"
        # else:
        #     size = (self.input_dim, self.output_dim)
        #     self.coef_shape = "Dd"
        self.num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads
        if self.prefix == "attn" and self.config.fourier_separate_head:
            size = (self.config.num_key_value_heads, self.input_dim, self.output_dim)
            self.coef_shape = "hDd"
        else:
            size = (self.input_dim, self.output_dim)
            self.coef_shape = "Dd"
        
        if self.config.fourier_separate_basis:
            self.sin_coef = nn.Parameter(
                torch.randn(size=size, device=_non_meta_init_device(self.config), dtype=torch.float),
                requires_grad=self.config.fourier_learnable
            )
            self.cos_coef = nn.Parameter(
                torch.randn(size=size, device=_non_meta_init_device(self.config), dtype=torch.float),
                requires_grad=self.config.fourier_learnable
            )
        else:
            self.fourier_coef = nn.Parameter(
                torch.randn(size=size, device=_non_meta_init_device(self.config), dtype=torch.float),
                requires_grad=self.config.fourier_learnable
            )
        
        self.reset_parameters()
    
    # def apply_rotary_pos_emb(self, pos_sin, pos_cos, t, inverse = False):
    #     if self.config.fourier_separate_basis:
    #         if self.config.fourier_norm:
    #             fourier_sin = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_sin, self.sin_coef / self.sin_coef.sum(dim=-2, keepdim=True))
    #             fourier_cos = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_cos, self.cos_coef / self.cos_coef.sum(dim=-2, keepdim=True))
    #         else:
    #             fourier_sin = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_sin, self.sin_coef)
    #             fourier_cos = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_cos, self.cos_coef)
    #     else:
    #         if self.config.fourier_norm:
    #             fourier_sin = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_sin, self.fourier_coef / self.fourier_coef.sum(dim=-2, keepdim=True))
    #             fourier_cos = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_cos, self.fourier_coef / self.fourier_coef.sum(dim=-2, keepdim=True))
    #         else:
    #             fourier_sin = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_sin, self.fourier_coef)
    #             fourier_cos = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_cos, self.fourier_coef)
        
    #     if self.config.fourier_ignore_zero:
    #         fourier_sin = F.pad(input=fourier_sin, pad=(0, self.head_dim//2-fourier_sin.size(-1)), mode="constant", value=1)
    #         fourier_cos = F.pad(input=fourier_cos, pad=(0, self.head_dim//2-fourier_cos.size(-1)), mode="constant", value=1)
        
    #     fourier_sin = torch.cat((fourier_sin, fourier_sin), dim=-1)
    #     fourier_cos = torch.cat((fourier_cos, fourier_cos), dim=-1)
        
    #     if not inverse:
    #         return ((t * fourier_cos) + (self.rotate_half(t) * fourier_sin)).to(t.dtype)
    #     else:
    #         return ((t * fourier_cos) - (self.rotate_half(t) * fourier_sin)).to(t.dtype)

    def _expand_coef_for_heads(self, coef: torch.Tensor, h_in: int) -> torch.Tensor:
        if self.prefix != "attn" or not self.config.fourier_separate_head:
            return coef
        H_kv = self.config.num_key_value_heads
        H_q = self.config.num_attention_heads
        G = self.num_key_value_groups
        if h_in == H_kv:
            return coef
        elif h_in == H_q:
            return coef.repeat_interleave(G, dim=0) 
        elif h_in == 1:
            return coef
        else:
            raise ValueError(f"Unexpected head size: got {h_in}, expected {H_kv}, {H_q}, or 1.")

    def _maybe_expand_pos_h(self, pos: torch.Tensor, target_h: int) -> torch.Tensor:
        # pos shape: (b, h, t, D)
        if pos.size(1) == target_h:
            return pos
        if pos.size(1) == 1:
            return pos.expand(pos.size(0), target_h, pos.size(2), pos.size(3))
        raise ValueError(f"pos head dim {pos.size(1)} cannot be matched to target {target_h}.")

    def apply_rotary_pos_emb(self, pos_sin, pos_cos, t, inverse=False):
        ## Add the following code
        if self.prefix == "attn" and self.config.fourier_separate_head:
            h_in = t.size(1) 
            if self.config.fourier_separate_basis:
                sin_coef = self._expand_coef_for_heads(self.sin_coef, h_in)
                cos_coef = self._expand_coef_for_heads(self.cos_coef, h_in)

                target_h = sin_coef.size(0)
                pos_sin = self._maybe_expand_pos_h(pos_sin, target_h)
                pos_cos = self._maybe_expand_pos_h(pos_cos, target_h)
                if self.config.fourier_norm:
                    sin_coef_use = sin_coef / sin_coef.sum(dim=-2, keepdim=True)
                    cos_coef_use = cos_coef / cos_coef.sum(dim=-2, keepdim=True)
                else:
                    sin_coef_use = sin_coef
                    cos_coef_use = cos_coef
                sin_coef_use = sin_coef_use.to(dtype=pos_sin.dtype, device=pos_sin.device)
                cos_coef_use = cos_coef_use.to(dtype=pos_cos.dtype, device=pos_cos.device)
                fourier_sin = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_sin, sin_coef_use)
                fourier_cos = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_cos, cos_coef_use)
            else:
                fourier_coef = self._expand_coef_for_heads(self.fourier_coef, h_in)
                target_h = fourier_coef.size(0)
                pos_sin = self._maybe_expand_pos_h(pos_sin, target_h)
                pos_cos = self._maybe_expand_pos_h(pos_cos, target_h)
                if self.config.fourier_norm:
                    coef_use = fourier_coef / fourier_coef.sum(dim=-2, keepdim=True)
                else:
                    coef_use = fourier_coef
                coef_use = coef_use.to(dtype=pos_sin.dtype, device=pos_sin.device)
                fourier_sin = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_sin, coef_use)
                fourier_cos = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_cos, coef_use)
        else:
            ## Original
            if self.config.fourier_separate_basis:
                if self.config.fourier_norm:
                    fourier_sin = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_sin, self.sin_coef / self.sin_coef.sum(dim=-2, keepdim=True))
                    fourier_cos = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_cos, self.cos_coef / self.cos_coef.sum(dim=-2, keepdim=True))
                else:
                    fourier_sin = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_sin, self.sin_coef)
                    fourier_cos = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_cos, self.cos_coef)
            else:
                if self.config.fourier_norm:
                    fourier_sin = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_sin, self.fourier_coef / self.fourier_coef.sum(dim=-2, keepdim=True))
                    fourier_cos = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_cos, self.fourier_coef / self.fourier_coef.sum(dim=-2, keepdim=True))
                else:
                    fourier_sin = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_sin, self.fourier_coef)
                    fourier_cos = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_cos, self.fourier_coef)

        if self.config.fourier_ignore_zero:
            fourier_sin = F.pad(input=fourier_sin, pad=(0, self.head_dim//2-fourier_sin.size(-1)), mode="constant", value=1)
            fourier_cos = F.pad(input=fourier_cos, pad=(0, self.head_dim//2-fourier_cos.size(-1)), mode="constant", value=1)

        fourier_sin = torch.cat((fourier_sin, fourier_sin), dim=-1)
        fourier_cos = torch.cat((fourier_cos, fourier_cos), dim=-1)

        if not inverse:
            return ((t * fourier_cos) + (self.rotate_half(t) * fourier_sin)).to(t.dtype)
        else:
            return ((t * fourier_cos) - (self.rotate_half(t) * fourier_sin)).to(t.dtype)
    
    def get_step_eye(self, _param):
        _param = torch.zeros_like(_param)
        
        step = math.ceil(self.input_dim / self.output_dim)
        for i in range(self.output_dim):
            if i*step < self.input_dim:
                _param[..., i*step, i] = 1.0
        
        return _param
    
    def reset_parameters(self):
        with torch.no_grad():
            if self.config.fourier_separate_basis:
                if self.config.fourier_init == "eye":
                    if self.input_dim == self.output_dim:
                        if self.prefix == "attn" and self.config.fourier_separate_head:
                            for i in range(self.sin_coef.size(0)):
                                torch.nn.init.eye_(self.sin_coef[i]) 
                                torch.nn.init.eye_(self.cos_coef[i])
                        else:
                            torch.nn.init.eye_(self.sin_coef)
                            torch.nn.init.eye_(self.cos_coef)
                    else:   
                        self.sin_coef.data = self.get_step_eye(self.sin_coef)
                        self.cos_coef.data = self.get_step_eye(self.cos_coef)
                        
                elif self.config.fourier_init == "eye_norm":
                    torch.nn.init.normal_(self.sin_coef, std=self.config.fourier_init_norm_gain)
                    torch.nn.init.normal_(self.cos_coef, std=self.config.fourier_init_norm_gain)
                    
                    if self.input_dim == self.output_dim:
                        self.sin_coef += torch.eye(self.input_dim, device=self.sin_coef.device)
                        self.cos_coef += torch.eye(self.input_dim, device=self.cos_coef.device)
                    else:
                        self.sin_coef += self.get_step_eye(self.sin_coef)
                        self.cos_coef += self.get_step_eye(self.cos_coef)
                    
                elif self.config.fourier_init == "eye_xavier_norm":
                    torch.nn.init.xavier_normal_(self.sin_coef, gain=self.config.fourier_init_norm_gain)
                    torch.nn.init.xavier_normal_(self.cos_coef, gain=self.config.fourier_init_norm_gain)
                    
                    if self.input_dim == self.output_dim:    
                        self.sin_coef += torch.eye(self.input_dim, device=self.sin_coef.device)
                        self.cos_coef += torch.eye(self.input_dim, device=self.cos_coef.device)
                    else:
                        self.sin_coef += self.get_step_eye(self.sin_coef)
                        self.cos_coef += self.get_step_eye(self.cos_coef)
                
                elif self.config.fourier_init == "eye_xavier_uniform":
                    torch.nn.init.xavier_uniform_(self.sin_coef, gain=self.config.fourier_init_norm_gain)
                    torch.nn.init.xavier_uniform_(self.cos_coef, gain=self.config.fourier_init_norm_gain)
                    
                    if self.input_dim == self.output_dim:
                        self.sin_coef += torch.eye(self.input_dim, device=self.sin_coef.device)
                        self.cos_coef += torch.eye(self.input_dim, device=self.cos_coef.device)
                    else:
                        self.sin_coef += self.get_step_eye(self.sin_coef)
                        self.cos_coef += self.get_step_eye(self.cos_coef)
                
                elif self.config.fourier_init == "xavier_norm":
                    torch.nn.init.xavier_normal_(self.sin_coef)
                    torch.nn.init.xavier_normal_(self.cos_coef)
                elif self.config.fourier_init == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(self.sin_coef)
                    torch.nn.init.xavier_uniform_(self.cos_coef)
                else:
                    raise ValueError(f"Unsupported init method: {self.config.fourier_init}")
            else:
                if self.config.fourier_init == "eye":
                    
                    if self.input_dim == self.output_dim:
                        if self.prefix == "attn" and self.config.fourier_separate_head:
                            for i in range(self.fourier_coef.size(0)):
                                torch.nn.init.eye_(self.fourier_coef[i])
                        else:
                            torch.nn.init.eye_(self.fourier_coef)
                    else:
                            
                        self.fourier_coef.data = self.get_step_eye(self.fourier_coef)
                        
                elif self.config.fourier_init == "eye_norm":
                    torch.nn.init.normal_(self.fourier_coef, std=self.config.fourier_init_norm_gain)
                    
                    if self.input_dim == self.output_dim:    
                        self.fourier_coef += torch.eye(self.input_dim, device=self.fourier_coef.device)
                    else:
                        self.fourier_coef += self.get_step_eye(self.fourier_coef)
                    
                elif self.config.fourier_init == "eye_xavier_norm":
                    torch.nn.init.xavier_normal_(self.fourier_coef, gain=self.config.fourier_init_norm_gain)
                    
                    if self.input_dim == self.output_dim:
                        self.fourier_coef += torch.eye(self.input_dim, device=self.fourier_coef.device)
                    else:
                        self.fourier_coef += self.get_step_eye(self.fourier_coef)
                    
                elif self.config.fourier_init == "eye_xavier_uniform":
                    torch.nn.init.xavier_uniform_(self.fourier_coef, gain=self.config.fourier_init_norm_gain) 
                    
                    if self.input_dim == self.output_dim:
                        self.fourier_coef += torch.eye(self.input_dim, device=self.fourier_coef.device)
                    else:
                        self.fourier_coef += self.get_step_eye(self.fourier_coef)
                            
                elif self.config.fourier_init == "xavier_norm":
                    torch.nn.init.xavier_normal_(self.fourier_coef)
                elif self.config.fourier_init == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(self.fourier_coef)
                else:
                    raise ValueError(f"Unsupported init method: {self.config.fourier_init}")

    def __repr__(self):
        return f"{self.__class__.__name__}(fourier_dim={self.input_dim})"

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        # if self.config.rope_config['1d']:
        #     if self.config.rope_config['1d_mode'] == '1d1':
        #         self.qk_head_dim = self.head_dim // 2
        #         self.scaling = self.qk_head_dim**-0.5
        #     elif self.config.rope_config['1d_mode'] in ['1d2', '1dl']:
        #         self.qk_head_dim = self.head_dim
        #         self.scaling = self.qk_head_dim**-0.5
        #     else:
        #         raise KeyError(f"1d mode '{self.config.rope_config['1d_mode']}' not supported.")
        # else:
        self.qk_head_dim = self.head_dim
        self.scaling = self.qk_head_dim**-0.5
        
        # if self.config.rope_config['imag']:
        #     if self.config.rope_config['imag_mode'] == 'imag1':
        #         num_q_heads, num_kv_heads = config.num_attention_heads // 2, config.num_key_value_heads // 2
        #         self.num_key_value_groups = self.num_key_value_groups * 2
        #         num_attention_heads = config.num_attention_heads
        #     elif self.config.rope_config['imag_mode'] == 'imag2':
        #         num_q_heads, num_kv_heads = config.num_attention_heads, config.num_key_value_heads
        #         self.num_key_value_groups = self.num_key_value_groups * 2
        #         num_attention_heads = config.num_attention_heads * 2
        #     elif self.config.rope_config['imag_mode'] in ['imagh', 'imago']:
        #         num_q_heads, num_kv_heads = config.num_attention_heads, config.num_key_value_heads
        #         num_attention_heads = config.num_attention_heads
        #     else:
        #         raise KeyError(f"imag mode '{self.config.rope_config['imag_mode']}' not supported.")
        # else:
        num_q_heads, num_kv_heads = config.num_attention_heads, config.num_key_value_heads
        num_attention_heads = config.num_attention_heads

        if self.config.fourier:
            self.pos_emb = FourierEmbedding(config)
        else:
            self.pos_emb = RotaryEmbedding(config)
        
        self.q_proj = nn.Linear(
            config.hidden_size, num_q_heads * self.qk_head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, num_kv_heads * self.qk_head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
    
    # def apply_rotary_pos_emb(self, q, k, cos, sin, unsqueeze_dim=1):
    #     cos = cos.unsqueeze(unsqueeze_dim)
    #     sin = sin.unsqueeze(unsqueeze_dim)
    
    #     if self.config.rope_config['1d'] and self.config.rope_config['1d_mode'] == '1dl':
    #         q_shape = q.shape
    #         q = torch.stack([q, q], dim=-2).reshape(*q_shape[:-1], -1)
    #         k_shape = k.shape
    #         k = torch.stack([k, k], dim=-2).reshape(*k_shape[:-1], -1)
        
    #     if self.config.rope_config['imag'] and self.config.rope_config['imag_mode'] == 'imagh':
    #         assert not self.config.rope_config['1d']
    #         batch, num_attention_heads, slen, head_dim = q.shape
    #         q_rope = q * cos + rotate_half(q) * sin
    #         k_rope = k * cos + rotate_half(k) * sin
    #         q_rope[:, num_attention_heads // 2:, ...] = q[:, num_attention_heads // 2:, ...] * sin - rotate_half(q[:, num_attention_heads // 2:, ...]) * cos
    #     elif self.config.rope_config['imag'] and self.config.rope_config['imag_mode'] == 'imago':
    #         assert not self.config.rope_config['1d']
    #         k_rope = k * cos + rotate_half(k) * sin
    #         q_rope = q * sin - rotate_half(q) * cos
    #     else:
    #         if self.config.rope_config['1d'] and self.config.rope_config['1d_mode'] in ['1d2', '1d1']:
    #             q_rope = torch.cat([q * cos, q * sin], dim=-1)
    #             k_rope = torch.cat([k * cos, k * sin], dim=-1)
    #             q_imag = torch.cat([q * sin, -q * cos], dim=-1) if self.config.rope_config['imag'] else None
    #         else:  # '1dl'
    #             q_rope = q * cos + rotate_half(q) * sin
    #             k_rope = k * cos + rotate_half(k) * sin
    #             q_imag = q * sin - rotate_half(q) * cos if self.config.rope_config['imag'] else None
    #         if q_imag is not None:
    #             batch, num_attention_heads, slen, head_dim = q_rope.shape
    #             q_rope = q_rope[:, :, None, :, :].expand(batch, num_attention_heads, 2, slen, head_dim)
    #             q_rope[:, :, 1, :, :] = q_imag
    #             q_rope = q_rope.reshape(batch, num_attention_heads * 2, slen, head_dim)

    #     #   (q0 cost - q1 sint) * (k0 coss - k1 sins) + (q1 cost + q0 sint) * (k1 coss + k0 sins)
    #     # = (q0 k0 + q1 k1) * (cost coss + sint sins) + (q0 k1 - q1 k0)(sint coss - cost sins)
    #     # = (q · k) cos(t-s) + (q * k) sin(t-s)
    #     #   (q0 sint + q1 cost) * (k0 coss - k1 sins) + (q1 sint - q0 cost) * (k1 coss + k0 sins)
    #     # = (q0 k0 + q1 k1) * (sint coss - cost sins) - (q0 k1 - q1 k0)(cost coss + sint sins)
    #     # = (q · k) sin(t-s) - (q * k) cos(t-s)

    #     return q_rope, k_rope

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]

        query_states = self.q_proj(hidden_states).view((*input_shape, -1, self.qk_head_dim)).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view((*input_shape, -1, self.qk_head_dim)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view((*input_shape, -1, self.head_dim)).transpose(1, 2)

        # cos, sin = position_embeddings
        # query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # if self.config.rope_config['1d'] and self.config.rope_config['1d_mode'] in ['1d2', '1dl']:
        #     value_states = torch.cat([value_states, value_states], dim=-1)

        query_len, key_len = query_states.shape[-2], key_states.shape[-2]  # could be different if layer_past not None
        all_len = max(query_len, key_len)

        query_states = self.pos_emb(query_states, all_len, layer_idx=self.layer_idx)
        key_states = self.pos_emb(key_states, all_len, layer_idx=self.layer_idx)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        # if self.config.rope_config['1d'] and self.config.rope_config['1d_mode'] in ['1d2', '1dl']:
        #     attn_output = attn_output[..., :self.head_dim]

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        # position_embeddings = None

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            if isinstance(attention_mask, BlockMask):
                return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to place the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.ignore_index = config.ignore_index

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, 
                                      ignore_index=self.ignore_index, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @can_return_tuple
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> SequenceClassifierOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        transformer_outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = transformer_outputs.last_hidden_state
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
The Llama Model transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForQuestionAnswering(LlamaPreTrainedModel):
    base_model_prefix = "transformer"

    # Copied from transformers.models.bloom.modeling_bloom.BloomForQuestionAnswering.__init__ with Bloom->Llama
    def __init__(self, config):
        super().__init__(config)
        self.transformer = LlamaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    @can_return_tuple
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> QuestionAnsweringModelOutput:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """

        outputs: BaseModelOutputWithPast = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs.last_hidden_state

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss = None
        if start_positions is not None and end_positions is not None:
            loss = self.loss_function(start_logits, end_logits, start_positions, end_positions, **kwargs)

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    The Llama Model transformer with a token classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForTokenClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @can_return_tuple
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> TokenClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
    "LlamaForSequenceClassification",
    "LlamaForQuestionAnswering",
    "LlamaForTokenClassification",
]