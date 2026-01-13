"""
Standard GPT-2 implementation and baseline Transformer models for comparison.

This module contains:
- GPT2: Original GPT-2 implementation
- BaseTransformer: Size-matched Transformer for fair comparison with Grassmann
- BaseTransformerBlock: Standard transformer block used by BaseTransformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Small Transformer Baseline (size-matched to Grassmann)
# -----------------------------------------------------------------------------

class BaseTransformerBlock(nn.Module):
    """Standard transformer block. Uses Flash Attention via F.scaled_dot_product_attention."""

    def __init__(self, model_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        """Initialize the transformer block.

        Args:
            model_dim (int): Dimension of the model embeddings.
            num_heads (int): Number of attention heads.
            ff_dim (int): Dimension of the feed-forward network.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        
        self.ln1 = nn.LayerNorm(model_dim)
        
        # QKV projections (combined for efficiency)
        self.qkv_proj = nn.Linear(model_dim, 3 * model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)
        
        self.ln2 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass. Uses Flash Attention via F.scaled_dot_product_attention (5-6x faster).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, model_dim).
            attn_mask (torch.Tensor, optional): Attention mask for causal attention (unused, kept for compatibility).

        Returns:
            torch.Tensor: Output tensor after self-attention and FFN.
        """
        batch_size, seq_len, _ = x.shape
        
        # Self-attention with residual
        normed = self.ln1(x)
        
        # Compute Q, K, V
        qkv = self.qkv_proj(normed)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Flash Attention (automatically enabled with bfloat16)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        # Reshape and project
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, self.model_dim)
        attn_out = self.out_proj(attn_out)
        x = x + self.dropout(attn_out)

        # FFN with residual
        normed = self.ln2(x)
        x = x + self.ffn(normed)

        return x


class BaseTransformer(nn.Module):
    """Size-matched Transformer for fair comparison with Grassmann."""

    def __init__(
        self,
        vocab_size: int = 50257,
        max_seq_len: int = 256,
        model_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = None,
        dropout: float = 0.1,
        tie_weights: bool = True,
    ):
        """Initialize the size-matched transformer model.

        Args:
            vocab_size (int): Size of the vocabulary.
            max_seq_len (int): Maximum sequence length.
            model_dim (int): Dimension of the model embeddings.
            num_layers (int): Number of transformer layers.
            num_heads (int): Number of attention heads.
            ff_dim (int, optional): Dimension of the feed-forward network.
            dropout (float): Dropout probability.
            tie_weights (bool): Whether to tie input and output embeddings.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.model_dim = model_dim

        ff_dim = ff_dim or 4 * model_dim

        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.position_embedding = nn.Embedding(max_seq_len, model_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            BaseTransformerBlock(model_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(model_dim)
        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize model weights.

        Args:
            module: The module to initialize.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids, labels=None):
        """Forward pass of the transformer model.

        Args:
            input_ids (torch.Tensor): Input token ids of shape (batch_size, seq_len).
            labels (torch.Tensor, optional): Target token ids for loss computation.

        Returns:
            tuple: (logits, loss) where loss is None if labels not provided.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(torch.arange(seq_len, device=device))
        hidden_states = self.embedding_dropout(tok_emb + pos_emb)

        for block in self.blocks:
            hidden_states = block(hidden_states)

        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    def get_num_params(self) -> int:
        """Get the total number of parameters in the model.

        Returns:
            int: Number of parameters.
        """
        return sum(p.numel() for p in self.parameters())

