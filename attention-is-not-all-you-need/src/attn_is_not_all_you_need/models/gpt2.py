"""
Standard GPT-2 implementation for baseline comparison.

This is a clean implementation matching the structure of GrassmannGPT
to ensure fair comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class CausalSelfAttention(nn.Module):


    def __init__(
        self,
        model_dim: int,
        num_heads: int = 12,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        assert model_dim % num_heads == 0

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V projections (combined for efficiency)
        self.qkv_proj = nn.Linear(model_dim, 3 * model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(
                1, 1, max_seq_len, max_seq_len
            )
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Compute Q, K, V
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        attn_weights = attn_weights.masked_fill(causal_mask == 0, float('-inf'))

        # Optional padding mask
        if attention_mask is not None:
            # attention_mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.model_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output


class TransformerBlock(nn.Module):
  
    def __init__(
        self,
        model_dim: int,
        num_heads: int = 12,
        ff_dim: int = None,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        ff_dim = ff_dim or 4 * model_dim

        # Self-attention
        self.attention = CausalSelfAttention(
            model_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
            nn.Dropout(dropout),
        )

        # Layer norms (pre-norm architecture like GPT-2)
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm attention with residual
        normed = self.ln1(hidden_states)
        attn_out = self.attention(normed, attention_mask)
        hidden_states = hidden_states + attn_out

        # Pre-norm FFN with residual
        normed = self.ln2(hidden_states)
        ff_out = self.ffn(normed)
        hidden_states = hidden_states + ff_out

        return hidden_states


class GPT2(nn.Module):
    """
    Standard GPT-2 .
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        max_seq_len: int = 1024,
        model_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        ff_dim: int = None,
        dropout: float = 0.1,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.model_dim = model_dim
        self.num_layers = num_layers

        ff_dim = ff_dim or 4 * model_dim

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.position_embedding = nn.Embedding(max_seq_len, model_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                model_dim=model_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                max_seq_len=max_seq_len,
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(model_dim)

        # LM head
        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)

        # Tie weights
        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
     
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token embeddings
        tok_emb = self.token_embedding(input_ids)

        # Position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)

        # Combined embeddings
        hidden_states = self.embedding_dropout(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        # LM head
        logits = self.lm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    def get_num_params(self) -> int:
       
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Simple greedy/sampling generation."""
        for _ in range(max_new_tokens):
            # Truncate to max_seq_len
            idx_cond = input_ids[:, -self.max_seq_len:]

            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Optional top-k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
