"""
Grassmann Flow layers architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple


class PluckerEncoder(nn.Module):
    """
    Plucker coordinate encoder per paper.

    p_ij^(delta)(t) = z_{t,i} * z_{t-delta,j} - z_{t,j} * z_{t-delta,i}

    With L2 normalization: p_hat = p / max(||p||_2, eps)
    """

    def __init__(self, reduced_dim: int, eps: float = 1e-8):
        """Initialize the Plucker coordinate encoder.

        Args:
            reduced_dim (int): Dimension of the reduced vectors (r in paper).
            eps (float): Epsilon for L2 normalization.
        """
        super().__init__()
        self.reduced_dim = reduced_dim
        self.eps = eps
        self.plucker_dim = reduced_dim * (reduced_dim - 1) // 2

        # Create index tensors for upper triangular elements
        indices_i, indices_j = [], []
        for i in range(reduced_dim):
            for j in range(i + 1, reduced_dim):
                indices_i.append(i)
                indices_j.append(j)
        self.register_buffer('idx_i', torch.tensor(indices_i, dtype=torch.long))
        self.register_buffer('idx_j', torch.tensor(indices_j, dtype=torch.long))

    def forward(self, z_current: torch.Tensor, z_past: torch.Tensor) -> torch.Tensor:
        """
        Compute L2-normalized Plucker coordinates.

        Args:
            z_current: (batch, seq, r) current position vectors
            z_past: (batch, seq, r) past position vectors

        Returns:
            L2-normalized Plucker coordinates (batch, seq, plucker_dim)
        """
        # p_ij = z_current_i * z_past_j - z_current_j * z_past_i
        p = (z_current[..., self.idx_i] * z_past[..., self.idx_j] -
             z_current[..., self.idx_j] * z_past[..., self.idx_i])

        # L2 normalize: p_hat = p / max(||p||_2, eps)
        norm = torch.norm(p, dim=-1, keepdim=True)
        p_normalized = p / torch.clamp(norm, min=self.eps)

        return p_normalized


class CausalGrassmannMixing(nn.Module):
    """
    Causal Grassmann Mixing layer - exact paper implementation.

    Paper's forward pass:
    1. z_t = W_red * h_t + b_red
    2. For each delta: p_ij = z_t_i * z_{t-delta}_j - z_t_j * z_{t-delta}_i
    3. p_hat = p / max(||p||_2, eps)  [L2 normalize]
    4. g_t^(delta) = W_plu * p_hat + b_plu  [project each window]
    5. g_t = average(g_t^(delta)) across valid deltas
    6. alpha = sigmoid(W_gate * [h_t; g_t] + b_gate)  [gate from concat]
    7. h_mix = alpha * h_t + (1-alpha) * g_t  [blend, not add]
    8. Apply LayerNorm
    """

    def __init__(
        self,
        model_dim: int,
        reduced_dim: int = 32,  # Paper uses r=32
        window_sizes: List[int] = None,
        dropout: float = 0.1,
    ):
        """Initialize the Causal Grassmann Mixing layer.

        Args:
            model_dim (int): Dimension of the model embeddings.
            reduced_dim (int): Dimension of the reduced vectors (r=32 in paper).
            window_sizes (List[int], optional): Window sizes for looking back.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.model_dim = model_dim
        self.reduced_dim = reduced_dim
        # Paper: {1, 2, 4, 8, 12, 16} for 6-layer
        self.window_sizes = window_sizes or [1, 2, 4, 8, 12, 16]
        self.num_windows = len(self.window_sizes)
        self.plucker_dim = reduced_dim * (reduced_dim - 1) // 2

        # Step 1: Linear reduction z = W_red * h + b_red
        self.W_red = nn.Linear(model_dim, reduced_dim)

        # Plucker encoder with L2 normalization
        self.plucker = PluckerEncoder(reduced_dim)

        # Step 4: Project Plucker to model dim: g = W_plu * p + b_plu
        self.W_plu = nn.Linear(self.plucker_dim, model_dim)

        # Step 6: Gate from concatenated [h; g]
        # Input is 2*model_dim, output is model_dim
        self.W_gate = nn.Linear(2 * model_dim, model_dim)

        # Step 8: LayerNorm after mixing
        self.layer_norm = nn.LayerNorm(model_dim)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for the Grassmann mixing layer."""
        # Paper doesn't specify init, use reasonable defaults
        nn.init.xavier_uniform_(self.W_red.weight)
        nn.init.zeros_(self.W_red.bias)
        nn.init.xavier_uniform_(self.W_plu.weight)
        nn.init.zeros_(self.W_plu.bias)
        nn.init.xavier_uniform_(self.W_gate.weight)
        nn.init.zeros_(self.W_gate.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply Causal Grassmann Mixing per paper.

        Args:
            hidden_states: (batch, seq_len, model_dim)

        Returns:
            Mixed hidden states (batch, seq_len, model_dim)
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Step 1: Reduce to low dimension
        z = self.W_red(hidden_states)  # (batch, seq_len, reduced_dim)

        # Steps 2-5: Compute Plucker features for all windows, then average
        geo_accum = torch.zeros(batch_size, seq_len, self.model_dim, device=device, dtype=dtype)
        counts = torch.zeros(batch_size, seq_len, 1, device=device, dtype=dtype)

        for delta in self.window_sizes:
            if delta >= seq_len:
                continue

            # Current positions (from delta onwards can look back)
            z_current = z[:, delta:, :]   # (batch, seq_len-delta, r)
            z_past = z[:, :-delta, :]     # (batch, seq_len-delta, r)

            # Steps 2-3: Plucker coordinates with L2 normalization
            p_hat = self.plucker(z_current, z_past)  # (batch, seq_len-delta, plucker_dim)

            # Step 4: Project to model dim
            g_delta = self.W_plu(p_hat)  # (batch, seq_len-delta, model_dim)

            # Accumulate for averaging
            geo_accum[:, delta:, :] = geo_accum[:, delta:, :] + g_delta
            counts[:, delta:, :] = counts[:, delta:, :] + 1

        # Step 5: Average across valid windows
        counts = counts.clamp(min=1)
        g = geo_accum / counts  # (batch, seq_len, model_dim)

        # Step 6: Gating - concatenate [h; g] then sigmoid
        concat = torch.cat([hidden_states, g], dim=-1)  # (batch, seq_len, 2*model_dim)
        alpha = torch.sigmoid(self.W_gate(concat))  # (batch, seq_len, model_dim)

        # Step 7: Blend (not add!) - h_mix = alpha * h + (1-alpha) * g
        h_mix = alpha * hidden_states + (1 - alpha) * g

        # Step 8: LayerNorm and dropout
        output = self.layer_norm(h_mix)
        output = self.dropout(output)

        return output


class GrassmannBlock(nn.Module):
    """
    Transformer block with Causal Grassmann Mixing.

    Structure: LN -> Grassmann -> Residual -> LN -> FFN -> Residual
    """

    def __init__(
        self,
        model_dim: int,
        reduced_dim: int = 32,
        ff_dim: int = None,
        window_sizes: List[int] = None,
        dropout: float = 0.1,
    ):
        """Initialize the Grassmann block.

        Args:
            model_dim (int): Dimension of the model embeddings.
            reduced_dim (int): Dimension of the reduced vectors.
            ff_dim (int, optional): Dimension of the feed-forward network.
            window_sizes (List[int], optional): Window sizes for Grassmann mixing.
            dropout (float): Dropout probability.
        """
        super().__init__()
        ff_dim = ff_dim or 4 * model_dim

        self.ln1 = nn.LayerNorm(model_dim)
        self.grassmann = CausalGrassmannMixing(
            model_dim=model_dim,
            reduced_dim=reduced_dim,
            window_sizes=window_sizes,
            dropout=dropout,
        )

        self.ln2 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Grassmann block.

        Args:
            hidden_states (torch.Tensor): Input hidden states.

        Returns:
            torch.Tensor: Output hidden states after Grassmann mixing and FFN.
        """
        # Grassmann mixing with residual
        normed = self.ln1(hidden_states)
        hidden_states = hidden_states + self.grassmann(normed)

        # FFN with residual
        normed = self.ln2(hidden_states)
        hidden_states = hidden_states + self.ffn(normed)

        return hidden_states


class GrassmannGPT(nn.Module):
    """
    GrassmannGPT with exact paper architecture (v4).

    Key differences from v3:
    - reduced_dim = 32 (paper's value)
    - window_sizes = [1, 2, 4, 8, 12, 16]
    - Blend gating: alpha * h + (1-alpha) * g
    - Gate input: concatenate [h; g]
    - L2 normalize Plucker before projection
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        max_seq_len: int = 1024,
        model_dim: int = 768,
        num_layers: int = 12,
        reduced_dim: int = 32,  # Paper's value
        ff_dim: int = None,
        window_sizes: List[int] = None,
        dropout: float = 0.1,
        tie_weights: bool = True,
    ):
        """Initialize the GrassmannGPT model.

        Args:
            vocab_size (int): Size of the vocabulary.
            max_seq_len (int): Maximum sequence length.
            model_dim (int): Dimension of the model embeddings.
            num_layers (int): Number of layers.
            reduced_dim (int): Dimension of the reduced vectors (r=32 in paper).
            ff_dim (int, optional): Dimension of the feed-forward network.
            window_sizes (List[int], optional): Window sizes for Grassmann mixing.
            dropout (float): Dropout probability.
            tie_weights (bool): Whether to tie input and output embeddings.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.model_dim = model_dim

        ff_dim = ff_dim or 4 * model_dim
        # Paper's window sizes for 6-layer (use same for all)
        window_sizes = window_sizes or [1, 2, 4, 8, 12, 16]

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.position_embedding = nn.Embedding(max_seq_len, model_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        # Blocks
        self.blocks = nn.ModuleList([
            GrassmannBlock(
                model_dim=model_dim,
                reduced_dim=reduced_dim,
                ff_dim=ff_dim,
                window_sizes=window_sizes,
                dropout=dropout,
            )
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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of the GrassmannGPT model.

        Args:
            input_ids (torch.Tensor): Input token ids.
            attention_mask (torch.Tensor, optional): Attention mask.
            labels (torch.Tensor, optional): Target token ids for loss.

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

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate text autoregressively.

        Args:
            input_ids (torch.Tensor): Input token ids.
            max_new_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature.
            top_k (int, optional): Top-k sampling.

        Returns:
            torch.Tensor: Generated token ids.
        """
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
