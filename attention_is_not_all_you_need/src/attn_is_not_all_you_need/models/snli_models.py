"""
SNLI Classification Models for Paper Reproduction.

Implements:
- Transformer head: 2-layer Transformer-style classifier  
- Grassmann-Plucker head: Grassmann mixing module classifier

Both use DistilBERT-base-uncased backbone with frozen or fine-tuned features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel
from typing import List


class TransformerNLIHead(nn.Module):
    """
    Transformer-based NLI classifier head.
    
    Paper spec: 2-layer Transformer-style classifier with self-attention
    over pooled features and final linear layer for 3-way classification.
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,  # DistilBERT hidden size
        model_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 512,
        dropout: float = 0.1,
        num_classes: int = 3,
    ):
        """Initialize the Transformer NLI head.
        
        Args:
            hidden_dim: Input dimension from backbone (768 for DistilBERT)
            model_dim: Internal model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            ff_dim: Feed-forward dimension
            dropout: Dropout probability
            num_classes: Number of output classes (3 for SNLI)
        """
        super().__init__()
        
        # Project backbone features to model_dim
        self.input_projection = nn.Linear(hidden_dim, model_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_classes)
        )
        
    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            pooled_output: (batch_size, hidden_dim) pooled features from backbone
            
        Returns:
            logits: (batch_size, num_classes) classification logits
        """
        # Project to model_dim
        x = self.input_projection(pooled_output)  # (batch, model_dim)
        x = x.unsqueeze(1)  # (batch, 1, model_dim) - add sequence dimension
        
        # Apply transformer layers
        x = self.transformer(x)  # (batch, 1, model_dim)
        
        # Pool and classify
        x = x.squeeze(1)  # (batch, model_dim)
        logits = self.classifier(x)  # (batch, num_classes)
        
        return logits


class GrassmannPluckerNLIHead(nn.Module):
    """
    Grassmann-Plucker NLI classifier head.
    
    Paper spec:
    - dproj = 64 (reduced dimension)
    - window = 8, stride = 8
    - dmodel = 256
    - 2 mixing layers
    - 4 mixing heads (for grouping pairs)
    - dff = 512
    - dropout = 0.1
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,  # DistilBERT hidden size  
        dproj: int = 64,  # Reduced dimension for Grassmann
        dmodel: int = 256,
        window_size: int = 8,
        stride: int = 8,
        num_layers: int = 2,
        num_heads: int = 4,
        dff: int = 512,
        dropout: float = 0.1,
        num_classes: int = 3,
    ):
        """Initialize the Grassmann-Plucker NLI head.
        
        Args:
            hidden_dim: Input dimension from backbone (768 for DistilBERT)
            dproj: Reduced dimension for Grassmann projection
            dmodel: Model dimension
            window_size: Window size for Grassmann mixing
            stride: Stride for windowing
            num_layers: Number of mixing layers
            num_heads: Number of mixing heads
            dff: Feed-forward dimension
            dropout: Dropout probability
            num_classes: Number of output classes (3 for SNLI)
        """
        super().__init__()
        
        self.window_size = window_size
        self.stride = stride
        self.dproj = dproj
        self.plucker_dim = dproj * (dproj - 1) // 2
        
        # Project backbone features to dmodel
        self.input_projection = nn.Linear(hidden_dim, dmodel)
        
        # Grassmann mixing layers
        self.mixing_layers = nn.ModuleList([
            GrassmannMixingLayer(
                model_dim=dmodel,
                reduced_dim=dproj,
                num_heads=num_heads,
                ff_dim=dff,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(dmodel, dmodel),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dmodel, num_classes)
        )
        
    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            pooled_output: (batch_size, hidden_dim) pooled features from backbone
            
        Returns:
            logits: (batch_size, num_classes) classification logits
        """
        # Project to dmodel
        x = self.input_projection(pooled_output)  # (batch, dmodel)
        x = x.unsqueeze(1)  # (batch, 1, dmodel) - add sequence dimension
        
        # Apply Grassmann mixing layers
        for layer in self.mixing_layers:
            x = layer(x)  # (batch, 1, dmodel)
        
        # Pool and classify
        x = x.squeeze(1)  # (batch, dmodel)
        logits = self.classifier(x)  # (batch, num_classes)
        
        return logits


class GrassmannMixingLayer(nn.Module):
    """Single Grassmann mixing layer for NLI head."""
    
    def __init__(
        self,
        model_dim: int,
        reduced_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.model_dim = model_dim
        self.reduced_dim = reduced_dim
        self.plucker_dim = reduced_dim * (reduced_dim - 1) // 2
        
        # Projection to reduced dimension
        self.to_reduced = nn.Linear(model_dim, reduced_dim)
        
        # Plucker coordinate encoder
        self.plucker_encoder = PluckerEncoderSimple(reduced_dim)
        
        # Project Plucker coordinates back to model_dim
        self.from_plucker = nn.Linear(self.plucker_dim, model_dim)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(2 * model_dim, model_dim),
            nn.Sigmoid()
        )
        
        # Layer norm and FFN
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: (batch, seq_len, model_dim)
            
        Returns:
            output: (batch, seq_len, model_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to reduced dimension
        z = self.to_reduced(x)  # (batch, seq_len, reduced_dim)
        
        # Compute Plucker coordinates (use identity for simplicity in classification)
        # For classification, we can use self-attention-like mechanism
        p = self.plucker_encoder(z, z)  # (batch, seq_len, plucker_dim)
        
        # Project back to model_dim
        g = self.from_plucker(p)  # (batch, seq_len, model_dim)
        
        # Gating
        concat = torch.cat([x, g], dim=-1)  # (batch, seq_len, 2*model_dim)
        alpha = self.gate(concat)  # (batch, seq_len, model_dim)
        mixed = alpha * x + (1 - alpha) * g  # (batch, seq_len, model_dim)
        
        # Residual and layer norm
        x = self.ln1(x + mixed)
        
        # FFN with residual
        x = self.ln2(x + self.ffn(x))
        
        return x


class PluckerEncoderSimple(nn.Module):
    """Simplified Plucker encoder for NLI."""
    
    def __init__(self, reduced_dim: int, eps: float = 1e-8):
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
        """Compute L2-normalized Plucker coordinates.
        
        Args:
            z_current: (batch, seq, r) current vectors
            z_past: (batch, seq, r) past vectors
            
        Returns:
            p_normalized: (batch, seq, plucker_dim)
        """
        # p_ij = z_current_i * z_past_j - z_current_j * z_past_i
        p = (z_current[..., self.idx_i] * z_past[..., self.idx_j] -
             z_current[..., self.idx_j] * z_past[..., self.idx_i])
        
        # L2 normalize
        norm = torch.norm(p, dim=-1, keepdim=True)
        p_normalized = p / torch.clamp(norm, min=self.eps)
        
        return p_normalized


class SNLIModel(nn.Module):
    """Complete SNLI model with DistilBERT backbone and classification head."""
    
    def __init__(
        self,
        backbone_name: str = "distilbert-base-uncased",
        head_type: str = "transformer",
        freeze_backbone: bool = False,
        **head_kwargs
    ):
        """Initialize SNLI model.
        
        Args:
            backbone_name: Name of the DistilBERT model
            head_type: Type of head ('transformer' or 'grassmann')
            freeze_backbone: Whether to freeze backbone weights
            **head_kwargs: Additional arguments for the head
        """
        super().__init__()
        
        # Load DistilBERT backbone
        self.backbone = DistilBertModel.from_pretrained(backbone_name)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Create classification head
        hidden_dim = self.backbone.config.hidden_size  # 768
        
        if head_type == "transformer":
            self.head = TransformerNLIHead(hidden_dim=hidden_dim, **head_kwargs)
        elif head_type == "grassmann":
            self.head = GrassmannPluckerNLIHead(hidden_dim=hidden_dim, **head_kwargs)
        else:
            raise ValueError(f"Unknown head_type: {head_type}")
        
        self.head_type = head_type
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        """Forward pass.
        
        Args:
            input_ids: (batch, seq_len) input token ids
            attention_mask: (batch, seq_len) attention mask
            labels: (batch,) labels for loss computation
            
        Returns:
            dict with 'logits' and optional 'loss'
        """
        # Get backbone features
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_dim)
        
        # Classification head
        logits = self.head(pooled_output)  # (batch, 3)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        return {"logits": logits, "loss": loss}
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
