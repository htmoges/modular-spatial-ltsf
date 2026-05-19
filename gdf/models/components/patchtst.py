"""
PatchTST Backbone
=================

Channel-Independent PatchTST for long-term time series forecasting.
Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" (ICLR 2023)

Architecture:
    Input: (B, seq_len, N)
    → RevIN normalization
    → Patching: unfold into (B*N, patch_num, patch_len)
    → Linear projection to d_model
    → Positional encoding (learnable)
    → Transformer encoder (n_layers × [Self-Attention + FFN])
    → Flatten + Linear head → (B*N, pred_len)
    → RevIN denormalization
    Output: (B, pred_len, N)

Channel-independent: each variable processed independently through shared weights.
This matches the CI protocol of DLinear/NLinear for fair comparison.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    """Reversible Instance Normalization.
    
    Kim et al., "Reversible Instance Normalization for Accurate Time-Series Forecasting
    against Distribution Shift" (ICLR 2022)
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))
        # Store statistics for denormalization
        self.mean = None
        self.stdev = None
    
    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Args:
            x: (B, T, N)
            mode: 'norm' or 'denorm'
        """
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self.stdev + self.mean
        return x


class PatchTST(nn.Module):
    """PatchTST: Patch Time Series Transformer (Channel-Independent).
    
    Matches the (B, seq_len, N) → (B, pred_len, N) interface required
    by the GDF backbone registry.
    
    Args:
        seq_len: Input sequence length
        pred_len: Prediction horizon
        n_features: Number of variables (N) — used for RevIN
        patch_len: Length of each patch
        stride: Stride between patches
        d_model: Transformer model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        n_layers: Number of transformer encoder layers
        dropout: Dropout rate
        fc_dropout: Dropout on final linear head
        head_dropout: Dropout on flatten head
        activation: Activation function ('gelu' or 'relu')
        norm: Normalization type ('BatchNorm' or 'LayerNorm')
        use_revin: Whether to use RevIN
        padding_patch: Whether to pad input for one extra patch ('end' or None)
    
    Shape:
        Input:  (B, seq_len, N)
        Output: (B, pred_len, N)
    """
    
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_features: int = 1,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 16,
        d_ff: int = 256,
        n_layers: int = 3,
        dropout: float = 0.2,
        fc_dropout: float = 0.2,
        head_dropout: float = 0.0,
        activation: str = 'gelu',
        norm: str = 'LayerNorm',
        use_revin: bool = True,
        padding_patch: str = 'end',
        **kwargs,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.use_revin = use_revin
        self.padding_patch = padding_patch
        
        # RevIN
        if use_revin:
            self.revin = RevIN(n_features, affine=True)
        
        # Compute number of patches
        self.patch_num = int((seq_len - patch_len) / stride + 1)
        if padding_patch == 'end':
            self.padding_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1
        
        # Patch embedding: patch_len → d_model
        self.W_P = nn.Linear(patch_len, d_model)
        
        # Positional encoding (learnable)
        self.W_pos = nn.Parameter(torch.zeros(1, self.patch_num, d_model))
        nn.init.uniform_(self.W_pos, -0.02, 0.02)
        
        self.dropout_embed = nn.Dropout(dropout)
        
        # Transformer encoder  
        encoder_norm = nn.LayerNorm(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=False,  # post-norm like original PatchTST
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=encoder_norm,
        )
        
        # Flatten head: d_model * patch_num → pred_len
        head_nf = d_model * self.patch_num
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),         # [B*N, d_model * patch_num]
            nn.Linear(head_nf, pred_len),
            nn.Dropout(head_dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, N)
        Returns:
            (B, pred_len, N)
        """
        B, T, N = x.shape
        
        # 1. RevIN normalization
        if self.use_revin:
            x = self.revin(x, 'norm')       # (B, T, N)
        
        # 2. Permute to (B, N, T) for CI processing
        x = x.permute(0, 2, 1)              # (B, N, T)
        
        # 3. Padding
        if self.padding_patch == 'end':
            x = self.padding_layer(x)        # (B, N, T+stride)
        
        # 4. Patching via unfold
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x: (B, N, patch_num, patch_len)
        
        # 5. Flatten B and N for channel-independent processing
        x = x.reshape(B * N, self.patch_num, self.patch_len)
        # x: (B*N, patch_num, patch_len)
        
        # 6. Patch embedding + positional encoding
        x = self.W_P(x)                      # (B*N, patch_num, d_model)
        x = self.dropout_embed(x + self.W_pos)
        
        # 7. Transformer encoder
        x = self.encoder(x)                  # (B*N, patch_num, d_model)
        
        # 8. Flatten head → prediction
        x = self.head(x)                     # (B*N, pred_len)
        
        # 9. Reshape back
        x = x.reshape(B, N, self.pred_len)   # (B, N, pred_len)
        x = x.permute(0, 2, 1)               # (B, pred_len, N)
        
        # 10. RevIN denormalization
        if self.use_revin:
            x = self.revin(x, 'denorm')
        
        return x
