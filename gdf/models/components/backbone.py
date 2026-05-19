"""
Temporal Backbones
==================

Clean implementations of temporal forecasting backbones.
Each backbone: (B, T_in, N) → (B, T_out, N)

Backbones:
  - DLinear: Decomposition + Linear (Zeng et al., AAAI 2023)
  - NLinear: Last-value normalization + Linear (Zeng et al., AAAI 2023)
  - Linear: Plain single linear layer (Zeng et al., AAAI 2023)
  - RLinear: RevIN + Linear (Li et al., 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Lazy import to avoid circular — PatchTST is in its own module
_PatchTST = None
def _get_patchtst():
    global _PatchTST
    if _PatchTST is None:
        from .patchtst import PatchTST
        _PatchTST = PatchTST
    return _PatchTST


# =============================================================================
# DLinear
# =============================================================================

class DLinear(nn.Module):
    """DLinear: Moving-average decomposition + separate linear projections.
    
    Supports both shared (individual=False) and per-feature (individual=True)
    projections. Shared is default to match Lite-STGNN.
    
    Args:
        seq_len: Input sequence length
        pred_len: Prediction horizon
        n_features: Number of input features (required for individual=True)
        individual: If True, learn per-feature projections
        kernel_size: Moving average kernel for trend/seasonal decomposition
    
    Shape:
        Input:  (B, seq_len, N)
        Output: (B, pred_len, N)
    """
    
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_features: int = 1,
        individual: bool = False,
        kernel_size: int = 25,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.individual = individual
        self.kernel_size = kernel_size
        
        if individual:
            self.Linear_Trend = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(n_features)
            ])
            self.Linear_Seasonal = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(n_features)
            ])
        else:
            self.Linear_Trend = nn.Linear(seq_len, pred_len)
            self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
    
    @staticmethod
    def moving_average(x: torch.Tensor, kernel_size: int = 25) -> torch.Tensor:
        """Moving average for trend extraction."""
        if kernel_size <= 1:
            return x
        
        pad = (kernel_size - 1) // 2
        weight = torch.ones(1, 1, kernel_size, device=x.device) / kernel_size
        
        b, l, n = x.shape
        x_ = x.permute(0, 2, 1).reshape(b * n, 1, l)
        trend = torch.conv1d(
            F.pad(x_, (pad, pad), mode='replicate'),
            weight
        )
        trend = trend.squeeze(1).reshape(b, n, l).permute(0, 2, 1)
        return trend
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self.moving_average(x, self.kernel_size)
        seasonal = x - trend
        
        if self.individual:
            out_t = torch.zeros(x.size(0), self.pred_len, self.n_features, device=x.device)
            out_s = torch.zeros(x.size(0), self.pred_len, self.n_features, device=x.device)
            for i in range(self.n_features):
                out_t[:, :, i] = self.Linear_Trend[i](trend[:, :, i])
                out_s[:, :, i] = self.Linear_Seasonal[i](seasonal[:, :, i])
        else:
            # Shared projection: transpose to (B, N, T) → project → transpose back
            out_t = self.Linear_Trend(trend.transpose(1, 2)).transpose(1, 2)
            out_s = self.Linear_Seasonal(seasonal.transpose(1, 2)).transpose(1, 2)
        
        return out_t + out_s


# =============================================================================
# RLinear
# =============================================================================

class RLinear(nn.Module):
    """RLinear: Reversible Instance Normalization + Linear projection.
    
    Simple but effective: normalize per-feature, project, denormalize.
    
    Args:
        seq_len: Input sequence length
        pred_len: Prediction horizon
        n_features: Number of features (unused, for API consistency)
    
    Shape:
        Input:  (B, seq_len, N)
        Output: (B, pred_len, N)
    """
    
    def __init__(self, seq_len: int, pred_len: int, n_features: int = 1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.Linear = nn.Linear(seq_len, pred_len)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Instance normalization (per sample, per feature)
        means = x.mean(dim=1, keepdim=True)
        stdev = torch.sqrt(x.var(dim=1, keepdim=True) + 1e-5)
        x_norm = (x - means) / stdev
        
        # Project: (B, N, T_in) → (B, N, T_out)
        y_norm = self.Linear(x_norm.transpose(1, 2)).transpose(1, 2)
        
        # Denormalize
        return y_norm * stdev + means




# =============================================================================
# NLinear
# =============================================================================

class NLinear(nn.Module):
    """NLinear: Normalization by last-value subtraction + Linear projection.
    
    "To boost the performance of Linear when there is a distribution shift,
    NLinear first subtracts the input by the last value of the sequence."
    — Zeng et al., AAAI 2023
    
    Supports both shared and per-feature projections.
    
    Args:
        seq_len: Input sequence length
        pred_len: Prediction horizon
        n_features: Number of input features (required for individual=True)
        individual: If True, learn per-feature projections
    
    Shape:
        Input:  (B, seq_len, N)
        Output: (B, pred_len, N)
    """
    
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_features: int = 1,
        individual: bool = False,
        **kwargs,  # Accept and ignore extra kwargs (e.g., kernel_size)
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.individual = individual
        
        if individual:
            self.Linear = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(n_features)
            ])
        else:
            self.Linear = nn.Linear(seq_len, pred_len)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N]
        seq_last = x[:, -1:, :].detach()  # [B, 1, N]
        x = x - seq_last
        
        if self.individual:
            output = torch.zeros(
                x.size(0), self.pred_len, self.n_features,
                dtype=x.dtype, device=x.device
            )
            for i in range(self.n_features):
                output[:, :, i] = self.Linear[i](x[:, :, i])
        else:
            output = self.Linear(x.transpose(1, 2)).transpose(1, 2)
        
        output = output + seq_last
        return output  # [B, pred_len, N]


# =============================================================================
# Linear (Plain)
# =============================================================================

class PlainLinear(nn.Module):
    """Linear: Just one linear layer. The simplest possible baseline.
    
    "It is just a one-layer linear model, but it outperforms Transformers."
    — Zeng et al., AAAI 2023
    
    Args:
        seq_len: Input sequence length
        pred_len: Prediction horizon
        n_features: Number of input features (required for individual=True)
        individual: If True, learn per-feature projections
    
    Shape:
        Input:  (B, seq_len, N)
        Output: (B, pred_len, N)
    """
    
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_features: int = 1,
        individual: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.individual = individual
        
        if individual:
            self.Linear = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(n_features)
            ])
        else:
            self.Linear = nn.Linear(seq_len, pred_len)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N]
        if self.individual:
            output = torch.zeros(
                x.size(0), self.pred_len, self.n_features,
                dtype=x.dtype, device=x.device
            )
            for i in range(self.n_features):
                output[:, :, i] = self.Linear[i](x[:, :, i])
        else:
            output = self.Linear(x.transpose(1, 2)).transpose(1, 2)
        
        return output  # [B, pred_len, N]


# =============================================================================
# Backbone Registry
# =============================================================================

BACKBONE_REGISTRY = {
    'dlinear': {
        'class': DLinear,
        'default_lr': 8e-4,
        'description': 'DLinear (decomposition + shared linear)',
    },
    'dlinear_individual': {
        'class': DLinear,
        'default_lr': 8e-4,
        'description': 'DLinear (decomposition + per-feature linear)',
        'individual': True,
    },
    'rlinear': {
        'class': RLinear,
        'default_lr': 8e-4,
        'description': 'RLinear (RevIN + linear)',
    },
    'nlinear': {
        'class': NLinear,
        'default_lr': 8e-4,
        'description': 'NLinear (last-value subtraction + linear)',
    },
    'nlinear_individual': {
        'class': NLinear,
        'default_lr': 8e-4,
        'description': 'NLinear (last-value subtraction + per-feature linear)',
        'individual': True,
    },
    'linear': {
        'class': PlainLinear,
        'default_lr': 8e-4,
        'description': 'Linear (plain single linear layer)',
    },
    'linear_individual': {
        'class': PlainLinear,
        'default_lr': 8e-4,
        'description': 'Linear (plain per-feature linear)',
        'individual': True,
    },
    'patchtst': {
        'class': 'PatchTST',  # string ref, resolved lazily
        'default_lr': 1e-4,
        'description': 'PatchTST (CI Transformer with patching + RevIN)',
    },
}


def create_backbone(name: str, seq_len: int, pred_len: int, n_features: int, **kwargs) -> nn.Module:
    """Factory function to create a backbone by name.
    
    Args:
        name: Backbone name (from BACKBONE_REGISTRY)
        seq_len: Input sequence length
        pred_len: Prediction horizon
        n_features: Number of features
        **kwargs: Override any registry defaults
    
    Returns:
        nn.Module backbone
    """
    if name not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone: {name}. Available: {list(BACKBONE_REGISTRY.keys())}")
    
    info = BACKBONE_REGISTRY[name]
    cls = info['class']
    
    # Resolve lazy string references
    if isinstance(cls, str) and cls == 'PatchTST':
        cls = _get_patchtst()
    
    # Merge registry defaults with kwargs
    init_kwargs = {k: v for k, v in info.items() if k not in ('class', 'default_lr', 'description')}
    init_kwargs.update(kwargs)
    
    return cls(seq_len=seq_len, pred_len=pred_len, n_features=n_features, **init_kwargs)
