"""
Evaluation Metrics
==================

MSE and MAE computed over the full prediction horizon and at standard
checkpoints (96, 192, 336, 720).
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional


STANDARD_HORIZONS = [96, 192, 336, 720]


def compute_metrics(
    y_hat: torch.Tensor,
    y: torch.Tensor,
) -> Dict[str, float]:
    """Compute MSE and MAE over the full prediction.
    
    Args:
        y_hat: Predictions [B, L, N]
        y: Targets [B, L, N]
    
    Returns:
        Dictionary with 'mse' and 'mae'
    """
    mse = F.mse_loss(y_hat, y).item()
    mae = F.l1_loss(y_hat, y).item()
    return {'mse': mse, 'mae': mae}


def per_horizon_metrics(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    horizons: Optional[List[int]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute MSE and MAE at specific horizon checkpoints.
    
    Standard LTSF protocol: evaluate at cumulative horizons [96, 192, 336, 720].
    That is, MSE@96 means MSE over timesteps [0:96].
    
    Args:
        y_hat: Predictions [B, L, N]
        y: Targets [B, L, N]
        horizons: List of horizons to evaluate (default: [96, 192, 336, 720])
    
    Returns:
        Dict mapping horizon → {'mse': float, 'mae': float}
    """
    if horizons is None:
        horizons = STANDARD_HORIZONS
    
    L = y_hat.shape[1]
    results = {}
    
    for h in horizons:
        if h <= L:
            mse = F.mse_loss(y_hat[:, :h, :], y[:, :h, :]).item()
            mae = F.l1_loss(y_hat[:, :h, :], y[:, :h, :]).item()
            results[h] = {'mse': mse, 'mae': mae}
    
    return results


def mean_horizon_metric(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    metric: str = 'mse',
    horizons: Optional[List[int]] = None,
) -> float:
    """Average metric across standard horizons.
    
    This is the primary ranking metric for LTSF papers:
    mean of MSE@{96, 192, 336, 720}.
    
    Args:
        y_hat: Predictions [B, L, N]
        y: Targets [B, L, N]
        metric: 'mse' or 'mae'
        horizons: List of horizons
    
    Returns:
        Mean metric value
    """
    per_h = per_horizon_metrics(y_hat, y, horizons)
    values = [v[metric] for v in per_h.values()]
    return sum(values) / len(values) if values else float('nan')
