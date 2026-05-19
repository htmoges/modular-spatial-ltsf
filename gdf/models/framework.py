"""Modular Forecasting Framework for Spatial Attribution Study.

Architecture (GraphDiffusionForecaster):
    1. Temporal backbone:         Y_base = backbone(X)
    2. Prediction head [opt]:     Y_ref  = temporal_head(Y_base)
    3. Pre-norm:                  Y_norm = LayerNorm(Y_ref)
    4. Spatial correction:        delta  = Σ_p (A^p - I) · Y_norm
    5. Post-norm + dropout:       delta  = Dropout(LayerNorm(delta))
    6. Band-wise gating:          delta  = σ(g) · delta
    7. Output:                    Y_hat  = Y_norm + delta
    8. Input residual [opt]:      Y_hat += proj(X)

Attribution configurations (controlled via CLI flags):
    L         : backbone only  (--no-use-temporal-head --no-use-input-residual --n-bands 0)
    L+PH      : + pred. head   (--use-temporal-head    --no-use-input-residual --n-bands 0)
    L+PH+R    : + input skip   (--use-temporal-head    --use-input-residual    --n-bands 0)
    L+PH+R+S  : + spatial      (--use-temporal-head    --use-input-residual    --n-bands 1)
"""

import torch
import torch.nn as nn
from typing import Dict

from .components.adjacency import LowRankTopKAdjacency, BandAdjacency
from .components.backbone import create_backbone


class ResidualGating(nn.Module):
    """Band-wise sigmoid gating for the spatial residual.

    Horizon bands (pred_len=720): [0,96) [96,192) [192,336) [336,720).
    Each band gets an independent gate σ(g_k) ∈ [0, 1].
    Initialised conservatively at g = -4.0 → σ ≈ 0.018 (nearly closed).
    """

    BAND_BOUNDARIES = [96, 192, 336]

    def __init__(self, pred_len: int, init_value: float = -4.0):
        super().__init__()
        self.pred_len = pred_len
        self.g_params = nn.Parameter(torch.full((4,), init_value))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        B, L, N = r.shape
        gates = torch.sigmoid(self.g_params)
        idx = torch.arange(L, device=r.device)
        band_idx = torch.zeros(L, dtype=torch.long, device=r.device)
        for boundary in self.BAND_BOUNDARIES:
            band_idx = band_idx + (idx >= boundary).long()
        band_idx = torch.clamp(band_idx, 0, 3)
        gate_h = gates[band_idx].view(1, L, 1)
        return gate_h * r


class TemporalHead(nn.Module):
    """Per-node 2-layer GELU MLP applied along the temporal dimension.

    Transforms backbone output before spatial processing.
    Applied per-node: [B, N, L] → [B, N, L].
    """

    def __init__(self, pred_len: int, ratio: float = 0.5, dropout: float = 0.1):
        super().__init__()
        hidden = max(1, int(pred_len * ratio))
        self.mlp = nn.Sequential(
            nn.Linear(pred_len, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, pred_len),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.mlp(y.transpose(1, 2)).transpose(1, 2)


class GraphDiffusionForecaster(nn.Module):
    """Graph Diffusion Forecasting model.

    Supports the four attribution configurations (L / L+PH / L+PH+R / L+PH+R+S)
    via `use_temporal_head`, `use_input_residual`, and `n_bands` flags.
    Set n_bands=0 to disable the spatial module entirely.

    Args:
        n_nodes: Number of variables (N).
        seq_len: Input window length (T).
        pred_len: Forecast horizon (L).
        backbone: Temporal backbone name ('dlinear', 'nlinear', 'rlinear', 'patchtst').
        backbone_individual: Per-feature linear projections in backbone.
        n_bands: Spatial bands. 1=single learned graph, 0=disabled.
        prop_orders: Multi-hop order (P=1 in paper).
        adj_rank: Low-rank embedding dimension for adjacency.
        adj_topk: Top-K connections per node.
        adj_tau: Softmax temperature for adjacency.
        self_loop_alpha: Self-loop weight.
        use_temporal_head: Enable prediction head MLP (PH component).
        temporal_head_ratio: Hidden dim ratio for prediction head.
        temporal_head_dropout: Dropout in prediction head.
        residual_dropout: Dropout on spatial delta.
        gate_init: Initial gating logit (σ(-4) ≈ 0.018).
        use_input_residual: Enable input skip connection (R component).
        input_residual_dropout: Dropout on input residual path.
        extra_backbone_kwargs: Passed directly to backbone (e.g. PatchTST kwargs).
    """

    def __init__(
        self,
        n_nodes: int,
        seq_len: int = 96,
        pred_len: int = 720,
        backbone: str = 'dlinear',
        backbone_individual: bool = False,
        n_bands: int = 1,
        prop_orders: int = 1,
        adj_rank: int = 16,
        adj_topk: int = 10,
        adj_tau: float = 1.0,
        self_loop_alpha: float = 0.2,
        use_temporal_head: bool = False,
        temporal_head_ratio: float = 0.5,
        temporal_head_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        gate_init: float = -4.0,
        use_input_residual: bool = True,
        input_residual_dropout: float = 0.0,
        extra_backbone_kwargs: dict = None,
        # internal compat. — ignored in release
        **kwargs,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_bands = n_bands
        self.prop_orders = prop_orders
        self.use_temporal_head = use_temporal_head
        self.use_input_residual = use_input_residual

        # Backbone
        bkw = {'individual': True} if backbone_individual else {}
        bkw.update(extra_backbone_kwargs or {})
        self.backbone = create_backbone(
            backbone, seq_len=seq_len, pred_len=pred_len,
            n_features=n_nodes, **bkw
        )

        # Prediction head (PH)
        if use_temporal_head:
            self.temporal_head = TemporalHead(
                pred_len=pred_len,
                ratio=temporal_head_ratio,
                dropout=temporal_head_dropout,
            )

        # Normalisation
        self.pre_norm = nn.LayerNorm(n_nodes)
        self.post_norm = nn.LayerNorm(n_nodes)
        self.residual_drop = (
            nn.Dropout(residual_dropout) if residual_dropout > 0
            else nn.Identity()
        )

        # Learnable adjacency (S component — skipped when n_bands=0)
        if n_bands == 1:
            self.adjacency = LowRankTopKAdjacency(
                n_nodes, rank=adj_rank, tau=adj_tau, k=adj_topk,
                self_loop_alpha=self_loop_alpha
            )
        elif n_bands > 1:
            self.adjacency = BandAdjacency(
                n_nodes, n_bands=n_bands, rank=adj_rank, tau=adj_tau,
                k=adj_topk, self_loop_alpha=self_loop_alpha
            )
        else:
            self.adjacency = None  # spatial disabled

        # Gating
        self.gate = ResidualGating(pred_len=pred_len, init_value=gate_init)

        # Input residual (R)
        if use_input_residual:
            self.input_proj = nn.Linear(seq_len, pred_len)
            self.input_drop = (
                nn.Dropout(input_residual_dropout) if input_residual_dropout > 0
                else nn.Identity()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Temporal backbone
        y_base = self.backbone(x)

        # 2. Prediction head (optional)
        y_ref = self.temporal_head(y_base) if self.use_temporal_head else y_base

        # 3. Pre-norm
        y_norm = self.pre_norm(y_ref)

        # 4–6. Spatial correction, post-norm, gating
        delta = self._spatial_correction(y_norm)
        delta = self.gate(self.residual_drop(self.post_norm(delta)))

        # 7. Add spatial correction
        y_hat = y_norm + delta

        # 8. Input residual (optional)
        if self.use_input_residual:
            inp = self.input_proj(x.transpose(1, 2)).transpose(1, 2)
            y_hat = y_hat + self.input_drop(inp)

        return y_hat

    def _spatial_correction(self, y_norm: torch.Tensor) -> torch.Tensor:
        """Compute Σ_p (A^p - I) · y. Returns zeros when n_bands=0."""
        if self.n_bands == 0 or self.adjacency is None:
            return torch.zeros_like(y_norm)

        delta = torch.zeros_like(y_norm)
        z = y_norm.transpose(1, 2)                          # [B, N, L]

        if self.n_bands == 1:
            A = self.adjacency()
            A_pow_z = z
            for _ in range(self.prop_orders):
                A_pow_z = torch.matmul(A, A_pow_z)
                delta += (A_pow_z - z).transpose(1, 2)
        else:
            for k, (start, end) in enumerate(self._band_boundaries()):
                A_k = self.adjacency.adjacencies[k]()
                z_band = z[:, :, start:end]                 # [B, N, band]
                A_pow = z_band
                band_delta = torch.zeros_like(z_band)
                for _ in range(self.prop_orders):
                    A_pow = torch.matmul(A_k, A_pow)
                    band_delta += A_pow - z_band
                delta[:, start:end, :] = band_delta.transpose(1, 2)

        return delta

    def _band_boundaries(self):
        L = self.pred_len
        n = self.n_bands
        if n == 4 and L == 720:
            return [(0, 96), (96, 192), (192, 336), (336, 720)]
        size = L // n
        return [(i * size, (i + 1) * size if i < n - 1 else L)
                for i in range(n)]

    def count_params(self) -> Dict[str, int]:
        def _n(module):
            return sum(p.numel() for p in module.parameters())
        return {
            'backbone': _n(self.backbone),
            'temporal_head': _n(self.temporal_head) if self.use_temporal_head else 0,
            'adjacency': _n(self.adjacency) if self.adjacency else 0,
            'gating': _n(self.gate),
            'input_residual': _n(self.input_proj) if self.use_input_residual else 0,
            'total': _n(self),
        }
