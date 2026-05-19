"""
Low-Rank TopK Adjacency Module
===============================

Proven component from Lite-STGNN (ICAART 2026).
Produces a sparse, row-normalized adjacency matrix via low-rank factorization.

    A = TopK(softmax(E₁ E₂ᵀ / τ))  + α·I    (row-normalized)

Complexity: O(N·r) parameters, O(N²) forward (but sparse).

For GDF, this is extended to support K independent adjacency matrices
(one per horizon band), each producing its own graph Laplacian.
"""

import torch
import torch.nn as nn
from typing import Optional


class LowRankTopKAdjacency(nn.Module):
    """Single low-rank factorized sparse adjacency matrix.
    
    Args:
        n_nodes: Number of graph nodes (N = number of features/variables)
        rank: Embedding dimension for low-rank factorization
        tau: Temperature for softmax (higher = more uniform)
        k: Number of top connections per node
        self_loop_alpha: Weight of self-loop (added before normalization)
    
    Returns:
        A: [N, N] row-normalized adjacency matrix
    """
    
    def __init__(
        self,
        n_nodes: int,
        rank: int = 16,
        tau: float = 1.0,
        k: int = 10,
        self_loop_alpha: float = 0.2,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.rank = rank
        self.tau = tau
        self.k = k
        self.self_loop_alpha = self_loop_alpha
        
        # Low-rank factorization: A ≈ E₁ @ E₂ᵀ
        self.E1 = nn.Parameter(torch.randn(n_nodes, rank) * 0.02)
        self.E2 = nn.Parameter(torch.randn(n_nodes, rank) * 0.02)
    
    def forward(self) -> torch.Tensor:
        """Compute sparse row-normalized adjacency matrix.
        
        Returns:
            A: [N, N] adjacency matrix with self-loops, row-normalized
        """
        # Similarity via low-rank factorization
        sim = (self.E1 @ self.E2.t()) / max(self.tau, 1e-6)
        A_soft = torch.softmax(sim, dim=-1)
        
        n = A_soft.size(0)
        device = A_soft.device
        
        # Remove diagonal (self-loops handled separately)
        A_soft = A_soft - torch.diag_embed(torch.diag(A_soft))
        
        # TopK sparsification per row
        if self.k < n:
            _, topk_idx = torch.topk(A_soft, k=self.k, dim=1)
            mask = torch.zeros_like(A_soft)
            mask.scatter_(1, topk_idx, 1.0)
            sparse = A_soft * mask
        else:
            sparse = A_soft
        
        # Add self-loops and normalize
        I = torch.eye(n, device=device)
        A = sparse + self.self_loop_alpha * I
        row_sum = A.sum(dim=1, keepdim=True).clamp(min=1e-6)
        A = A / row_sum
        
        return A
    
    def get_laplacian(self) -> torch.Tensor:
        """Compute graph Laplacian L = I - A.
        
        Returns:
            L: [N, N] graph Laplacian
        """
        A = self.forward()
        I = torch.eye(self.n_nodes, device=A.device)
        return I - A
    
    def get_sparse_adjacency(self) -> torch.Tensor:
        """Get adjacency without self-loops (for visualization/analysis)."""
        sim = (self.E1 @ self.E2.t()) / max(self.tau, 1e-6)
        A_soft = torch.softmax(sim, dim=-1)
        A_soft = A_soft - torch.diag_embed(torch.diag(A_soft))
        if self.k < self.n_nodes:
            _, topk_idx = torch.topk(A_soft, k=self.k, dim=1)
            mask = torch.zeros_like(A_soft)
            mask.scatter_(1, topk_idx, 1.0)
            return A_soft * mask
        return A_soft
    
    def sparsity(self) -> float:
        """Fraction of zero entries in the adjacency."""
        A = self.get_sparse_adjacency()
        return (A == 0).float().mean().item()


class BandAdjacency(nn.Module):
    """K independent adjacency matrices, one per horizon band.
    
    Each band learns its own graph structure, enabling horizon-dependent
    spatial relationships. This is the key extension over Lite-STGNN's
    static graph.
    
    Args:
        n_nodes: Number of graph nodes
        n_bands: Number of horizon bands (K)
        rank: Embedding dimension per band
        tau: Temperature for softmax
        k: TopK connections per node
        self_loop_alpha: Self-loop weight
    
    Returns:
        List of K adjacency matrices, each [N, N]
    """
    
    def __init__(
        self,
        n_nodes: int,
        n_bands: int = 4,
        rank: int = 16,
        tau: float = 1.0,
        k: int = 10,
        self_loop_alpha: float = 0.2,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_bands = n_bands
        
        self.adjacencies = nn.ModuleList([
            LowRankTopKAdjacency(n_nodes, rank=rank, tau=tau, k=k,
                                 self_loop_alpha=self_loop_alpha)
            for _ in range(n_bands)
        ])
    
    def forward(self) -> list:
        """Compute all K adjacency matrices.
        
        Returns:
            List of K tensors, each [N, N]
        """
        return [adj() for adj in self.adjacencies]
    
    def get_laplacians(self) -> list:
        """Compute K graph Laplacians L_k = I - A_k."""
        return [adj.get_laplacian() for adj in self.adjacencies]
    
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
