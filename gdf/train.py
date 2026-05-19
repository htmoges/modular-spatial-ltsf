"""
GDF Training Script
====================

Unified training entry point for both baselines and GDF models.
Config-driven via YAML files. Supports multi-seed evaluation.

Usage:
    # Single run
    python -m gdf.train --config configs/gdf/gdf_static_electricity.yaml

    # Override config values via CLI
    python -m gdf.train --config configs/gdf/gdf_static_electricity.yaml --seed 1

    # Multi-seed
    python -m gdf.train --config configs/gdf/gdf_static_electricity.yaml --seeds 42 1 2
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from gdf.data import create_dataloaders
from gdf.models.framework import GraphDiffusionForecaster
from gdf.utils.metrics import STANDARD_HORIZONS


# =============================================================================
# Lightning Module
# =============================================================================

class GDFLightning(pl.LightningModule):
    """Lightning wrapper for GDF and baseline models.
    
    Handles:
      - Training with MSE loss
      - Validation with per-horizon metrics
      - Testing with comprehensive metric logging
      - Optimizer and scheduler configuration
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        super().__init__()
        self.model = model
        self.config = config
        self.criterion = nn.MSELoss()
        self.test_step_outputs = []
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        
        mae = F.l1_loss(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_mae', mae, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        
        mse = F.mse_loss(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        self.log('val_mse', mse, on_step=False, on_epoch=True)
        self.log('val_mae', mae, on_step=False, on_epoch=True, prog_bar=True)
        
        # Per-horizon metrics
        L = y_hat.shape[1]
        mse_list, mae_list = [], []
        for h in STANDARD_HORIZONS:
            if h <= L:
                mse_h = F.mse_loss(y_hat[:, :h, :], y[:, :h, :])
                mae_h = F.l1_loss(y_hat[:, :h, :], y[:, :h, :])
                mse_list.append(mse_h)
                mae_list.append(mae_h)
                self.log(f'val_mse@{h}', mse_h, on_step=False, on_epoch=True)
                self.log(f'val_mae@{h}', mae_h, on_step=False, on_epoch=True)
        
        # Mean across horizons (primary ranking metric)
        if mse_list:
            val_mse_mean = torch.stack(mse_list).mean()
            val_mae_mean = torch.stack(mae_list).mean()
            self.log('val_mse_mean', val_mse_mean, on_step=False, on_epoch=True, prog_bar=True)
            self.log('val_mae_mean', val_mae_mean, on_step=False, on_epoch=True)
        
        # Log gate diagnostics if GDF model
        if hasattr(self.model, 'get_gate_values'):
            gates = self.model.get_gate_values()
            for k, g in enumerate(gates):
                self.log(f'gate_band{k}', g.item(), on_step=False, on_epoch=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        
        mse = F.mse_loss(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        self.log('test_mse', mse, on_step=False, on_epoch=True)
        self.log('test_mae', mae, on_step=False, on_epoch=True)
        
        L = y_hat.shape[1]
        for h in STANDARD_HORIZONS:
            if h <= L:
                mse_h = F.mse_loss(y_hat[:, :h, :], y[:, :h, :])
                mae_h = F.l1_loss(y_hat[:, :h, :], y[:, :h, :])
                self.log(f'test_mse@{h}', mse_h, on_step=False, on_epoch=True)
                self.log(f'test_mae@{h}', mae_h, on_step=False, on_epoch=True)
        
        self.test_step_outputs.append({'mse': mse.item(), 'mae': mae.item()})
    
    def on_test_epoch_end(self):
        if self.test_step_outputs:
            avg_mse = np.mean([d['mse'] for d in self.test_step_outputs])
            avg_mae = np.mean([d['mae'] for d in self.test_step_outputs])
            print(f"\n{'='*60}")
            print(f"  TEST RESULTS: MSE={avg_mse:.6f}  MAE={avg_mae:.6f}")
            print(f"{'='*60}\n")
            self.test_step_outputs.clear()
    
    def configure_optimizers(self):
        lr = self.config.get('learning_rate', 8e-4)
        weight_decay = self.config.get('weight_decay', 0.0)
        
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        
        # Multi-step scheduler (decay at 60% and 80% of training)
        max_epochs = self.config.get('max_epochs', 50)
        milestones = self.config.get(
            'lr_milestones',
            [int(0.6 * max_epochs), int(0.8 * max_epochs)]
        )
        gamma = self.config.get('lr_gamma', 0.5)
        
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma
        )
        
        return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch'}]


# =============================================================================
# Model Factory
# =============================================================================

def create_model(config: Dict[str, Any], n_features: int) -> nn.Module:
    """Create a GraphDiffusionForecaster from config."""
    backbone_kwargs = config.get('backbone_kwargs', {})
    n_bands = config.get('n_bands', 1)

    return GraphDiffusionForecaster(
        n_nodes=n_features,
        seq_len=config.get('seq_len', 96),
        pred_len=config.get('pred_len', 720),
        backbone=config.get('backbone', 'dlinear'),
        backbone_individual=config.get('backbone_individual', False),
        extra_backbone_kwargs=backbone_kwargs if backbone_kwargs else None,
        n_bands=max(n_bands, 1),        # n_bands=0 → adjacency=None (no spatial)
        prop_orders=config.get('prop_orders', 1),
        adj_rank=config.get('adj_rank', 16),
        adj_topk=config.get('adj_topk', 10),
        adj_tau=config.get('adj_tau', 1.0),
        self_loop_alpha=config.get('self_loop_alpha', 0.2),
        use_temporal_head=config.get('use_temporal_head', False),
        temporal_head_ratio=config.get('temporal_head_ratio', 0.5),
        temporal_head_dropout=config.get('temporal_head_dropout', 0.1),
        residual_dropout=config.get('residual_dropout', 0.0),
        gate_init=config.get('gate_init', config.get('diffusion_init', -4.0)),
        use_input_residual=config.get('use_input_residual', True),
        input_residual_dropout=config.get('input_residual_dropout', 0.0),
    )


# =============================================================================
# Single Training Run
# =============================================================================

def train_single(config: Dict[str, Any], seed: int = 42) -> Dict[str, Any]:
    """Execute a single training run.
    
    Returns:
        Dictionary with test metrics and training info
    """
    # Seed everything
    pl.seed_everything(seed, workers=True)
    
    # Data
    data_root = config.get('data_root', 'data')
    train_loader, val_loader, test_loader, n_features = create_dataloaders(
        dataset=config['dataset'],
        data_root=data_root,
        seq_len=config.get('seq_len', 96),
        pred_len=config.get('pred_len', 720),
        batch_size=config.get('batch_size', None),
        num_workers=config.get('num_workers', 4),
    )
    
    # Model
    model = create_model(config, n_features)
    
    # Print model info
    param_info = model.count_params()
    model_type = config.get('model', 'gdf')
    print(f"\n{'='*60}")
    print(f"  Model: {model_type.upper()} | Backbone: {config.get('backbone', 'dlinear')}")
    print(f"  Dataset: {config['dataset']} (N={n_features})")
    print(f"  Seed: {seed}")
    print(f"  Parameters:")
    for k, v in param_info.items():
        print(f"    {k}: {v:,}")
    if hasattr(model, 'n_bands'):
        print(f"  Horizon bands (K): {model.n_bands}")
        print(f"  Prop orders (P): {model.prop_orders}")
    if hasattr(model, 'use_temporal_head'):
        print(f"  Temporal head: {model.use_temporal_head}")
    if hasattr(model, 'gate'):
        print(f"  Gating: band-wise sigmoid")
    print(f"{'='*60}\n")
    
    # Lightning module
    lightning_model = GDFLightning(model, config)
    
    # Callbacks
    max_epochs = config.get('max_epochs', 50)
    patience = config.get('patience', 10)
    monitor = config.get('monitor', 'val_mse_mean')
    
    results_dir = config.get('results_dir', 'results')
    exp_name = config.get('experiment_name', f"{model_type}_{config['dataset']}_seed{seed}")
    exp_dir = os.path.join(results_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(exp_dir, 'checkpoints'),
        filename=f'best_{{epoch:02d}}_{{{monitor}:.4f}}',
        save_top_k=1,
        monitor=monitor,
        mode='min',
    )
    
    early_stop_cb = EarlyStopping(
        monitor=monitor,
        patience=patience,
        mode='min',
        verbose=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision=config.get('precision', 32),
        logger=None,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor],
        enable_progress_bar=True,
        enable_model_summary=False,
        gradient_clip_val=config.get('gradient_clip', 1.0),
        num_sanity_val_steps=0,
    )
    
    # Train
    t0 = time.time()
    trainer.fit(lightning_model, train_loader, val_loader)
    train_time = (time.time() - t0) / 60.0
    
    # Test
    test_results = trainer.test(lightning_model, test_loader, ckpt_path='best')
    test_metrics = dict(test_results[0]) if test_results else {}
    
    # Save results
    result = {
        'seed': seed,
        'config': config,
        'test_metrics': test_metrics,
        'param_info': param_info,
        'train_time_minutes': round(train_time, 2),
        'best_checkpoint': checkpoint_cb.best_model_path,
        'best_val_score': float(checkpoint_cb.best_model_score) if checkpoint_cb.best_model_score else None,
    }
    
    result_path = os.path.join(exp_dir, f'results_seed{seed}.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"Results saved to: {result_path}")
    print(f"Training time: {train_time:.1f} min")
    
    return result


# =============================================================================
# Multi-seed Evaluation
# =============================================================================

def train_multi_seed(config: Dict[str, Any], seeds: List[int]) -> Dict[str, Any]:
    """Run training across multiple seeds and aggregate results.
    
    Returns:
        Aggregated results with mean ± std
    """
    all_results = []
    
    for seed in seeds:
        print(f"\n{'#'*60}")
        print(f"  SEED {seed} / {seeds}")
        print(f"{'#'*60}\n")
        
        result = train_single(config, seed=seed)
        all_results.append(result)
    
    # Aggregate
    test_keys = all_results[0]['test_metrics'].keys()
    aggregated = {}
    for key in test_keys:
        values = [r['test_metrics'][key] for r in all_results if key in r['test_metrics']]
        if values:
            aggregated[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': values,
            }
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"  MULTI-SEED SUMMARY ({len(seeds)} seeds)")
    print(f"{'='*60}")
    for key, stats in aggregated.items():
        print(f"  {key}: {stats['mean']:.6f} ± {stats['std']:.6f}")
    print(f"{'='*60}\n")
    
    # Save aggregated results
    results_dir = config.get('results_dir', 'results')
    model_type = config.get('model', 'gdf')
    exp_name = config.get('experiment_name', f"{model_type}_{config['dataset']}")
    exp_dir = os.path.join(results_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    agg_path = os.path.join(exp_dir, 'results_aggregated.json')
    with open(agg_path, 'w') as f:
        json.dump({
            'seeds': seeds,
            'aggregated': aggregated,
            'per_seed': [r['test_metrics'] for r in all_results],
        }, f, indent=2, default=str)
    
    print(f"Aggregated results saved to: {agg_path}")
    
    return {'aggregated': aggregated, 'per_seed': all_results}


# =============================================================================
# CLI
# =============================================================================

def load_config(config_path: str, overrides: dict = None) -> Dict[str, Any]:
    """Load YAML config and apply CLI overrides."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                config[k] = v
    
    return config


def main():
    parser = argparse.ArgumentParser(description='GDF Training')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    
    # CLI overrides (take precedence over config file)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--backbone', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Multiple seeds for aggregated evaluation')
    parser.add_argument('--max-epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--data-root', type=str, default=None)
    parser.add_argument('--results-dir', type=str, default=None)
    parser.add_argument('--n-bands', type=int, default=None)
    parser.add_argument('--seq-len', type=int, default=None)
    parser.add_argument('--pred-len', type=int, default=None)
    parser.add_argument('--diffusion-init', type=float, default=None)
    parser.add_argument('--adj-rank', type=int, default=None)
    parser.add_argument('--adj-topk', type=int, default=None)
    parser.add_argument('--experiment-name', type=str, default=None)
    parser.add_argument('--prop-orders', type=int, default=None,
                        help='Multi-hop propagation orders')
    parser.add_argument('--use-temporal-head', action=argparse.BooleanOptionalAction, default=None,
                        help='Enable/disable prediction head (--use-temporal-head / --no-temporal-head)')
    parser.add_argument('--use-input-residual', action=argparse.BooleanOptionalAction, default=None,
                        help='Enable/disable input skip connection (--use-input-residual / --no-input-residual)')
    parser.add_argument('--temporal-head-ratio', type=float, default=None,
                        help='Hidden dim ratio for temporal head')
    parser.add_argument('--residual-dropout', type=float, default=None,
                        help='Dropout on spatial residual')
    parser.add_argument('--gate-mode', type=str, default=None,
                        choices=['band', 'flat'],
                        help='Gating mode')
    
    args = parser.parse_args()
    
    # Build overrides dict (only non-None values)
    overrides = {}
    for key in ['dataset', 'backbone', 'seed', 'max_epochs', 'batch_size',
                'learning_rate', 'data_root', 'results_dir', 'n_bands',
                'seq_len', 'adj_rank', 'adj_topk',
                'experiment_name', 'prop_orders',
                'use_temporal_head', 'use_input_residual',
                'temporal_head_ratio', 'residual_dropout', 'gate_mode']:
        cli_key = key.replace('_', '-')
        val = getattr(args, key.replace('-', '_'), None)
        if val is not None:
            overrides[key] = val
    
    config = load_config(args.config, overrides)
    
    # Default data root
    if 'data_root' not in config:
        config['data_root'] = 'data'
    if 'results_dir' not in config:
        config['results_dir'] = 'results'
    
    # Multi-seed or single seed
    seeds = args.seeds
    if seeds:
        train_multi_seed(config, seeds)
    else:
        seed = config.get('seed', 42)
        train_single(config, seed)


if __name__ == '__main__':
    main()
