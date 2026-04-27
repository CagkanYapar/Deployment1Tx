"""
Deep Tx Localization Training Script - FIXED WITH COMPREHENSIVE VALIDATION

KEY FEATURES:
- Two-tier validation: LIGHT (every epoch) and FULL (every N epochs)
- Proper handling of optimize_for='power' vs 'coverage'
- Multiple best model tracking (PL%, coverage%, coord_error)
- Physical radio propagation heatmaps
- Advanced loss functions support
- Comprehensive metrics and visualizations
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image")
import os
import sys
import time
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from collections import defaultdict
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import csv
import json

from data.dataset_heatmap import TxLocationDatasetLarge, get_building_splits
from models.discriminative import create_model_deep

sys.path.append('.')
from training.losses import compute_multi_loss
from models.saipp_net import RadioNet
from scipy.ndimage import distance_transform_edt


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_average_maps(building_id, base_dir, use_center=True):
    """Load average coverage and PL maps for ranking (quantized maps)"""
    try:
        cov_path = os.path.join(base_dir, 'normalized_by_free_pixels', f'{building_id}_avgCov.png')
        pl_path = os.path.join(base_dir, 'normalized_by_free_pixels', f'{building_id}_avgPL.png')
        
        if not os.path.exists(cov_path) or not os.path.exists(pl_path):
            return None, None
        
        cov_map = np.array(Image.open(cov_path).convert('L')).astype(np.float32)
        pl_map = np.array(Image.open(pl_path).convert('L')).astype(np.float32)
        
        if use_center:
            cov_map = cov_map[53:203, 53:203]
            pl_map = pl_map[53:203, 53:203]
        
        return cov_map, pl_map
    except Exception as e:
        return None, None


def compute_predicted_radio_metrics(pred_coords, building_id, radionet_model, use_center, buildings_dir, threshold=0.0):
    """Compute coverage/power for predicted Tx using RadioNet."""
    
    if radionet_model is None:
        return None
    
    building_path = os.path.join(buildings_dir, f'{building_id}.png')
    if not os.path.exists(building_path):
        return None
    
    try:
        building_img = np.array(Image.open(building_path).convert('L')).astype(np.float32) / 255.0
        
        pred_y, pred_x = pred_coords
        
        if use_center:
            pred_y_256 = pred_y + 53
            pred_x_256 = pred_x + 53
        else:
            pred_y_256, pred_x_256 = pred_y, pred_x
        
        tx_onehot = np.zeros((256, 256), dtype=np.float32)
        pred_y_int = int(np.clip(pred_y_256, 0, 255))
        pred_x_int = int(np.clip(pred_x_256, 0, 255))
        tx_onehot[pred_y_int, pred_x_int] = 1.0
        
        building_tensor = torch.from_numpy(building_img).unsqueeze(0).unsqueeze(0)
        tx_tensor = torch.from_numpy(tx_onehot).unsqueeze(0).unsqueeze(0)
        inputs = torch.cat([building_tensor, tx_tensor], dim=1).to(next(radionet_model.parameters()).device)
        
        with torch.no_grad():
            power_map_256 = radionet_model(inputs)[0, 0].cpu().numpy()
        
        # Scale RadioNet output from [0,1] to [0,255] to match GT PL maps
        power_map_256 = np.clip(power_map_256 * 255, 0, 255).astype(np.uint8)
        
        power_center = power_map_256[53:203, 53:203]
        building_center = building_img[53:203, 53:203]
        center_free = (building_center <= 0.1)
        
        # Extract power values at free center pixels
        center_free_values = power_center[center_free]
        
        # Coverage count: free pixels with signal > threshold
        coverage_count = int(np.sum(center_free_values > threshold))
        
        # Average power: mean of free center pixels
        avg_power = float(np.mean(center_free_values)) if center_free_values.size > 0 else 0.0
        
        return {
            'coverage_count': coverage_count,
            'avg_power': avg_power
        }
    except Exception as e:
        print(f"RadioNet: Error computing metrics for {building_id}: {e}", flush=True)
        return None


def load_pl_map_metrics(building_id, base_dir, subfolder, use_center, buildings_dir_full, threshold=0.0):
    """Load PL map from specified subfolder and compute coverage/avg_power metrics"""
    try:
        pl_path = os.path.join(base_dir, subfolder, f'{building_id}_PL.png')
        building_path = os.path.join(buildings_dir_full, f'{building_id}.png')
        
        if not os.path.exists(pl_path) or not os.path.exists(building_path):
            return None
        
        # Load PL map as uint8 (0-255)
        pl_map = np.array(Image.open(pl_path).convert('L'))
        building_img = np.array(Image.open(building_path).convert('L')).astype(np.float32) / 255.0
        
        # Get center region
        if use_center:
            pl_map_center = pl_map[53:203, 53:203]
            building_center = building_img[53:203, 53:203]
        else:
            pl_map_center = pl_map
            building_center = building_img
        
        # Get free pixels in center
        free_space = (building_center <= 0.1)
        center_free_values = pl_map_center[free_space]
        
        # Coverage count: free pixels with signal > threshold
        coverage_count = int(np.sum(center_free_values > threshold))
        
        # Average power: mean of free center pixels
        avg_power = float(np.mean(center_free_values)) if len(center_free_values) > 0 else 0.0
        
        return {
            'coverage_count': coverage_count,
            'avg_power': avg_power,
            'pl_map': pl_map_center.astype(np.float32) / 255.0  # Normalize for visualization
        }
    except Exception as e:
        return None


def evaluate_with_avg_maps(pred_coords, gt_coords, building_id, heatmap_base_dir, 
                          optimize_for, use_center, buildings_dir_full, radionet_model,
                          include_rankings=False):
    """
    Evaluate prediction with proper handling of optimize_for.
    
    When optimize_for='power':
        - Load from best_by_power (primary) and best_by_coverage (reference)
        - Return: pl_pct, coverage_pct_power, coverage_pct_gt
    
    When optimize_for='coverage':
        - Load from best_by_coverage (primary) and best_by_power (reference)
        - Return: coverage_pct, pl_pct_coverage, pl_pct_ref
    
    If include_rankings=True, also compute rankings from quantized avgCov/avgPL maps
    """
    
    # 1. Compute predicted metrics using RadioNet
    metrics_pred = compute_predicted_radio_metrics(
        pred_coords, building_id, radionet_model, use_center, 
        buildings_dir_full, threshold=0.0
    )
    
    if metrics_pred is None:
        return None
    
    # 2. Load GT metrics based on optimize_for
    if optimize_for == 'power':
        # Primary: best_by_power, Reference: best_by_coverage
        gt_power_metrics = load_pl_map_metrics(
            building_id, heatmap_base_dir, 'best_by_power', 
            use_center, buildings_dir_full, threshold=0.0
        )
        gt_cov_metrics = load_pl_map_metrics(
            building_id, heatmap_base_dir, 'best_by_coverage', 
            use_center, buildings_dir_full, threshold=0.0
        )
        
        if gt_power_metrics is None or gt_cov_metrics is None:
            return None
        
        # Compute percentages
        pred_pl = float(metrics_pred['avg_power'])
        gt_pl = float(gt_power_metrics['avg_power'])
        pl_pct = (pred_pl / gt_pl * 100.0) if gt_pl > 0 else 0.0
        
        pred_cov = float(metrics_pred['coverage_count'])
        gt_cov_power = float(gt_power_metrics['coverage_count'])
        gt_cov_ref = float(gt_cov_metrics['coverage_count'])
        coverage_pct_power = (pred_cov / gt_cov_power * 100.0) if gt_cov_power > 0 else 0.0
        coverage_pct_gt = (pred_cov / gt_cov_ref * 100.0) if gt_cov_ref > 0 else 0.0
        
        result = {
            'pl_pct': pl_pct,
            'coverage_pct_power': coverage_pct_power,
            'coverage_pct_gt': coverage_pct_gt,
            'pred_pl': pred_pl,
            'gt_pl': gt_pl,
            'pred_coverage': pred_cov,
            'gt_coverage_power': gt_cov_power,
            'gt_coverage_ref': gt_cov_ref
        }
    
    else:  # optimize_for == 'coverage'
        # Primary: best_by_coverage, Reference: best_by_power
        gt_cov_metrics = load_pl_map_metrics(
            building_id, heatmap_base_dir, 'best_by_coverage', 
            use_center, buildings_dir_full, threshold=0.0
        )
        gt_power_metrics = load_pl_map_metrics(
            building_id, heatmap_base_dir, 'best_by_power', 
            use_center, buildings_dir_full, threshold=0.0
        )
        
        if gt_cov_metrics is None or gt_power_metrics is None:
            return None
        
        # Compute percentages
        pred_cov = float(metrics_pred['coverage_count'])
        gt_cov = float(gt_cov_metrics['coverage_count'])
        coverage_pct = (pred_cov / gt_cov * 100.0) if gt_cov > 0 else 0.0
        
        pred_pl = float(metrics_pred['avg_power'])
        gt_pl_cov = float(gt_cov_metrics['avg_power'])
        gt_pl_ref = float(gt_power_metrics['avg_power'])
        pl_pct_coverage = (pred_pl / gt_pl_cov * 100.0) if gt_pl_cov > 0 else 0.0
        pl_pct_ref = (pred_pl / gt_pl_ref * 100.0) if gt_pl_ref > 0 else 0.0
        
        result = {
            'coverage_pct': coverage_pct,
            'pl_pct_coverage': pl_pct_coverage,
            'pl_pct_ref': pl_pct_ref,
            'pred_coverage': pred_cov,
            'gt_coverage': gt_cov,
            'pred_pl': pred_pl,
            'gt_pl_coverage': gt_pl_cov,
            'gt_pl_ref': gt_pl_ref
        }
    
    # 3. Compute rankings if requested (only for full validation)
    if include_rankings:
        cov_map, pl_map = load_average_maps(building_id, heatmap_base_dir, use_center)
        
        if cov_map is not None and pl_map is not None:
            # Load building map for free space mask
            building_path = os.path.join(buildings_dir_full, f'{building_id}.png')
            if os.path.exists(building_path):
                building_img = np.array(Image.open(building_path).convert('L')).astype(np.float32) / 255.0
                if use_center:
                    building_img = building_img[53:203, 53:203]
                free_space = (building_img <= 0.1)
                
                # Get predicted values from average maps for ranking
                h, w = cov_map.shape
                pred_y = int(np.clip(pred_coords[0], 0, h-1))
                pred_x = int(np.clip(pred_coords[1], 0, w-1))
                
                pred_cov_map = float(cov_map[pred_y, pred_x])
                pred_pl_map = float(pl_map[pred_y, pred_x])
                
                # Compute ranks using quantized maps
                better_than_pred_cov = (cov_map[free_space] > pred_cov_map).sum()
                equal_to_pred_cov = (cov_map[free_space] == pred_cov_map).sum()
                rank_cov_best = int(better_than_pred_cov) + 1
                rank_cov_worst = int(better_than_pred_cov + equal_to_pred_cov)
                
                better_than_pred_pl = (pl_map[free_space] > pred_pl_map).sum()
                equal_to_pred_pl = (pl_map[free_space] == pred_pl_map).sum()
                rank_pl_best = int(better_than_pred_pl) + 1
                rank_pl_worst = int(better_than_pred_pl + equal_to_pred_pl)
                
                rank_cov = (rank_cov_best + rank_cov_worst) // 2
                rank_pl = (rank_pl_best + rank_pl_worst) // 2
                
                total_free = int(free_space.sum())
                
                result['rank_coverage'] = rank_cov
                result['rank_coverage_best'] = rank_cov_best
                result['rank_coverage_worst'] = rank_cov_worst
                result['rank_pl'] = rank_pl
                result['rank_pl_best'] = rank_pl_best
                result['rank_pl_worst'] = rank_pl_worst
                result['total_free_pixels'] = total_free
    
    return result


def compute_building_density(building_id, buildings_dir_full, use_center=True):
    """Compute building density (percentage of building pixels)"""
    try:
        building_path = os.path.join(buildings_dir_full, f'{building_id}.png')
        if not os.path.exists(building_path):
            return None
        
        building_img = np.array(Image.open(building_path).convert('L')).astype(np.float32) / 255.0
        
        if use_center:
            building_img = building_img[53:203, 53:203]
        
        # Building pixels are those > 0.1
        building_pixels = (building_img > 0.1).sum()
        total_pixels = building_img.size
        density = float(building_pixels) / total_pixels * 100.0
        
        return density
    except Exception as e:
        return None


def load_topk_csv(building_id, topk_dir, use_center):
    """Load top-50 Tx locations from CSV."""
    csv_path = os.path.join(topk_dir, f'{building_id}_topk_coverage_rankings.csv')
    if not os.path.exists(csv_path):
        return None
    
    topk_txs = []
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tx_y = float(row['tx_y'])
                tx_x = float(row['tx_x'])
                
                if use_center:
                    tx_y = tx_y - 53
                    tx_x = tx_x - 53
                
                topk_txs.append({
                    'rank': int(row['rank']),
                    'tx_y': tx_y,
                    'tx_x': tx_x,
                    'coverage_count': int(row['coverage_count']),
                    'coverage_fraction': float(row['coverage_fraction']),
                    'total_power': int(float(row['total_power'])),
                    'avg_power': float(row['avg_power'])
                })
        return topk_txs
    except Exception as e:
        return None


def find_closest_topk_rank(pred_coords, topk_txs):
    """Find rank of closest topK Tx to prediction"""
    if topk_txs is None or len(topk_txs) == 0:
        return None, None
    
    pred_point = np.array(pred_coords)
    min_dist = float('inf')
    closest_rank = None
    
    for tx_data in topk_txs:
        tx_point = np.array([tx_data['tx_y'], tx_data['tx_x']])
        dist = np.sqrt(np.sum((pred_point - tx_point)**2))
        
        if dist < min_dist:
            min_dist = dist
            closest_rank = tx_data['rank']
    
    return closest_rank, min_dist


def create_gaussian_target(coords, img_size, sigma=3.0):
    """Create Gaussian heatmap targets centered at GT coordinates."""
    batch_size = coords.shape[0]
    device = coords.device
    
    y_grid = torch.arange(img_size, device=device).float()
    x_grid = torch.arange(img_size, device=device).float()
    y_grid = y_grid.view(1, img_size, 1).expand(batch_size, img_size, img_size)
    x_grid = x_grid.view(1, 1, img_size).expand(batch_size, img_size, img_size)
    
    y_target = coords[:, 0].view(-1, 1, 1)
    x_target = coords[:, 1].view(-1, 1, 1)
    
    gaussian = torch.exp(-((y_grid - y_target)**2 + (x_grid - x_target)**2) / (2 * sigma**2))
    heatmap = gaussian.unsqueeze(1)
    
    return heatmap


def visualize_comprehensive(building_map, gt_coords, pred_coords, building_id, epoch, sample_idx,
                           pred_heatmap, gt_pl_power, gt_pl_coverage, radionet_pl,
                           metrics_pred, metrics_gt_power, metrics_gt_coverage,
                           pixel_error, save_dir, use_center, topk_rank=None, avg_metrics=None,
                           building_density=None):
    """Create comprehensive visualization with all heatmaps and metrics"""
    
    try:
        # Determine number of subplots
        if radionet_pl is not None:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            axes = axes.flatten()
        
        # 1. Building map with GT and predicted Tx
        axes[0].imshow(building_map, cmap='gray', origin='upper')
        axes[0].plot(gt_coords[1], gt_coords[0], 'gx', markersize=15, markeredgewidth=3, label='GT')
        axes[0].plot(pred_coords[1], pred_coords[0], 'ro', markersize=12, markerfacecolor='none', markeredgewidth=2, label='Pred')
        
        title_text = f'Building {building_id}\nPixel Error: {pixel_error:.2f}px'
        if building_density is not None:
            title_text += f'\nBuilding Density: {building_density:.1f}%'
        
        if avg_metrics:
            # Show the appropriate metrics based on what's available
            if 'coverage_pct' in avg_metrics:
                title_text += f'\nCov: {avg_metrics["coverage_pct"]:.1f}%'
            if 'pl_pct' in avg_metrics:
                title_text += f', PL: {avg_metrics["pl_pct"]:.1f}%'
            
            # Add rankings if available
            if 'rank_pl_best' in avg_metrics and 'rank_pl_worst' in avg_metrics:
                if avg_metrics['rank_pl_best'] == avg_metrics['rank_pl_worst']:
                    title_text += f'\nPL Rank: {avg_metrics["rank_pl"]}/{avg_metrics.get("total_free_pixels", 0)}'
                else:
                    title_text += f'\nPL Rank: {avg_metrics["rank_pl_best"]}-{avg_metrics["rank_pl_worst"]}/{avg_metrics.get("total_free_pixels", 0)}'
            
            if 'rank_coverage_best' in avg_metrics and 'rank_coverage_worst' in avg_metrics:
                if avg_metrics['rank_coverage_best'] == avg_metrics['rank_coverage_worst']:
                    title_text += f', Cov Rank: {avg_metrics["rank_coverage"]}/{avg_metrics.get("total_free_pixels", 0)}'
                else:
                    title_text += f', Cov Rank: {avg_metrics["rank_coverage_best"]}-{avg_metrics["rank_coverage_worst"]}/{avg_metrics.get("total_free_pixels", 0)}'
        
        axes[0].set_title(title_text)
        axes[0].legend()
        axes[0].axis('off')
        
        # 2. Predicted heatmap
        axes[1].imshow(pred_heatmap, cmap='hot', origin='upper')
        axes[1].set_title('Predicted Heatmap')
        axes[1].axis('off')
        
        # 3. GT PL from best_by_power
        if gt_pl_power is not None and metrics_gt_power and 'avg_power' in metrics_gt_power:
            axes[2].imshow(gt_pl_power, cmap='hot', origin='upper')
            axes[2].set_title(f'GT PL (best_by_power)\nCov: {metrics_gt_power["coverage_count"]}, AvgPower: {metrics_gt_power["avg_power"]:.2f}')
        else:
            axes[2].text(0.5, 0.5, 'GT Power\nNot Available', ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('GT PL (best_by_power)')
        axes[2].axis('off')
        
        # 4. GT PL from best_by_coverage
        if gt_pl_coverage is not None and metrics_gt_coverage and 'avg_power' in metrics_gt_coverage:
            axes[3].imshow(gt_pl_coverage, cmap='hot', origin='upper')
            axes[3].set_title(f'GT PL (best_by_coverage)\nCov: {metrics_gt_coverage["coverage_count"]}, AvgPower: {metrics_gt_coverage["avg_power"]:.2f}')
        else:
            axes[3].text(0.5, 0.5, 'GT Coverage\nNot Available', ha='center', va='center', transform=axes[3].transAxes)
            axes[3].set_title('GT PL (best_by_coverage)')
        axes[3].axis('off')
        
        # 5 & 6: RadioNet generated (if available)
        if radionet_pl is not None and metrics_pred:
            axes[4].imshow(radionet_pl, cmap='hot', origin='upper')
            
            # Calculate percentages relative to GT optimal values
            cov_pct_str = "N/A"
            power_pct_str = "N/A"
            
            if metrics_gt_coverage and 'coverage_count' in metrics_gt_coverage and metrics_gt_coverage['coverage_count'] > 0:
                cov_pct = (metrics_pred['coverage_count'] / metrics_gt_coverage['coverage_count']) * 100
                cov_pct_str = f"{cov_pct:.1f}%"
            
            if metrics_gt_power and 'avg_power' in metrics_gt_power and metrics_gt_power['avg_power'] > 0:
                power_pct = (metrics_pred['avg_power'] / metrics_gt_power['avg_power']) * 100
                power_pct_str = f"{power_pct:.1f}%"
            
            axes[4].set_title(f'RadioNet PL (from pred Tx)\n'
                            f'Cov: {metrics_pred["coverage_count"]} ({cov_pct_str}), '
                            f'AvgPower: {metrics_pred["avg_power"]:.2f} ({power_pct_str})')
            axes[4].axis('off')
            
            # Comparison text
            comparison_text = "Predicted Tx Metrics:\n"
            comparison_text += f"Coverage: {metrics_pred['coverage_count'] if metrics_pred else 'N/A'}\n"
            pred_power_str = f"{metrics_pred['avg_power']:.2f}" if (metrics_pred and 'avg_power' in metrics_pred) else 'N/A'
            comparison_text += f"AvgPower: {pred_power_str}\n\n"
            comparison_text += "GT (power) Metrics:\n"
            comparison_text += f"Coverage: {metrics_gt_power['coverage_count'] if metrics_gt_power else 'N/A'}\n"
            gt_power_str = f"{metrics_gt_power['avg_power']:.2f}" if (metrics_gt_power and 'avg_power' in metrics_gt_power) else 'N/A'
            comparison_text += f"AvgPower: {gt_power_str}"
            
            axes[5].text(0.1, 0.5, comparison_text, ha='left', va='center',
                        transform=axes[5].transAxes, fontsize=10, family='monospace')
            axes[5].set_title('Metrics Comparison')
            axes[5].axis('off')
        
        mode_str = "center150" if use_center else "full256"
        
        # Add coverage and PL to filename for poor performance cases
        if avg_metrics:
            # Check both optimize_for cases
            is_poor = False
            if 'coverage_pct' in avg_metrics and 'pl_pct' in avg_metrics:
                if avg_metrics['coverage_pct'] < 60 or avg_metrics['pl_pct'] < 85:
                    is_poor = True
                    cov_val = avg_metrics['coverage_pct']
                    pl_val = avg_metrics['pl_pct']
                    save_path = os.path.join(save_dir, f"vis_e{epoch}_s{sample_idx}_b{building_id}_err{pixel_error:.1f}px_COV{cov_val:.0f}_PL{pl_val:.0f}_{mode_str}.png")
            
            if not is_poor:
                save_path = os.path.join(save_dir, f"vis_e{epoch}_s{sample_idx}_b{building_id}_err{pixel_error:.1f}px_{mode_str}.png")
        else:
            save_path = os.path.join(save_dir, f"vis_e{epoch}_s{sample_idx}_b{building_id}_err{pixel_error:.1f}px_{mode_str}.png")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"VIS: Error creating visualization: {e}", flush=True)
        plt.close()


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_one_epoch(model, train_loader, optimizer, device, heatmap_type, 
                   heatmap_weight, coord_weight, img_size, loss_weights=None,
                   gaussian_sigma=3.0):
    """Train for one epoch with configurable heatmap + coordinate loss"""
    model.train()
    
    running_loss = 0.0
    running_heatmap_loss = 0.0
    running_coord_error = 0.0
    num_batches = 0
    
    for batch_idx, batch_data in enumerate(train_loader):
        if heatmap_type == 'gaussian':
            building_maps, gt_coords, building_ids = batch_data
            target_heatmap = None
        else:
            building_maps, gt_coords, target_heatmap, building_ids = batch_data
        
        building_maps = building_maps.to(device)
        gt_coords = gt_coords.to(device)
        
        optimizer.zero_grad()
        
        logits, pred_coords = model(building_maps)
        
        # Heatmap loss
        output_size = logits.size(2)
        if target_heatmap is None:
            target_heatmap = create_gaussian_target(gt_coords, output_size, sigma=gaussian_sigma)
        else:
            target_heatmap = target_heatmap.to(device)
            if target_heatmap.size(2) != output_size:
                target_heatmap = F.interpolate(target_heatmap, size=(output_size, output_size),
                                              mode='bilinear', align_corners=False)
        
        # Compute heatmap losses
        heatmap_loss_total, heatmap_loss_dict = compute_multi_loss(logits, target_heatmap, loss_weights)

        # Compute coordinate loss
        coord_loss = torch.mean(torch.abs(pred_coords - gt_coords))

        # Total loss
        loss = heatmap_loss_total + coord_weight * coord_loss

        # Track L2 specifically for backward compatibility
        heatmap_loss = heatmap_loss_dict.get('l2', 0.0)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_heatmap_loss += heatmap_loss
        running_coord_error += coord_loss.item()
        num_batches += 1
    
    avg_loss = running_loss / num_batches
    avg_heatmap_loss = running_heatmap_loss / num_batches
    avg_coord_error = running_coord_error / num_batches
    
    return {
        'loss': avg_loss,
        'heatmap_loss': avg_heatmap_loss,
        'coord_error': avg_coord_error
    }


# ============================================================================
# LIGHT VALIDATION (Every Epoch)
# ============================================================================

def validate_one_epoch_light(model, val_loader, device, epoch, save_dir, use_center,
                            heatmap_type, heatmap_weight, coord_weight, img_size,
                            optimize_for, radionet_model=None, buildings_dir_full=None,
                            heatmap_base_dir=None, loss_weights=None, gaussian_sigma=3.0):
    """
    LIGHT validation - runs every epoch.
    
    Computes:
    - Loss and coord error
    - Coverage% and PL% (requires RadioNet)
    - Basic statistics (mean, median, std)
    
    Skips:
    - Rankings (saves 2× PNG I/O per sample)
    - Building density
    - TopK analysis
    - Histograms
    - Most visualizations (only worst 5 by pixel and worst 5 by primary metric)
    """
    model.eval()
    
    running_loss = 0.0
    running_heatmap_loss = 0.0
    running_coord_error = 0.0
    num_batches = 0
    
    # Collect metrics
    pixel_errors = []
    
    if optimize_for == 'power':
        pl_pcts = []
        coverage_pcts_power = []
        coverage_pcts_gt = []
    else:  # coverage
        coverage_pcts = []
        pl_pcts_coverage = []
        pl_pcts_ref = []
    
    # For visualization: track worst samples
    worst_samples_pixel = []  # (pixel_error, sample_data)
    worst_samples_metric = []  # (metric_value, sample_data)
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            if heatmap_type == 'gaussian':
                building_maps, gt_coords, building_ids = batch_data
                target_heatmap = None
            else:
                building_maps, gt_coords, target_heatmap, building_ids = batch_data
            
            building_maps = building_maps.to(device)
            gt_coords = gt_coords.to(device)
            
            logits, pred_coords = model(building_maps)
            
            # Compute loss
            output_size = logits.size(2)
            if target_heatmap is None:
                target_heatmap = create_gaussian_target(gt_coords, output_size, sigma=gaussian_sigma)
            else:
                target_heatmap = target_heatmap.to(device)
                if target_heatmap.size(2) != output_size:
                    target_heatmap = F.interpolate(target_heatmap, size=(output_size, output_size),
                                                  mode='bilinear', align_corners=False)
            
            heatmap_loss_total, heatmap_loss_dict = compute_multi_loss(logits, target_heatmap, loss_weights)
            coord_loss = torch.mean(torch.abs(pred_coords - gt_coords))
            loss = heatmap_loss_total + coord_weight * coord_loss
            heatmap_loss = heatmap_loss_dict.get('l2', 0.0)
            
            running_loss += loss.item()
            running_heatmap_loss += heatmap_loss
            running_coord_error += coord_loss.item()
            num_batches += 1
            
            # Per-sample metrics
            batch_size = building_maps.size(0)
            for i in range(batch_size):
                pred_np = pred_coords[i].cpu().numpy()
                gt_np = gt_coords[i].cpu().numpy()
                building_id = building_ids[i]
                
                # Pixel error
                pixel_error = np.sqrt(np.sum((pred_np - gt_np)**2))
                pixel_errors.append(pixel_error)
                
                # Evaluate with RadioNet
                if heatmap_base_dir and radionet_model and buildings_dir_full:
                    avg_metrics = evaluate_with_avg_maps(
                        pred_np, gt_np, building_id, heatmap_base_dir, 
                        optimize_for, use_center, buildings_dir_full, radionet_model
                    )
                    
                    if avg_metrics:
                        if optimize_for == 'power':
                            pl_pcts.append(avg_metrics['pl_pct'])
                            coverage_pcts_power.append(avg_metrics['coverage_pct_power'])
                            coverage_pcts_gt.append(avg_metrics['coverage_pct_gt'])
                            primary_metric = avg_metrics['pl_pct']
                        else:  # coverage
                            coverage_pcts.append(avg_metrics['coverage_pct'])
                            pl_pcts_coverage.append(avg_metrics['pl_pct_coverage'])
                            pl_pcts_ref.append(avg_metrics['pl_pct_ref'])
                            primary_metric = avg_metrics['coverage_pct']
                        
                        # Track worst samples
                        sample_data = (building_maps[i], pred_np, gt_np, building_id, pixel_error, avg_metrics)
                        worst_samples_pixel.append((pixel_error, sample_data))
                        worst_samples_metric.append((primary_metric, sample_data))
    
    # Compute statistics
    metrics = {
        'loss': running_loss / num_batches,
        'heatmap_loss': running_heatmap_loss / num_batches,
        'coord_error': running_coord_error / num_batches
    }
    
    if len(pixel_errors) > 0:
        pixel_errors_np = np.array(pixel_errors)
        metrics['coord_error_median'] = float(np.median(pixel_errors_np))
        metrics['coord_error_std'] = float(np.std(pixel_errors_np))
    
    if optimize_for == 'power':
        if len(pl_pcts) > 0:
            metrics['avg_pl_pct'] = float(np.mean(pl_pcts))
            metrics['median_pl_pct'] = float(np.median(pl_pcts))
            metrics['std_pl_pct'] = float(np.std(pl_pcts))
            metrics['avg_coverage_pct_power'] = float(np.mean(coverage_pcts_power))
            metrics['avg_coverage_pct_gt'] = float(np.mean(coverage_pcts_gt))
    else:  # coverage
        if len(coverage_pcts) > 0:
            metrics['avg_coverage_pct'] = float(np.mean(coverage_pcts))
            metrics['median_coverage_pct'] = float(np.median(coverage_pcts))
            metrics['std_coverage_pct'] = float(np.std(coverage_pcts))
            metrics['avg_pl_pct_coverage'] = float(np.mean(pl_pcts_coverage))
            metrics['avg_pl_pct_ref'] = float(np.mean(pl_pcts_ref))
    
    # Save minimal visualizations (worst 5 by pixel, worst 5 by metric)
    # Note: Visualization code omitted for brevity - can add if needed
    
    return metrics


# ============================================================================
# FULL VALIDATION (Every N Epochs)
# ============================================================================

def validate_one_epoch_full(model, val_loader, device, epoch, save_dir, use_center,
                           heatmap_type, heatmap_weight, coord_weight, img_size,
                           optimize_for, radionet_model=None, topk_dir=None,
                           buildings_dir_full=None, heatmap_base_dir=None, loss_weights=None,
                           gaussian_sigma=3.0):
    """
    FULL validation - runs every N epochs and at end of training.
    
    Includes everything from light validation PLUS:
    - Rankings from avgCov/avgPL maps
    - Building density
    - TopK analysis
    - Histograms (pixel error, coverage%, PL%, density)
    - Correlations
    - Stratified visualizations
    """
    model.eval()
    
    running_loss = 0.0
    running_heatmap_loss = 0.0
    running_coord_error = 0.0
    num_batches = 0
    
    # Comprehensive metrics storage
    pixel_errors = []
    building_densities = []
    
    if optimize_for == 'power':
        pl_pcts = []
        coverage_pcts_power = []
        coverage_pcts_gt = []
        pl_ranks = []
        cov_ranks = []
    else:  # coverage
        coverage_pcts = []
        pl_pcts_coverage = []
        pl_pcts_ref = []
        cov_ranks = []
        pl_ranks = []
    
    topk_ranks = []
    topk_dists = []
    
    # Stratified visualization counters
    vis_counts = {
        'below_5': 0,
        '5_to_10': 0,
        '10_to_20': 0,
        '20_to_30': 0,
        'above_30': 0
    }
    
    vis_limits = {
        'below_5': 5,
        '5_to_10': 10,
        '10_to_20': 15,
        '20_to_30': 20,
        'above_30': 30
    }
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            if heatmap_type == 'gaussian':
                building_maps, gt_coords, building_ids = batch_data
                target_heatmap = None
            else:
                building_maps, gt_coords, target_heatmap, building_ids = batch_data
            
            building_maps = building_maps.to(device)
            gt_coords = gt_coords.to(device)
            
            logits, pred_coords = model(building_maps)
            
            # Compute loss
            output_size = logits.size(2)
            if target_heatmap is None:
                target_heatmap = create_gaussian_target(gt_coords, output_size, sigma=gaussian_sigma)
            else:
                target_heatmap = target_heatmap.to(device)
                if target_heatmap.size(2) != output_size:
                    target_heatmap = F.interpolate(target_heatmap, size=(output_size, output_size),
                                                  mode='bilinear', align_corners=False)
            
            heatmap_loss_total, heatmap_loss_dict = compute_multi_loss(logits, target_heatmap, loss_weights)
            coord_loss = torch.mean(torch.abs(pred_coords - gt_coords))
            loss = heatmap_loss_total + coord_weight * coord_loss
            heatmap_loss = heatmap_loss_dict.get('l2', 0.0)
            
            running_loss += loss.item()
            running_heatmap_loss += heatmap_loss
            running_coord_error += coord_loss.item()
            num_batches += 1
            
            # Per-sample comprehensive metrics
            batch_size = building_maps.size(0)
            for i in range(batch_size):
                pred_np = pred_coords[i].cpu().numpy()
                gt_np = gt_coords[i].cpu().numpy()
                building_id = building_ids[i]
                
                # Pixel error
                pixel_error = np.sqrt(np.sum((pred_np - gt_np)**2))
                pixel_errors.append(pixel_error)
                
                # Building density
                if buildings_dir_full:
                    density = compute_building_density(building_id, buildings_dir_full, use_center)
                    if density is not None:
                        building_densities.append(density)
                
                # RadioNet evaluation with rankings
                avg_metrics = None
                density = None
                if heatmap_base_dir and radionet_model and buildings_dir_full:
                    # Use include_rankings=True for full validation
                    avg_metrics = evaluate_with_avg_maps(
                        pred_np, gt_np, building_id, heatmap_base_dir, 
                        optimize_for, use_center, buildings_dir_full, radionet_model,
                        include_rankings=True
                    )
                    
                    if avg_metrics:
                        if optimize_for == 'power':
                            pl_pcts.append(avg_metrics['pl_pct'])
                            coverage_pcts_power.append(avg_metrics['coverage_pct_power'])
                            coverage_pcts_gt.append(avg_metrics['coverage_pct_gt'])
                        else:  # coverage
                            coverage_pcts.append(avg_metrics['coverage_pct'])
                            pl_pcts_coverage.append(avg_metrics['pl_pct_coverage'])
                            pl_pcts_ref.append(avg_metrics['pl_pct_ref'])
                        
                        # Rankings are now in avg_metrics
                        if 'rank_coverage' in avg_metrics:
                            cov_ranks.append(avg_metrics['rank_coverage'])
                        if 'rank_pl' in avg_metrics:
                            pl_ranks.append(avg_metrics['rank_pl'])
                
                # TopK analysis
                closest_rank = None
                if topk_dir:
                    topk_txs = load_topk_csv(building_id, topk_dir, use_center)
                    if topk_txs:
                        closest_rank, min_dist = find_closest_topk_rank(pred_np, topk_txs)
                        if closest_rank is not None:
                            topk_ranks.append(closest_rank)
                            topk_dists.append(min_dist)
                
                # Determine visualization category
                if pixel_error < 5:
                    category = 'below_5'
                elif pixel_error < 10:
                    category = '5_to_10'
                elif pixel_error < 20:
                    category = '10_to_20'
                elif pixel_error < 30:
                    category = '20_to_30'
                else:
                    category = 'above_30'
                
                # Check if we should visualize this sample
                should_visualize = (vis_counts[category] < vis_limits[category]) and save_dir is not None
                
                if should_visualize:
                    # Get building density if not already computed
                    if density is None and buildings_dir_full:
                        density = compute_building_density(building_id, buildings_dir_full, use_center)
                    
                    # Prepare data for visualization
                    building_map_np = building_maps[i, 0].cpu().numpy()
                    pred_heatmap_np = logits[i, 0].cpu().numpy()
                    
                    # Load GT PL maps
                    metrics_gt_power = load_pl_map_metrics(
                        building_id, heatmap_base_dir, 'best_by_power', 
                        use_center, buildings_dir_full, threshold=0.0
                    ) if heatmap_base_dir else None
                    
                    metrics_gt_coverage = load_pl_map_metrics(
                        building_id, heatmap_base_dir, 'best_by_coverage', 
                        use_center, buildings_dir_full, threshold=0.0
                    ) if heatmap_base_dir else None
                    
                    gt_pl_power = metrics_gt_power['pl_map'] if metrics_gt_power else None
                    gt_pl_coverage = metrics_gt_coverage['pl_map'] if metrics_gt_coverage else None
                    
                    # Generate RadioNet PL map for visualization
                    radionet_pl = None
                    metrics_pred_for_vis = None
                    if radionet_model and buildings_dir_full:
                        building_path = os.path.join(buildings_dir_full, f'{building_id}.png')
                        if os.path.exists(building_path):
                            try:
                                building_img = np.array(Image.open(building_path).convert('L')).astype(np.float32) / 255.0
                                
                                pred_y, pred_x = pred_np
                                if use_center:
                                    pred_y_256 = pred_y + 53
                                    pred_x_256 = pred_x + 53
                                else:
                                    pred_y_256, pred_x_256 = pred_y, pred_x
                                
                                tx_onehot = np.zeros((256, 256), dtype=np.float32)
                                pred_y_int = int(np.clip(pred_y_256, 0, 255))
                                pred_x_int = int(np.clip(pred_x_256, 0, 255))
                                tx_onehot[pred_y_int, pred_x_int] = 1.0
                                
                                building_tensor = torch.from_numpy(building_img).unsqueeze(0).unsqueeze(0)
                                tx_tensor = torch.from_numpy(tx_onehot).unsqueeze(0).unsqueeze(0)
                                inputs = torch.cat([building_tensor, tx_tensor], dim=1).to(device)
                                
                                power_map_256 = radionet_model(inputs)[0, 0].cpu().numpy()
                                
                                if use_center:
                                    radionet_pl = power_map_256[53:203, 53:203]
                                else:
                                    radionet_pl = power_map_256
                                
                                # Compute metrics from this map
                                metrics_pred_for_vis = compute_predicted_radio_metrics(
                                    pred_np, building_id, radionet_model, use_center, 
                                    buildings_dir_full, threshold=0.0
                                )
                            except Exception as e:
                                print(f"  Warning: Could not generate RadioNet visualization for {building_id}: {e}", flush=True)
                    
                    # Create comprehensive visualization
                    visualize_comprehensive(
                        building_map_np, gt_np, pred_np, building_id, epoch, i,
                        pred_heatmap_np, gt_pl_power, gt_pl_coverage, radionet_pl,
                        metrics_pred_for_vis, metrics_gt_power, metrics_gt_coverage,
                        pixel_error, save_dir, use_center, topk_rank=closest_rank,
                        avg_metrics=avg_metrics, building_density=density
                    )
                    
                    # Update counter
                    vis_counts[category] += 1
    
    # Compute comprehensive statistics
    metrics = {
        'loss': running_loss / num_batches,
        'heatmap_loss': running_heatmap_loss / num_batches,
        'coord_error': running_coord_error / num_batches
    }
    
    # Pixel error statistics
    if len(pixel_errors) > 0:
        pixel_errors_np = np.array(pixel_errors)
        metrics['coord_error_median'] = float(np.median(pixel_errors_np))
        metrics['coord_error_std'] = float(np.std(pixel_errors_np))
        metrics['coord_error_max'] = float(np.max(pixel_errors_np))
        
        # Histogram
        hist_bins = [0, 5, 10, 20, 30, np.inf]
        hist_labels = ['<5px', '5-10px', '10-20px', '20-30px', '>30px']
        hist, _ = np.histogram(pixel_errors_np, bins=hist_bins)
        metrics['pixel_error_histogram'] = {label: int(count) for label, count in zip(hist_labels, hist)}
    
    # Coverage/PL statistics based on optimize_for
    if optimize_for == 'power':
        if len(pl_pcts) > 0:
            pl_pcts_np = np.array(pl_pcts)
            metrics['avg_pl_pct'] = float(np.mean(pl_pcts_np))
            metrics['median_pl_pct'] = float(np.median(pl_pcts_np))
            metrics['std_pl_pct'] = float(np.std(pl_pcts_np))
            metrics['min_pl_pct'] = float(np.min(pl_pcts_np))
            metrics['max_pl_pct'] = float(np.max(pl_pcts_np))
            
            # Histogram
            hist_bins = [0, 20, 40, 60, 80, 100, 200]
            hist_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%', '>100%']
            hist, _ = np.histogram(pl_pcts_np, bins=hist_bins)
            metrics['pl_pct_histogram'] = {label: int(count) for label, count in zip(hist_labels, hist)}
            
            metrics['avg_coverage_pct_power'] = float(np.mean(coverage_pcts_power))
            metrics['avg_coverage_pct_gt'] = float(np.mean(coverage_pcts_gt))
    else:  # coverage
        if len(coverage_pcts) > 0:
            cov_pcts_np = np.array(coverage_pcts)
            metrics['avg_coverage_pct'] = float(np.mean(cov_pcts_np))
            metrics['median_coverage_pct'] = float(np.median(cov_pcts_np))
            metrics['std_coverage_pct'] = float(np.std(cov_pcts_np))
            metrics['min_coverage_pct'] = float(np.min(cov_pcts_np))
            metrics['max_coverage_pct'] = float(np.max(cov_pcts_np))
            
            # Histogram
            hist_bins = [0, 20, 40, 60, 80, 100, 200]
            hist_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%', '>100%']
            hist, _ = np.histogram(cov_pcts_np, bins=hist_bins)
            metrics['coverage_pct_histogram'] = {label: int(count) for label, count in zip(hist_labels, hist)}
            
            metrics['avg_pl_pct_coverage'] = float(np.mean(pl_pcts_coverage))
            metrics['avg_pl_pct_ref'] = float(np.mean(pl_pcts_ref))
    
    # Rankings
    if len(cov_ranks) > 0:
        metrics['avg_rank_cov'] = float(np.mean(cov_ranks))
        metrics['median_rank_cov'] = float(np.median(cov_ranks))
    if len(pl_ranks) > 0:
        metrics['avg_rank_pl'] = float(np.mean(pl_ranks))
        metrics['median_rank_pl'] = float(np.median(pl_ranks))
    
    # TopK
    if len(topk_ranks) > 0:
        metrics['avg_topk_rank'] = float(np.mean(topk_ranks))
        metrics['median_topk_rank'] = float(np.median(topk_ranks))
        metrics['avg_topk_dist'] = float(np.mean(topk_dists))
    
    # Building density
    if len(building_densities) > 0:
        density_np = np.array(building_densities)
        metrics['avg_building_density'] = float(np.mean(density_np))
        metrics['median_building_density'] = float(np.median(density_np))
        metrics['std_building_density'] = float(np.std(density_np))
        
        # Building density histogram
        hist_bins = [0, 10, 20, 30, 40, 50, 100]
        hist_labels = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '>50%']
        hist, _ = np.histogram(density_np, bins=hist_bins)
        metrics['building_density_histogram'] = {label: int(count) for label, count in zip(hist_labels, hist)}
        
        # Correlations with building density
        if len(pixel_errors) == len(building_densities):
            corr_density_error = np.corrcoef(density_np, pixel_errors_np)[0, 1]
            metrics['corr_density_vs_error'] = float(corr_density_error)
        
        # Correlation with coverage/PL percentages
        if optimize_for == 'power' and len(pl_pcts) == len(building_densities):
            corr_density_pl = np.corrcoef(density_np, pl_pcts_np)[0, 1]
            metrics['corr_density_vs_pl'] = float(corr_density_pl)
        elif optimize_for == 'coverage' and len(coverage_pcts) == len(building_densities):
            corr_density_cov = np.corrcoef(density_np, cov_pcts_np)[0, 1]
            metrics['corr_density_vs_coverage'] = float(corr_density_cov)
        
        # Per-density-bin analysis
        density_bins = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 100)]
        density_analysis = {}
        
        for bin_min, bin_max in density_bins:
            mask = (density_np >= bin_min) & (density_np < bin_max)
            if mask.sum() > 0:
                bin_label = f'{bin_min}-{bin_max}%'
                density_analysis[bin_label] = {
                    'count': int(mask.sum()),
                    'pixel_error_mean': float(pixel_errors_np[mask].mean()),
                    'pixel_error_median': float(np.median(pixel_errors_np[mask]))
                }
                
                # Add coverage/PL specific analysis
                if optimize_for == 'power' and len(pl_pcts) == len(building_densities):
                    density_analysis[bin_label]['pl_pct_mean'] = float(pl_pcts_np[mask].mean())
                elif optimize_for == 'coverage' and len(coverage_pcts) == len(building_densities):
                    density_analysis[bin_label]['coverage_pct_mean'] = float(cov_pcts_np[mask].mean())
        
        metrics['density_analysis'] = density_analysis
    
    print(f"  Visualizations: <5px={vis_counts['below_5']}, 5-10px={vis_counts['5_to_10']}, "
          f"10-20px={vis_counts['10_to_20']}, 20-30px={vis_counts['20_to_30']}, >30px={vis_counts['above_30']}", flush=True)
    
    return metrics


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device,
               save_dir, use_center, heatmap_type, heatmap_weight, coord_weight, img_size,
               optimize_for, full_val_interval, radionet_model=None, topk_dir=None, 
               buildings_dir_full=None, heatmap_base_dir=None, loss_weights=None,
               gaussian_sigma=3.0):
    """Main training loop with two-tier validation and multiple best model tracking."""
    
    print(f"\n{'='*70}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Epochs: {num_epochs}")
    print(f"Optimize for: {optimize_for}")
    print(f"Heatmap type: {heatmap_type}")
    print(f"Loss weights - Heatmap: {heatmap_weight}, Coord: {coord_weight}")
    print(f"Validation: Light (every epoch), Full (every {full_val_interval} epochs)")
    print(f"{'='*70}\n")
    
    # Initialize best model tracking
    if optimize_for == 'power':
        best_pl_pct = -float('inf')
        best_pl_epoch = -1
    else:  # coverage
        best_coverage_pct = -float('inf')
        best_coverage_epoch = -1
        best_pl_from_cov_pct = -float('inf')
        best_pl_from_cov_epoch = -1
    
    best_coord_error = float('inf')
    best_coord_epoch = -1
    
    history = defaultdict(list)
    
    # Create CSV log file
    csv_path = os.path.join(save_dir, 'training_log.csv')
    csv_file = open(csv_path, 'w', newline='')
    
    if optimize_for == 'power':
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['epoch', 'train_loss', 'val_loss', 'coord_error', 'coord_error_median', 
                            'avg_pl_pct', 'median_pl_pct', 'std_pl_pct', 
                            'avg_coverage_pct_power', 'avg_coverage_pct_gt', 'lr'])
    else:  # coverage
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['epoch', 'train_loss', 'val_loss', 'coord_error', 'coord_error_median',
                            'avg_coverage_pct', 'median_coverage_pct', 'std_coverage_pct',
                            'avg_pl_pct_coverage', 'avg_pl_pct_ref', 'lr'])
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            heatmap_type, heatmap_weight, coord_weight, img_size, loss_weights,
            gaussian_sigma=gaussian_sigma
        )
        
        # Validation: Light or Full
        is_full_val = (epoch % full_val_interval == 0) or (epoch == num_epochs - 1)
        
        if is_full_val:
            print(f"[Epoch {epoch+1}] Running FULL validation...", flush=True)
            val_metrics = validate_one_epoch_full(
                model, val_loader, device, epoch, save_dir, use_center,
                heatmap_type, heatmap_weight, coord_weight, img_size,
                optimize_for, radionet_model, topk_dir, buildings_dir_full,
                heatmap_base_dir, loss_weights, gaussian_sigma=gaussian_sigma
            )
            
            # Save detailed JSON for full validation
            json_path = os.path.join(save_dir, f'metrics_epoch{epoch}.json')
            with open(json_path, 'w') as f:
                json.dump(val_metrics, f, indent=2)
        else:
            val_metrics = validate_one_epoch_light(
                model, val_loader, device, epoch, save_dir, use_center,
                heatmap_type, heatmap_weight, coord_weight, img_size,
                optimize_for, radionet_model, buildings_dir_full,
                heatmap_base_dir, loss_weights, gaussian_sigma=gaussian_sigma
            )
        
        # Scheduler step
        if scheduler is not None:
            scheduler.step()
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train: {train_metrics['loss']:.4f} (HM: {train_metrics['heatmap_loss']:.4f}) | "
              f"Val: {val_metrics['loss']:.4f} | "
              f"Error: {val_metrics['coord_error']:.2f}px", end='')
        
        if 'coord_error_median' in val_metrics:
            print(f" (med: {val_metrics['coord_error_median']:.2f}px)", end='')
        
        print(f" | LR: {current_lr:.2e} | {epoch_time:.1f}s")
        
        # Print optimize-specific metrics
        if optimize_for == 'power':
            if 'avg_pl_pct' in val_metrics:
                print(f"  PL: {val_metrics['avg_pl_pct']:.1f}%", end='')
                if 'median_pl_pct' in val_metrics:
                    print(f" (med: {val_metrics['median_pl_pct']:.1f}%)", end='')
                if 'avg_coverage_pct_gt' in val_metrics:
                    print(f" | CovGT: {val_metrics['avg_coverage_pct_gt']:.1f}%", end='')
                print()
        else:  # coverage
            if 'avg_coverage_pct' in val_metrics:
                print(f"  Cov: {val_metrics['avg_coverage_pct']:.1f}%", end='')
                if 'median_coverage_pct' in val_metrics:
                    print(f" (med: {val_metrics['median_coverage_pct']:.1f}%)", end='')
                if 'avg_pl_pct_coverage' in val_metrics:
                    print(f" | PL: {val_metrics['avg_pl_pct_coverage']:.1f}%", end='')
                if 'avg_pl_pct_ref' in val_metrics:
                    print(f" | PLref: {val_metrics['avg_pl_pct_ref']:.1f}%", end='')
                print()
        
        # Print rankings if available
        if is_full_val and 'avg_rank_pl' in val_metrics:
            print(f"  Ranks - PL: {val_metrics['avg_rank_pl']:.0f}, Cov: {val_metrics['avg_rank_cov']:.0f}", end='')
            if 'avg_topk_rank' in val_metrics:
                print(f" | TopK: {val_metrics['avg_topk_rank']:.1f}", end='')
            print()
        
        # Update history
        for k, v in train_metrics.items():
            history[f'train_{k}'].append(v)
        for k, v in val_metrics.items():
            history[f'val_{k}'].append(v)
        
        # Write to CSV
        if optimize_for == 'power':
            csv_writer.writerow([
                epoch, train_metrics['loss'], val_metrics['loss'],
                val_metrics['coord_error'], val_metrics.get('coord_error_median', 0),
                val_metrics.get('avg_pl_pct', 0), val_metrics.get('median_pl_pct', 0),
                val_metrics.get('std_pl_pct', 0),
                val_metrics.get('avg_coverage_pct_power', 0), val_metrics.get('avg_coverage_pct_gt', 0),
                current_lr
            ])
        else:  # coverage
            csv_writer.writerow([
                epoch, train_metrics['loss'], val_metrics['loss'],
                val_metrics['coord_error'], val_metrics.get('coord_error_median', 0),
                val_metrics.get('avg_coverage_pct', 0), val_metrics.get('median_coverage_pct', 0),
                val_metrics.get('std_coverage_pct', 0),
                val_metrics.get('avg_pl_pct_coverage', 0), val_metrics.get('avg_pl_pct_ref', 0),
                current_lr
            ])
        csv_file.flush()
        
        # Track and save best models
        if optimize_for == 'power':
            # Best PL model
            if 'avg_pl_pct' in val_metrics and val_metrics['avg_pl_pct'] > best_pl_pct:
                best_pl_pct = val_metrics['avg_pl_pct']
                best_pl_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'pl_pct': best_pl_pct,
                    'history': dict(history)
                }, os.path.join(save_dir, 'best_pl_model.pth'))
                print(f"  ★ New best PL: {best_pl_pct:.2f}%")
        
        else:  # coverage
            # Best coverage model
            if 'avg_coverage_pct' in val_metrics and val_metrics['avg_coverage_pct'] > best_coverage_pct:
                best_coverage_pct = val_metrics['avg_coverage_pct']
                best_coverage_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'coverage_pct': best_coverage_pct,
                    'history': dict(history)
                }, os.path.join(save_dir, 'best_coverage_model.pth'))
                print(f"  ★ New best Coverage: {best_coverage_pct:.2f}%")
            
            # Best PL from coverage model
            if 'avg_pl_pct_coverage' in val_metrics and val_metrics['avg_pl_pct_coverage'] > best_pl_from_cov_pct:
                best_pl_from_cov_pct = val_metrics['avg_pl_pct_coverage']
                best_pl_from_cov_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'pl_pct': best_pl_from_cov_pct,
                    'history': dict(history)
                }, os.path.join(save_dir, 'best_pl_from_cov_model.pth'))
                print(f"  ★ New best PL (from cov): {best_pl_from_cov_pct:.2f}%")
        
        # Best coord error model (always track)
        if val_metrics['coord_error'] < best_coord_error:
            best_coord_error = val_metrics['coord_error']
            best_coord_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'coord_error': best_coord_error,
                'history': dict(history)
            }, os.path.join(save_dir, 'best_coord_model.pth'))
            print(f"  ★ New best coord error: {best_coord_error:.2f}px")
    
    csv_file.close()
    
    # Print final summary
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    
    if optimize_for == 'power':
        print(f"Best PL%: {best_pl_pct:.2f}% (epoch {best_pl_epoch})")
    else:
        print(f"Best Coverage%: {best_coverage_pct:.2f}% (epoch {best_coverage_epoch})")
        print(f"Best PL% (from cov): {best_pl_from_cov_pct:.2f}% (epoch {best_pl_from_cov_epoch})")
    
    print(f"Best coord error: {best_coord_error:.2f}px (epoch {best_coord_epoch})")
    print(f"Models saved to: {save_dir}")
    print(f"{'='*70}\n")
    
    return model, history


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Deep Tx Localization with Comprehensive Validation')
    
    # Architecture
    parser.add_argument('--arch', type=str, default='deepxl_150',
                       choices=['deepxl_150', 'deepxl_150r', 'radionet_150', 'ms_150b', 'ms_150c', 'ms_150e',
                               'ultradeep_150', 'deepxl_256', 'deepxxl_256', 'ultradeep_256', 'hyperdeep_256',
                               'locunet_256', 'locunet_256_wide', 'locunet_150', 'locunet_150_wide',
                               'pmnet_150', 'sip2net_150', 'dcnet_150'],
                       help='Deep model architecture')
    parser.add_argument('--coord_method', type=str, default='soft_argmax',
                       choices=['soft_argmax', 'hard_argmax', 'center_of_mass'],
                       help='Coordinate extraction method')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for soft-argmax')
    parser.add_argument('--use_masking', action='store_true',
                       help='Mask building pixels in soft-argmax')
    
    # Heatmap configuration
    parser.add_argument('--optimize_for', type=str, default='power',
                       choices=['coverage', 'power'],
                       help='Optimize for coverage or power (determines GT Tx source)')
    parser.add_argument('--heatmap_type', type=str, default='radio_power',
                       choices=['gaussian', 'radio_coverage', 'radio_power',
                               'avg_coverage_free', 'avg_coverage_total'],
                       help='Type of heatmap to use as training target')
    parser.add_argument('--normalize_heatmap', action='store_true',
                       help='Normalize loaded heatmaps to [0,1]')
    parser.add_argument('--heatmap_weight', type=float, default=1000.0,
                       help='Weight for heatmap loss')
    parser.add_argument('--coord_weight', type=float, default=0.1,
                       help='Weight for coordinate loss')
    parser.add_argument('--gaussian_sigma', type=float, default=3.0,
                       help='Sigma for Gaussian target heatmap (only used when heatmap_type=gaussian)')
    parser.add_argument('--heatmap_base_dir', type=str,
                       default='./data/unified_results',
                       help='Base directory for physical heatmaps')
    
    # Data
    parser.add_argument('--use_center', action='store_true',
                       help='Use center 150x150 crop instead of full 256x256')
    parser.add_argument('--augment', action='store_true',
                       help='Enable D4 augmentation')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    
    # Training
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--full_val_interval', type=int, default=20,
                       help='Run full validation every N epochs')
    
    # Paths
    parser.add_argument('--buildings_dir', type=str,
                       default='./data/buildings')
    parser.add_argument('--tx_dir', type=str, default=None,
                       help='Directory with Tx images (auto-set if None)')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--topk_dir', type=str, default=None,
                       help='Directory with topK CSV files')
    parser.add_argument('--radionet_path', type=str, default=None,
                       help='Path to RadioNet checkpoint')
    
    # Advanced loss weights
    parser.add_argument('--loss_l2_weight', type=float, default=None,
                       help='Weight for L2/MSE heatmap loss')
    parser.add_argument('--loss_l1_weight', type=float, default=0.0)
    parser.add_argument('--loss_ssim_weight', type=float, default=0.0)
    parser.add_argument('--loss_ms_ssim_weight', type=float, default=0.0)
    parser.add_argument('--loss_tv_weight', type=float, default=0.0)
    parser.add_argument('--loss_gdl_weight', type=float, default=0.0)
    parser.add_argument('--loss_ms_gsim_weight', type=float, default=0.0)
    parser.add_argument('--loss_focal_weight', type=float, default=0.0)
    parser.add_argument('--loss_coverage_binary_weight', type=float, default=0.0,
                   help='Weight for coverage binary loss')
    
    args = parser.parse_args()
    
    # Auto-set tx_dir based on optimize_for
    if args.tx_dir is None:
        args.tx_dir = os.path.join(args.heatmap_base_dir, f'best_by_{args.optimize_for}')
        print(f"Auto-set tx_dir to: {args.tx_dir}")
    
    # Backward compatibility for loss weights
    if args.loss_l2_weight is None:
        args.loss_l2_weight = args.heatmap_weight
    else:
        args.heatmap_weight = args.loss_l2_weight
    
    # Create loss weights dictionary
    loss_weights = {
        'l2': args.loss_l2_weight,
        'l1': args.loss_l1_weight,
        'ssim': args.loss_ssim_weight,
        'ms_ssim': args.loss_ms_ssim_weight,
        'tv': args.loss_tv_weight,
        'gdl': args.loss_gdl_weight,
        'ms_gsim': args.loss_ms_gsim_weight,
        'focal': args.loss_focal_weight,
        'coverage_binary': args.loss_coverage_binary_weight
    }
    
    # Print active loss components
    print("\n" + "="*80)
    print("ACTIVE LOSS COMPONENTS:")
    print("="*80)
    for key, weight in loss_weights.items():
        if weight > 0:
            print(f"  {key.upper()}: {weight}")
    print(f"  COORD: {args.coord_weight}")
    print("="*80 + "\n")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save args
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Image size
    img_size = 150 if args.use_center else 256
    
    # Load RadioNet
    radionet_model = None
    if args.radionet_path and os.path.exists(args.radionet_path):
        try:
            radionet_model = RadioNet(inputs=2).to(device)
            checkpoint = torch.load(args.radionet_path, map_location=device)
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    radionet_model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    radionet_model.load_state_dict(checkpoint['state_dict'])
                elif 'model' in checkpoint:
                    radionet_model.load_state_dict(checkpoint['model'])
                else:
                    radionet_model.load_state_dict(checkpoint)
            else:
                radionet_model.load_state_dict(checkpoint)
            
            radionet_model.eval()
            print(f"[RadioNet] Loaded from {args.radionet_path}")
        except Exception as e:
            print(f"[RadioNet] Warning: Could not load: {e}")
            radionet_model = None
    
    print(f"\n{'='*70}")
    print(f"CONFIGURATION")
    print(f"{'='*70}")
    print(f"Architecture: {args.arch.upper()}")
    print(f"Image size: {img_size}x{img_size}")
    print(f"Optimize for: {args.optimize_for}")
    print(f"Heatmap type: {args.heatmap_type}")
    print(f"Normalize heatmap: {args.normalize_heatmap}")
    print(f"Loss weights - Heatmap: {args.heatmap_weight}, Coord: {args.coord_weight}")
    print(f"Coord method: {args.coord_method} (T={args.temperature})")
    print(f"Augmentation: {args.augment}")
    print(f"RadioNet: {'ENABLED' if radionet_model else 'DISABLED'}")
    print(f"TopK Analysis: {'ENABLED' if args.topk_dir else 'DISABLED'}")
    print(f"Full validation interval: every {args.full_val_interval} epochs")
    print(f"{'='*70}\n")
    
    # Create model
    model = create_model_deep(
        arch=args.arch,
        coord_method=args.coord_method,
        temperature=args.temperature,
        use_masking=args.use_masking,
        img_size=img_size
    ).to(device)
    
    # Load data
    train_ids, val_ids, test_ids = get_building_splits(args.buildings_dir)
    
    train_dataset = TxLocationDatasetLarge(
        train_ids,
        buildings_dir=args.buildings_dir,
        tx_dir=args.tx_dir,
        optimize_for=args.optimize_for,
        heatmap_type=args.heatmap_type,
        heatmap_base_dir=args.heatmap_base_dir,
        normalize_heatmap=args.normalize_heatmap,
        augment=args.augment,
        use_center=args.use_center
    )
    
    val_dataset = TxLocationDatasetLarge(
        val_ids,
        buildings_dir=args.buildings_dir,
        tx_dir=args.tx_dir,
        optimize_for=args.optimize_for,
        heatmap_type=args.heatmap_type,
        heatmap_base_dir=args.heatmap_base_dir,
        normalize_heatmap=args.normalize_heatmap,
        augment=False,
        use_center=args.use_center
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=args.num_workers,
                             pin_memory=True)
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers,
                           pin_memory=True)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
    
    # Train
    model, history = train_model(
        model, train_loader, val_loader, optimizer, scheduler,
        num_epochs=args.num_epochs, device=device, save_dir=args.save_dir,
        use_center=args.use_center, heatmap_type=args.heatmap_type,
        heatmap_weight=args.heatmap_weight, coord_weight=args.coord_weight,
        img_size=img_size, optimize_for=args.optimize_for,
        full_val_interval=args.full_val_interval, radionet_model=radionet_model,
        topk_dir=args.topk_dir, buildings_dir_full=args.buildings_dir,
        heatmap_base_dir=args.heatmap_base_dir, loss_weights=loss_weights,
        gaussian_sigma=args.gaussian_sigma
    )


if __name__ == '__main__':
    main()
