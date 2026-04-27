"""
Shared Utilities for AvgMap Evaluation

Functions used by both discriminative and diffusion avgmap evaluations:
- Top-N candidate extraction from predicted average maps
- Batched RadioNet evaluation  
- Dual-GT metric computation
- Results aggregation

This module provides common infrastructure so discriminative and diffusion
evaluation scripts can focus on their model-specific forward pass logic.
"""

import numpy as np
import torch
from PIL import Image
import os
import sys

sys.path.append('.')
from models.saipp_net import RadioNet


def extract_topn_candidates(pred_avgmap, building_mask, n=100, use_center=True):
    """
    Extract top-N candidate TX locations from predicted average map.
    
    Args:
        pred_avgmap: Predicted average map (numpy array, 150x150 or 256x256)
        building_mask: Boolean mask of valid building pixels
        n: Number of top candidates to extract
        use_center: Whether using center crop
    
    Returns:
        List of (y, x) coordinates
    """
    # Mask out non-building pixels
    masked_map = pred_avgmap.copy()
    masked_map[~building_mask] = -np.inf
    
    # Flatten and get top-N indices
    flat_indices = np.argsort(masked_map.flatten())[::-1]
    
    # Convert to 2D coordinates
    img_size = pred_avgmap.shape[0]
    candidates = []
    
    for flat_idx in flat_indices:
        y = flat_idx // img_size
        x = flat_idx % img_size
        
        # Verify it's a valid building pixel
        if building_mask[y, x]:
            candidates.append((float(y), float(x)))
        
        if len(candidates) >= n:
            break
    
    return candidates


def evaluate_single_location_radionet(coord, building_id, buildings_dir, radionet_model, 
                                     use_center, device):
    """
    Evaluate a single TX location using RadioNet.
    Matches compute_predicted_radio_metrics() in training/train_scoremap.py exactly.
    
    Returns dict with pred_coverage and pred_pl (both in 0-255 uint8 scale).
    """
    building_path = os.path.join(buildings_dir, f'{building_id}.png')
    if not os.path.exists(building_path):
        return None
    
    try:
        # Load building
        building_img = np.array(Image.open(building_path).convert('L')).astype(np.float32) / 255.0
        
        pred_y, pred_x = coord
        
        # Convert to 256x256 coordinates if using center
        if use_center:
            pred_y_256 = pred_y + 53
            pred_x_256 = pred_x + 53
        else:
            pred_y_256, pred_x_256 = pred_y, pred_x
        
        # Create TX one-hot
        tx_onehot = np.zeros((256, 256), dtype=np.float32)
        pred_y_int = int(np.clip(pred_y_256, 0, 255))
        pred_x_int = int(np.clip(pred_x_256, 0, 255))
        tx_onehot[pred_y_int, pred_x_int] = 1.0
        
        # RadioNet forward — matching training exactly
        building_tensor = torch.from_numpy(building_img).unsqueeze(0).unsqueeze(0)
        tx_tensor = torch.from_numpy(tx_onehot).unsqueeze(0).unsqueeze(0)
        inputs = torch.cat([building_tensor, tx_tensor], dim=1).to(next(radionet_model.parameters()).device)
        
        with torch.no_grad():
            power_map_256 = radionet_model(inputs)[0, 0].cpu().numpy()
        
        # Scale RadioNet output from [0,1] to [0,255] to match GT PL maps
        power_map_256 = np.clip(power_map_256 * 255, 0, 255).astype(np.uint8)
        
        # Center crop and compute metrics — matching training exactly
        power_center = power_map_256[53:203, 53:203]
        building_center = building_img[53:203, 53:203]
        center_free = (building_center <= 0.1)
        
        # Extract power values at free center pixels
        center_free_values = power_center[center_free]
        
        # Coverage count: free pixels with signal > threshold
        coverage_count = int(np.sum(center_free_values > 0.0))
        
        # Average power: mean of free center pixels
        avg_power = float(np.mean(center_free_values)) if center_free_values.size > 0 else 0.0
        
        return {
            'coord': coord,
            'pred_coverage': float(coverage_count),
            'pred_pl': avg_power
        }
        
    except Exception as e:
        print(f"Warning: RadioNet evaluation failed for building {building_id}: {e}", flush=True)
        return None


def evaluate_candidates_batch_radionet(candidates, building_id, buildings_dir, radionet_model,
                                      use_center, device, batch_size=64):
    """
    Evaluate multiple candidates with RadioNet.
    
    Args:
        candidates: List of (y, x) coordinates
        building_id: Building ID
        buildings_dir: Path to building images
        radionet_model: RadioNet model
        use_center: Whether using center crop
        device: torch device
        batch_size: Ignored (kept for API compatibility)
    
    Returns:
        List of dicts with metrics for each candidate
    """
    results = []
    
    for coord in candidates:
        metrics = evaluate_single_location_radionet(
            coord, building_id, buildings_dir, radionet_model, use_center, device
        )
        if metrics is not None:
            results.append(metrics)
    
    return results


def load_pl_map_metrics(building_id, heatmap_base_dir, optimize_dir, use_center, 
                       buildings_dir, threshold=0.0):
    """
    Load pre-computed PL map metrics for GT location.
    Matches load_pl_map_metrics() in training/train_scoremap.py exactly.
    
    Args:
        building_id: Building ID
        heatmap_base_dir: Base directory for heatmaps
        optimize_dir: 'best_by_power' or 'best_by_coverage'
        use_center: Whether using center crop
        buildings_dir: Path to building images
        threshold: Power threshold for coverage
    
    Returns:
        Dict with avg_power and coverage_count, or None if not found
    """
    pl_path = os.path.join(heatmap_base_dir, optimize_dir, f'{building_id}_PL.png')
    building_path = os.path.join(buildings_dir, f'{building_id}.png')
    
    if not os.path.exists(pl_path) or not os.path.exists(building_path):
        return None
    
    try:
        # Load PL map as uint8 (0-255) — matching training
        pl_map = np.array(Image.open(pl_path).convert('L'))
        building_img = np.array(Image.open(building_path).convert('L')).astype(np.float32) / 255.0
        
        # Get center region — matching training
        if use_center:
            pl_map_center = pl_map[53:203, 53:203]
            building_center = building_img[53:203, 53:203]
        else:
            pl_map_center = pl_map
            building_center = building_img
        
        # Get free pixels — matching training: <= 0.1
        free_space = (building_center <= 0.1)
        center_free_values = pl_map_center[free_space]
        
        # Coverage count: free pixels with signal > threshold
        coverage_count = int(np.sum(center_free_values > threshold))
        
        # Average power: mean of free center pixels
        avg_power = float(np.mean(center_free_values)) if len(center_free_values) > 0 else 0.0
        
        return {
            'avg_power': avg_power,
            'coverage_count': float(coverage_count)
        }
        
    except Exception as e:
        print(f"Warning: Could not load PL map for building {building_id}: {e}", flush=True)
        return None


def compute_dual_gt_metrics(pred_coord, power_gt, coverage_gt, candidate_metrics,
                           building_id, heatmap_base_dir, optimize_for, use_center, 
                           buildings_dir):
    """
    Compute all coordinate errors and radio metrics against BOTH GTs.
    
    Args:
        pred_coord: Predicted (y, x) coordinates
        power_gt: Power-optimal GT coordinates
        coverage_gt: Coverage-optimal GT coordinates
        candidate_metrics: RadioNet metrics for predicted location
        building_id: Building ID
        heatmap_base_dir: Base directory for heatmaps
        optimize_for: 'power' or 'coverage'
        use_center: Whether using center crop
        buildings_dir: Path to building images
    
    Returns:
        Dict with all metrics
    """
    pred_coord_np = np.array(pred_coord, dtype=np.float32)
    power_gt_np = np.array(power_gt, dtype=np.float32)
    coverage_gt_np = np.array(coverage_gt, dtype=np.float32)
    
    # Coordinate errors
    coord_error_power = float(np.sqrt(np.sum((pred_coord_np - power_gt_np)**2)))
    coord_error_coverage = float(np.sqrt(np.sum((pred_coord_np - coverage_gt_np)**2)))
    coord_error_power_L1_train = float(np.mean(np.abs(pred_coord_np - power_gt_np)))
    coord_error_coverage_L1_train = float(np.mean(np.abs(pred_coord_np - coverage_gt_np)))
    coord_error_power_L1_manhattan = float(np.sum(np.abs(pred_coord_np - power_gt_np)))
    coord_error_coverage_L1_manhattan = float(np.sum(np.abs(pred_coord_np - coverage_gt_np)))
    
    # Radio metrics from RadioNet (now in 0-255 scale, matching GT)
    pred_coverage = candidate_metrics['pred_coverage']
    pred_pl = candidate_metrics['pred_pl']
    
    # Load GT metrics
    power_gt_metrics = load_pl_map_metrics(
        building_id, heatmap_base_dir, 'best_by_power',
        use_center, buildings_dir, threshold=0.0
    )
    
    coverage_gt_metrics = load_pl_map_metrics(
        building_id, heatmap_base_dir, 'best_by_coverage',
        use_center, buildings_dir, threshold=0.0
    )
    
    if power_gt_metrics is None or coverage_gt_metrics is None:
        return None
    
    # Compute all 4 cross metrics
    gt_coverage_ref = coverage_gt_metrics['coverage_count']
    gt_coverage_power = power_gt_metrics['coverage_count']
    gt_pl_ref = power_gt_metrics['avg_power']
    gt_pl_cov = coverage_gt_metrics['avg_power']
    
    coverage_pct_ref = (pred_coverage / gt_coverage_ref * 100.0) if gt_coverage_ref > 0 else 0.0
    coverage_pct_power = (pred_coverage / gt_coverage_power * 100.0) if gt_coverage_power > 0 else 0.0
    pl_pct_ref = (pred_pl / gt_pl_ref * 100.0) if gt_pl_ref > 0 else 0.0
    pl_pct_cov = (pred_pl / gt_pl_cov * 100.0) if gt_pl_cov > 0 else 0.0
    
    return {
        'coord_error_power': coord_error_power,
        'coord_error_coverage': coord_error_coverage,
        'coord_error_power_L1_train': coord_error_power_L1_train,
        'coord_error_coverage_L1_train': coord_error_coverage_L1_train,
        'coord_error_power_L1_manhattan': coord_error_power_L1_manhattan,
        'coord_error_coverage_L1_manhattan': coord_error_coverage_L1_manhattan,
        'coverage_pct_ref': coverage_pct_ref,
        'coverage_pct_power': coverage_pct_power,
        'pl_pct_ref': pl_pct_ref,
        'pl_pct_cov': pl_pct_cov,
        'pred_coord': pred_coord,
        'pred_coverage': pred_coverage,
        'pred_pl': pred_pl
    }


def select_best_candidate(candidates_metrics, criterion='coverage'):
    """
    Select best candidate from list based on criterion.
    
    Args:
        candidates_metrics: List of dicts with RadioNet metrics
        criterion: 'coverage' or 'power'
    
    Returns:
        (best_index, best_metrics)
    """
    if criterion == 'coverage':
        best_idx = max(range(len(candidates_metrics)), 
                      key=lambda i: candidates_metrics[i]['pred_coverage'])
    else:  # power
        best_idx = max(range(len(candidates_metrics)),
                      key=lambda i: candidates_metrics[i]['pred_pl'])
    
    return best_idx, candidates_metrics[best_idx]


def aggregate_results(results_list):
    """
    Aggregate list of per-building results into mean/std/median statistics.
    
    Args:
        results_list: List of dicts with per-building metrics
    
    Returns:
        Dict with aggregated statistics
    """
    if not results_list:
        return {}
    
    # Extract all metrics
    metrics = {}
    for key in results_list[0].keys():
        if key not in ['pred_coord', 'building_id', 'model_rank']:  # Skip non-numeric
            values = [r[key] for r in results_list if key in r and r[key] is not None]
            if values:
                metrics[f'{key}_mean'] = float(np.mean(values))
                metrics[f'{key}_std'] = float(np.std(values))
                metrics[f'{key}_median'] = float(np.median(values))
    
    metrics['num_buildings'] = len(results_list)
    
    return metrics


def convert_to_json_serializable(obj):
    """Convert numpy types to JSON-serializable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    else:
        return obj
