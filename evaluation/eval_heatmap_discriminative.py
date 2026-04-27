"""
Evaluate Fixed (Non-Diffusion) Models on Test Set - DUAL GT VERSION

Evaluates trained models from training/train_heatmap.py on the test set
using the same dual ground truth approach as the diffusion model evaluation.

Key Features:
- Evaluates on TEST set (not validation)
- Computes coordinate errors against BOTH power-optimal and coverage-optimal GTs
- Three distance metrics: L2, L1-train, L1-manhattan
- Radio propagation metrics: coverage%, PL%
- Generates quantile-based visualizations

Usage:
    python evaluate_fixed_DUAL_GT_test_set.py \
        --model_path /path/to/best_pl_model.pth \
        --arch deepxl_150 \
        --optimize_for power \
        --heatmap_type radio_power \
        --use_center \
        --save_dir evaluation_results/model_name
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json
import time
from tqdm import tqdm


def get_hardware_info():
    """Capture hardware information for timing context."""
    import subprocess
    
    hardware_info = {
        'device_type': 'cpu',
        'gpu_name': None,
        'gpu_memory_gb': None,
        'cuda_version': None,
        'gpu_driver_version': None,
        'gpu_count': 0
    }
    
    if torch.cuda.is_available():
        hardware_info['device_type'] = 'cuda'
        hardware_info['gpu_count'] = torch.cuda.device_count()
        
        # Get GPU name and memory
        device_props = torch.cuda.get_device_properties(0)
        hardware_info['gpu_name'] = device_props.name
        hardware_info['gpu_memory_gb'] = round(device_props.total_memory / 1024**3, 2)
        
        # Get CUDA version
        hardware_info['cuda_version'] = torch.version.cuda
        
        # Try to get driver version from nvidia-smi
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                hardware_info['gpu_driver_version'] = result.stdout.strip().split('\n')[0]
        except:
            pass
    
    return hardware_info


def format_hardware_info(hw_info):
    """Format hardware info for display."""
    if hw_info['device_type'] == 'cuda':
        info = f"GPU: {hw_info['gpu_name']} ({hw_info['gpu_memory_gb']}GB)"
        if hw_info['cuda_version']:
            info += f", CUDA {hw_info['cuda_version']}"
        if hw_info['gpu_driver_version']:
            info += f", Driver {hw_info['gpu_driver_version']}"
        return info
    else:
        return "CPU"


# Import model creation and evaluation functions from training script
from models.discriminative import create_model_deep
from data.dataset_heatmap import TxLocationDatasetLarge, get_building_splits

# Import RadioNet
sys.path.append('.')
from models.saipp_net import RadioNet

# Import evaluation functions from training script
from training.train_heatmap import (
    evaluate_with_avg_maps,
    load_pl_map_metrics,
)

# Import helper functions from multisample evaluation
from evaluation.eval_heatmap_utils import (
    convert_to_json_serializable,
    generate_radio_maps,
    compute_coverage_map,
    save_simple_grayscale,
    save_map_with_tx_overlay,
    save_info_txt
)


def evaluate_fixed_model_dual_gt(model, test_dataset, power_dataset, coverage_dataset,
                                 device, optimize_for, use_center, buildings_dir,
                                 radionet_model, heatmap_base_dir):
    """
    Evaluate a fixed (non-diffusion) model on test set with dual GT.
    
    Args:
        model: Trained model
        test_dataset: Test dataset (based on optimize_for)
        power_dataset: Test dataset with power-optimal GTs
        coverage_dataset: Test dataset with coverage-optimal GTs
        device: torch device
        optimize_for: 'power' or 'coverage'
        use_center: Whether center cropping was used
        buildings_dir: Directory with building images
        radionet_model: RadioNet for computing coverage/PL
        heatmap_base_dir: Base directory for heatmaps
    
    Returns:
        dict: Evaluation results with dual GT errors and all metrics
    """
    model.eval()
    
    # Storage for results
    results = {
        'coord_errors_power': [],  # L2 error vs power-optimal GT
        'coord_errors_coverage': [],  # L2 error vs coverage-optimal GT
        'coord_errors_power_L1_train': [],  # L1 training-style vs power GT
        'coord_errors_coverage_L1_train': [],  # L1 training-style vs coverage GT
        'coord_errors_power_L1_manhattan': [],  # L1 Manhattan vs power GT
        'coord_errors_coverage_L1_manhattan': [],  # L1 Manhattan vs coverage GT
        'coverage_pcts': [],  # Coverage vs coverage-optimal GT
        'coverage_pct_powers': [],  # Coverage vs power-optimal GT
        'pl_pct_covs': [],  # PL vs coverage-optimal GT's PL
        'pl_pct_refs': [],  # PL vs power-optimal GT's PL
        # Per-building data for visualization
        'building_ids': [],
        'pred_coords': [],
        # Timing data
        'model_times': [],  # Model inference time per building
        'radionet_times': [],  # RadioNet evaluation time per building
        'total_times': []  # Total time per building
    }
    
    print(f"\nEvaluating {len(test_dataset)} buildings on test set...")
    
    import time
    overall_start = time.time()
    
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="Evaluating"):
            building_start = time.time()
            
            # Get building data - handle both gaussian (3 items) and radio (4 items) heatmap types
            test_data = test_dataset[idx]
            if len(test_data) == 3:
                # Gaussian heatmap: (building_map, gt_coords, building_id)
                building_map, gt_coords, building_id = test_data
            else:
                # Radio heatmap: (building_map, gt_coords, heatmap, building_id)
                building_map, gt_coords, _, building_id = test_data
            
            # Get GTs from both datasets (always radio heatmaps, so 4 items)
            power_data = power_dataset[idx]
            if len(power_data) == 3:
                _, power_gt, power_building_id = power_data
            else:
                _, power_gt, _, power_building_id = power_data
            
            coverage_data = coverage_dataset[idx]
            if len(coverage_data) == 3:
                _, coverage_gt, coverage_building_id = coverage_data
            else:
                _, coverage_gt, _, coverage_building_id = coverage_data
            
            # Verify building IDs match (they should be integers, not tensors)
            assert building_id == power_building_id == coverage_building_id, \
                f"Building ID mismatch: {building_id} vs {power_building_id} vs {coverage_building_id}"
            
            # Move to device
            building_map = building_map.unsqueeze(0).to(device)
            
            # Get prediction - TIME THIS
            model_start = time.time()
            logits, pred_coords = model(building_map)
            model_time = time.time() - model_start
            
            pred_coord_np = pred_coords[0].cpu().numpy()
            
            # Get GTs as numpy
            power_gt_np = power_gt.numpy()
            coverage_gt_np = coverage_gt.numpy()
            
            # Compute coordinate errors against BOTH GTs
            # L2 (Euclidean)
            coord_error_power = float(np.sqrt(np.sum((pred_coord_np - power_gt_np)**2)))
            coord_error_coverage = float(np.sqrt(np.sum((pred_coord_np - coverage_gt_np)**2)))
            
            # L1 training-style (mean of absolute differences)
            coord_error_power_L1_train = float(np.mean(np.abs(pred_coord_np - power_gt_np)))
            coord_error_coverage_L1_train = float(np.mean(np.abs(pred_coord_np - coverage_gt_np)))
            
            # L1 Manhattan (sum of absolute differences)
            coord_error_power_L1_manhattan = float(np.sum(np.abs(pred_coord_np - power_gt_np)))
            coord_error_coverage_L1_manhattan = float(np.sum(np.abs(pred_coord_np - coverage_gt_np)))
            
            # Evaluate with RadioNet - TIME THIS
            gt_coords_np = gt_coords.numpy()
            radionet_start = time.time()
            metrics = evaluate_with_avg_maps(
                pred_coord_np, gt_coords_np, building_id, heatmap_base_dir,
                optimize_for, use_center, buildings_dir, radionet_model
            )
            radionet_time = time.time() - radionet_start
            
            building_total_time = time.time() - building_start
            
            if metrics is not None:
                # Compute all 4 dual-GT metrics for complete evaluation:
                # 1. Coverage%(ref): pred coverage / coverage-optimal GT's coverage
                # 2. Coverage%(power): pred coverage / power-optimal GT's coverage  
                # 3. PL%(from cov): pred PL / coverage-optimal GT's PL
                # 4. PL%(reference): pred PL / power-optimal GT's PL
                
                if optimize_for == 'power':
                    # Has: pl_pct, coverage_pct_power, coverage_pct_gt
                    coverage_pct_ref = metrics['coverage_pct_gt']
                    coverage_pct_power = metrics['coverage_pct_power']
                    pl_pct_ref = metrics['pl_pct']
                    
                    # Missing: PL vs coverage-optimal GT's PL
                    # Need to get coverage GT's PL
                    gt_cov_metrics = load_pl_map_metrics(
                        building_id, heatmap_base_dir, 'best_by_coverage',
                        use_center, buildings_dir, threshold=0.0
                    )
                    if gt_cov_metrics is not None:
                        pred_pl = float(metrics['pred_pl'])
                        gt_pl_cov = float(gt_cov_metrics['avg_power'])
                        pl_pct_cov = (pred_pl / gt_pl_cov * 100.0) if gt_pl_cov > 0 else 0.0
                    else:
                        pl_pct_cov = None
                        
                else:  # coverage
                    # Has: coverage_pct, pl_pct_coverage, pl_pct_ref
                    coverage_pct_ref = metrics['coverage_pct']
                    pl_pct_cov = metrics['pl_pct_coverage']
                    pl_pct_ref = metrics['pl_pct_ref']
                    
                    # Missing: coverage vs power GT's coverage
                    gt_power_metrics = load_pl_map_metrics(
                        building_id, heatmap_base_dir, 'best_by_power',
                        use_center, buildings_dir, threshold=0.0
                    )
                    if gt_power_metrics is not None:
                        pred_cov = float(metrics['pred_coverage'])
                        gt_cov_power = float(gt_power_metrics['coverage_count'])
                        coverage_pct_power = (pred_cov / gt_cov_power * 100.0) if gt_cov_power > 0 else 0.0
                    else:
                        coverage_pct_power = None
                
                # Skip this building if we can't compute all metrics
                if coverage_pct_power is None or pl_pct_cov is None:
                    continue
                
                # Store all metrics
                results['coord_errors_power'].append(coord_error_power)
                results['coord_errors_coverage'].append(coord_error_coverage)
                results['coord_errors_power_L1_train'].append(coord_error_power_L1_train)
                results['coord_errors_coverage_L1_train'].append(coord_error_coverage_L1_train)
                results['coord_errors_power_L1_manhattan'].append(coord_error_power_L1_manhattan)
                results['coord_errors_coverage_L1_manhattan'].append(coord_error_coverage_L1_manhattan)
                
                # Store all 4 dual-GT radio metrics
                results['coverage_pcts'].append(coverage_pct_ref)
                results['coverage_pct_powers'].append(coverage_pct_power)
                results['pl_pct_covs'].append(pl_pct_cov)
                results['pl_pct_refs'].append(pl_pct_ref)
                
                # Store per-building data
                results['building_ids'].append(building_id)
                results['pred_coords'].append(pred_coord_np)
                
                # Store timing data
                results['model_times'].append(model_time)
                results['radionet_times'].append(radionet_time)
                results['total_times'].append(building_total_time)
    
    overall_time = time.time() - overall_start
    
    # Compute aggregate statistics
    print("\nComputing aggregate statistics...")
    
    aggregated = {
        # Coordinate errors
        'coord_error_power_mean': float(np.mean(results['coord_errors_power'])),
        'coord_error_power_median': float(np.median(results['coord_errors_power'])),
        'coord_error_power_std': float(np.std(results['coord_errors_power'])),
        'coord_error_coverage_mean': float(np.mean(results['coord_errors_coverage'])),
        'coord_error_coverage_median': float(np.median(results['coord_errors_coverage'])),
        'coord_error_coverage_std': float(np.std(results['coord_errors_coverage'])),
        # L1 train errors
        'coord_error_power_L1_train_mean': float(np.mean(results['coord_errors_power_L1_train'])),
        'coord_error_power_L1_train_median': float(np.median(results['coord_errors_power_L1_train'])),
        'coord_error_power_L1_train_std': float(np.std(results['coord_errors_power_L1_train'])),
        'coord_error_coverage_L1_train_mean': float(np.mean(results['coord_errors_coverage_L1_train'])),
        'coord_error_coverage_L1_train_median': float(np.median(results['coord_errors_coverage_L1_train'])),
        'coord_error_coverage_L1_train_std': float(np.std(results['coord_errors_coverage_L1_train'])),
        # L1 Manhattan errors
        'coord_error_power_L1_manhattan_mean': float(np.mean(results['coord_errors_power_L1_manhattan'])),
        'coord_error_power_L1_manhattan_median': float(np.median(results['coord_errors_power_L1_manhattan'])),
        'coord_error_power_L1_manhattan_std': float(np.std(results['coord_errors_power_L1_manhattan'])),
        'coord_error_coverage_L1_manhattan_mean': float(np.mean(results['coord_errors_coverage_L1_manhattan'])),
        'coord_error_coverage_L1_manhattan_median': float(np.median(results['coord_errors_coverage_L1_manhattan'])),
        'coord_error_coverage_L1_manhattan_std': float(np.std(results['coord_errors_coverage_L1_manhattan'])),
        # Radio metrics (4 total: 2 coverage + 2 PL)
        'coverage_pct_mean': float(np.mean(results['coverage_pcts'])),
        'coverage_pct_std': float(np.std(results['coverage_pcts'])),
        'coverage_pct_power_mean': float(np.mean(results['coverage_pct_powers'])),
        'coverage_pct_power_std': float(np.std(results['coverage_pct_powers'])),
        'pl_pct_cov_mean': float(np.mean(results['pl_pct_covs'])),
        'pl_pct_cov_std': float(np.std(results['pl_pct_covs'])),
        'pl_pct_ref_mean': float(np.mean(results['pl_pct_refs'])),
        'pl_pct_ref_std': float(np.std(results['pl_pct_refs'])),
        # Timing metrics
        'model_time_mean': float(np.mean(results['model_times'])),
        'model_time_median': float(np.median(results['model_times'])),
        'model_time_std': float(np.std(results['model_times'])),
        'radionet_time_mean': float(np.mean(results['radionet_times'])),
        'radionet_time_median': float(np.median(results['radionet_times'])),
        'radionet_time_std': float(np.std(results['radionet_times'])),
        'total_time_per_building_mean': float(np.mean(results['total_times'])),
        'total_time_per_building_median': float(np.median(results['total_times'])),
        'total_time_per_building_std': float(np.std(results['total_times'])),
        'overall_time': float(overall_time),
        'buildings_per_second': float(len(results['total_times']) / overall_time),
        'buildings_per_hour': float((len(results['total_times']) / overall_time) * 3600),
        # Metadata
        'num_buildings': len(results['coord_errors_power']),
        # Per-building data
        'building_ids': [int(bid) if isinstance(bid, np.integer) else bid for bid in results['building_ids']],
        'pred_coords': [coord.tolist() if isinstance(coord, np.ndarray) else coord for coord in results['pred_coords']],
        'coord_errors_power': [float(e) for e in results['coord_errors_power']],
        'coord_errors_coverage': [float(e) for e in results['coord_errors_coverage']],
        'coverage_pcts': [float(c) for c in results['coverage_pcts']],
        'coverage_pct_powers': [float(c) for c in results['coverage_pct_powers']],
        'pl_pct_covs': [float(p) for p in results['pl_pct_covs']],
        'pl_pct_refs': [float(p) for p in results['pl_pct_refs']]
    }
    
    return aggregated


def save_results_txt(results, save_path, optimize_for, model_info, hardware_info=None):
    """Save results in human-readable text format."""
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FIXED MODEL EVALUATION RESULTS - DUAL GT - TEST SET\n")
        f.write("="*80 + "\n\n")
        
        f.write("Model Information:\n")
        f.write(f"  Architecture: {model_info['arch']}\n")
        f.write(f"  Optimize for: {model_info['optimize_for']}\n")
        f.write(f"  Heatmap type: {model_info['heatmap_type']}\n")
        if model_info.get('model_group'):
            f.write(f"  Model group: {model_info['model_group']}\n")
        f.write(f"  Model path: {model_info['model_path']}\n")
        f.write(f"  Buildings evaluated: {results['num_buildings']}\n")
        
        # Add hardware information
        if hardware_info:
            f.write(f"\nHardware Information:\n")
            f.write(f"  Device: {hardware_info['device_type']}\n")
            if hardware_info.get('gpu_name'):
                f.write(f"  GPU: {hardware_info['gpu_name']}\n")
                f.write(f"  GPU Memory: {hardware_info['gpu_memory_gb']} GB\n")
                if hardware_info.get('cuda_version'):
                    f.write(f"  CUDA Version: {hardware_info['cuda_version']}\n")
                if hardware_info.get('gpu_driver_version'):
                    f.write(f"  Driver Version: {hardware_info['gpu_driver_version']}\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("COORDINATE ERRORS (Dual GT)\n")
        f.write("="*80 + "\n\n")
        
        f.write("L2 (Euclidean) Errors:\n")
        f.write(f"  vs Power-optimal GT:    {results['coord_error_power_mean']:.2f} ± {results['coord_error_power_std']:.2f} px (median: {results['coord_error_power_median']:.2f})\n")
        f.write(f"  vs Coverage-optimal GT: {results['coord_error_coverage_mean']:.2f} ± {results['coord_error_coverage_std']:.2f} px (median: {results['coord_error_coverage_median']:.2f})\n\n")
        
        f.write("L1 Training-style Errors (mean of |Δy|+|Δx|):\n")
        f.write(f"  vs Power-optimal GT:    {results['coord_error_power_L1_train_mean']:.2f} ± {results['coord_error_power_L1_train_std']:.2f} px (median: {results['coord_error_power_L1_train_median']:.2f})\n")
        f.write(f"  vs Coverage-optimal GT: {results['coord_error_coverage_L1_train_mean']:.2f} ± {results['coord_error_coverage_L1_train_std']:.2f} px (median: {results['coord_error_coverage_L1_train_median']:.2f})\n\n")
        
        f.write("L1 Manhattan Errors (|Δy|+|Δx|):\n")
        f.write(f"  vs Power-optimal GT:    {results['coord_error_power_L1_manhattan_mean']:.2f} ± {results['coord_error_power_L1_manhattan_std']:.2f} px (median: {results['coord_error_power_L1_manhattan_median']:.2f})\n")
        f.write(f"  vs Coverage-optimal GT: {results['coord_error_coverage_L1_manhattan_mean']:.2f} ± {results['coord_error_coverage_L1_manhattan_std']:.2f} px (median: {results['coord_error_coverage_L1_manhattan_median']:.2f})\n\n")
        
        f.write("="*80 + "\n")
        f.write("RADIO PROPAGATION METRICS\n")
        f.write("="*80 + "\n\n")
        
        # All 4 dual-GT metrics (2 coverage + 2 PL)
        f.write("Coverage Metrics:\n")
        f.write(f"  Coverage%(ref):     {results['coverage_pct_mean']:.2f} ± {results['coverage_pct_std']:.2f}%  [vs coverage-optimal GT]\n")
        f.write(f"  Coverage%(power):   {results['coverage_pct_power_mean']:.2f} ± {results['coverage_pct_power_std']:.2f}%  [vs power-optimal GT]\n\n")
        
        f.write("Path Loss Metrics:\n")
        f.write(f"  PL%(from cov):      {results['pl_pct_cov_mean']:.2f} ± {results['pl_pct_cov_std']:.2f}%  [vs coverage-optimal GT]\n")
        f.write(f"  PL%(reference):     {results['pl_pct_ref_mean']:.2f} ± {results['pl_pct_ref_std']:.2f}%  [vs power-optimal GT]\n")
        
        f.write("\n" + "="*80 + "\n")
        
        f.write("⏱️  TIMING METRICS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Model inference per building:    {results['model_time_mean']:.4f} ± {results['model_time_std']:.4f} s (median: {results['model_time_median']:.4f})\n")
        f.write(f"RadioNet per building:           {results['radionet_time_mean']:.4f} ± {results['radionet_time_std']:.4f} s (median: {results['radionet_time_median']:.4f})\n")
        f.write(f"Total per building:              {results['total_time_per_building_mean']:.2f} ± {results['total_time_per_building_std']:.2f} s (median: {results['total_time_per_building_median']:.2f})\n")
        f.write(f"Buildings/second:                {results['buildings_per_second']:.2f}\n")
        f.write(f"Buildings/hour:                  {results['buildings_per_hour']:.1f}\n")
        f.write(f"Overall evaluation time:         {results['overall_time']/60:.1f} minutes ({results['overall_time']/3600:.2f} hours)\n")
        
        f.write("\n" + "="*80 + "\n")
        
        f.write("⏱️  TIMING METRICS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Model inference per building:    {results['model_time_mean']:.4f} ± {results['model_time_std']:.4f} s (median: {results['model_time_median']:.4f})\n")
        f.write(f"RadioNet per building:           {results['radionet_time_mean']:.4f} ± {results['radionet_time_std']:.4f} s (median: {results['radionet_time_median']:.4f})\n")
        f.write(f"Total per building:              {results['total_time_per_building_mean']:.2f} ± {results['total_time_per_building_std']:.2f} s (median: {results['total_time_per_building_median']:.2f})\n")
        f.write(f"Buildings/second:                {results['buildings_per_second']:.2f}\n")
        f.write(f"Buildings/hour:                  {results['buildings_per_hour']:.1f}\n")
        f.write(f"Overall evaluation time:         {results['overall_time']/60:.1f} minutes ({results['overall_time']/3600:.2f} hours)\n")
        
        f.write("\n" + "="*80 + "\n")


def generate_quantile_visualizations_fixed(results, test_dataset, power_dataset, coverage_dataset,
                                          buildings_dir, radionet_model, use_center, save_dir, optimize_for):
    """
    Generate visualizations for buildings sampled from L2 error quantiles.
    Exactly matches diffusion evaluation approach with uniform bins in error space.
    
    Key differences from multisample evaluation:
    - Single strategy per model (based on optimize_for)
    - Single predicted location per building (not multiple samples)
    """
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap
    
    viz_dir = os.path.join(save_dir, 'quantile_visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    print(f"\nGenerating quantile visualizations...")
    
    # Determine which error to use for quantiles based on optimize_for
    if optimize_for == 'coverage':
        errors = np.array(results['coord_errors_coverage'])
        error_type = 'L2_Err_Coverage'
        strategy_name = 'best_coverage'
    else:  # power
        errors = np.array(results['coord_errors_power'])
        error_type = 'L2_Err_Power'
        strategy_name = 'best_power'
    
    building_ids = results['building_ids']
    pred_coords_list = results['pred_coords']
    
    # Compute uniform bins in error space (not percentile-based) - MATCH DIFFUSION EXACTLY
    min_error = np.min(errors)
    max_error = np.max(errors)
    n_bins = 10  # Match diffusion evaluation
    quantiles = np.linspace(min_error, max_error, n_bins + 1)
    
    strategy_dir = os.path.join(viz_dir, f"{strategy_name}_strategy")
    os.makedirs(strategy_dir, exist_ok=True)
    
    selected_buildings_data = []
    
    # Sample 5 buildings from each quantile - MATCH DIFFUSION EXACTLY
    for q_idx in range(len(quantiles) - 1):
        q_low, q_high = quantiles[q_idx], quantiles[q_idx + 1]
        
        # Find buildings in this quantile range
        in_quantile = (errors >= q_low) & (errors <= q_high)
        quantile_indices = np.where(in_quantile)[0]
        
        if len(quantile_indices) == 0:
            continue
        
        # Sample up to 5 buildings from this quantile - MATCH DIFFUSION EXACTLY
        n_samples_viz = min(5, len(quantile_indices))
        sampled_indices = np.random.choice(quantile_indices, size=n_samples_viz, replace=False)
        
        # Create quantile directory - MATCH DIFFUSION NAMING
        quantile_name = f"q{q_idx:02d}_err{q_low:.1f}-{q_high:.1f}px"
        if q_idx == 0:
            quantile_name += "_best"
        elif q_idx == len(quantiles) - 2:
            quantile_name += "_worst"
        
        quantile_dir = os.path.join(strategy_dir, quantile_name)
        os.makedirs(quantile_dir, exist_ok=True)
        
        # Generate visualizations for sampled buildings
        quantile_buildings = []
        for idx in sampled_indices:
            building_id = building_ids[idx]
            pred_coord = np.array(pred_coords_list[idx])
            error_val = errors[idx]
            
            # Get coverage and power errors for this building
            error_cov = results['coord_errors_coverage'][idx]
            error_pow = results['coord_errors_power'][idx]
            coverage_pct = results['coverage_pcts'][idx]
            pl_pct_ref = results['pl_pct_refs'][idx]
            
            quantile_buildings.append(building_id)
            
            # Find building index in datasets
            building_idx = None
            for i in range(len(test_dataset)):
                test_data = test_dataset[i]
                if len(test_data) == 3:
                    _, _, bid = test_data
                else:
                    _, _, _, bid = test_data
                if bid == building_id:
                    building_idx = i
                    break
            
            if building_idx is None:
                continue
            
            # Load ground truth coordinates
            power_data = power_dataset[building_idx]
            if len(power_data) == 3:
                _, power_gt_coords, _ = power_data
            else:
                _, power_gt_coords, _, _ = power_data
            
            coverage_data = coverage_dataset[building_idx]
            if len(coverage_data) == 3:
                _, coverage_gt_coords, _ = coverage_data
            else:
                _, coverage_gt_coords, _, _ = coverage_data
                
            power_gt = power_gt_coords.cpu().numpy() if torch.is_tensor(power_gt_coords) else power_gt_coords
            coverage_gt = coverage_gt_coords.cpu().numpy() if torch.is_tensor(coverage_gt_coords) else coverage_gt_coords
            
            # Generate visualizations using the exact same function as diffusion eval
            visualize_building(
                building_id, pred_coord, power_gt, coverage_gt,
                error_pow, error_cov, coverage_pct, pl_pct_ref,
                buildings_dir, radionet_model, use_center,
                quantile_dir
            )
            
            # Track for summary
            selected_buildings_data.append({
                'building_id': building_id,
                'strategy': optimize_for,
                'quantile_range': quantile_name,
                'L2_error_coverage': error_cov,
                'L2_error_power': error_pow,
                'coverage_pct': coverage_pct,
                'pl_pct_ref': pl_pct_ref
            })
        
        # Create summary file for this quantile - MATCH DIFFUSION FORMAT
        summary_path = os.path.join(quantile_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Quantile Range: {q_low:.2f} - {q_high:.2f} pixels\n")
            f.write(f"Samples: {len(quantile_buildings)}\n")
            f.write(f"Average L2 Error: {errors[sampled_indices].mean():.2f} pixels\n")
            if optimize_for == 'coverage':
                f.write(f"Average Coverage%: {np.array([results['coverage_pcts'][i] for i in sampled_indices]).mean():.2f}%\n")
            else:
                f.write(f"Average PL%: {np.array([results['pl_pct_refs'][i] for i in sampled_indices]).mean():.2f}%\n")
            f.write(f"Building IDs: {', '.join(map(str, quantile_buildings))}\n")
    
    # Save selected buildings list - MATCH DIFFUSION FORMAT
    if len(selected_buildings_data) > 0:
        import pandas as pd
        df = pd.DataFrame(selected_buildings_data)
        df.to_csv(os.path.join(viz_dir, 'selected_buildings.csv'), index=False)
        
        # Create simple text file
        with open(os.path.join(viz_dir, 'selected_buildings.txt'), 'w') as f:
            f.write("# Buildings visualized for quantile analysis\n\n")
            f.write(f"# {optimize_for.capitalize()} Optimization\n")
            for quantile in df['quantile_range'].unique():
                quant_buildings = df[df['quantile_range'] == quantile]
                building_ids_str = ', '.join(quant_buildings['building_id'].astype(str).tolist())
                f.write(f"# {quantile}: {building_ids_str}\n")
        
        print(f"\n✓ Generated quantile visualizations: {viz_dir}")
        print(f"  - Total buildings visualized: {len(selected_buildings_data)}")
        print(f"  - Selected buildings saved to: {os.path.join(viz_dir, 'selected_buildings.csv')}")


def visualize_building(building_id, pred_coord, power_gt, coverage_gt,
                       error_pow, error_cov, coverage_pct, pl_pct_ref,
                       buildings_dir, radionet_model, use_center, save_dir):
    """
    Generate 9 visualization files + 1 text file for a single building.
    Exactly matches diffusion evaluation format.
    """
    # Load building image
    building_path = os.path.join(buildings_dir, f'{building_id}.png')
    if not os.path.exists(building_path):
        return
    
    building_img = np.array(Image.open(building_path).convert('L')).astype(np.float32) / 255.0
    
    # Convert coordinates to 256x256 space if using center crop
    if use_center:
        pred_y_256 = pred_coord[0] + 53
        pred_x_256 = pred_coord[1] + 53
        power_gt_256 = (power_gt[0] + 53, power_gt[1] + 53)
        coverage_gt_256 = (coverage_gt[0] + 53, coverage_gt[1] + 53)
    else:
        pred_y_256 = pred_coord[0]
        pred_x_256 = pred_coord[1]
        power_gt_256 = power_gt
        coverage_gt_256 = coverage_gt
    
    # Generate power maps using RadioNet (returns uint8 [0-255])
    pred_power_map = generate_radio_maps(building_img, pred_y_256, pred_x_256, radionet_model)
    power_gt_power_map = generate_radio_maps(building_img, power_gt_256[0], power_gt_256[1], radionet_model)
    coverage_gt_power_map = generate_radio_maps(building_img, coverage_gt_256[0], coverage_gt_256[1], radionet_model)
    
    # Generate coverage maps (binary: 255 where power > 0)
    pred_coverage_map = compute_coverage_map(pred_power_map)
    power_gt_coverage_map = compute_coverage_map(power_gt_power_map)
    coverage_gt_coverage_map = compute_coverage_map(coverage_gt_power_map)
    
    # 1-2: Predicted maps (simple grayscale)
    save_simple_grayscale(pred_power_map, 
                         os.path.join(save_dir, f'{building_id}_pred_power.png'))
    save_simple_grayscale(pred_coverage_map, 
                         os.path.join(save_dir, f'{building_id}_pred_coverage.png'))
    
    # 3-4: Power-optimal GT maps (simple grayscale)
    save_simple_grayscale(power_gt_power_map, 
                         os.path.join(save_dir, f'{building_id}_gt_powerOpt_power.png'))
    save_simple_grayscale(power_gt_coverage_map, 
                         os.path.join(save_dir, f'{building_id}_gt_powerOpt_coverage.png'))
    
    # 5-6: Coverage-optimal GT maps (simple grayscale)
    save_simple_grayscale(coverage_gt_power_map, 
                         os.path.join(save_dir, f'{building_id}_gt_covOpt_power.png'))
    save_simple_grayscale(coverage_gt_coverage_map, 
                         os.path.join(save_dir, f'{building_id}_gt_covOpt_coverage.png'))
    
    # Prepare Tx locations for overlays
    tx_locations = {
        'pred': (pred_y_256, pred_x_256),
        'power_gt': power_gt_256,
        'coverage_gt': coverage_gt_256
    }
    
    # 7-8: Maps with Tx overlays
    save_map_with_tx_overlay(pred_power_map, tx_locations, 
                             os.path.join(save_dir, f'{building_id}_power_tx.png'))
    save_map_with_tx_overlay(pred_coverage_map, tx_locations, 
                             os.path.join(save_dir, f'{building_id}_coverage_tx.png'))
    
    # 9: Building layout with Tx overlays
    building_uint8 = (building_img * 255).astype(np.uint8)
    save_map_with_tx_overlay(building_uint8, tx_locations, 
                             os.path.join(save_dir, f'{building_id}_building_tx.png'))
    
    # 10: Info text file
    save_info_txt(building_id, pred_coord, power_gt, coverage_gt,
                 error_pow, error_cov, coverage_pct, pl_pct_ref,
                 os.path.join(save_dir, f'{building_id}_info.txt'))


def main():
    parser = argparse.ArgumentParser(description='Evaluate Fixed Model on Test Set - Dual GT')
    
    # Model
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint (best_pl_model.pth or best_coverage_model.pth)')
    parser.add_argument('--arch', type=str, required=True,
                       choices=['deepxl_150', 'deepxl_150r', 'radionet_150', 'ms_150b', 'ms_150c', 'ms_150e',
                               'ultradeep_150', 'deepxl_256', 'deepxxl_256', 'ultradeep_256', 'hyperdeep_256',
                               'locunet_256', 'locunet_256_wide', 'locunet_150', 'locunet_150_wide',
                               'pmnet_150', 'sip2net_150', 'dcnet_150'],
                       help='Model architecture')
    parser.add_argument('--coord_method', type=str, default='hard_argmax',
                       choices=['soft_argmax', 'hard_argmax', 'center_of_mass'],
                       help='Coordinate extraction method')
    parser.add_argument('--temperature', type=float, default=0.01,
                       help='Temperature for soft-argmax')
    
    # Data
    parser.add_argument('--optimize_for', type=str, required=True,
                       choices=['power', 'coverage'],
                       help='What the model was optimized for')
    parser.add_argument('--heatmap_type', type=str, required=True,
                       choices=['gaussian', 'radio_coverage', 'radio_power'],
                       help='Heatmap type used during training')
    parser.add_argument('--model_group', type=str, default=None,
                       help='Model group identifier (e.g., FixedCorrCov, DeepXLL2, OthersL2, AdvLoss)')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate on (default: test)')
    parser.add_argument('--use_center', action='store_true',
                       help='Use center 150x150 crop')
    
    # Paths
    parser.add_argument('--buildings_dir', type=str,
                       default='./data/buildings')
    parser.add_argument('--heatmap_base_dir', type=str,
                       default='./data/unified_results')
    parser.add_argument('--radionet_path', type=str,
                       default='./checkpoints/saipp_net.pt')
    parser.add_argument('--save_dir', type=str, required=True)
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Capture hardware information
    hardware_info = get_hardware_info()
    print(f"Using device: {device}")
    print(f"Hardware: {format_hardware_info(hardware_info)}\n")
    
    # Image size
    img_size = 150 if args.use_center else 256
    
    # Load model
    print("Loading model...")
    model = create_model_deep(
        arch=args.arch,
        coord_method=args.coord_method,
        temperature=args.temperature,
        use_masking=False,
        img_size=img_size
    ).to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from: {args.model_path}")
    print(f"Trained for {checkpoint.get('epoch', '?')} epochs\n")
    
    # Load RadioNet
    print("Loading RadioNet...")
    radionet_model = RadioNet(inputs=2).to(device)
    radionet_checkpoint = torch.load(args.radionet_path, map_location=device)
    
    if isinstance(radionet_checkpoint, dict):
        if 'model_state_dict' in radionet_checkpoint:
            radionet_model.load_state_dict(radionet_checkpoint['model_state_dict'])
        elif 'state_dict' in radionet_checkpoint:
            radionet_model.load_state_dict(radionet_checkpoint['state_dict'])
        else:
            radionet_model.load_state_dict(radionet_checkpoint)
    else:
        radionet_model.load_state_dict(radionet_checkpoint)
    
    radionet_model.eval()
    print(f"Loaded RadioNet from: {args.radionet_path}\n")
    
    # Load dataset splits
    print(f"Loading {args.split} dataset...")
    train_ids, val_ids, test_ids = get_building_splits(args.buildings_dir)
    
    # Select split
    if args.split == 'train':
        eval_ids = train_ids
    elif args.split == 'val':
        eval_ids = val_ids
    else:  # test
        eval_ids = test_ids
    
    print(f"{args.split.capitalize()} set size: {len(eval_ids)} buildings\n")
    
    # Create dataset for evaluation (based on optimize_for)
    tx_dir = os.path.join(args.heatmap_base_dir, f'best_by_{args.optimize_for}')
    
    test_dataset = TxLocationDatasetLarge(
        eval_ids,
        buildings_dir=args.buildings_dir,
        tx_dir=tx_dir,
        optimize_for=args.optimize_for,
        heatmap_type=args.heatmap_type,
        heatmap_base_dir=args.heatmap_base_dir,
        normalize_heatmap=False,
        augment=False,
        use_center=args.use_center
    )
    
    # Load BOTH power and coverage datasets for dual GT evaluation
    print("Loading power-optimal GT dataset...")
    power_dataset = TxLocationDatasetLarge(
        eval_ids,
        buildings_dir=args.buildings_dir,
        tx_dir=os.path.join(args.heatmap_base_dir, 'best_by_power'),
        optimize_for='power',
        heatmap_type='radio_power',
        heatmap_base_dir=args.heatmap_base_dir,
        normalize_heatmap=False,
        augment=False,
        use_center=args.use_center
    )
    
    print("Loading coverage-optimal GT dataset...")
    coverage_dataset = TxLocationDatasetLarge(
        eval_ids,
        buildings_dir=args.buildings_dir,
        tx_dir=os.path.join(args.heatmap_base_dir, 'best_by_coverage'),
        optimize_for='coverage',
        heatmap_type='radio_power',
        heatmap_base_dir=args.heatmap_base_dir,
        normalize_heatmap=False,
        augment=False,
        use_center=args.use_center
    )
    
    print(f"{'='*80}")
    print(f"FIXED MODEL EVALUATION - DUAL GT - {args.split.upper()} SET")
    print(f"{'='*80}")
    print(f"Model: {args.model_path}")
    print(f"Architecture: {args.arch}")
    print(f"Optimize for: {args.optimize_for}")
    print(f"Heatmap type: {args.heatmap_type}")
    print(f"{args.split.capitalize()} set: {len(test_dataset)} buildings")
    print(f"{'='*80}\n")
    
    # Run evaluation
    start_time = time.time()
    results = evaluate_fixed_model_dual_gt(
        model, test_dataset, power_dataset, coverage_dataset,
        device, args.optimize_for, args.use_center, args.buildings_dir,
        radionet_model, args.heatmap_base_dir
    )
    elapsed_time = time.time() - start_time
    
    print(f"\nEvaluation completed in {elapsed_time/60:.1f} minutes")
    
    # Save results as JSON
    json_path = os.path.join(args.save_dir, 'results.json')
    
    # Add model_info to results for the CSV parser
    results_with_info = convert_to_json_serializable(results)
    results_with_info['model_info'] = {
        'arch': args.arch,
        'optimize_for': args.optimize_for,
        'heatmap_type': args.heatmap_type,
        'model_path': args.model_path,
        'model_group': args.model_group  # Add model_group for tracking
    }
    results_with_info['hardware_info'] = hardware_info  # Add hardware context for timing
    
    with open(json_path, 'w') as f:
        json.dump(results_with_info, f, indent=2)
    print(f"Saved JSON results to: {json_path}")
    
    # Save results as TXT
    txt_path = os.path.join(args.save_dir, 'results.txt')
    model_info = {
        'arch': args.arch,
        'optimize_for': args.optimize_for,
        'heatmap_type': args.heatmap_type,
        'model_path': args.model_path,
        'model_group': args.model_group
    }
    save_results_txt(results, txt_path, args.optimize_for, model_info, hardware_info)
    print(f"Saved TXT results to: {txt_path}")
    
    # Generate quantile visualizations
    print(f"\n{'='*80}")
    print("Generating quantile visualizations...")
    print(f"{'='*80}\n")
    
    generate_quantile_visualizations_fixed(
        results, test_dataset, power_dataset, coverage_dataset,
        args.buildings_dir, radionet_model, args.use_center, args.save_dir, args.optimize_for
    )
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Results saved to: {args.save_dir}")
    print(f"  - results.json (machine-readable)")
    print(f"  - results.txt (human-readable)")
    print(f"  - quantile_visualizations/ (example buildings)")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
