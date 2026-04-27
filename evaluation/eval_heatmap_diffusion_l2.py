"""
Multi-Sample Evaluation for Diffusion Models - DUAL GT VERSION

KEY FIX: Calculates coordinate errors against BOTH power-optimal and coverage-optimal
GT locations, regardless of which optimization target the model was trained on.

This allows fair comparison across models trained with different objectives.
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
import pandas as pd
import json
import time
from tqdm import tqdm
from contextlib import contextmanager

# Import models and utilities
from models.diffusion import create_diffusion_model
from training.train_diffusion_heatmap import DDIMScheduler, DDPMScheduler, sample_diffusion, hard_argmax
from data.dataset_heatmap import TxLocationDatasetLarge, get_building_splits

# Import RadioNet
sys.path.append('.')
from models.saipp_net import RadioNet

# Import existing validation functions
from training.train_heatmap import (
    evaluate_with_avg_maps,
    compute_building_density,
    load_pl_map_metrics,
    compute_predicted_radio_metrics
)


@contextmanager
def timer(name="Operation"):
    """Context manager for timing operations"""
    start_time = time.time()
    yield lambda: time.time() - start_time
    # elapsed = time.time() - start_time




def convert_to_json_serializable(obj):
    """Convert numpy arrays and other non-serializable objects to JSON-serializable types."""
    import numpy as np
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    else:
        return obj


def evaluate_multisample(model, val_dataset, power_dataset, coverage_dataset, num_samples, 
                        scheduler, num_inference_steps, use_ddim, device, optimize_for, 
                        use_center, buildings_dir, radionet_model, heatmap_base_dir, max_buildings=None):
    """
    Evaluate model with multiple samples per building
    
    KEY CHANGE: Now tracks coordinate errors against BOTH power and coverage GT
    """
    model.eval()
    
    # Determine how many buildings to evaluate
    if max_buildings is not None:
        num_buildings = min(max_buildings, len(val_dataset))
    else:
        num_buildings = len(val_dataset)
    
    # Storage for each strategy
    strategies = {
        'Single': {
            'coord_errors_power': [],  # L2 error vs power-optimal GT
            'coord_errors_coverage': [],  # L2 error vs coverage-optimal GT
            'coord_errors_power_L1_train': [],  # L1 training-style vs power GT
            'coord_errors_coverage_L1_train': [],  # L1 training-style vs coverage GT
            'coord_errors_power_L1_manhattan': [],  # L1 Manhattan vs power GT
            'coord_errors_coverage_L1_manhattan': [],  # L1 Manhattan vs coverage GT
            'coverage_pcts': [],
            'pl_pct_coverages': [],
            'pl_pct_refs': [],
            # Timing metrics
            'diffusion_times': [],  # Per-sample diffusion time
            'radionet_times': [],   # Per-sample RadioNet time
            'building_times': [],    # Total per-building time
            # Per-building data for visualization
            'building_ids': [],
            'pred_coords': []  # Predicted coordinates for this strategy
        },
        'Best-Coverage': {
            'coord_errors_power': [],
            'coord_errors_coverage': [],
            'coord_errors_power_L1_train': [],
            'coord_errors_coverage_L1_train': [],
            'coord_errors_power_L1_manhattan': [],
            'coord_errors_coverage_L1_manhattan': [],
            'coverage_pcts': [],
            'pl_pct_coverages': [],
            'pl_pct_refs': [],
            'diffusion_times': [],
            'radionet_times': [],
            'building_times': [],
            # Per-building data for visualization
            'building_ids': [],
            'pred_coords': []
        },
        'Best-Power': {
            'coord_errors_power': [],
            'coord_errors_coverage': [],
            'coord_errors_power_L1_train': [],
            'coord_errors_coverage_L1_train': [],
            'coord_errors_power_L1_manhattan': [],
            'coord_errors_coverage_L1_manhattan': [],
            'coverage_pcts': [],
            'pl_pct_coverages': [],
            'pl_pct_refs': [],
            'diffusion_times': [],
            'radionet_times': [],
            'building_times': [],
            # Per-building data for visualization
            'building_ids': [],
            'pred_coords': []
        },
        'Average': {
            'coord_errors_power': [],
            'coord_errors_coverage': [],
            'coord_errors_power_L1_train': [],
            'coord_errors_coverage_L1_train': [],
            'coord_errors_power_L1_manhattan': [],
            'coord_errors_coverage_L1_manhattan': [],
            'coverage_pcts': [],
            'pl_pct_coverages': [],
            'pl_pct_refs': [],
            'diffusion_times': [],
            'radionet_times': [],
            'building_times': [],
            # Per-building data for visualization  
            'building_ids': [],
            'pred_coords': []
        },
        'Best-L2': {
            'coord_errors_power': [],
            'coord_errors_coverage': [],
            'coord_errors_power_L1_train': [],
            'coord_errors_coverage_L1_train': [],
            'coord_errors_power_L1_manhattan': [],
            'coord_errors_coverage_L1_manhattan': [],
            'coverage_pcts': [],
            'pl_pct_coverages': [],
            'pl_pct_refs': [],
            'diffusion_times': [],
            'radionet_times': [],
            'building_times': [],
            # Per-building data for visualization
            'building_ids': [],
            'pred_coords': []
        }
    }
    
    print(f"\nEvaluating {num_buildings} buildings with {num_samples} samples each...")
    
    with torch.no_grad():
        for idx in tqdm(range(num_buildings), desc=f"Samples={num_samples}", mininterval=5.0, disable=not sys.stdout.isatty()):
            building_start_time = time.time()
            
            building_map, gt_coords, target_heatmap, building_id = val_dataset[idx]
            
            # Load BOTH ground truth coordinates using the datasets
            _, power_gt_coords, _, _ = power_dataset[idx]
            _, coverage_gt_coords, _, _ = coverage_dataset[idx]
            
            power_gt = power_gt_coords.cpu().numpy()
            coverage_gt = coverage_gt_coords.cpu().numpy()
            
            # Prepare batch
            building_map = building_map.unsqueeze(0).to(device)
            optimize_for_list = [optimize_for]
            
            # Generate num_samples for this building
            sample_metrics_list = []
            sample_diffusion_times = []
            sample_radionet_times = []
            
            for k in range(num_samples):
                # Time: Diffusion generation
                diffusion_start = time.time()
                pred_heatmap = sample_diffusion(
                    model, building_map, optimize_for_list,
                    scheduler, num_inference_steps, use_ddim, device
                )
                diffusion_time = time.time() - diffusion_start
                sample_diffusion_times.append(diffusion_time)
                
                # Extract coordinates
                building_mask = (building_map > 0.1)
                img_size = 150 if use_center else 256
                y, x = hard_argmax(pred_heatmap, building_mask, img_size=img_size)
                pred_coord = torch.stack([y, x], dim=1)[0]
                pred_coord_np = pred_coord.cpu().numpy()
                
                # Time: RadioNet evaluation
                radionet_start = time.time()
                gt_coords_np = gt_coords.cpu().numpy()
                metrics = evaluate_with_avg_maps(
                    pred_coord_np, gt_coords_np, building_id, heatmap_base_dir,
                    optimize_for, use_center, buildings_dir, radionet_model
                )
                radionet_time = time.time() - radionet_start
                sample_radionet_times.append(radionet_time)
                
                if metrics is not None:
                    # Compute coordinate errors against BOTH GTs
                    # L2 (Euclidean) - existing
                    coord_error_power = float(np.sqrt(np.sum((pred_coord_np - power_gt)**2)))
                    coord_error_coverage = float(np.sqrt(np.sum((pred_coord_np - coverage_gt)**2)))
                    
                    # L1 training-style (mean of all coordinate differences)
                    coord_error_power_L1_train = float(np.mean(np.abs(pred_coord_np - power_gt)))
                    coord_error_coverage_L1_train = float(np.mean(np.abs(pred_coord_np - coverage_gt)))
                    
                    # L1 Manhattan (sum of absolute differences per instance)
                    coord_error_power_L1_manhattan = float(np.sum(np.abs(pred_coord_np - power_gt)))
                    coord_error_coverage_L1_manhattan = float(np.sum(np.abs(pred_coord_np - coverage_gt)))
                    
                    # Store sample data with metrics depending on optimize_for
                    if optimize_for == 'power':
                        sample_metrics_list.append({
                            'pred_coord': pred_coord_np,  # Store for visualization
                            'coord_error_power': coord_error_power,
                            'coord_error_coverage': coord_error_coverage,
                            'coord_error_power_L1_train': coord_error_power_L1_train,
                            'coord_error_coverage_L1_train': coord_error_coverage_L1_train,
                            'coord_error_power_L1_manhattan': coord_error_power_L1_manhattan,
                            'coord_error_coverage_L1_manhattan': coord_error_coverage_L1_manhattan,
                            'pl_pct': metrics['pl_pct'],
                            'coverage_pct_power': metrics['coverage_pct_power'],
                            'coverage_pct_gt': metrics['coverage_pct_gt']
                        })
                    else:  # optimize_for == 'coverage'
                        sample_metrics_list.append({
                            'pred_coord': pred_coord_np,  # Store for visualization
                            'coord_error_power': coord_error_power,
                            'coord_error_coverage': coord_error_coverage,
                            'coord_error_power_L1_train': coord_error_power_L1_train,
                            'coord_error_coverage_L1_train': coord_error_coverage_L1_train,
                            'coord_error_power_L1_manhattan': coord_error_power_L1_manhattan,
                            'coord_error_coverage_L1_manhattan': coord_error_coverage_L1_manhattan,
                            'coverage_pct': metrics['coverage_pct'],
                            'pl_pct_coverage': metrics['pl_pct_coverage'],
                            'pl_pct_ref': metrics['pl_pct_ref']
                        })
            
            # Skip this building if no valid samples
            if len(sample_metrics_list) == 0:
                continue
            
            # Calculate total building time
            building_total_time = time.time() - building_start_time
            
            # Calculate average timing per sample for this building
            avg_diffusion_time = float(np.mean(sample_diffusion_times))
            avg_radionet_time = float(np.mean(sample_radionet_times))
            
            # Handle strategies based on optimize_for
            if optimize_for == 'power':
                # For power optimization
                # Strategy 1: Single (first sample)
                single_metrics = sample_metrics_list[0]
                strategies['Single']['coord_errors_power'].append(single_metrics['coord_error_power'])
                strategies['Single']['coord_errors_coverage'].append(single_metrics['coord_error_coverage'])
                strategies['Single']['coord_errors_power_L1_train'].append(single_metrics['coord_error_power_L1_train'])
                strategies['Single']['coord_errors_coverage_L1_train'].append(single_metrics['coord_error_coverage_L1_train'])
                strategies['Single']['coord_errors_power_L1_manhattan'].append(single_metrics['coord_error_power_L1_manhattan'])
                strategies['Single']['coord_errors_coverage_L1_manhattan'].append(single_metrics['coord_error_coverage_L1_manhattan'])
                strategies['Single']['coverage_pcts'].append(single_metrics['coverage_pct_gt'])
                strategies['Single']['pl_pct_coverages'].append(single_metrics['coverage_pct_power'])
                strategies['Single']['pl_pct_refs'].append(single_metrics['pl_pct'])
                strategies['Single']['diffusion_times'].append(avg_diffusion_time)
                strategies['Single']['radionet_times'].append(avg_radionet_time)
                strategies['Single']['building_times'].append(building_total_time)
                strategies['Single']['building_ids'].append(building_id)
                strategies['Single']['pred_coords'].append(single_metrics['pred_coord'])
                
                # Strategy 2: Best-Coverage (highest coverage_pct_gt)
                best_cov_idx = np.argmax([m['coverage_pct_gt'] for m in sample_metrics_list])
                best_cov_metrics = sample_metrics_list[best_cov_idx]
                strategies['Best-Coverage']['coord_errors_power'].append(best_cov_metrics['coord_error_power'])
                strategies['Best-Coverage']['coord_errors_coverage'].append(best_cov_metrics['coord_error_coverage'])
                strategies['Best-Coverage']['coord_errors_power_L1_train'].append(best_cov_metrics['coord_error_power_L1_train'])
                strategies['Best-Coverage']['coord_errors_coverage_L1_train'].append(best_cov_metrics['coord_error_coverage_L1_train'])
                strategies['Best-Coverage']['coord_errors_power_L1_manhattan'].append(best_cov_metrics['coord_error_power_L1_manhattan'])
                strategies['Best-Coverage']['coord_errors_coverage_L1_manhattan'].append(best_cov_metrics['coord_error_coverage_L1_manhattan'])
                strategies['Best-Coverage']['coverage_pcts'].append(best_cov_metrics['coverage_pct_gt'])
                strategies['Best-Coverage']['pl_pct_coverages'].append(best_cov_metrics['coverage_pct_power'])
                strategies['Best-Coverage']['pl_pct_refs'].append(best_cov_metrics['pl_pct'])
                strategies['Best-Coverage']['diffusion_times'].append(avg_diffusion_time)
                strategies['Best-Coverage']['radionet_times'].append(avg_radionet_time)
                strategies['Best-Coverage']['building_times'].append(building_total_time)
                strategies['Best-Coverage']['building_ids'].append(building_id)
                strategies['Best-Coverage']['pred_coords'].append(best_cov_metrics['pred_coord'])
                
                # Strategy 3: Best-Power (highest pl_pct)
                best_pwr_idx = np.argmax([m['pl_pct'] for m in sample_metrics_list])
                best_pwr_metrics = sample_metrics_list[best_pwr_idx]
                strategies['Best-Power']['coord_errors_power'].append(best_pwr_metrics['coord_error_power'])
                strategies['Best-Power']['coord_errors_coverage'].append(best_pwr_metrics['coord_error_coverage'])
                strategies['Best-Power']['coord_errors_power_L1_train'].append(best_pwr_metrics['coord_error_power_L1_train'])
                strategies['Best-Power']['coord_errors_coverage_L1_train'].append(best_pwr_metrics['coord_error_coverage_L1_train'])
                strategies['Best-Power']['coord_errors_power_L1_manhattan'].append(best_pwr_metrics['coord_error_power_L1_manhattan'])
                strategies['Best-Power']['coord_errors_coverage_L1_manhattan'].append(best_pwr_metrics['coord_error_coverage_L1_manhattan'])
                strategies['Best-Power']['coverage_pcts'].append(best_pwr_metrics['coverage_pct_gt'])
                strategies['Best-Power']['pl_pct_coverages'].append(best_pwr_metrics['coverage_pct_power'])
                strategies['Best-Power']['pl_pct_refs'].append(best_pwr_metrics['pl_pct'])
                strategies['Best-Power']['diffusion_times'].append(avg_diffusion_time)
                strategies['Best-Power']['radionet_times'].append(avg_radionet_time)
                strategies['Best-Power']['building_times'].append(building_total_time)
                strategies['Best-Power']['building_ids'].append(building_id)
                strategies['Best-Power']['pred_coords'].append(best_pwr_metrics['pred_coord'])
                
                # Strategy 4: Best-L2 (lowest L2 distance from ideal point)
                # Calculate L2 distance from ideal (100%, 100%) for each sample
                l2_distances = [
                    np.sqrt((100 - m['coverage_pct_gt'])**2 + (100 - m['pl_pct'])**2)
                    for m in sample_metrics_list
                ]
                best_l2_idx = np.argmin(l2_distances)  # Lower is better
                best_l2_metrics = sample_metrics_list[best_l2_idx]
                strategies['Best-L2']['coord_errors_power'].append(best_l2_metrics['coord_error_power'])
                strategies['Best-L2']['coord_errors_coverage'].append(best_l2_metrics['coord_error_coverage'])
                strategies['Best-L2']['coord_errors_power_L1_train'].append(best_l2_metrics['coord_error_power_L1_train'])
                strategies['Best-L2']['coord_errors_coverage_L1_train'].append(best_l2_metrics['coord_error_coverage_L1_train'])
                strategies['Best-L2']['coord_errors_power_L1_manhattan'].append(best_l2_metrics['coord_error_power_L1_manhattan'])
                strategies['Best-L2']['coord_errors_coverage_L1_manhattan'].append(best_l2_metrics['coord_error_coverage_L1_manhattan'])
                strategies['Best-L2']['coverage_pcts'].append(best_l2_metrics['coverage_pct_gt'])
                strategies['Best-L2']['pl_pct_coverages'].append(best_l2_metrics['coverage_pct_power'])
                strategies['Best-L2']['pl_pct_refs'].append(best_l2_metrics['pl_pct'])
                strategies['Best-L2']['diffusion_times'].append(avg_diffusion_time)
                strategies['Best-L2']['radionet_times'].append(avg_radionet_time)
                strategies['Best-L2']['building_times'].append(building_total_time)
                strategies['Best-L2']['building_ids'].append(building_id)
                strategies['Best-L2']['pred_coords'].append(best_l2_metrics['pred_coord'])
                
                # Strategy 5: Average
                avg_coord_error_power = float(np.mean([m['coord_error_power'] for m in sample_metrics_list]))
                avg_coord_error_coverage = float(np.mean([m['coord_error_coverage'] for m in sample_metrics_list]))
                avg_coord_error_power_L1_train = float(np.mean([m['coord_error_power_L1_train'] for m in sample_metrics_list]))
                avg_coord_error_coverage_L1_train = float(np.mean([m['coord_error_coverage_L1_train'] for m in sample_metrics_list]))
                avg_coord_error_power_L1_manhattan = float(np.mean([m['coord_error_power_L1_manhattan'] for m in sample_metrics_list]))
                avg_coord_error_coverage_L1_manhattan = float(np.mean([m['coord_error_coverage_L1_manhattan'] for m in sample_metrics_list]))
                avg_coverage = float(np.mean([m['coverage_pct_gt'] for m in sample_metrics_list]))
                avg_cov_power = float(np.mean([m['coverage_pct_power'] for m in sample_metrics_list]))
                avg_pl = float(np.mean([m['pl_pct'] for m in sample_metrics_list]))
                strategies['Average']['coord_errors_power'].append(avg_coord_error_power)
                strategies['Average']['coord_errors_coverage'].append(avg_coord_error_coverage)
                strategies['Average']['coord_errors_power_L1_train'].append(avg_coord_error_power_L1_train)
                strategies['Average']['coord_errors_coverage_L1_train'].append(avg_coord_error_coverage_L1_train)
                strategies['Average']['coord_errors_power_L1_manhattan'].append(avg_coord_error_power_L1_manhattan)
                strategies['Average']['coord_errors_coverage_L1_manhattan'].append(avg_coord_error_coverage_L1_manhattan)
                strategies['Average']['coverage_pcts'].append(avg_coverage)
                strategies['Average']['pl_pct_coverages'].append(avg_cov_power)
                strategies['Average']['pl_pct_refs'].append(avg_pl)
                strategies['Average']['diffusion_times'].append(avg_diffusion_time)
                strategies['Average']['radionet_times'].append(avg_radionet_time)
                strategies['Average']['building_times'].append(building_total_time)
                
            else:  # optimize_for == 'coverage'
                # For coverage optimization
                # Strategy 1: Single (first sample)
                single_metrics = sample_metrics_list[0]
                strategies['Single']['coord_errors_power'].append(single_metrics['coord_error_power'])
                strategies['Single']['coord_errors_coverage'].append(single_metrics['coord_error_coverage'])
                strategies['Single']['coord_errors_power_L1_train'].append(single_metrics['coord_error_power_L1_train'])
                strategies['Single']['coord_errors_coverage_L1_train'].append(single_metrics['coord_error_coverage_L1_train'])
                strategies['Single']['coord_errors_power_L1_manhattan'].append(single_metrics['coord_error_power_L1_manhattan'])
                strategies['Single']['coord_errors_coverage_L1_manhattan'].append(single_metrics['coord_error_coverage_L1_manhattan'])
                strategies['Single']['coverage_pcts'].append(single_metrics['coverage_pct'])
                strategies['Single']['pl_pct_coverages'].append(single_metrics['pl_pct_coverage'])
                strategies['Single']['pl_pct_refs'].append(single_metrics['pl_pct_ref'])
                strategies['Single']['diffusion_times'].append(avg_diffusion_time)
                strategies['Single']['radionet_times'].append(avg_radionet_time)
                strategies['Single']['building_times'].append(building_total_time)
                strategies['Single']['building_ids'].append(building_id)
                strategies['Single']['pred_coords'].append(single_metrics['pred_coord'])
                
                # Strategy 2: Best-Coverage (highest coverage_pct)
                best_cov_idx = np.argmax([m['coverage_pct'] for m in sample_metrics_list])
                best_cov_metrics = sample_metrics_list[best_cov_idx]
                strategies['Best-Coverage']['coord_errors_power'].append(best_cov_metrics['coord_error_power'])
                strategies['Best-Coverage']['coord_errors_coverage'].append(best_cov_metrics['coord_error_coverage'])
                strategies['Best-Coverage']['coord_errors_power_L1_train'].append(best_cov_metrics['coord_error_power_L1_train'])
                strategies['Best-Coverage']['coord_errors_coverage_L1_train'].append(best_cov_metrics['coord_error_coverage_L1_train'])
                strategies['Best-Coverage']['coord_errors_power_L1_manhattan'].append(best_cov_metrics['coord_error_power_L1_manhattan'])
                strategies['Best-Coverage']['coord_errors_coverage_L1_manhattan'].append(best_cov_metrics['coord_error_coverage_L1_manhattan'])
                strategies['Best-Coverage']['coverage_pcts'].append(best_cov_metrics['coverage_pct'])
                strategies['Best-Coverage']['pl_pct_coverages'].append(best_cov_metrics['pl_pct_coverage'])
                strategies['Best-Coverage']['pl_pct_refs'].append(best_cov_metrics['pl_pct_ref'])
                strategies['Best-Coverage']['diffusion_times'].append(avg_diffusion_time)
                strategies['Best-Coverage']['radionet_times'].append(avg_radionet_time)
                strategies['Best-Coverage']['building_times'].append(building_total_time)
                strategies['Best-Coverage']['building_ids'].append(building_id)
                strategies['Best-Coverage']['pred_coords'].append(best_cov_metrics['pred_coord'])
                
                # Strategy 3: Best-Power (highest pl_pct_ref)
                best_pwr_idx = np.argmax([m['pl_pct_ref'] for m in sample_metrics_list])
                best_pwr_metrics = sample_metrics_list[best_pwr_idx]
                strategies['Best-Power']['coord_errors_power'].append(best_pwr_metrics['coord_error_power'])
                strategies['Best-Power']['coord_errors_coverage'].append(best_pwr_metrics['coord_error_coverage'])
                strategies['Best-Power']['coord_errors_power_L1_train'].append(best_pwr_metrics['coord_error_power_L1_train'])
                strategies['Best-Power']['coord_errors_coverage_L1_train'].append(best_pwr_metrics['coord_error_coverage_L1_train'])
                strategies['Best-Power']['coord_errors_power_L1_manhattan'].append(best_pwr_metrics['coord_error_power_L1_manhattan'])
                strategies['Best-Power']['coord_errors_coverage_L1_manhattan'].append(best_pwr_metrics['coord_error_coverage_L1_manhattan'])
                strategies['Best-Power']['coverage_pcts'].append(best_pwr_metrics['coverage_pct'])
                strategies['Best-Power']['pl_pct_coverages'].append(best_pwr_metrics['pl_pct_coverage'])
                strategies['Best-Power']['pl_pct_refs'].append(best_pwr_metrics['pl_pct_ref'])
                strategies['Best-Power']['diffusion_times'].append(avg_diffusion_time)
                strategies['Best-Power']['radionet_times'].append(avg_radionet_time)
                strategies['Best-Power']['building_times'].append(building_total_time)
                strategies['Best-Power']['building_ids'].append(building_id)
                strategies['Best-Power']['pred_coords'].append(best_pwr_metrics['pred_coord'])
                
                # Strategy 4: Best-L2 (lowest L2 distance from ideal point)
                # Calculate L2 distance from ideal (100%, 100%) for each sample
                l2_distances = [
                    np.sqrt((100 - m['coverage_pct'])**2 + (100 - m['pl_pct_ref'])**2)
                    for m in sample_metrics_list
                ]
                best_l2_idx = np.argmin(l2_distances)  # Lower is better
                best_l2_metrics = sample_metrics_list[best_l2_idx]
                strategies['Best-L2']['coord_errors_power'].append(best_l2_metrics['coord_error_power'])
                strategies['Best-L2']['coord_errors_coverage'].append(best_l2_metrics['coord_error_coverage'])
                strategies['Best-L2']['coord_errors_power_L1_train'].append(best_l2_metrics['coord_error_power_L1_train'])
                strategies['Best-L2']['coord_errors_coverage_L1_train'].append(best_l2_metrics['coord_error_coverage_L1_train'])
                strategies['Best-L2']['coord_errors_power_L1_manhattan'].append(best_l2_metrics['coord_error_power_L1_manhattan'])
                strategies['Best-L2']['coord_errors_coverage_L1_manhattan'].append(best_l2_metrics['coord_error_coverage_L1_manhattan'])
                strategies['Best-L2']['coverage_pcts'].append(best_l2_metrics['coverage_pct'])
                strategies['Best-L2']['pl_pct_coverages'].append(best_l2_metrics['pl_pct_coverage'])
                strategies['Best-L2']['pl_pct_refs'].append(best_l2_metrics['pl_pct_ref'])
                strategies['Best-L2']['diffusion_times'].append(avg_diffusion_time)
                strategies['Best-L2']['radionet_times'].append(avg_radionet_time)
                strategies['Best-L2']['building_times'].append(building_total_time)
                strategies['Best-L2']['building_ids'].append(building_id)
                strategies['Best-L2']['pred_coords'].append(best_l2_metrics['pred_coord'])
                
                # Strategy 5: Average
                avg_coord_error_power = float(np.mean([m['coord_error_power'] for m in sample_metrics_list]))
                avg_coord_error_coverage = float(np.mean([m['coord_error_coverage'] for m in sample_metrics_list]))
                avg_coord_error_power_L1_train = float(np.mean([m['coord_error_power_L1_train'] for m in sample_metrics_list]))
                avg_coord_error_coverage_L1_train = float(np.mean([m['coord_error_coverage_L1_train'] for m in sample_metrics_list]))
                avg_coord_error_power_L1_manhattan = float(np.mean([m['coord_error_power_L1_manhattan'] for m in sample_metrics_list]))
                avg_coord_error_coverage_L1_manhattan = float(np.mean([m['coord_error_coverage_L1_manhattan'] for m in sample_metrics_list]))
                avg_coverage = float(np.mean([m['coverage_pct'] for m in sample_metrics_list]))
                avg_pl_cov = float(np.mean([m['pl_pct_coverage'] for m in sample_metrics_list]))
                avg_pl_ref = float(np.mean([m['pl_pct_ref'] for m in sample_metrics_list]))
                strategies['Average']['coord_errors_power'].append(avg_coord_error_power)
                strategies['Average']['coord_errors_coverage'].append(avg_coord_error_coverage)
                strategies['Average']['coord_errors_power_L1_train'].append(avg_coord_error_power_L1_train)
                strategies['Average']['coord_errors_coverage_L1_train'].append(avg_coord_error_coverage_L1_train)
                strategies['Average']['coord_errors_power_L1_manhattan'].append(avg_coord_error_power_L1_manhattan)
                strategies['Average']['coord_errors_coverage_L1_manhattan'].append(avg_coord_error_coverage_L1_manhattan)
                strategies['Average']['coverage_pcts'].append(avg_coverage)
                strategies['Average']['pl_pct_coverages'].append(avg_pl_cov)
                strategies['Average']['pl_pct_refs'].append(avg_pl_ref)
                strategies['Average']['diffusion_times'].append(avg_diffusion_time)
                strategies['Average']['radionet_times'].append(avg_radionet_time)
                strategies['Average']['building_times'].append(building_total_time)
    
    # Compute statistics for each strategy
    results = {}
    for strategy_name, strategy_data in strategies.items():
        if len(strategy_data['coord_errors_power']) > 0:
            results[strategy_name] = {
                # L2 (Euclidean) coordinate errors
                'coord_error_power_mean': float(np.mean(strategy_data['coord_errors_power'])),
                'coord_error_power_median': float(np.median(strategy_data['coord_errors_power'])),
                'coord_error_power_std': float(np.std(strategy_data['coord_errors_power'])),
                'coord_error_coverage_mean': float(np.mean(strategy_data['coord_errors_coverage'])),
                'coord_error_coverage_median': float(np.median(strategy_data['coord_errors_coverage'])),
                'coord_error_coverage_std': float(np.std(strategy_data['coord_errors_coverage'])),
                # L1 training-style coordinate errors
                'coord_error_power_L1_train_mean': float(np.mean(strategy_data['coord_errors_power_L1_train'])),
                'coord_error_power_L1_train_median': float(np.median(strategy_data['coord_errors_power_L1_train'])),
                'coord_error_power_L1_train_std': float(np.std(strategy_data['coord_errors_power_L1_train'])),
                'coord_error_coverage_L1_train_mean': float(np.mean(strategy_data['coord_errors_coverage_L1_train'])),
                'coord_error_coverage_L1_train_median': float(np.median(strategy_data['coord_errors_coverage_L1_train'])),
                'coord_error_coverage_L1_train_std': float(np.std(strategy_data['coord_errors_coverage_L1_train'])),
                # L1 Manhattan coordinate errors
                'coord_error_power_L1_manhattan_mean': float(np.mean(strategy_data['coord_errors_power_L1_manhattan'])),
                'coord_error_power_L1_manhattan_median': float(np.median(strategy_data['coord_errors_power_L1_manhattan'])),
                'coord_error_power_L1_manhattan_std': float(np.std(strategy_data['coord_errors_power_L1_manhattan'])),
                'coord_error_coverage_L1_manhattan_mean': float(np.mean(strategy_data['coord_errors_coverage_L1_manhattan'])),
                'coord_error_coverage_L1_manhattan_median': float(np.median(strategy_data['coord_errors_coverage_L1_manhattan'])),
                'coord_error_coverage_L1_manhattan_std': float(np.std(strategy_data['coord_errors_coverage_L1_manhattan'])),
                # Radio metrics
                'coverage_pct_mean': float(np.mean(strategy_data['coverage_pcts'])),
                'coverage_pct_std': float(np.std(strategy_data['coverage_pcts'])),
                'pl_pct_coverage_mean': float(np.mean(strategy_data['pl_pct_coverages'])),
                'pl_pct_coverage_std': float(np.std(strategy_data['pl_pct_coverages'])),
                'pl_pct_ref_mean': float(np.mean(strategy_data['pl_pct_refs'])),
                'pl_pct_ref_std': float(np.std(strategy_data['pl_pct_refs'])),
                # Timing metrics
                'diffusion_time_per_sample_mean': float(np.mean(strategy_data['diffusion_times'])),
                'diffusion_time_per_sample_median': float(np.median(strategy_data['diffusion_times'])),
                'diffusion_time_per_sample_std': float(np.std(strategy_data['diffusion_times'])),
                'radionet_time_per_sample_mean': float(np.mean(strategy_data['radionet_times'])),
                'radionet_time_per_sample_median': float(np.median(strategy_data['radionet_times'])),
                'radionet_time_per_sample_std': float(np.std(strategy_data['radionet_times'])),
                'total_time_per_building_mean': float(np.mean(strategy_data['building_times'])),
                'total_time_per_building_median': float(np.median(strategy_data['building_times'])),
                'total_time_per_building_std': float(np.std(strategy_data['building_times'])),
                # Derived timing metrics
                'samples_per_second': float(num_samples / np.mean(strategy_data['building_times'])) if np.mean(strategy_data['building_times']) > 0 else 0.0,
                'buildings_per_hour': float(3600.0 / np.mean(strategy_data['building_times'])) if np.mean(strategy_data['building_times']) > 0 else 0.0,
                # Metadata
                'num_buildings': len(strategy_data['coord_errors_power']),
                'num_samples_per_building': num_samples,
                # Per-building data for visualization (convert numpy arrays to lists)
                'building_ids': [int(bid) if isinstance(bid, np.integer) else bid for bid in strategy_data['building_ids']],
                'pred_coords': [coord.tolist() if isinstance(coord, np.ndarray) else coord for coord in strategy_data['pred_coords']],
                'coord_errors_power': [float(e) for e in strategy_data['coord_errors_power']],
                'coord_errors_coverage': [float(e) for e in strategy_data['coord_errors_coverage']],
                'coverage_pcts': [float(c) for c in strategy_data['coverage_pcts']],
                'pl_pct_refs': [float(p) for p in strategy_data['pl_pct_refs']]
            }
        else:
            results[strategy_name] = None
    
    return results


def print_results(results, num_samples, optimize_for):
    """Print results in a readable format with DUAL coordinate errors and timing"""
    print(f"\n{'='*80}")
    print(f"RESULTS FOR {num_samples} SAMPLES PER BUILDING (optimize_for={optimize_for})")
    print(f"{'='*80}")
    
    if optimize_for == 'power':
        for strategy_name in ['Single', 'Best-Coverage', 'Best-Power', 'Average']:
            if strategy_name in results and results[strategy_name] is not None:
                r = results[strategy_name]
                print(f"\n{strategy_name}:")
                print(f"  L2 Coord Error (vs Power GT):        {r['coord_error_power_mean']:.2f} ± {r['coord_error_power_std']:.2f} px (median: {r['coord_error_power_median']:.2f})")
                print(f"  L2 Coord Error (vs Coverage GT):     {r['coord_error_coverage_mean']:.2f} ± {r['coord_error_coverage_std']:.2f} px (median: {r['coord_error_coverage_median']:.2f})")
                print(f"  L1-train Coord Error (vs Power GT):  {r['coord_error_power_L1_train_mean']:.2f} ± {r['coord_error_power_L1_train_std']:.2f} px (median: {r['coord_error_power_L1_train_median']:.2f})")
                print(f"  L1-train Coord Error (vs Cov GT):    {r['coord_error_coverage_L1_train_mean']:.2f} ± {r['coord_error_coverage_L1_train_std']:.2f} px (median: {r['coord_error_coverage_L1_train_median']:.2f})")
                print(f"  L1-manh Coord Error (vs Power GT):   {r['coord_error_power_L1_manhattan_mean']:.2f} ± {r['coord_error_power_L1_manhattan_std']:.2f} px (median: {r['coord_error_power_L1_manhattan_median']:.2f})")
                print(f"  L1-manh Coord Error (vs Cov GT):     {r['coord_error_coverage_L1_manhattan_mean']:.2f} ± {r['coord_error_coverage_L1_manhattan_std']:.2f} px (median: {r['coord_error_coverage_L1_manhattan_median']:.2f})")
                print(f"  Coverage%(ref):                       {r['coverage_pct_mean']:.2f} ± {r['coverage_pct_std']:.2f}%  [vs coverage-optimal GT]")
                print(f"  Coverage%(power):                     {r['pl_pct_coverage_mean']:.2f} ± {r['pl_pct_coverage_std']:.2f}%  [vs power-optimal GT]")
                print(f"  PL%:                                  {r['pl_pct_ref_mean']:.2f} ± {r['pl_pct_ref_std']:.2f}%  [vs power-optimal GT]")
                print(f"  ")
                print(f"  ⏱️  Timing:")
                print(f"    Diffusion per sample:    {r['diffusion_time_per_sample_mean']:.3f} ± {r['diffusion_time_per_sample_std']:.3f} s (median: {r['diffusion_time_per_sample_median']:.3f})")
                print(f"    RadioNet per sample:     {r['radionet_time_per_sample_mean']:.3f} ± {r['radionet_time_per_sample_std']:.3f} s (median: {r['radionet_time_per_sample_median']:.3f})")
                print(f"    Total per building:      {r['total_time_per_building_mean']:.2f} ± {r['total_time_per_building_std']:.2f} s (median: {r['total_time_per_building_median']:.2f})")
                print(f"    Samples/second:          {r['samples_per_second']:.2f}")
                print(f"    Buildings/hour:          {r['buildings_per_hour']:.1f}")
    else:
        for strategy_name in ['Single', 'Best-Coverage', 'Best-Power', 'Average']:
            if strategy_name in results and results[strategy_name] is not None:
                r = results[strategy_name]
                print(f"\n{strategy_name}:")
                print(f"  L2 Coord Error (vs Power GT):        {r['coord_error_power_mean']:.2f} ± {r['coord_error_power_std']:.2f} px (median: {r['coord_error_power_median']:.2f})")
                print(f"  L2 Coord Error (vs Coverage GT):     {r['coord_error_coverage_mean']:.2f} ± {r['coord_error_coverage_std']:.2f} px (median: {r['coord_error_coverage_median']:.2f})")
                print(f"  L1-train Coord Error (vs Power GT):  {r['coord_error_power_L1_train_mean']:.2f} ± {r['coord_error_power_L1_train_std']:.2f} px (median: {r['coord_error_power_L1_train_median']:.2f})")
                print(f"  L1-train Coord Error (vs Cov GT):    {r['coord_error_coverage_L1_train_mean']:.2f} ± {r['coord_error_coverage_L1_train_std']:.2f} px (median: {r['coord_error_coverage_L1_train_median']:.2f})")
                print(f"  L1-manh Coord Error (vs Power GT):   {r['coord_error_power_L1_manhattan_mean']:.2f} ± {r['coord_error_power_L1_manhattan_std']:.2f} px (median: {r['coord_error_power_L1_manhattan_median']:.2f})")
                print(f"  L1-manh Coord Error (vs Cov GT):     {r['coord_error_coverage_L1_manhattan_mean']:.2f} ± {r['coord_error_coverage_L1_manhattan_std']:.2f} px (median: {r['coord_error_coverage_L1_manhattan_median']:.2f})")
                print(f"  Coverage%:                            {r['coverage_pct_mean']:.2f} ± {r['coverage_pct_std']:.2f}%")
                print(f"  PL%(cov):                             {r['pl_pct_coverage_mean']:.2f} ± {r['pl_pct_coverage_std']:.2f}%")
                print(f"  PL%(ref):                             {r['pl_pct_ref_mean']:.2f} ± {r['pl_pct_ref_std']:.2f}%")
                print(f"  ")
                print(f"  ⏱️  Timing:")
                print(f"    Diffusion per sample:    {r['diffusion_time_per_sample_mean']:.3f} ± {r['diffusion_time_per_sample_std']:.3f} s (median: {r['diffusion_time_per_sample_median']:.3f})")
                print(f"    RadioNet per sample:     {r['radionet_time_per_sample_mean']:.3f} ± {r['radionet_time_per_sample_std']:.3f} s (median: {r['radionet_time_per_sample_median']:.3f})")
                print(f"    Total per building:      {r['total_time_per_building_mean']:.2f} ± {r['total_time_per_building_std']:.2f} s (median: {r['total_time_per_building_median']:.2f})")
                print(f"    Samples/second:          {r['samples_per_second']:.2f}")
                print(f"    Buildings/hour:          {r['buildings_per_hour']:.1f}")


def create_comparison_table(all_results, save_path, optimize_for):
    """Create comparison table with dual coordinate errors and timing"""
    rows = []
    for num_samples, results in sorted(all_results.items()):
        for strategy_name in ['Single', 'Best-Coverage', 'Best-Power', 'Average']:
            if strategy_name in results and results[strategy_name] is not None:
                r = results[strategy_name]
                row = {
                    'N_Samples': num_samples,
                    'Strategy': strategy_name,
                    'L2_Err_Power': f"{r['coord_error_power_mean']:.2f}",
                    'L2_Err_Coverage': f"{r['coord_error_coverage_mean']:.2f}",
                    'L1train_Err_Power': f"{r['coord_error_power_L1_train_mean']:.2f}",
                    'L1train_Err_Cov': f"{r['coord_error_coverage_L1_train_mean']:.2f}",
                    'L1manh_Err_Power': f"{r['coord_error_power_L1_manhattan_mean']:.2f}",
                    'L1manh_Err_Cov': f"{r['coord_error_coverage_L1_manhattan_mean']:.2f}",
                    'Coverage%': f"{r['coverage_pct_mean']:.2f}",
                    'PL%(cov)': f"{r['pl_pct_coverage_mean']:.2f}",
                    'PL%(ref)': f"{r['pl_pct_ref_mean']:.2f}",
                    'Diffusion_s': f"{r['diffusion_time_per_sample_mean']:.3f}",
                    'RadioNet_s': f"{r['radionet_time_per_sample_mean']:.3f}",
                    'Total_s': f"{r['total_time_per_building_mean']:.1f}",
                    'Bldg/hr': f"{r['buildings_per_hour']:.1f}"
                }
                rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save to text file
    with open(save_path, 'w') as f:
        f.write("MULTI-SAMPLE EVALUATION COMPARISON WITH DUAL GT AND THREE DISTANCE METRICS\n")
        f.write("="*100 + "\n\n")
        f.write(f"Optimization target: {optimize_for}\n\n")
        f.write("Metrics Explained:\n")
        f.write("  Coordinate Errors (three distance metrics):\n")
        f.write("    - L2_Err_Power/Coverage: L2 (Euclidean) distance from GT (pixels)\n")
        f.write("    - L1train_Err_Power/Cov: L1 training-style (mean of |Δy|+|Δx|) from GT\n")
        f.write("    - L1manh_Err_Power/Cov: L1 Manhattan distance (|Δy|+|Δx|) from GT\n")
        f.write("  Radio Performance:\n")
        f.write("    - Coverage%: Predicted coverage / GT coverage-optimal × 100%\n")
        f.write("    - PL%(cov): Predicted power / GT coverage-optimal power × 100%\n")
        f.write("    - PL%(ref): Predicted power / GT power-optimal power × 100%\n")
        f.write("  Timing:\n")
        f.write("    - Diffusion_s: Diffusion inference time per sample (seconds)\n")
        f.write("    - RadioNet_s: RadioNet evaluation time per sample (seconds)\n")
        f.write("    - Total_s: Total time per building for all samples (seconds)\n")
        f.write("    - Bldg/hr: Buildings processed per hour (throughput)\n\n")
        f.write("="*100 + "\n\n")
        f.write(df.to_string(index=False))
    
    # Save CSV
    csv_path = save_path.replace('.txt', '.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"\nComparison table saved to:")
    print(f"  - {save_path}")
    print(f"  - {csv_path}")
    
    return df



# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def generate_quantile_visualizations(results, num_samples, eval_dataset, power_dataset, coverage_dataset,
                                      buildings_dir, radionet_model, use_center, save_dir):
    """
    Generate visualizations for buildings sampled from L2 error quantiles.
    
    Creates visualizations for Best-Coverage and Best-Power strategies, showing:
    - Predicted radio power and coverage maps
    - Ground truth maps
    - Tx location overlays
    
    Args:
        results: Evaluation results dictionary
        num_samples: Number of samples used in evaluation
        eval_dataset, power_dataset, coverage_dataset: Datasets
        buildings_dir: Directory containing building images
        radionet_model: RadioNet model for generating predictions
        use_center: Whether center cropping was used
        save_dir: Base directory for saving visualizations
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap
    
    viz_dir = os.path.join(save_dir, 'quantile_visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Process Best-Coverage and Best-Power strategies
    strategies_to_viz = ['Best-Coverage', 'Best-Power']
    
    selected_buildings_data = []
    
    for strategy in strategies_to_viz:
        if strategy not in results or results[strategy] is None:
            continue
            
        strategy_data = results[strategy]
        
        # Determine which error to use for quantiles
        if strategy == 'Best-Coverage':
            errors = np.array(strategy_data['coord_errors_coverage'])
            error_type = 'L2_Err_Coverage'
        else:  # Best-Power
            errors = np.array(strategy_data['coord_errors_power'])
            error_type = 'L2_Err_Power'
        
        building_ids = strategy_data['building_ids']
        pred_coords_list = strategy_data['pred_coords']
        
        # Compute uniform bins in error space (not percentile-based)
        min_error = np.min(errors)
        max_error = np.max(errors)
        n_bins = 10
        quantiles = np.linspace(min_error, max_error, n_bins + 1)
        
        strategy_dir = os.path.join(viz_dir, f"{strategy.lower().replace('-', '_')}_strategy")
        os.makedirs(strategy_dir, exist_ok=True)
        
        # Sample 5 buildings from each quantile
        for q_idx in range(len(quantiles) - 1):
            q_low, q_high = quantiles[q_idx], quantiles[q_idx + 1]
            
            # Find buildings in this quantile range
            in_quantile = (errors >= q_low) & (errors <= q_high)
            quantile_indices = np.where(in_quantile)[0]
            
            if len(quantile_indices) == 0:
                continue
            
            # Sample up to 5 buildings from this quantile
            n_samples_viz = min(5, len(quantile_indices))
            sampled_indices = np.random.choice(quantile_indices, size=n_samples_viz, replace=False)
            
            # Create quantile directory
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
                pred_coord = pred_coords_list[idx]
                error_val = errors[idx]
                
                # Get coverage and power errors for this building
                error_cov = strategy_data['coord_errors_coverage'][idx]
                error_pow = strategy_data['coord_errors_power'][idx]
                coverage_pct = strategy_data['coverage_pcts'][idx]
                pl_pct_ref = strategy_data['pl_pct_refs'][idx]
                
                quantile_buildings.append(building_id)
                
                # Find building index in datasets
                building_idx = None
                for i in range(len(eval_dataset)):
                    _, _, _, bid = eval_dataset[i]
                    if bid == building_id:
                        building_idx = i
                        break
                
                if building_idx is None:
                    continue
                
                # Load ground truth coordinates
                _, power_gt_coords, _, _ = power_dataset[building_idx]
                _, coverage_gt_coords, _, _ = coverage_dataset[building_idx]
                power_gt = power_gt_coords.cpu().numpy()
                coverage_gt = coverage_gt_coords.cpu().numpy()
                
                # Generate visualizations
                visualize_building(
                    building_id, pred_coord, power_gt, coverage_gt,
                    error_pow, error_cov, coverage_pct, pl_pct_ref,
                    buildings_dir, radionet_model, use_center,
                    quantile_dir
                )
                
                # Track for summary
                selected_buildings_data.append({
                    'building_id': building_id,
                    'strategy': strategy,
                    'quantile_range': quantile_name,
                    'L2_error_coverage': error_cov,
                    'L2_error_power': error_pow,
                    'coverage_pct': coverage_pct,
                    'pl_pct_ref': pl_pct_ref
                })
            
            # Create summary file for this quantile
            summary_path = os.path.join(quantile_dir, 'summary.txt')
            with open(summary_path, 'w') as f:
                f.write(f"Quantile Range: {q_low:.2f} - {q_high:.2f} pixels\n")
                f.write(f"Samples: {len(quantile_buildings)}\n")
                f.write(f"Average L2 Error: {errors[sampled_indices].mean():.2f} pixels\n")
                if strategy == 'Best-Coverage':
                    f.write(f"Average Coverage%: {np.array([strategy_data['coverage_pcts'][i] for i in sampled_indices]).mean():.2f}%\n")
                else:
                    f.write(f"Average PL%: {np.array([strategy_data['pl_pct_refs'][i] for i in sampled_indices]).mean():.2f}%\n")
                f.write(f"Building IDs: {', '.join(map(str, quantile_buildings))}\n")
    
    # Save selected buildings list
    if len(selected_buildings_data) > 0:
        df = pd.DataFrame(selected_buildings_data)
        df.to_csv(os.path.join(viz_dir, 'selected_buildings.csv'), index=False)
        
        # Create simple text file
        with open(os.path.join(viz_dir, 'selected_buildings.txt'), 'w') as f:
            f.write("# Buildings visualized for quantile analysis\n\n")
            for strategy in strategies_to_viz:
                strategy_buildings = df[df['strategy'] == strategy]
                if len(strategy_buildings) > 0:
                    f.write(f"# {strategy} Strategy\n")
                    for quantile in strategy_buildings['quantile_range'].unique():
                        quant_buildings = strategy_buildings[strategy_buildings['quantile_range'] == quantile]
                        building_ids_str = ', '.join(quant_buildings['building_id'].astype(str).tolist())
                        f.write(f"# {quantile}: {building_ids_str}\n")
                    f.write("\n")
        
        print(f"\n✓ Generated quantile visualizations: {viz_dir}")
        print(f"  - Total buildings visualized: {len(selected_buildings_data)}")
        print(f"  - Selected buildings saved to: {os.path.join(viz_dir, 'selected_buildings.csv')}")



def generate_radio_maps(building_img, tx_y, tx_x, radionet_model):
    """Generate radio power map using RadioNet. Returns uint8 [0-255] power map."""
    tx_onehot = np.zeros((256, 256), dtype=np.float32)
    tx_y_int = int(np.clip(tx_y, 0, 255))
    tx_x_int = int(np.clip(tx_x, 0, 255))
    tx_onehot[tx_y_int, tx_x_int] = 1.0
    
    building_tensor = torch.from_numpy(building_img).unsqueeze(0).unsqueeze(0)
    tx_tensor = torch.from_numpy(tx_onehot).unsqueeze(0).unsqueeze(0)
    inputs = torch.cat([building_tensor, tx_tensor], dim=1).to(next(radionet_model.parameters()).device)
    
    with torch.no_grad():
        power_map_float = radionet_model(inputs)[0, 0].cpu().numpy()
    
    # Scale RadioNet output from [0,1] to [0,255] to match GT PL maps
    power_map_uint8 = np.clip(power_map_float * 255, 0, 255).astype(np.uint8)
    
    return power_map_uint8


def compute_coverage_map(power_map_uint8):
    """Generate binary coverage map: 255 where power > 0, else 0."""
    return (power_map_uint8 > 0).astype(np.uint8) * 255


def save_simple_grayscale(image_uint8, filepath):
    """Save a simple grayscale image with no decorations."""
    from PIL import Image
    Image.fromarray(image_uint8).save(filepath)


def save_map_with_tx_overlay(image_uint8, tx_locations, filepath, building_img=None):
    """
    Save grayscale image with colored Tx location dots.
    Uses PIL for direct pixel manipulation to ensure exact 256x256 output.
    
    tx_locations: dict with keys 'pred', 'power_gt', 'coverage_gt'
                  each value is (y, x) tuple or None
    """
    from PIL import Image, ImageDraw
    
    # Convert to RGB for colored dots
    img = Image.fromarray(image_uint8).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # Draw colored dots (circles) for Tx locations
    dot_radius = 4
    
    # Red dot for predicted
    if tx_locations.get('pred') is not None:
        y, x = tx_locations['pred']
        x, y = int(x), int(y)
        draw.ellipse([x-dot_radius, y-dot_radius, x+dot_radius, y+dot_radius], 
                     fill=(255, 0, 0), outline=(255, 255, 255), width=1)
    
    # Blue dot for power-optimal GT
    if tx_locations.get('power_gt') is not None:
        y, x = tx_locations['power_gt']
        x, y = int(x), int(y)
        draw.ellipse([x-dot_radius, y-dot_radius, x+dot_radius, y+dot_radius], 
                     fill=(0, 0, 255), outline=(255, 255, 255), width=1)
    
    # Green dot for coverage-optimal GT
    if tx_locations.get('coverage_gt') is not None:
        y, x = tx_locations['coverage_gt']
        x, y = int(x), int(y)
        draw.ellipse([x-dot_radius, y-dot_radius, x+dot_radius, y+dot_radius], 
                     fill=(0, 255, 0), outline=(255, 255, 255), width=1)
    
    # Save as PNG (exact 256x256)
    img.save(filepath)


def save_info_txt(building_id, pred_coord, power_gt, coverage_gt, 
                  error_pow, error_cov, coverage_pct, pl_pct, filepath):
    """Save metadata text file."""
    with open(filepath, 'w') as f:
        f.write(f"Building ID: {building_id}\n")
        f.write(f"Predicted Tx (y,x): ({pred_coord[0]:.2f}, {pred_coord[1]:.2f})\n")
        f.write(f"Power-optimal GT (y,x): ({power_gt[0]:.2f}, {power_gt[1]:.2f})\n")
        f.write(f"Coverage-optimal GT (y,x): ({coverage_gt[0]:.2f}, {coverage_gt[1]:.2f})\n")
        f.write(f"L2 Error vs Power GT: {error_pow:.2f} px\n")
        f.write(f"L2 Error vs Coverage GT: {error_cov:.2f} px\n")
        f.write(f"Coverage %: {coverage_pct:.2f}\n")
        f.write(f"PL %: {pl_pct:.2f}\n")


def visualize_building(building_id, pred_coord, power_gt, coverage_gt,
                       error_pow, error_cov, coverage_pct, pl_pct_ref,
                       buildings_dir, radionet_model, use_center, save_dir):
    """
    Generate 9 visualization files + 1 text file for a single building.
    """
    from PIL import Image
    
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
    parser = argparse.ArgumentParser(description='Multi-Sample Evaluation - DUAL GT VERSION')
    
    # Model
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--conditioning', type=str, default='concat',
                       choices=['concat', 'film', 'cross_attn'])
    
    # Evaluation setup
    parser.add_argument('--num_samples', type=int, nargs='+', default=[1, 5, 10, 20, 50, 100])
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate on')
    parser.add_argument('--val_set', type=str, default='small',
                       choices=['small', 'full'])
    parser.add_argument('--small_val_size', type=int, default=1000)
    
    # Diffusion parameters
    parser.add_argument('--num_diffusion_steps', type=int, default=1000)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--use_ddim', action='store_true', default=True)
    
    # Data
    parser.add_argument('--optimize_for', type=str, default='coverage',
                       choices=['power', 'coverage'])
    parser.add_argument('--use_center', action='store_true', default=True)
    
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
    print(f"Using device: {device}\n")
    
    # Load model
    print("Loading diffusion model...")
    img_size = 150 if args.use_center else 256
    model = create_diffusion_model(conditioning=args.conditioning, img_size=img_size).to(device)
    
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
    
    # Create scheduler
    if args.use_ddim:
        scheduler = DDIMScheduler(num_train_timesteps=args.num_diffusion_steps)
        print(f"Using DDIM scheduler with {args.num_inference_steps} steps\n")
    else:
        scheduler = DDPMScheduler(num_train_timesteps=args.num_diffusion_steps)
        print(f"Using DDPM scheduler with {args.num_inference_steps} steps\n")
    
    # Load dataset splits
    print(f"Loading {args.split} dataset...")
    train_ids, val_ids, test_ids = get_building_splits(args.buildings_dir)
    
    # Select the appropriate split
    if args.split == 'train':
        split_ids = train_ids
    elif args.split == 'val':
        split_ids = val_ids
    else:  # test
        split_ids = test_ids
    
    tx_dir = os.path.join(args.heatmap_base_dir, f'best_by_{args.optimize_for}')
    
    # Create dataset for the selected split
    eval_dataset = TxLocationDatasetLarge(
        split_ids,
        buildings_dir=args.buildings_dir,
        tx_dir=tx_dir,
        optimize_for=args.optimize_for,
        heatmap_type='radio_power',
        heatmap_base_dir=args.heatmap_base_dir,
        normalize_heatmap=False,
        augment=False,
        use_center=args.use_center
    )
    
    # Load BOTH power and coverage datasets for dual GT evaluation
    print("Loading power-optimal GT dataset...")
    power_dataset = TxLocationDatasetLarge(
        split_ids,
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
        split_ids,
        buildings_dir=args.buildings_dir,
        tx_dir=os.path.join(args.heatmap_base_dir, 'best_by_coverage'),
        optimize_for='coverage',
        heatmap_type='radio_power',
        heatmap_base_dir=args.heatmap_base_dir,
        normalize_heatmap=False,
        augment=False,
        use_center=args.use_center
    )
    
    # Subset for small evaluation set
    if args.val_set == 'small':
        max_buildings = min(args.small_val_size, len(eval_dataset))
        print(f"Using small {args.split} set: {max_buildings} buildings")
    else:
        max_buildings = None
        print(f"Using full {args.split} set: {len(eval_dataset)} buildings")
    
    print(f"\n{'='*80}")
    print(f"MULTI-SAMPLE EVALUATION - DUAL GT VERSION")
    print(f"{'='*80}")
    print(f"Model: {args.model_path}")
    print(f"Conditioning: {args.conditioning}")
    print(f"Optimize for: {args.optimize_for}")
    print(f"Dataset split: {args.split}")
    print(f"Evaluation set: {args.val_set} ({max_buildings or len(eval_dataset)} buildings)")
    print(f"Sample counts to test: {args.num_samples}")
    print(f"{'='*80}\n")
    
    # Run evaluation for each sample count
    all_results = {}
    
    for num_samples in args.num_samples:
        print(f"\n{'#'*80}")
        print(f"# Testing with {num_samples} samples per building")
        print(f"{'#'*80}")
        
        start_time = time.time()
        results = evaluate_multisample(
            model, eval_dataset, power_dataset, coverage_dataset, num_samples, 
            scheduler, args.num_inference_steps, args.use_ddim, device, args.optimize_for, 
            args.use_center, args.buildings_dir, radionet_model, args.heatmap_base_dir,
            max_buildings
        )
        elapsed_time = time.time() - start_time
        print(f"\nEvaluation completed in {elapsed_time/60:.1f} minutes")
        
        # Print results
        print_results(results, num_samples, args.optimize_for)
        
        # Save JSON
        json_path = os.path.join(args.save_dir, f'results_samples{num_samples}.json')
        with open(json_path, 'w') as f:
            json.dump(convert_to_json_serializable(results), f, indent=2)
        print(f"Saved results to: {json_path}")

        # Generate quantile visualizations
        print(f"\n{'='*80}")
        print("Generating quantile visualizations...")
        print(f"{'='*80}\n")
        
        if 'building_ids' in results.get('Best-Coverage', {}) and len(results['Best-Coverage']['building_ids']) > 0:
            generate_quantile_visualizations(
                results, num_samples, eval_dataset, power_dataset, coverage_dataset,
                args.buildings_dir, radionet_model, args.use_center, args.save_dir
            )
        else:
            print("⚠ Skipping visualization: No per-building data available")
        
        all_results[num_samples] = results
    
    # Create comparison table
    print(f"\n{'='*80}")
    print("Creating comparison table across all sample counts...")
    print(f"{'='*80}\n")
    
    table_path = os.path.join(args.save_dir, 'comparison_table_dual_gt.txt')
    df = create_comparison_table(all_results, table_path, args.optimize_for)
    
    print("\n" + df.to_string(index=False))
    
    # Create summary
    summary = {
        'model_path': args.model_path,
        'conditioning': args.conditioning,
        'optimize_for': args.optimize_for,
        'split': args.split,
        'val_set': args.val_set,
        'num_buildings': max_buildings or len(eval_dataset),
        'num_inference_steps': args.num_inference_steps,  # ADDED FOR L2 EXPERIMENTS
        'sample_counts_tested': args.num_samples,
        'all_results': {str(k): v for k, v in all_results.items()}
    }
    
    summary_path = os.path.join(args.save_dir, 'summary_dual_gt.json')
    with open(summary_path, 'w') as f:
        json.dump(convert_to_json_serializable(summary), f, indent=2)
    
    print("EVALUATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Results saved to: {args.save_dir}")
    print(f"  - results_samples*.json (detailed statistics with dual GT errors)")
    print(f"  - comparison_table_dual_gt.txt/csv (comparison with both coordinate errors)")
    print(f"  - summary_dual_gt.json (overall summary)")
    print(f"  - quantile_visualizations/ (building visualizations by error quantile)")
    print(f"{'='*80}\n")




if __name__ == '__main__':
    import sys
    import traceback
    try:
        # Force unbuffered output
        sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
        sys.stderr = open(sys.stderr.fileno(), 'w', buffering=1)
        
        print("DEBUG: Starting main()", flush=True)
        main()
        print("DEBUG: main() completed successfully", flush=True)
    except Exception as e:
        print(f"\n{'='*80}", flush=True, file=sys.stderr)
        print(f"FATAL ERROR IN MAIN:", flush=True, file=sys.stderr)
        print(f"{'='*80}", flush=True, file=sys.stderr)
        print(f"{type(e).__name__}: {e}", flush=True, file=sys.stderr)
        print(f"\nFull traceback:", flush=True, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print(f"{'='*80}", flush=True, file=sys.stderr)
        sys.exit(1)

