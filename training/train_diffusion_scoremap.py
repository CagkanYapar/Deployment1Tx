"""
Diffusion Model Training Script for Average Maps - COMPLETE VERSION

Trains diffusion models on average power/coverage maps with:
1. Training using modular functions from training/train_diffusion_heatmap.py
2. AUTOMATIC test set evaluation after training with multi-sample support
3. Results saved in standard format matching evaluate_fixed_DUAL_GT_test_set.py

Uses concat conditioning only (most successful in previous experiments)
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image")
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from PIL import Image
import json
from tqdm import tqdm

# Import new dataset class
from data.dataset_scoremap import TxLocationDatasetAvgMap, get_building_splits
from models.diffusion import create_diffusion_model

# Import existing diffusion training functions - REUSE!
from training.train_diffusion_heatmap import (
    DDPMScheduler,
    train_one_epoch_diffusion,
    validate_one_epoch_diffusion,
    sample_diffusion
)

# Import validation helpers
from training.train_heatmap import (
    evaluate_with_avg_maps,
    load_pl_map_metrics
)

# Import multi-sample evaluation utilities
from evaluation.eval_utils import (
    extract_topn_candidates,
    evaluate_candidates_batch_radionet,
    select_best_candidate
)

sys.path.append('.')
from models.saipp_net import RadioNet


def get_hardware_info():
    """Capture hardware information for timing context"""
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
        
        device_props = torch.cuda.get_device_properties(0)
        hardware_info['gpu_name'] = device_props.name
        hardware_info['gpu_memory_gb'] = round(device_props.total_memory / 1024**3, 2)
        
        hardware_info['cuda_version'] = torch.version.cuda
        
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                hardware_info['gpu_driver_version'] = result.stdout.strip().split('\n')[0]
        except:
            pass
    
    return hardware_info


def convert_to_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def train_diffusion_model(model, train_loader, val_loader, optimizer, scheduler, scheduler_diffusion,
                          num_epochs, device, save_dir, optimize_for, use_center, buildings_dir,
                          heatmap_base_dir, radionet_model, num_inference_steps=50):
    """
    Train diffusion model with validation - adapted from training/train_diffusion_heatmap.py
    """
    
    best_primary_metric = -float('inf')
    best_coord_error = float('inf')
    best_primary_model = None
    best_coord_model = None
    
    training_history = {
        'train_loss': [],
        'val_coord_error': [],
        'val_primary_metric': []
    }
    
    print(f"\n{'='*80}")
    print("STARTING DIFFUSION TRAINING")
    print(f"{'='*80}\n")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 80)
        
        # Train
        train_loss = train_one_epoch_diffusion(
            model, train_loader, optimizer, device, 
            scheduler_diffusion, epoch
        )
        
        training_history['train_loss'].append(train_loss)
        
        # Validate
        val_metrics = validate_one_epoch_diffusion(
            model, val_loader, device, epoch, save_dir,
            use_center, optimize_for, scheduler_diffusion,
            radionet_model, buildings_dir, heatmap_base_dir,
            num_inference_steps=num_inference_steps
        )
        
        # Extract metrics
        coord_error = val_metrics['coord_error']
        
        if optimize_for == 'power':
            primary_metric = val_metrics.get('avg_pl_pct', 0.0)
        else:  # coverage
            primary_metric = val_metrics.get('avg_coverage_pct', 0.0)
        
        training_history['val_coord_error'].append(coord_error)
        training_history['val_primary_metric'].append(primary_metric)
        
        # Save best models
        if primary_metric > best_primary_metric:
            best_primary_metric = primary_metric
            best_primary_model = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'primary_metric': primary_metric,
                'coord_error': coord_error
            }
            
            # Save checkpoint
            if optimize_for == 'power':
                checkpoint_path = os.path.join(save_dir, 'best_pl_model.pth')
            else:
                checkpoint_path = os.path.join(save_dir, 'best_coverage_model.pth')
            
            torch.save(best_primary_model, checkpoint_path)
            print(f"\n✓ Saved best primary model: {checkpoint_path}")
            print(f"  Primary metric: {primary_metric:.2f}%")
        
        if coord_error < best_coord_error:
            best_coord_error = coord_error
            best_coord_model = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'primary_metric': primary_metric,
                'coord_error': coord_error
            }
            
            checkpoint_path = os.path.join(save_dir, 'best_coord_model.pth')
            torch.save(best_coord_model, checkpoint_path)
            print(f"\n✓ Saved best coord model: {checkpoint_path}")
            print(f"  Coord error: {coord_error:.2f} px")
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
        
        print(f"\nEpoch Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Coord Error: {coord_error:.2f} px")
        print(f"  Val Primary Metric: {primary_metric:.2f}%")
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETED")
    print(f"{'='*80}")
    print(f"Best primary metric: {best_primary_metric:.2f}%")
    print(f"Best coord error: {best_coord_error:.2f} px")
    
    return training_history


def evaluate_test_set_multisample_diffusion(model, test_dataset, power_dataset, coverage_dataset,
                                            optimize_for, use_center, buildings_dir, radionet_model,
                                            heatmap_base_dir, device, scheduler_diffusion,
                                            num_inference_steps=50,
                                            sample_counts=[1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]):
    """
    Evaluate diffusion model on test set with multi-sample support.
    
    For diffusion: Sample from noise multiple times to get different predictions.
    """
    model.eval()
    
    print(f"\n{'='*80}")
    print("TEST SET EVALUATION - DIFFUSION WITH MULTI-SAMPLE")
    print(f"{'='*80}")
    print(f"Sample counts: {sample_counts}")
    print(f"Optimize for: {optimize_for}")
    print(f"Test buildings: {len(test_dataset)}")
    print(f"Inference steps: {num_inference_steps}")
    print(f"{'='*80}\n")
    
    results_by_n = {}
    max_n = max(sample_counts)
    
    overall_start = time.time()
    
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="Evaluating"):
            building_start = time.time()
            
            # Get data
            building_map, gt_coords, target_heatmap, building_id = test_dataset[idx]
            _, power_gt, _, power_building_id = power_dataset[idx]
            _, coverage_gt, _, coverage_building_id = coverage_dataset[idx]
            
            assert building_id == power_building_id == coverage_building_id
            
            # Generate multiple samples from diffusion model
            building_map_input = building_map.unsqueeze(0).to(device)
            
            model_start = time.time()
            
            # Sample max_n predictions from diffusion
            all_pred_coords = []
            for _ in range(max_n):
                pred_heatmap = sample_diffusion(
                    model, building_map_input, optimize_for, scheduler_diffusion,
                    num_inference_steps=num_inference_steps, guidance_scale=1.0
                )
                
                # Extract coordinate from predicted heatmap
                pred_heatmap_np = pred_heatmap[0, 0].cpu().numpy()
                pred_y, pred_x = np.unravel_index(pred_heatmap_np.argmax(), pred_heatmap_np.shape)
                all_pred_coords.append((float(pred_y), float(pred_x)))
            
            model_time = time.time() - model_start
            
            # Evaluate all candidates with RadioNet (batched)
            radionet_start = time.time()
            candidate_metrics = evaluate_candidates_batch_radionet(
                all_pred_coords, building_id, buildings_dir, radionet_model,
                use_center=use_center, batch_size=64, device=device
            )
            radionet_time = time.time() - radionet_start
            
            building_total_time = time.time() - building_start
            
            # GTs
            power_gt_np = power_gt.numpy()
            coverage_gt_np = coverage_gt.numpy()
            
            # For each sample count
            for n in sample_counts:
                if n not in results_by_n:
                    results_by_n[n] = {
                        'coord_errors_power': [],
                        'coord_errors_coverage': [],
                        'coord_errors_power_L1_train': [],
                        'coord_errors_coverage_L1_train': [],
                        'coord_errors_power_L1_manhattan': [],
                        'coord_errors_coverage_L1_manhattan': [],
                        'coverage_pcts': [],
                        'coverage_pct_powers': [],
                        'pl_pct_covs': [],
                        'pl_pct_refs': [],
                        'model_ranks': [],
                        'building_ids': [],
                        'pred_coords': [],
                        'model_times': [],
                        'radionet_times': [],
                        'total_times': []
                    }
                
                # Get first n samples
                candidates_n = all_pred_coords[:n]
                metrics_n = candidate_metrics[:n]
                
                # Select best by RadioNet
                criterion = 'coverage' if optimize_for == 'coverage' else 'power'
                best_idx, best_metrics = select_best_candidate(metrics_n, criterion=criterion)
                
                pred_coord_np = np.array(best_metrics['coord'], dtype=np.float32)
                model_rank = best_idx + 1
                
                # Coordinate errors
                coord_error_power = float(np.sqrt(np.sum((pred_coord_np - power_gt_np)**2)))
                coord_error_coverage = float(np.sqrt(np.sum((pred_coord_np - coverage_gt_np)**2)))
                coord_error_power_L1_train = float(np.mean(np.abs(pred_coord_np - power_gt_np)))
                coord_error_coverage_L1_train = float(np.mean(np.abs(pred_coord_np - coverage_gt_np)))
                coord_error_power_L1_manhattan = float(np.sum(np.abs(pred_coord_np - power_gt_np)))
                coord_error_coverage_L1_manhattan = float(np.sum(np.abs(pred_coord_np - coverage_gt_np)))
                
                # Radio metrics
                gt_coords_np = gt_coords.numpy()
                metrics = evaluate_with_avg_maps(
                    pred_coord_np, gt_coords_np, building_id, heatmap_base_dir,
                    optimize_for, use_center, buildings_dir, radionet_model
                )
                
                if metrics is not None:
                    # Compute all 4 dual-GT metrics
                    if optimize_for == 'power':
                        coverage_pct_ref = metrics['coverage_pct_gt']
                        coverage_pct_power = metrics['coverage_pct_power']
                        pl_pct_ref = metrics['pl_pct']
                        
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
                    else:
                        coverage_pct_ref = metrics['coverage_pct']
                        pl_pct_cov = metrics['pl_pct_coverage']
                        pl_pct_ref = metrics['pl_pct_ref']
                        
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
                    
                    if coverage_pct_power is None or pl_pct_cov is None:
                        continue
                    
                    # Store results
                    results_by_n[n]['coord_errors_power'].append(coord_error_power)
                    results_by_n[n]['coord_errors_coverage'].append(coord_error_coverage)
                    results_by_n[n]['coord_errors_power_L1_train'].append(coord_error_power_L1_train)
                    results_by_n[n]['coord_errors_coverage_L1_train'].append(coord_error_coverage_L1_train)
                    results_by_n[n]['coord_errors_power_L1_manhattan'].append(coord_error_power_L1_manhattan)
                    results_by_n[n]['coord_errors_coverage_L1_manhattan'].append(coord_error_coverage_L1_manhattan)
                    results_by_n[n]['coverage_pcts'].append(coverage_pct_ref)
                    results_by_n[n]['coverage_pct_powers'].append(coverage_pct_power)
                    results_by_n[n]['pl_pct_covs'].append(pl_pct_cov)
                    results_by_n[n]['pl_pct_refs'].append(pl_pct_ref)
                    results_by_n[n]['model_ranks'].append(model_rank)
                    results_by_n[n]['building_ids'].append(building_id)
                    results_by_n[n]['pred_coords'].append(pred_coord_np)
                    results_by_n[n]['model_times'].append(model_time * n / max_n)
                    results_by_n[n]['radionet_times'].append(radionet_time * n / max_n)
                    results_by_n[n]['total_times'].append((model_time + radionet_time) * n / max_n)
    
    overall_time = time.time() - overall_start
    
    # Compute aggregates
    print("\nComputing aggregate statistics...")
    
    aggregated_by_n = {}
    for n in sample_counts:
        if n not in results_by_n or len(results_by_n[n]['coord_errors_power']) == 0:
            continue
        
        results = results_by_n[n]
        
        aggregated_by_n[n] = {
            'coord_error_power_mean': float(np.mean(results['coord_errors_power'])),
            'coord_error_power_median': float(np.median(results['coord_errors_power'])),
            'coord_error_power_std': float(np.std(results['coord_errors_power'])),
            'coord_error_coverage_mean': float(np.mean(results['coord_errors_coverage'])),
            'coord_error_coverage_median': float(np.median(results['coord_errors_coverage'])),
            'coord_error_coverage_std': float(np.std(results['coord_errors_coverage'])),
            'coord_error_power_L1_train_mean': float(np.mean(results['coord_errors_power_L1_train'])),
            'coord_error_power_L1_train_median': float(np.median(results['coord_errors_power_L1_train'])),
            'coord_error_power_L1_train_std': float(np.std(results['coord_errors_power_L1_train'])),
            'coord_error_coverage_L1_train_mean': float(np.mean(results['coord_errors_coverage_L1_train'])),
            'coord_error_coverage_L1_train_median': float(np.median(results['coord_errors_coverage_L1_train'])),
            'coord_error_coverage_L1_train_std': float(np.std(results['coord_errors_coverage_L1_train'])),
            'coord_error_power_L1_manhattan_mean': float(np.mean(results['coord_errors_power_L1_manhattan'])),
            'coord_error_power_L1_manhattan_median': float(np.median(results['coord_errors_power_L1_manhattan'])),
            'coord_error_power_L1_manhattan_std': float(np.std(results['coord_errors_power_L1_manhattan'])),
            'coord_error_coverage_L1_manhattan_mean': float(np.mean(results['coord_errors_coverage_L1_manhattan'])),
            'coord_error_coverage_L1_manhattan_median': float(np.median(results['coord_errors_coverage_L1_manhattan'])),
            'coord_error_coverage_L1_manhattan_std': float(np.std(results['coord_errors_coverage_L1_manhattan'])),
            'coverage_pct_mean': float(np.mean(results['coverage_pcts'])),
            'coverage_pct_std': float(np.std(results['coverage_pcts'])),
            'coverage_pct_power_mean': float(np.mean(results['coverage_pct_powers'])),
            'coverage_pct_power_std': float(np.std(results['coverage_pct_powers'])),
            'pl_pct_cov_mean': float(np.mean(results['pl_pct_covs'])),
            'pl_pct_cov_std': float(np.std(results['pl_pct_covs'])),
            'pl_pct_ref_mean': float(np.mean(results['pl_pct_refs'])),
            'pl_pct_ref_std': float(np.std(results['pl_pct_refs'])),
            'model_rank_mean': float(np.mean(results['model_ranks'])),
            'model_rank_median': float(np.median(results['model_ranks'])),
            'model_rank_std': float(np.std(results['model_ranks'])),
            'model_time_mean': float(np.mean(results['model_times'])),
            'model_time_median': float(np.median(results['model_times'])),
            'model_time_std': float(np.std(results['model_times'])),
            'radionet_time_mean': float(np.mean(results['radionet_times'])),
            'radionet_time_median': float(np.median(results['radionet_times'])),
            'radionet_time_std': float(np.std(results['radionet_times'])),
            'total_time_per_building_mean': float(np.mean(results['total_times'])),
            'total_time_per_building_median': float(np.median(results['total_times'])),
            'total_time_per_building_std': float(np.std(results['total_times'])),
            'num_buildings': len(results['coord_errors_power']),
            'sample_count': n,
            'overall_time': overall_time,
            'buildings_per_second': len(results['coord_errors_power']) / overall_time,
            'buildings_per_hour': (len(results['coord_errors_power']) / overall_time) * 3600
        }
    
    return aggregated_by_n, results_by_n


def save_test_results(aggregated_by_n, raw_results_by_n, save_dir, model_info, hardware_info, optimize_for):
    """Save test results in standard format"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save multi-sample results
    json_path = os.path.join(save_dir, 'results_multisample.json')
    results_dict = {
        'model_info': model_info,
        'hardware_info': hardware_info,
        'results_by_sample_count': convert_to_json_serializable(aggregated_by_n)
    }
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"Saved multi-sample results: {json_path}")
    
    # Save comparison table
    txt_path = os.path.join(save_dir, 'results_comparison.txt')
    with open(txt_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("DIFFUSION AVERAGE MAP MODEL - MULTI-SAMPLE TEST RESULTS\n")
        f.write("="*100 + "\n\n")
        
        f.write("Model Information:\n")
        f.write(f"  Model type: Diffusion\n")
        f.write(f"  Conditioning: {model_info['conditioning']}\n")
        f.write(f"  Optimize for: {model_info['optimize_for']}\n")
        f.write(f"  Heatmap type: {model_info['heatmap_type']}\n")
        f.write(f"  Model path: {model_info['model_path']}\n\n")
        
        if hardware_info:
            f.write(f"Hardware: {hardware_info.get('gpu_name', 'CPU')}\n\n")
        
        f.write("="*100 + "\n")
        f.write("COMPARISON ACROSS SAMPLE COUNTS\n")
        f.write("="*100 + "\n\n")
        
        sample_counts = sorted(aggregated_by_n.keys())
        
        f.write(f"{'N':<6} | {'L2 Err':>8} | {'Cov%':>7} | {'PL%':>7} | {'Rank':>6} | {'Time(s)':>8}\n")
        f.write("-"*60 + "\n")
        
        for n in sample_counts:
            res = aggregated_by_n[n]
            err = res['coord_error_power_mean'] if optimize_for == 'power' else res['coord_error_coverage_mean']
            f.write(f"{n:<6} | {err:>8.2f} | {res['coverage_pct_mean']:>7.2f} | ")
            f.write(f"{res['pl_pct_ref_mean']:>7.2f} | {res['model_rank_mean']:>6.2f} | ")
            f.write(f"{res['total_time_per_building_mean']:>8.3f}\n")
    
    print(f"Saved comparison table: {txt_path}")
    
    # Save n=1 in standard format
    if 1 in aggregated_by_n:
        save_standard_results(aggregated_by_n[1], raw_results_by_n[1], 
                            save_dir, model_info, hardware_info)


def save_standard_results(agg_results, raw_results, save_dir, model_info, hardware_info):
    """Save n=1 results in exact format as evaluate_fixed_DUAL_GT_test_set.py"""
    
    json_path = os.path.join(save_dir, 'results.json')
    
    results_dict = {
        'coord_error_power_mean': agg_results['coord_error_power_mean'],
        'coord_error_power_median': agg_results['coord_error_power_median'],
        'coord_error_power_std': agg_results['coord_error_power_std'],
        'coord_error_coverage_mean': agg_results['coord_error_coverage_mean'],
        'coord_error_coverage_median': agg_results['coord_error_coverage_median'],
        'coord_error_coverage_std': agg_results['coord_error_coverage_std'],
        'coord_error_power_L1_train_mean': agg_results['coord_error_power_L1_train_mean'],
        'coord_error_power_L1_train_median': agg_results['coord_error_power_L1_train_median'],
        'coord_error_power_L1_train_std': agg_results['coord_error_power_L1_train_std'],
        'coord_error_coverage_L1_train_mean': agg_results['coord_error_coverage_L1_train_mean'],
        'coord_error_coverage_L1_train_median': agg_results['coord_error_coverage_L1_train_median'],
        'coord_error_coverage_L1_train_std': agg_results['coord_error_coverage_L1_train_std'],
        'coord_error_power_L1_manhattan_mean': agg_results['coord_error_power_L1_manhattan_mean'],
        'coord_error_power_L1_manhattan_median': agg_results['coord_error_power_L1_manhattan_median'],
        'coord_error_power_L1_manhattan_std': agg_results['coord_error_power_L1_manhattan_std'],
        'coord_error_coverage_L1_manhattan_mean': agg_results['coord_error_coverage_L1_manhattan_mean'],
        'coord_error_coverage_L1_manhattan_median': agg_results['coord_error_coverage_L1_manhattan_median'],
        'coord_error_coverage_L1_manhattan_std': agg_results['coord_error_coverage_L1_manhattan_std'],
        'coverage_pct_mean': agg_results['coverage_pct_mean'],
        'coverage_pct_std': agg_results['coverage_pct_std'],
        'coverage_pct_power_mean': agg_results['coverage_pct_power_mean'],
        'coverage_pct_power_std': agg_results['coverage_pct_power_std'],
        'pl_pct_cov_mean': agg_results['pl_pct_cov_mean'],
        'pl_pct_cov_std': agg_results['pl_pct_cov_std'],
        'pl_pct_ref_mean': agg_results['pl_pct_ref_mean'],
        'pl_pct_ref_std': agg_results['pl_pct_ref_std'],
        'model_time_mean': agg_results['model_time_mean'],
        'model_time_median': agg_results['model_time_median'],
        'model_time_std': agg_results['model_time_std'],
        'radionet_time_mean': agg_results['radionet_time_mean'],
        'radionet_time_median': agg_results['radionet_time_median'],
        'radionet_time_std': agg_results['radionet_time_std'],
        'total_time_per_building_mean': agg_results['total_time_per_building_mean'],
        'total_time_per_building_median': agg_results['total_time_per_building_median'],
        'total_time_per_building_std': agg_results['total_time_per_building_std'],
        'overall_time': agg_results['overall_time'],
        'buildings_per_second': agg_results['buildings_per_second'],
        'buildings_per_hour': agg_results['buildings_per_hour'],
        'num_buildings': agg_results['num_buildings'],
        'building_ids': [int(bid) for bid in raw_results['building_ids']],
        'pred_coords': [coord.tolist() if isinstance(coord, np.ndarray) else coord 
                       for coord in raw_results['pred_coords']],
        'coord_errors_power': [float(e) for e in raw_results['coord_errors_power']],
        'coord_errors_coverage': [float(e) for e in raw_results['coord_errors_coverage']],
        'coverage_pcts': [float(c) for c in raw_results['coverage_pcts']],
        'coverage_pct_powers': [float(c) for c in raw_results['coverage_pct_powers']],
        'pl_pct_covs': [float(p) for p in raw_results['pl_pct_covs']],
        'pl_pct_refs': [float(p) for p in raw_results['pl_pct_refs']],
        'model_info': model_info,
        'hardware_info': hardware_info
    }
    
    with open(json_path, 'w') as f:
        json.dump(convert_to_json_serializable(results_dict), f, indent=2)
    print(f"Saved standard results (n=1): {json_path}")


def run_test_evaluation(args, train_dir, device, scheduler_diffusion):
    """Run test evaluation on best model after training"""
    
    print(f"\n{'='*80}")
    print("STARTING TEST SET EVALUATION")
    print(f"{'='*80}\n")
    
    hardware_info = get_hardware_info()
    
    # Load RadioNet
    print(f"Loading RadioNet: {args.radionet_path}")
    radionet_model = RadioNet(inputs=2).to(device)
    radionet_model.load_state_dict(torch.load(args.radionet_path, map_location=device))
    radionet_model.eval()
    print("✓ RadioNet loaded\n")
    
    # Get splits
    _, _, test_ids = get_building_splits(args.buildings_dir, train_ratio=0.8, val_ratio=0.1, seed=42)
    
    # Create test datasets
    print("Creating test datasets...")
    
    test_dataset = TxLocationDatasetAvgMap(
        test_ids, args.buildings_dir, None, args.optimize_for,
        args.heatmap_type, args.heatmap_base_dir, False, args.use_center
    )
    
    power_dataset = TxLocationDatasetAvgMap(
        test_ids, args.buildings_dir, None, 'power',
        args.heatmap_type, args.heatmap_base_dir, False, args.use_center
    )
    
    coverage_dataset = TxLocationDatasetAvgMap(
        test_ids, args.buildings_dir, None, 'coverage',
        args.heatmap_type, args.heatmap_base_dir, False, args.use_center
    )
    
    print(f"✓ Test datasets created ({len(test_dataset)} buildings)\n")
    
    # Evaluate primary model
    checkpoint_name = 'best_pl_model.pth' if args.optimize_for == 'power' else 'best_coverage_model.pth'
    model_path = os.path.join(train_dir, checkpoint_name)
    
    if not os.path.exists(model_path):
        print(f"⚠ Model not found: {model_path}\n")
        return
    
    print(f"{'='*80}")
    print(f"Evaluating PRIMARY model: {checkpoint_name}")
    print(f"{'='*80}\n")
    
    # Load model
    img_size = 150 if args.use_center else 256
    model = create_diffusion_model(img_size=img_size, conditioning=args.conditioning).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Model loaded\n")
    
    # Run evaluation
    agg_results, raw_results = evaluate_test_set_multisample_diffusion(
        model, test_dataset, power_dataset, coverage_dataset,
        args.optimize_for, args.use_center, args.buildings_dir,
        radionet_model, args.heatmap_base_dir, device, scheduler_diffusion,
        num_inference_steps=args.num_inference_steps
    )
    
    # Save results
    checkpoint_suffix = 'best_pl' if args.optimize_for == 'power' else 'best_coverage'
    eval_dir_name = f"diffusion_{args.conditioning}_{args.optimize_for}_{args.heatmap_type}_{checkpoint_suffix}"
    eval_save_dir = os.path.join(args.test_results_dir, eval_dir_name)
    
    model_info = {
        'model_type': 'diffusion',
        'conditioning': args.conditioning,
        'optimize_for': args.optimize_for,
        'heatmap_type': args.heatmap_type,
        'model_path': model_path,
        'checkpoint_type': 'primary',
        'num_inference_steps': args.num_inference_steps
    }
    
    save_test_results(agg_results, raw_results, eval_save_dir, 
                     model_info, hardware_info, args.optimize_for)
    
    print(f"\n✓ Results saved to: {eval_save_dir}\n")
    
    print(f"\n{'='*80}")
    print("TEST EVALUATION COMPLETED")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Train Diffusion Models on Average Maps')
    
    # Model
    parser.add_argument('--conditioning', type=str, default='concat',
                       choices=['concat', 'film', 'cross_attn'],
                       help='Conditioning method (use concat for best results)')
    parser.add_argument('--optimize_for', type=str, required=True,
                       choices=['power', 'coverage'])
    parser.add_argument('--heatmap_type', type=str, required=True)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Diffusion
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--noise_schedule', type=str, default='linear')
    parser.add_argument('--num_inference_steps', type=int, default=50)
    
    # Paths
    parser.add_argument('--buildings_dir', type=str, default='./data/buildings')
    parser.add_argument('--heatmap_base_dir', type=str, default='./data/unified_results')
    parser.add_argument('--radionet_path', type=str, default='./checkpoints/saipp_net.pt')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--test_results_dir', type=str, default='evaluation_avgmap_test_results')
    
    # Options
    parser.add_argument('--use_center', action='store_true', default=True)
    parser.add_argument('--augment', action='store_true')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.test_results_dir, exist_ok=True)
    
    # Get splits
    train_ids, val_ids, test_ids = get_building_splits(args.buildings_dir)
    
    # Create datasets
    print(f"\n{'='*80}")
    print("CREATING DATASETS")
    print(f"{'='*80}\n")
    
    train_dataset = TxLocationDatasetAvgMap(
        train_ids, args.buildings_dir, None, args.optimize_for,
        args.heatmap_type, args.heatmap_base_dir, args.augment, args.use_center
    )
    
    val_dataset = TxLocationDatasetAvgMap(
        val_ids, args.buildings_dir, None, args.optimize_for,
        args.heatmap_type, args.heatmap_base_dir, False, args.use_center
    )
    
    print(f"\n✓ Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_ids)}\n")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # Model
    print(f"{'='*80}")
    print("CREATING DIFFUSION MODEL")
    print(f"{'='*80}\n")
    
    img_size = 150 if args.use_center else 256
    model = create_diffusion_model(img_size=img_size, conditioning=args.conditioning).to(device)
    print(f"✓ Diffusion model ({args.conditioning} conditioning, {img_size}x{img_size})\n")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    # Diffusion scheduler
    scheduler_diffusion = DDPMScheduler(num_train_timesteps=args.timesteps)
    
    # RadioNet
    print(f"{'='*80}")
    print("LOADING RADIONET")
    print(f"{'='*80}\n")
    
    radionet_model = RadioNet(inputs=2).to(device)
    radionet_model.load_state_dict(torch.load(args.radionet_path, map_location=device))
    radionet_model.eval()
    print(f"✓ RadioNet loaded\n")
    
    # TRAIN
    train_diffusion_model(
        model, train_loader, val_loader, optimizer, scheduler, scheduler_diffusion,
        args.num_epochs, device, args.save_dir, args.optimize_for, args.use_center,
        args.buildings_dir, args.heatmap_base_dir, radionet_model,
        num_inference_steps=args.num_inference_steps
    )
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETED")
    print(f"{'='*80}\n")
    
    # RUN TEST EVALUATION
    run_test_evaluation(args, args.save_dir, device, scheduler_diffusion)
    
    print(f"\n{'='*80}")
    print("ALL DONE!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
