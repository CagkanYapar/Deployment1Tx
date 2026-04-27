"""
AvgMap Discriminative Model Evaluation - Multi-Sample with Dual GT

Evaluates discriminative models trained on average maps (avg_coverage_per_map, avg_power_per_map)
with multi-sample candidate extraction and dual GT comparison.

Key features:
- Extracts top-N candidates from predicted average map
- Evaluates all candidates with RadioNet
- Selects best by coverage AND best by power
- Computes metrics against BOTH power-optimal and coverage-optimal GTs
- Supports all normalization variants (with_denorm, nonorm, norm11, norm11_nonorm)
"""

import os
import sys
import argparse
import numpy as np
import torch
import json
import time
from tqdm import tqdm
from PIL import Image

# Import model and dataset
from models.discriminative import create_model_deep
from data.dataset_heatmap import TxLocationDatasetLarge, get_building_splits

# Import RadioNet
sys.path.append('.')
from models.saipp_net import RadioNet

# Import shared utilities
from evaluation.eval_utils import (
    extract_topn_candidates,
    evaluate_candidates_batch_radionet,
    select_best_candidate,
    compute_dual_gt_metrics,
    aggregate_results,
    convert_to_json_serializable
)


def get_hardware_info():
    """Capture hardware information for timing context."""
    import subprocess
    
    hardware_info = {
        'device_type': 'cpu',
        'gpu_name': None,
        'gpu_memory_gb': None,
    }
    
    try:
        if torch.cuda.is_available():
            hardware_info['device_type'] = 'cuda'
            hardware_info['gpu_name'] = torch.cuda.get_device_name(0)
            hardware_info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
    except:
        pass
    
    return hardware_info


def evaluate_avgmap_discriminative_multisample(
    model, test_dataset, power_dataset, coverage_dataset,
    device, optimize_for, use_center, buildings_dir, radionet_model, heatmap_base_dir,
    n_values=[1, 5, 10, 20, 50, 100]
):
    """
    Evaluate discriminative avgmap model with multi-sample support.
    
    For each N in n_values:
        1. Extract top-N candidates from predicted average map
        2. Evaluate all N with RadioNet (batched)
        3. Select best-coverage and best-power
        4. Compute dual GT metrics for each strategy
    
    Args:
        model: Trained discriminative model
        test_dataset: Test dataset (based on optimize_for)
        power_dataset: Test dataset with power-optimal GTs
        coverage_dataset: Test dataset with coverage-optimal GTs
        device: torch device
        optimize_for: 'power' or 'coverage'
        use_center: Whether using center crop
        buildings_dir: Directory with building images
        radionet_model: RadioNet model
        heatmap_base_dir: Base directory for heatmaps
        n_values: List of N values to evaluate
    
    Returns:
        dict: Results for each N value with 3 strategies each
    """
    model.eval()
    radionet_model.eval()
    
    max_n = max(n_values)
    
    # Storage for each N
    results_by_n = {n: {
        'Single': [],
        'Best-Coverage': [],
        'Best-Power': []
    } for n in n_values}
    
    # Timing storage
    timing_data = {
        'model_times': [],
        'radionet_times': [],
        'total_times': []
    }
    
    print(f"\n{'='*80}")
    print(f"AVGMAP DISCRIMINATIVE EVALUATION - MULTI-SAMPLE")
    print(f"{'='*80}")
    print(f"N values: {n_values}")
    print(f"Optimize for: {optimize_for}")
    print(f"Test buildings: {len(test_dataset)}")
    print(f"{'='*80}\n")
    
    overall_start = time.time()
    
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="Evaluating", mininterval=2.0):
            building_start = time.time()
            
            # Get data
            building_map, gt_coords, target_heatmap, building_id = test_dataset[idx]
            _, power_gt, _, _ = power_dataset[idx]
            _, coverage_gt, _, _ = coverage_dataset[idx]
            
            # Model forward pass - SINGLE PREDICTION
            building_map_input = building_map.unsqueeze(0).to(device)
            
            model_start = time.time()
            logits, pred_coords_single = model(building_map_input)
            model_time = time.time() - model_start
            
            # Extract predicted average map
            pred_avgmap = logits[0, 0].cpu().numpy()
            
            # Extract free space mask — TX candidates must be in free space
            # Training convention: free space = building_center <= 0.1
            building_np = building_map.squeeze().cpu().numpy()
            free_space_mask = (building_np <= 0.1)
            
            # Extract top-max_n candidates from free space
            candidates = extract_topn_candidates(pred_avgmap, free_space_mask, n=max_n, use_center=use_center)
            
            # Evaluate all candidates with RadioNet
            radionet_start = time.time()
            candidates_metrics = evaluate_candidates_batch_radionet(
                candidates, building_id, buildings_dir, radionet_model,
                use_center, device, batch_size=64
            )
            radionet_time = time.time() - radionet_start
            
            building_total_time = time.time() - building_start
            
            # Store timing
            timing_data['model_times'].append(model_time)
            timing_data['radionet_times'].append(radionet_time)
            timing_data['total_times'].append(building_total_time)
            
            # Convert GTs to numpy
            power_gt_np = power_gt.cpu().numpy()
            coverage_gt_np = coverage_gt.cpu().numpy()
            
            # For each N value
            for n in n_values:
                # Get top-N candidates
                candidates_n = candidates[:n]
                metrics_n = candidates_metrics[:n]
                
                if len(metrics_n) == 0:
                    continue
                
                # Strategy 1: Single (N=1 uses first, N>1 uses model's top-1)
                single_metrics = metrics_n[0]
                single_dual_gt = compute_dual_gt_metrics(
                    single_metrics['coord'], power_gt_np, coverage_gt_np, single_metrics,
                    building_id, heatmap_base_dir, optimize_for, use_center, buildings_dir
                )
                
                if single_dual_gt:
                    single_dual_gt['building_id'] = building_id
                    results_by_n[n]['Single'].append(single_dual_gt)
                
                # Strategy 2: Best-Coverage (pick best by coverage from top-N)
                best_cov_idx, best_cov_metrics = select_best_candidate(metrics_n, criterion='coverage')
                best_cov_dual_gt = compute_dual_gt_metrics(
                    best_cov_metrics['coord'], power_gt_np, coverage_gt_np, best_cov_metrics,
                    building_id, heatmap_base_dir, optimize_for, use_center, buildings_dir
                )
                
                if best_cov_dual_gt:
                    best_cov_dual_gt['building_id'] = building_id
                    best_cov_dual_gt['model_rank'] = best_cov_idx + 1  # 1-indexed
                    results_by_n[n]['Best-Coverage'].append(best_cov_dual_gt)
                
                # Strategy 3: Best-Power (pick best by PL from top-N)
                best_pow_idx, best_pow_metrics = select_best_candidate(metrics_n, criterion='power')
                best_pow_dual_gt = compute_dual_gt_metrics(
                    best_pow_metrics['coord'], power_gt_np, coverage_gt_np, best_pow_metrics,
                    building_id, heatmap_base_dir, optimize_for, use_center, buildings_dir
                )
                
                if best_pow_dual_gt:
                    best_pow_dual_gt['building_id'] = building_id
                    best_pow_dual_gt['model_rank'] = best_pow_idx + 1  # 1-indexed
                    results_by_n[n]['Best-Power'].append(best_pow_dual_gt)
    
    overall_time = time.time() - overall_start
    
    # Aggregate results for each N and strategy
    final_results = {}
    
    for n in n_values:
        final_results[n] = {}
        
        for strategy in ['Single', 'Best-Coverage', 'Best-Power']:
            strategy_results = results_by_n[n][strategy]
            
            if len(strategy_results) > 0:
                aggregated = aggregate_results(strategy_results)
                
                # Add timing info (averaged over all buildings)
                aggregated['model_time_mean'] = float(np.mean(timing_data['model_times']))
                aggregated['radionet_time_mean'] = float(np.mean(timing_data['radionet_times']))
                aggregated['total_time_mean'] = float(np.mean(timing_data['total_times']))
                aggregated['total_evaluation_time'] = overall_time
                
                final_results[n][strategy] = aggregated
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate avgmap discriminative model on test set')
    
    # Model configuration
    parser.add_argument('--arch', type=str, required=True,
                       choices=['deepxl_150', 'pmnet_150', 'sip2net_150', 'dcnet_150'],
                       help='Model architecture')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--coord_method', type=str, default='hard_argmax',
                       help='Coordinate extraction method')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for coordinate extraction')
    
    # Dataset configuration
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--optimize_for', type=str, required=True,
                       choices=['power', 'coverage'],
                       help='Optimization target')
    parser.add_argument('--heatmap_type', type=str, required=True,
                       help='Heatmap type (e.g., avg_coverage_per_map, avg_power_per_map, etc.)')
    parser.add_argument('--use_center', action='store_true', default=True,
                       help='Use center 150x150 crop')
    
    # Paths
    parser.add_argument('--buildings_dir', type=str,
                       default='./data/buildings',
                       help='Path to building images')
    parser.add_argument('--heatmap_base_dir', type=str,
                       default='./data/unified_results',
                       help='Base directory for heatmaps')
    parser.add_argument('--radionet_checkpoint', type=str,
                       default='./checkpoints/saipp_net.pt',
                       help='Path to RadioNet checkpoint')
    
    # Output
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save results')
    
    # Multi-sample configuration
    parser.add_argument('--n_values', type=int, nargs='+', default=[1, 5, 10, 20, 50, 100],
                       help='N values for multi-sample evaluation')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Get hardware info
    hardware_info = get_hardware_info()
    print(f"Hardware: {hardware_info}")
    
    # Auto-set tx_dir if not provided (matching training behavior)
    if args.optimize_for == 'power':
        tx_dir = os.path.join(args.heatmap_base_dir, 'best_by_power')
    else:
        tx_dir = os.path.join(args.heatmap_base_dir, 'best_by_coverage')
    print(f"Tx directory: {tx_dir}")
    
    # Load building splits
    print(f"\nLoading {args.split} dataset split...")
    train_buildings, val_buildings, test_buildings = get_building_splits(args.buildings_dir)
    
    # Select appropriate split
    if args.split == 'train':
        eval_buildings = train_buildings
    elif args.split == 'val':
        eval_buildings = val_buildings
    else:  # test
        eval_buildings = test_buildings
    
    print(f"{args.split.capitalize()} set: {len(eval_buildings)} buildings")
    
    # Create datasets for BOTH GTs
    img_size = 150 if args.use_center else 256
    
    # Main dataset (based on optimize_for)
    print(f"\nCreating test dataset (optimize_for={args.optimize_for}, heatmap_type={args.heatmap_type})")
    test_dataset = TxLocationDatasetLarge(
        eval_buildings,
        buildings_dir=args.buildings_dir,
        tx_dir=tx_dir,
        optimize_for=args.optimize_for,
        heatmap_type=args.heatmap_type,
        heatmap_base_dir=args.heatmap_base_dir,
        normalize_heatmap=False,
        augment=False,
        use_center=args.use_center
    )
    
    # Power-optimal GT dataset
    print("Creating power-optimal GT dataset")
    power_tx_dir = os.path.join(args.heatmap_base_dir, 'best_by_power')
    power_dataset = TxLocationDatasetLarge(
        eval_buildings,
        buildings_dir=args.buildings_dir,
        tx_dir=power_tx_dir,
        optimize_for='power',
        heatmap_type=args.heatmap_type,
        heatmap_base_dir=args.heatmap_base_dir,
        normalize_heatmap=False,
        augment=False,
        use_center=args.use_center
    )
    
    # Coverage-optimal GT dataset
    print("Creating coverage-optimal GT dataset")
    coverage_tx_dir = os.path.join(args.heatmap_base_dir, 'best_by_coverage')
    coverage_dataset = TxLocationDatasetLarge(
        eval_buildings,
        buildings_dir=args.buildings_dir,
        tx_dir=coverage_tx_dir,
        optimize_for='coverage',
        heatmap_type=args.heatmap_type,
        heatmap_base_dir=args.heatmap_base_dir,
        normalize_heatmap=False,
        augment=False,
        use_center=args.use_center
    )
    
    # Create model
    print(f"\nCreating model: {args.arch}")
    model = create_model_deep(
        arch=args.arch,
        coord_method=args.coord_method,
        temperature=args.temperature,
        use_masking=False,
        img_size=img_size
    ).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Load RadioNet
    print(f"\nLoading RadioNet from {args.radionet_checkpoint}")
    radionet_model = RadioNet(inputs=2).to(device)
    radionet_checkpoint = torch.load(args.radionet_checkpoint, map_location=device)
    
    # Robust loading - try multiple possible keys
    if isinstance(radionet_checkpoint, dict):
        if 'model_state_dict' in radionet_checkpoint:
            radionet_model.load_state_dict(radionet_checkpoint['model_state_dict'])
        elif 'state_dict' in radionet_checkpoint:
            radionet_model.load_state_dict(radionet_checkpoint['state_dict'])
        elif 'model' in radionet_checkpoint:
            radionet_model.load_state_dict(radionet_checkpoint['model'])
        else:
            radionet_model.load_state_dict(radionet_checkpoint)
    else:
        radionet_model.load_state_dict(radionet_checkpoint)
    
    radionet_model.eval()
    print("[RadioNet] Loaded successfully")
    
    # Run evaluation
    print(f"\n{'='*80}")
    print("STARTING EVALUATION")
    print(f"{'='*80}")
    
    results = evaluate_avgmap_discriminative_multisample(
        model=model,
        test_dataset=test_dataset,
        power_dataset=power_dataset,
        coverage_dataset=coverage_dataset,
        device=device,
        optimize_for=args.optimize_for,
        use_center=args.use_center,
        buildings_dir=args.buildings_dir,
        radionet_model=radionet_model,
        heatmap_base_dir=args.heatmap_base_dir,
        n_values=args.n_values
    )
    
    # Prepare output
    output = {
        'model_info': {
            'architecture': args.arch,
            'model_path': args.model_path,
            'optimize_for': args.optimize_for,
            'heatmap_type': args.heatmap_type,
            'coord_method': args.coord_method,
            'temperature': args.temperature,
            'use_center': args.use_center
        },
        'dataset_info': {
            'split': args.split,
            'num_buildings': len(eval_buildings),
            'buildings_dir': args.buildings_dir,
            'heatmap_base_dir': args.heatmap_base_dir
        },
        'hardware_info': hardware_info,
        'n_values': args.n_values,
        'results': results
    }
    
    # Convert to JSON-serializable
    output = convert_to_json_serializable(output)
    
    # Save results
    output_path = os.path.join(args.output_dir, 'results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_path}")
    
    # Print summary
    print(f"\nSummary:")
    for n in args.n_values:
        if n in results:
            print(f"\nN={n}:")
            for strategy in ['Single', 'Best-Coverage', 'Best-Power']:
                if strategy in results[n]:
                    r = results[n][strategy]
                    print(f"  {strategy}:")
                    print(f"    Coord Error (power GT): {r.get('coord_error_power_mean', 0):.2f}px")
                    print(f"    Coord Error (coverage GT): {r.get('coord_error_coverage_mean', 0):.2f}px")
                    if 'coverage_pct_ref_mean' in r:
                        print(f"    Coverage %: {r['coverage_pct_ref_mean']:.2f}%")
                    if 'pl_pct_ref_mean' in r:
                        print(f"    PL %: {r['pl_pct_ref_mean']:.2f}%")


if __name__ == '__main__':
    main()
