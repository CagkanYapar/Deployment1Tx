"""
AvgMap Multi-Objective Evaluation

Evaluates models trained on power vs coverage objectives together.
Two-phase selection:
  Phase 1: Rank all free space pixels by both maps, select top-K by minimax
  Phase 2: Evaluate K candidates with RadioNet, select winner by L2 distance from ideal

Key differences from single-objective:
- Loads TWO models (one power-trained, one coverage-trained)
- Generates TWO average maps per building
- Intersection-based candidate selection
"""

import os
import sys
import argparse
import numpy as np
import torch
import json
import time
from tqdm import tqdm

# Import model and dataset
from models.discriminative import create_model_deep
from data.dataset_heatmap import TxLocationDatasetLarge, get_building_splits

# Import RadioNet
sys.path.append('.')
from models.saipp_net import RadioNet

# Import shared utilities
from evaluation.eval_utils import (
    evaluate_candidates_batch_radionet,
    compute_dual_gt_metrics,
    aggregate_results,
    convert_to_json_serializable
)


def get_hardware_info():
    """Capture hardware information for timing context."""
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


def rank_all_free_pixels(power_avgmap, coverage_avgmap, free_space_mask):
    """
    Rank all free space pixels by both power and coverage average maps.
    
    Args:
        power_avgmap: Power average map (H, W)
        coverage_avgmap: Coverage average map (H, W)
        free_space_mask: Boolean mask of free space pixels (H, W)
    
    Returns:
        List of dicts with {'coord': (y, x), 'rank_power': int, 'rank_coverage': int}
    """
    # Get coordinates and values of all free space pixels
    free_coords = np.argwhere(free_space_mask)  # (N, 2) array
    power_values = power_avgmap[free_space_mask]  # (N,) array
    coverage_values = coverage_avgmap[free_space_mask]  # (N,) array
    
    # Rank by descending value (higher is better)
    # argsort gives indices that would sort, argsort again gives ranks
    power_ranks = np.argsort(-power_values).argsort() + 1  # 1-indexed
    coverage_ranks = np.argsort(-coverage_values).argsort() + 1
    
    # Build list with ranks
    ranked_pixels = [
        {
            'coord': (float(free_coords[i, 0]), float(free_coords[i, 1])),
            'rank_power': int(power_ranks[i]),
            'rank_coverage': int(coverage_ranks[i])
        }
        for i in range(len(free_coords))
    ]
    
    return ranked_pixels


def select_topk_by_minimax(ranked_pixels, K):
    """
    Select top-K pixels by minimax criterion.
    Score = -max(rank_power, rank_coverage) — minimize worst-case rank.
    
    Args:
        ranked_pixels: List of dicts with rank_power and rank_coverage
        K: Number of candidates to select
    
    Returns:
        List of K dicts (subset of input)
    """
    # Compute minimax scores
    for pixel in ranked_pixels:
        pixel['minimax_score'] = -max(pixel['rank_power'], pixel['rank_coverage'])
    
    # Sort by score (descending) and take top-K
    sorted_pixels = sorted(ranked_pixels, key=lambda p: p['minimax_score'], reverse=True)
    topk = sorted_pixels[:K]
    
    return topk


def select_winner_by_l2(candidates_metrics):
    """
    Select best candidate by L2 distance from ideal point (100%, 100%).
    Score = -sqrt((100 - PL_pct)² + (100 - Cov_pct)²)
    
    Args:
        candidates_metrics: List of dicts with dual-GT metrics
    
    Returns:
        (best_index, best_metrics)
    """
    l2_scores = [
        -np.sqrt((100 - m['pl_pct_ref'])**2 + (100 - m['coverage_pct_ref'])**2)
        for m in candidates_metrics
    ]
    
    best_idx = int(np.argmax(l2_scores))
    best_metrics = candidates_metrics[best_idx].copy()
    best_metrics['l2_score'] = float(l2_scores[best_idx])
    
    return best_idx, best_metrics


def evaluate_multiobjective(
    model_power, model_coverage, test_dataset, power_dataset, coverage_dataset,
    device, use_center, buildings_dir, radionet_model, heatmap_base_dir, K=200
):
    """
    Evaluate multi-objective selection on test set.
    
    Args:
        model_power: Model trained on power objective
        model_coverage: Model trained on coverage objective
        test_dataset: Test dataset
        power_dataset: Dataset with power-optimal GTs
        coverage_dataset: Dataset with coverage-optimal GTs
        device: torch device
        use_center: Whether using center crop
        buildings_dir: Building images directory
        radionet_model: RadioNet model
        heatmap_base_dir: Base directory for heatmaps
        K: Number of candidates to evaluate with RadioNet
    
    Returns:
        dict with results
    """
    model_power.eval()
    model_coverage.eval()
    radionet_model.eval()
    
    results = []
    
    # Timing and statistics
    timing_data = {
        'model_times': [],
        'radionet_times': [],
        'total_times': []
    }
    
    intersection_stats = []
    
    print(f"\n{'='*80}")
    print(f"MULTI-OBJECTIVE AVGMAP EVALUATION")
    print(f"{'='*80}")
    print(f"K (candidates to evaluate): {K}")
    print(f"Test buildings: {len(test_dataset)}")
    print(f"Selection: Phase 1 = Minimax, Phase 2 = L2 distance")
    print(f"{'='*80}\n")
    
    overall_start = time.time()
    
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="Evaluating", mininterval=2.0):
            building_start = time.time()
            
            # Get data
            building_map, _, _, building_id = test_dataset[idx]
            _, power_gt, _, _ = power_dataset[idx]
            _, coverage_gt, _, _ = coverage_dataset[idx]
            
            # Model forward passes
            building_map_input = building_map.unsqueeze(0).to(device)
            
            model_start = time.time()
            
            # Generate power average map
            logits_power, _ = model_power(building_map_input)
            power_avgmap = logits_power[0, 0].cpu().numpy()
            
            # Generate coverage average map
            logits_coverage, _ = model_coverage(building_map_input)
            coverage_avgmap = logits_coverage[0, 0].cpu().numpy()
            
            model_time = time.time() - model_start
            
            # Get free space mask
            building_np = building_map.squeeze().cpu().numpy()
            free_space_mask = (building_np <= 0.1)
            
            # Rank all free space pixels by both maps
            ranked_pixels = rank_all_free_pixels(power_avgmap, coverage_avgmap, free_space_mask)
            
            intersection_stats.append(len(ranked_pixels))
            
            if len(ranked_pixels) == 0:
                print(f"Warning: Building {building_id} has no free space pixels, skipping")
                continue
            
            # Select top-K by minimax
            K_actual = min(K, len(ranked_pixels))
            topk = select_topk_by_minimax(ranked_pixels, K_actual)
            
            # Extract coordinates for RadioNet evaluation
            topk_coords = [pixel['coord'] for pixel in topk]
            
            # Evaluate with RadioNet
            radionet_start = time.time()
            candidates_metrics = evaluate_candidates_batch_radionet(
                topk_coords, building_id, buildings_dir, radionet_model,
                use_center, device, batch_size=64
            )
            radionet_time = time.time() - radionet_start
            
            building_total_time = time.time() - building_start
            
            # Store timing
            timing_data['model_times'].append(model_time)
            timing_data['radionet_times'].append(radionet_time)
            timing_data['total_times'].append(building_total_time)
            
            if len(candidates_metrics) == 0:
                print(f"Warning: Building {building_id} RadioNet evaluation failed, skipping")
                continue
            
            # Compute dual-GT metrics for each candidate
            power_gt_np = power_gt.cpu().numpy()
            coverage_gt_np = coverage_gt.cpu().numpy()
            
            candidates_with_metrics = []
            for metrics in candidates_metrics:
                dual_gt = compute_dual_gt_metrics(
                    metrics['coord'], power_gt_np, coverage_gt_np, metrics,
                    building_id, heatmap_base_dir, 'power',  # optimize_for doesn't matter for dual-GT
                    use_center, buildings_dir
                )
                
                if dual_gt is not None:
                    candidates_with_metrics.append(dual_gt)
            
            if len(candidates_with_metrics) == 0:
                print(f"Warning: Building {building_id} dual-GT computation failed, skipping")
                continue
            
            # Select winner by L2 distance from ideal
            best_idx, best_metrics = select_winner_by_l2(candidates_with_metrics)
            best_metrics['building_id'] = building_id
            
            results.append(best_metrics)
    
    overall_time = time.time() - overall_start
    
    # Aggregate results
    if len(results) == 0:
        print("\nERROR: No valid results collected!")
        return {}
    
    final_results = aggregate_results(results)
    
    # Add timing and statistics
    final_results['model_time_mean'] = float(np.mean(timing_data['model_times']))
    final_results['radionet_time_mean'] = float(np.mean(timing_data['radionet_times']))
    final_results['total_time_mean'] = float(np.mean(timing_data['total_times']))
    final_results['total_evaluation_time'] = overall_time
    
    final_results['free_pixels_mean'] = float(np.mean(intersection_stats))
    final_results['free_pixels_std'] = float(np.std(intersection_stats))
    final_results['free_pixels_min'] = float(np.min(intersection_stats))
    final_results['free_pixels_max'] = float(np.max(intersection_stats))
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description='Multi-objective avgmap evaluation')
    
    # Model configuration
    parser.add_argument('--arch', type=str, required=True,
                       choices=['deepxl_150', 'pmnet_150', 'sip2net_150', 'dcnet_150'],
                       help='Model architecture')
    parser.add_argument('--power_model_path', type=str, required=True,
                       help='Path to power-trained model checkpoint')
    parser.add_argument('--coverage_model_path', type=str, required=True,
                       help='Path to coverage-trained model checkpoint')
    parser.add_argument('--coord_method', type=str, default='hard_argmax',
                       help='Coordinate extraction method')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for coordinate extraction')
    
    # Dataset configuration
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--heatmap_type_power', type=str, default='avg_power_per_map',
                       help='Heatmap type for power models')
    parser.add_argument('--heatmap_type_coverage', type=str, default='avg_coverage_per_map',
                       help='Heatmap type for coverage models')
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
    
    # Multi-objective configuration
    parser.add_argument('--K', type=int, default=200,
                       help='Number of candidates to evaluate with RadioNet')
    
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
    
    # Create datasets
    img_size = 150 if args.use_center else 256
    
    # Test dataset (doesn't matter which optimize_for, we just need building maps)
    print(f"\nCreating test dataset...")
    test_dataset = TxLocationDatasetLarge(
        eval_buildings,
        buildings_dir=args.buildings_dir,
        tx_dir=os.path.join(args.heatmap_base_dir, 'best_by_power'),
        optimize_for='power',
        heatmap_type=args.heatmap_type_power,
        heatmap_base_dir=args.heatmap_base_dir,
        normalize_heatmap=False,
        augment=False,
        use_center=args.use_center
    )
    
    # Power-optimal GT dataset
    print("Creating power-optimal GT dataset...")
    power_dataset = TxLocationDatasetLarge(
        eval_buildings,
        buildings_dir=args.buildings_dir,
        tx_dir=os.path.join(args.heatmap_base_dir, 'best_by_power'),
        optimize_for='power',
        heatmap_type=args.heatmap_type_power,
        heatmap_base_dir=args.heatmap_base_dir,
        normalize_heatmap=False,
        augment=False,
        use_center=args.use_center
    )
    
    # Coverage-optimal GT dataset
    print("Creating coverage-optimal GT dataset...")
    coverage_dataset = TxLocationDatasetLarge(
        eval_buildings,
        buildings_dir=args.buildings_dir,
        tx_dir=os.path.join(args.heatmap_base_dir, 'best_by_coverage'),
        optimize_for='coverage',
        heatmap_type=args.heatmap_type_coverage,
        heatmap_base_dir=args.heatmap_base_dir,
        normalize_heatmap=False,
        augment=False,
        use_center=args.use_center
    )
    
    # Create power model
    print(f"\nCreating power model: {args.arch}")
    model_power = create_model_deep(
        arch=args.arch,
        coord_method=args.coord_method,
        temperature=args.temperature,
        use_masking=False,
        img_size=img_size
    ).to(device)
    
    # Load power checkpoint
    print(f"Loading power model from {args.power_model_path}")
    checkpoint = torch.load(args.power_model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model_power.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model_power.load_state_dict(checkpoint)
    
    model_power.eval()
    
    # Create coverage model
    print(f"\nCreating coverage model: {args.arch}")
    model_coverage = create_model_deep(
        arch=args.arch,
        coord_method=args.coord_method,
        temperature=args.temperature,
        use_masking=False,
        img_size=img_size
    ).to(device)
    
    # Load coverage checkpoint
    print(f"Loading coverage model from {args.coverage_model_path}")
    checkpoint = torch.load(args.coverage_model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model_coverage.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model_coverage.load_state_dict(checkpoint)
    
    model_coverage.eval()
    
    # Load RadioNet
    print(f"\nLoading RadioNet from {args.radionet_checkpoint}")
    radionet_model = RadioNet(inputs=2).to(device)
    radionet_checkpoint = torch.load(args.radionet_checkpoint, map_location=device)
    
    # Robust loading
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
    
    results = evaluate_multiobjective(
        model_power=model_power,
        model_coverage=model_coverage,
        test_dataset=test_dataset,
        power_dataset=power_dataset,
        coverage_dataset=coverage_dataset,
        device=device,
        use_center=args.use_center,
        buildings_dir=args.buildings_dir,
        radionet_model=radionet_model,
        heatmap_base_dir=args.heatmap_base_dir,
        K=args.K
    )
    
    # Prepare output
    output = {
        'model_info': {
            'model_type': 'multiobjective',
            'architecture': args.arch,
            'power_model_path': args.power_model_path,
            'coverage_model_path': args.coverage_model_path,
            'heatmap_type_power': args.heatmap_type_power,
            'heatmap_type_coverage': args.heatmap_type_coverage,
            'coord_method': args.coord_method,
            'temperature': args.temperature,
            'use_center': args.use_center,
            'K': args.K,
            'selection_phase1': 'minimax',
            'selection_phase2': 'L2_distance'
        },
        'dataset_info': {
            'split': args.split,
            'num_buildings': len(eval_buildings),
            'buildings_dir': args.buildings_dir,
            'heatmap_base_dir': args.heatmap_base_dir
        },
        'hardware_info': hardware_info,
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
    if results:
        print(f"\nSummary:")
        print(f"  Buildings evaluated: {results.get('num_buildings', 0)}")
        print(f"  Coord Error (power GT): {results.get('coord_error_power_mean', 0):.2f}px (±{results.get('coord_error_power_std', 0):.2f})")
        print(f"  Coord Error (coverage GT): {results.get('coord_error_coverage_mean', 0):.2f}px (±{results.get('coord_error_coverage_std', 0):.2f})")
        print(f"  PL %: {results.get('pl_pct_ref_mean', 0):.2f}% (±{results.get('pl_pct_ref_std', 0):.2f})")
        print(f"  Coverage %: {results.get('coverage_pct_ref_mean', 0):.2f}% (±{results.get('coverage_pct_ref_std', 0):.2f})")
        print(f"  Free pixels per building: {results.get('free_pixels_mean', 0):.0f} (±{results.get('free_pixels_std', 0):.0f})")


if __name__ == '__main__':
    main()
