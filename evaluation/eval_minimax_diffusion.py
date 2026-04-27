"""
AvgMap Multi-Objective Diffusion Evaluation

Evaluates diffusion models trained on power vs coverage objectives together.
Two-phase selection:
  Phase 1: Rank all free space pixels by both maps, select top-K by minimax
  Phase 2: Evaluate K candidates with RadioNet, select winner by L2 distance from ideal

Key differences from discriminative multi-objective:
- Loads TWO diffusion models (power-trained, coverage-trained)
- Uses sample_diffusion() for forward pass (single denoising run per model)
- Requires scheduler (DDPM or DDIM)
- Otherwise identical evaluation logic
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import json
import time
from tqdm import tqdm

# Import diffusion model and utilities
from models.diffusion import create_diffusion_model
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


# ============================================================================
# DIFFUSION SCHEDULERS (from single-objective diffusion evaluation)
# ============================================================================

class DDPMScheduler:
    """Denoising Diffusion Probabilistic Models noise scheduler"""
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_train_timesteps = num_train_timesteps
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )


class DDIMScheduler:
    """Denoising Diffusion Implicit Models scheduler (faster sampling)"""
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_train_timesteps = num_train_timesteps
        
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)


def sample_diffusion(model, building_map, optimize_for_list, scheduler, 
                    num_inference_steps, use_ddim, device):
    """
    Sample from diffusion model (single forward pass).
    
    Args:
        model: Diffusion model
        building_map: Building map tensor (batch, 1, H, W)
        optimize_for_list: List of optimization targets (e.g., ['power'] or ['coverage'])
        scheduler: DDPM or DDIM scheduler
        num_inference_steps: Number of denoising steps
        use_ddim: Whether to use DDIM sampling
        device: torch device
    
    Returns:
        Predicted average map (H, W) numpy array
    """
    batch_size = building_map.shape[0]
    img_size = building_map.shape[2]
    
    # Start from noise
    sample = torch.randn(batch_size, 1, img_size, img_size, device=device)
    
    # Create timestep schedule
    if use_ddim:
        step_ratio = scheduler.num_train_timesteps // num_inference_steps
        timesteps = list(range(0, scheduler.num_train_timesteps, step_ratio))[::-1]
    else:
        timesteps = list(range(scheduler.num_train_timesteps))[::-1]
        timesteps = timesteps[:num_inference_steps]
    
    # Denoising loop
    for i, t in enumerate(timesteps):
        timestep = torch.full((batch_size,), t, device=device, dtype=torch.long)
        
        # Predict noise
        with torch.no_grad():
            noise_pred = model(building_map, sample, timestep, optimize_for_list)
        
        # Update sample
        alpha_prod_t = scheduler.alphas_cumprod[t].to(device)
        alpha_prod_t_prev = scheduler.alphas_cumprod[timesteps[i+1]].to(device) if i < len(timesteps)-1 else torch.tensor(1.0, device=device)
        
        if use_ddim:
            # DDIM update
            sqrt_alpha_prod_t = torch.sqrt(alpha_prod_t)
            sqrt_one_minus_alpha_prod_t = torch.sqrt(1 - alpha_prod_t)
            
            pred_x0 = (sample - sqrt_one_minus_alpha_prod_t * noise_pred) / sqrt_alpha_prod_t
            
            sqrt_alpha_prod_t_prev = torch.sqrt(alpha_prod_t_prev)
            sqrt_one_minus_alpha_prod_t_prev = torch.sqrt(1 - alpha_prod_t_prev)
            
            sample = sqrt_alpha_prod_t_prev * pred_x0 + sqrt_one_minus_alpha_prod_t_prev * noise_pred
        else:
            # DDPM update
            beta_t = scheduler.betas[t].to(device)
            sqrt_one_minus_alpha_prod_t = torch.sqrt(1 - alpha_prod_t)
            
            pred_x0 = (sample - sqrt_one_minus_alpha_prod_t * noise_pred) / torch.sqrt(alpha_prod_t)
            
            variance = 0
            if t > 0:
                variance = torch.sqrt(scheduler.posterior_variance[t].to(device)) * torch.randn_like(sample)
            
            sample = torch.sqrt(alpha_prod_t_prev) * pred_x0 + torch.sqrt(1 - alpha_prod_t_prev) * noise_pred + variance
    
    # Return as numpy
    pred_heatmap = sample.squeeze().cpu().numpy()
    
    return pred_heatmap


# ============================================================================
# MULTI-OBJECTIVE SELECTION FUNCTIONS (from discriminative multi-objective)
# ============================================================================

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
        ranked_pixels: List of dicts with 'coord', 'rank_power', 'rank_coverage'
        K: Number of candidates to select
    
    Returns:
        List of top-K pixels (dicts with coord and ranks)
    """
    # Compute minimax score for each pixel
    for pixel in ranked_pixels:
        pixel['minimax_score'] = -max(pixel['rank_power'], pixel['rank_coverage'])
    
    # Sort by minimax score (descending = best first)
    sorted_pixels = sorted(ranked_pixels, key=lambda x: x['minimax_score'], reverse=True)
    
    # Return top K
    return sorted_pixels[:K]


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


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def evaluate_avgmap_diffusion_multiobjective(
    model_power, model_coverage, scheduler,
    test_dataset, power_dataset, coverage_dataset,
    device, use_center, buildings_dir, radionet_model,
    heatmap_base_dir, num_inference_steps, use_ddim, K=200
):
    """
    Evaluate two diffusion models (power + coverage) with multi-objective selection.
    
    Args:
        model_power: Diffusion model trained on power objective
        model_coverage: Diffusion model trained on coverage objective
        scheduler: DDPM or DDIM scheduler
        test_dataset, power_dataset, coverage_dataset: Datasets for evaluation
        device: torch device
        use_center: Whether using 150x150 center crop
        buildings_dir: Path to building images
        radionet_model: RadioNet model for evaluation
        heatmap_base_dir: Base directory for heatmaps
        num_inference_steps: Number of diffusion inference steps
        use_ddim: Whether using DDIM
        K: Number of candidates to evaluate with RadioNet (default: 200)
    
    Returns:
        Dictionary with aggregated results
    """
    results = []
    timing_data = {
        'model_times': [],
        'radionet_times': [],
        'total_times': []
    }
    intersection_stats = []
    
    print(f"\n{'='*80}")
    print(f"AVGMAP DIFFUSION MULTI-OBJECTIVE EVALUATION")
    print(f"{'='*80}")
    print(f"K (candidates to evaluate): {K}")
    print(f"Inference steps: {num_inference_steps}")
    print(f"Use DDIM: {use_ddim}")
    print(f"Test buildings: {len(test_dataset)}")
    print(f"Selection: Phase 1 = Minimax, Phase 2 = L2 distance")
    print(f"{'='*80}\n")
    
    overall_start = time.time()
    
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="Evaluating", mininterval=2.0):
            building_start = time.time()
            
            # Get data
            building_map, gt_coords, target_heatmap, building_id = test_dataset[idx]
            _, power_gt, _, _ = power_dataset[idx]
            _, coverage_gt, _, _ = coverage_dataset[idx]
            
            building_map_input = building_map.unsqueeze(0).to(device)
            
            model_start = time.time()
            
            # Generate power average map (single diffusion run)
            power_avgmap = sample_diffusion(
                model_power, building_map_input, ['power'],
                scheduler, num_inference_steps, use_ddim, device
            )
            
            # Generate coverage average map (single diffusion run)
            coverage_avgmap = sample_diffusion(
                model_coverage, building_map_input, ['coverage'],
                scheduler, num_inference_steps, use_ddim, device
            )
            
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
    parser = argparse.ArgumentParser(description='Multi-objective avgmap diffusion evaluation')
    
    # Model configuration
    parser.add_argument('--conditioning', type=str, required=True,
                       choices=['concat', 'film', 'cross_attn'],
                       help='Conditioning method (must be same for both models)')
    parser.add_argument('--power_model_path', type=str, required=True,
                       help='Path to power-trained diffusion model checkpoint')
    parser.add_argument('--coverage_model_path', type=str, required=True,
                       help='Path to coverage-trained diffusion model checkpoint')
    
    # Diffusion configuration
    parser.add_argument('--num_inference_steps', type=int, default=50,
                       help='Number of denoising steps')
    parser.add_argument('--use_ddim', action='store_true',
                       help='Use DDIM sampling (faster)')
    parser.add_argument('--num_train_timesteps', type=int, default=1000,
                       help='Number of training timesteps')
    
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
    
    # Dataset size limit
    parser.add_argument('--max_buildings', type=int, default=None,
                       help='Limit number of buildings to evaluate (for faster testing)')
    
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
    
    # Limit buildings if requested
    if args.max_buildings is not None and args.max_buildings < len(eval_buildings):
        eval_buildings = eval_buildings[:args.max_buildings]
        print(f"Limited to {len(eval_buildings)} buildings (--max_buildings={args.max_buildings})")
    
    # Create datasets
    img_size = 150 if args.use_center else 256
    
    # Power-optimal GT dataset
    print("Creating power-optimal GT dataset")
    power_tx_dir = os.path.join(args.heatmap_base_dir, 'best_by_power')
    power_dataset = TxLocationDatasetLarge(
        eval_buildings,
        buildings_dir=args.buildings_dir,
        tx_dir=power_tx_dir,
        optimize_for='power',
        heatmap_type=args.heatmap_type_power,
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
        heatmap_type=args.heatmap_type_coverage,
        heatmap_base_dir=args.heatmap_base_dir,
        normalize_heatmap=False,
        augment=False,
        use_center=args.use_center
    )
    
    # Test dataset (using power as reference, but doesn't matter for multi-objective)
    test_dataset = power_dataset
    
    # Create diffusion models
    print(f"\nCreating diffusion models (conditioning={args.conditioning})")
    
    # Power model
    print("Loading power-trained model...")
    model_power = create_diffusion_model(
        conditioning=args.conditioning,
        img_size=img_size
    ).to(device)
    
    checkpoint_power = torch.load(args.power_model_path, map_location=device)
    if 'model_state_dict' in checkpoint_power:
        model_power.load_state_dict(checkpoint_power['model_state_dict'])
        print(f"  Loaded from epoch {checkpoint_power.get('epoch', 'unknown')}")
    else:
        model_power.load_state_dict(checkpoint_power)
    model_power.eval()
    
    # Coverage model
    print("Loading coverage-trained model...")
    model_coverage = create_diffusion_model(
        conditioning=args.conditioning,
        img_size=img_size
    ).to(device)
    
    checkpoint_coverage = torch.load(args.coverage_model_path, map_location=device)
    if 'model_state_dict' in checkpoint_coverage:
        model_coverage.load_state_dict(checkpoint_coverage['model_state_dict'])
        print(f"  Loaded from epoch {checkpoint_coverage.get('epoch', 'unknown')}")
    else:
        model_coverage.load_state_dict(checkpoint_coverage)
    model_coverage.eval()
    
    # Create scheduler
    if args.use_ddim:
        scheduler = DDIMScheduler(num_train_timesteps=args.num_train_timesteps)
        print("\nUsing DDIM scheduler")
    else:
        scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)
        print("\nUsing DDPM scheduler")
    
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
    print("STARTING MULTI-OBJECTIVE EVALUATION")
    print(f"{'='*80}")
    
    results = evaluate_avgmap_diffusion_multiobjective(
        model_power=model_power,
        model_coverage=model_coverage,
        scheduler=scheduler,
        test_dataset=test_dataset,
        power_dataset=power_dataset,
        coverage_dataset=coverage_dataset,
        device=device,
        use_center=args.use_center,
        buildings_dir=args.buildings_dir,
        radionet_model=radionet_model,
        heatmap_base_dir=args.heatmap_base_dir,
        num_inference_steps=args.num_inference_steps,
        use_ddim=args.use_ddim,
        K=args.K
    )
    
    # Add model info
    results['model_info'] = {
        'power_model_path': args.power_model_path,
        'coverage_model_path': args.coverage_model_path,
        'conditioning': args.conditioning,
        'num_inference_steps': args.num_inference_steps,
        'use_ddim': args.use_ddim,
        'num_train_timesteps': args.num_train_timesteps,
        'K': args.K
    }
    
    results['hardware_info'] = hardware_info
    
    # Convert to JSON-serializable format
    results_json = convert_to_json_serializable(results)
    
    # Save results
    output_file = os.path.join(args.output_dir, 'results.json')
    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"\nKey metrics:")
    print(f"  Coverage: {results.get('coverage_pct_ref_mean', 0):.2f}%")
    print(f"  Power Loss: {results.get('pl_pct_ref_mean', 0):.2f}%")
    print(f"  L2 Distance: {results.get('l2_distance_mean', 0):.2f}")
    print(f"  Buildings evaluated: {results.get('num_buildings', 0)}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
