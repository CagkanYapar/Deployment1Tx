"""
Single-Objective Best-L2 with Union Evaluation

Evaluates two models (power + coverage) separately, then combines results.

For each building:
  1. Coverage model → top-M candidates → RadioNet → best L2 among M → L2_cov
  2. Power model → top-M candidates → RadioNet → best L2 among M → L2_pow
  3. L2_union = min(L2_cov, L2_pow)

Reports:
  - L2_cov_mean (coverage model performance)
  - L2_pow_mean (power model performance)
  - L2_union_mean (best of both per building)
  - union_size_mean (metadata: how many unique pixels in union)
"""

import os
import sys
import argparse
import numpy as np
import torch
import json
import time
from tqdm import tqdm

# Import discriminative model
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
# DIFFUSION SCHEDULERS
# ============================================================================

class DDPMScheduler:
    """Denoising Diffusion Probabilistic Models noise scheduler"""
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_train_timesteps = num_train_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.nn.functional.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

class DDIMScheduler:
    """Denoising Diffusion Implicit Models scheduler"""
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_train_timesteps = num_train_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

def sample_diffusion(model, building_map, optimize_for_list, scheduler, 
                    num_inference_steps, use_ddim, device):
    """Sample from diffusion model."""
    batch_size = building_map.shape[0]
    img_size = building_map.shape[2]
    sample = torch.randn(batch_size, 1, img_size, img_size, device=device)
    
    if use_ddim:
        step_ratio = scheduler.num_train_timesteps // num_inference_steps
        timesteps = list(range(0, scheduler.num_train_timesteps, step_ratio))[::-1]
    else:
        timesteps = list(range(scheduler.num_train_timesteps))[::-1][:num_inference_steps]
    
    for i, t in enumerate(timesteps):
        timestep = torch.full((batch_size,), t, device=device, dtype=torch.long)
        with torch.no_grad():
            noise_pred = model(building_map, sample, timestep, optimize_for_list)
        
        alpha_prod_t = scheduler.alphas_cumprod[t].to(device)
        alpha_prod_t_prev = scheduler.alphas_cumprod[timesteps[i+1]].to(device) if i < len(timesteps)-1 else torch.tensor(1.0, device=device)
        
        if use_ddim:
            sqrt_alpha_prod_t = torch.sqrt(alpha_prod_t)
            sqrt_one_minus_alpha_prod_t = torch.sqrt(1 - alpha_prod_t)
            pred_x0 = (sample - sqrt_one_minus_alpha_prod_t * noise_pred) / sqrt_alpha_prod_t
            sqrt_alpha_prod_t_prev = torch.sqrt(alpha_prod_t_prev)
            sqrt_one_minus_alpha_prod_t_prev = torch.sqrt(1 - alpha_prod_t_prev)
            sample = sqrt_alpha_prod_t_prev * pred_x0 + sqrt_one_minus_alpha_prod_t_prev * noise_pred
        else:
            beta_t = scheduler.betas[t].to(device)
            sqrt_one_minus_alpha_prod_t = torch.sqrt(1 - alpha_prod_t)
            pred_x0 = (sample - sqrt_one_minus_alpha_prod_t * noise_pred) / torch.sqrt(alpha_prod_t)
            variance = 0
            if t > 0:
                variance = torch.sqrt(scheduler.posterior_variance[t].to(device)) * torch.randn_like(sample)
            sample = torch.sqrt(alpha_prod_t_prev) * pred_x0 + torch.sqrt(1 - alpha_prod_t_prev) * noise_pred + variance
    
    return sample.squeeze().cpu().numpy()


# ============================================================================
# UTILITY FUNCTIONS
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


def extract_topn_candidates(heatmap, free_space_mask, n, use_center=True):
    """
    Extract top-N candidate coordinates from heatmap in free space.
    
    Args:
        heatmap: Predicted heatmap (H, W)
        free_space_mask: Boolean mask (H, W) - True for free space
        n: Number of candidates to extract
        use_center: Whether using 150x150 center crop
    
    Returns:
        List of (y, x) coordinates (top-N by heatmap value)
    """
    # Mask out buildings
    masked_heatmap = heatmap.copy()
    masked_heatmap[~free_space_mask] = -np.inf
    
    # Flatten and get top-N indices
    flat_indices = np.argsort(masked_heatmap.ravel())[::-1][:n]
    
    # Convert to 2D coordinates
    coords = []
    for idx in flat_indices:
        if masked_heatmap.ravel()[idx] == -np.inf:
            break
        y, x = np.unravel_index(idx, heatmap.shape)
        coords.append((float(y), float(x)))
    
    return coords


def select_best_by_l2(candidates_metrics):
    """
    Select best candidate by L2 distance from ideal point (100%, 100%).
    
    Args:
        candidates_metrics: List of dicts with dual-GT metrics
    
    Returns:
        (best_idx, best_metrics) tuple
    """
    l2_scores = [
        -np.sqrt((100 - m['pl_pct_ref'])**2 + (100 - m['coverage_pct_ref'])**2)
        for m in candidates_metrics
    ]
    
    best_idx = int(np.argmax(l2_scores))
    best_metrics = candidates_metrics[best_idx].copy()
    best_metrics['l2_score'] = float(l2_scores[best_idx])
    
    return best_idx, best_metrics


def compute_union_size(coords_cov, coords_pow):
    """
    Compute size of union of two coordinate sets.
    
    Args:
        coords_cov: List of (y, x) tuples from coverage model
        coords_pow: List of (y, x) tuples from power model
    
    Returns:
        Number of unique coordinates in union
    """
    coords_set = set(coords_cov) | set(coords_pow)
    return len(coords_set)


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def evaluate_singleobj_bestl2_union(
    model_coverage, model_power, scheduler, num_inference_steps, use_ddim,
    test_dataset, power_dataset, coverage_dataset,
    device, use_center, buildings_dir, radionet_model,
    heatmap_base_dir, M
):
    """
    Evaluate single-objective models with Best-L2 selection + union.
    
    Args:
        model_coverage: Coverage-optimized model
        model_power: Power-optimized model
    scheduler: Diffusion scheduler
    num_inference_steps: Diffusion inference steps
    use_ddim: Use DDIM
        test_dataset, power_dataset, coverage_dataset: Datasets
        device: torch device
        use_center: Whether using 150x150 center crop
        buildings_dir: Path to building images
        radionet_model: RadioNet model
        heatmap_base_dir: Base directory for heatmaps
        M: Number of candidates to extract from each model
    
    Returns:
        Dictionary with results
    """
    results_cov = []
    results_pow = []
    results_union = []
    
    timing_data = {
        'model_cov_times': [],
        'model_pow_times': [],
        'radionet_cov_times': [],
        'radionet_pow_times': [],
        'total_times': []
    }
    
    union_sizes = []
    
    print(f"\n{'='*80}")
    print(f"SINGLE-OBJECTIVE BEST-L2 WITH UNION")
    print(f"{'='*80}")
    print(f"M (candidates per model): {M}")
    print(f"Test buildings: {len(test_dataset)}")
    print(f"{'='*80}\n")
    
    # Checkpoint directory
    checkpoint_dir = heatmap_base_dir.replace('unified_results_v2_float32_K500_dual', 'checkpoints_bestl2_union_diffusion')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_M{M}.pkl')
    
    overall_start = time.time()
    
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="Evaluating", mininterval=2.0):
            building_start = time.time()
            
            # Get data
            building_map, gt_coords, target_heatmap, building_id = test_dataset[idx]
            _, power_gt, _, _ = power_dataset[idx]
            _, coverage_gt, _, _ = coverage_dataset[idx]
            
            building_map_input = building_map.unsqueeze(0).to(device)
            building_np = building_map.squeeze().cpu().numpy()
            free_space_mask = (building_np <= 0.1)
            
            # ================================================================
            # COVERAGE MODEL
            # ================================================================
            model_cov_start = time.time()
            pred_avgmap_cov = sample_diffusion(model_coverage, building_map_input, ['coverage'], scheduler, num_inference_steps, use_ddim, device)
            model_cov_time = time.time() - model_cov_start
            
            # Extract top-M from coverage map
            coords_cov = extract_topn_candidates(pred_avgmap_cov, free_space_mask, M, use_center)
            
            # Evaluate with RadioNet
            radionet_cov_start = time.time()
            candidates_cov = evaluate_candidates_batch_radionet(
                coords_cov, building_id, buildings_dir, radionet_model,
                use_center, device, batch_size=64
            )
            radionet_cov_time = time.time() - radionet_cov_start
            
            # Compute dual-GT for all candidates
            power_gt_np = power_gt.cpu().numpy()
            coverage_gt_np = coverage_gt.cpu().numpy()
            
            candidates_cov_with_metrics = []
            for metrics in candidates_cov:
                dual_gt = compute_dual_gt_metrics(
                    metrics['coord'], power_gt_np, coverage_gt_np, metrics,
                    building_id, heatmap_base_dir, 'coverage',
                    use_center, buildings_dir
                )
                if dual_gt is not None:
                    candidates_cov_with_metrics.append(dual_gt)
            
            # ================================================================
            # POWER MODEL
            # ================================================================
            model_pow_start = time.time()
            pred_avgmap_pow = sample_diffusion(model_power, building_map_input, ['power'], scheduler, num_inference_steps, use_ddim, device)
            model_pow_time = time.time() - model_pow_start
            
            # Extract top-M from power map
            coords_pow = extract_topn_candidates(pred_avgmap_pow, free_space_mask, M, use_center)
            
            # Evaluate with RadioNet
            radionet_pow_start = time.time()
            candidates_pow = evaluate_candidates_batch_radionet(
                coords_pow, building_id, buildings_dir, radionet_model,
                use_center, device, batch_size=64
            )
            radionet_pow_time = time.time() - radionet_pow_start
            
            # Compute dual-GT for all candidates
            candidates_pow_with_metrics = []
            for metrics in candidates_pow:
                dual_gt = compute_dual_gt_metrics(
                    metrics['coord'], power_gt_np, coverage_gt_np, metrics,
                    building_id, heatmap_base_dir, 'power',
                    use_center, buildings_dir
                )
                if dual_gt is not None:
                    candidates_pow_with_metrics.append(dual_gt)
            
            building_total_time = time.time() - building_start
            
            # Store timing
            timing_data['model_cov_times'].append(model_cov_time)
            timing_data['model_pow_times'].append(model_pow_time)
            timing_data['radionet_cov_times'].append(radionet_cov_time)
            timing_data['radionet_pow_times'].append(radionet_pow_time)
            timing_data['total_times'].append(building_total_time)
            
            # ================================================================
            # SELECTION
            # ================================================================
            if len(candidates_cov_with_metrics) == 0 or len(candidates_pow_with_metrics) == 0:
                print(f"Warning: Building {building_id} has no valid candidates, skipping")
                continue
            
            # Best L2 from coverage model
            best_idx_cov, best_metrics_cov = select_best_by_l2(candidates_cov_with_metrics)
            best_metrics_cov['building_id'] = building_id
            results_cov.append(best_metrics_cov)
            
            # Best L2 from power model
            best_idx_pow, best_metrics_pow = select_best_by_l2(candidates_pow_with_metrics)
            best_metrics_pow['building_id'] = building_id
            results_pow.append(best_metrics_pow)
            
            # Union: pick the better of the two
            l2_cov = -best_metrics_cov['l2_score']
            l2_pow = -best_metrics_pow['l2_score']
            
            if l2_cov <= l2_pow:
                best_union = best_metrics_cov.copy()
            else:
                best_union = best_metrics_pow.copy()
            
            results_union.append(best_union)
            
            # Compute union size
            union_size = compute_union_size(coords_cov, coords_pow)
            union_sizes.append(union_size)
            
            # Checkpoint every 100 buildings
            if (idx + 1) % 100 == 0:
                import pickle
                checkpoint_data = {
                    'results_cov': results_cov,
                    'results_pow': results_pow,
                    'results_union': results_union,
                    'union_sizes': union_sizes,
                    'timing_data': timing_data,
                    'buildings_processed': idx + 1
                }
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
                print(f"\nCheckpoint saved at building {idx + 1}")
    
    overall_time = time.time() - overall_start
    
    # Aggregate results
    if len(results_cov) == 0 or len(results_pow) == 0:
        print("\nERROR: No valid results collected!")
        return {}
    
    final_results = {}
    
    # Coverage model results
    final_results['coverage_model'] = aggregate_results(results_cov)
    
    # Power model results
    final_results['power_model'] = aggregate_results(results_pow)
    
    # Union results
    final_results['union'] = aggregate_results(results_union)
    
    # Add timing
    final_results['timing'] = {
        'model_cov_time_mean': float(np.mean(timing_data['model_cov_times'])),
        'model_pow_time_mean': float(np.mean(timing_data['model_pow_times'])),
        'radionet_cov_time_mean': float(np.mean(timing_data['radionet_cov_times'])),
        'radionet_pow_time_mean': float(np.mean(timing_data['radionet_pow_times'])),
        'total_time_mean': float(np.mean(timing_data['total_times'])),
        'total_evaluation_time': overall_time
    }
    
    # Add union statistics
    final_results['union_stats'] = {
        'union_size_mean': float(np.mean(union_sizes)),
        'union_size_std': float(np.std(union_sizes)),
        'union_size_min': float(np.min(union_sizes)),
        'union_size_max': float(np.max(union_sizes)),
        'M': M
    }
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description='Single-objective Best-L2 union evaluation')
    
    # Model configuration
    # Diffusion configuration
    parser.add_argument('--conditioning', type=str, default='concat',
                       choices=['concat', 'film', 'cross_attn'],
                       help='Conditioning method')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                       help='Number of diffusion inference steps')
    parser.add_argument('--use_ddim', action='store_true',
                       help='Use DDIM sampling')
    parser.add_argument('--num_train_timesteps', type=int, default=1000,
                       help='Number of training timesteps')
    
    parser.add_argument('--coverage_model_path', type=str, required=True,
                       help='Path to coverage-optimized model checkpoint')
    parser.add_argument('--power_model_path', type=str, required=True,
                       help='Path to power-optimized model checkpoint')
    
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
    
    # Evaluation configuration
    parser.add_argument('--M', type=int, default=200,
                       help='Number of candidates to evaluate per model')
    
    # Dataset size limit
    parser.add_argument('--max_buildings', type=int, default=None,
                       help='Limit number of buildings to evaluate')
    
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
    
    # Test dataset
    test_dataset = power_dataset
    
    # Create scheduler
    if args.use_ddim:
        scheduler = DDIMScheduler(num_train_timesteps=args.num_train_timesteps)
        print("Using DDIM scheduler")
    else:
        scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)
        print("Using DDPM scheduler")
    
    # Create coverage model
    print("\nCreating coverage model...")
    model_coverage = create_diffusion_model(conditioning=args.conditioning, img_size=img_size).to(device)
    
    # Load coverage checkpoint
    print(f"Loading coverage model from {args.coverage_model_path}")
    checkpoint_cov = torch.load(args.coverage_model_path, map_location=device)
    if 'model_state_dict' in checkpoint_cov:
        model_coverage.load_state_dict(checkpoint_cov['model_state_dict'])
        print(f"  Loaded from epoch {checkpoint_cov.get('epoch', 'unknown')}")
    else:
        model_coverage.load_state_dict(checkpoint_cov)
    model_coverage.eval()
    
    # Create power model
    print("\nCreating power model...")
    model_power = create_diffusion_model(conditioning=args.conditioning, img_size=img_size).to(device)
    
    # Load power checkpoint
    print(f"Loading power model from {args.power_model_path}")
    checkpoint_pow = torch.load(args.power_model_path, map_location=device)
    if 'model_state_dict' in checkpoint_pow:
        model_power.load_state_dict(checkpoint_pow['model_state_dict'])
        print(f"  Loaded from epoch {checkpoint_pow.get('epoch', 'unknown')}")
    else:
        model_power.load_state_dict(checkpoint_pow)
    model_power.eval()
    
    # Load RadioNet
    print(f"\nLoading RadioNet from {args.radionet_checkpoint}")
    radionet_model = RadioNet(inputs=2).to(device)
    radionet_checkpoint = torch.load(args.radionet_checkpoint, map_location=device)
    
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
    print("STARTING SINGLE-OBJECTIVE BEST-L2 UNION EVALUATION")
    print(f"{'='*80}")
    
    results = evaluate_singleobj_bestl2_union(
        model_coverage=model_coverage,
        model_power=model_power,
        scheduler=scheduler,
        num_inference_steps=args.num_inference_steps,
        use_ddim=args.use_ddim,
        test_dataset=test_dataset,
        power_dataset=power_dataset,
        coverage_dataset=coverage_dataset,
        device=device,
        use_center=args.use_center,
        buildings_dir=args.buildings_dir,
        radionet_model=radionet_model,
        heatmap_base_dir=args.heatmap_base_dir,
        M=args.M
    )
    
    # Add model info
    results['model_info'] = {
        'coverage_model_path': args.coverage_model_path,
        'power_model_path': args.power_model_path,
        'conditioning': args.conditioning,
        'num_inference_steps': args.num_inference_steps,
        'use_ddim': args.use_ddim,
        'M': args.M
    }
    
    results['hardware_info'] = hardware_info
    
    # Convert to JSON-serializable
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
    print(f"  Coverage model L2: {-results['coverage_model'].get('l2_score_mean', 0):.2f}")
    print(f"  Power model L2: {-results['power_model'].get('l2_score_mean', 0):.2f}")
    print(f"  Union L2: {-results['union'].get('l2_score_mean', 0):.2f}")
    print(f"  Average union size: {results['union_stats']['union_size_mean']:.1f} / {2*args.M}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
