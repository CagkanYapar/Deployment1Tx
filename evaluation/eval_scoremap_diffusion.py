"""
AvgMap Diffusion Model Evaluation - Multi-Sample with Dual GT

Evaluates diffusion models trained on average maps with multi-sample candidate extraction.

Key differences from discriminative:
- Uses sample_diffusion() for forward pass (single run, no multiple stochastic samples)
- Requires scheduler (DDPM or DDIM)
- Requires conditioning parameter
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
from PIL import Image

# Import diffusion model and utilities
from models.diffusion import create_diffusion_model
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


# ============================================================================
# DIFFUSION UTILITIES (from training script)
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
        optimize_for_list: List of optimization targets
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


def get_hardware_info():
    """Capture hardware information."""
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


def evaluate_avgmap_diffusion_multisample(
    model, scheduler, test_dataset, power_dataset, coverage_dataset,
    device, optimize_for, use_center, buildings_dir, radionet_model, heatmap_base_dir,
    num_inference_steps, use_ddim, n_values=[1, 5, 10, 20, 50, 100]
):
    """
    Evaluate diffusion avgmap model with multi-sample support.
    
    Same logic as discriminative, but uses sample_diffusion() for forward pass.
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
    print(f"AVGMAP DIFFUSION EVALUATION - MULTI-SAMPLE")
    print(f"{'='*80}")
    print(f"N values: {n_values}")
    print(f"Optimize for: {optimize_for}")
    print(f"Inference steps: {num_inference_steps}")
    print(f"Use DDIM: {use_ddim}")
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
            
            # Model forward pass - SINGLE DIFFUSION RUN
            building_map_input = building_map.unsqueeze(0).to(device)
            optimize_for_list = [optimize_for]
            
            model_start = time.time()
            pred_avgmap = sample_diffusion(
                model, building_map_input, optimize_for_list,
                scheduler, num_inference_steps, use_ddim, device
            )
            model_time = time.time() - model_start
            
            # Extract building mask
            building_np = building_map.squeeze().cpu().numpy()
            free_mask = (building_np <= 0.1)  # Free space, matching training convention
            
            # Extract top-max_n candidates from free space
            candidates = extract_topn_candidates(pred_avgmap, free_mask, n=max_n, use_center=use_center)
            
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
                
                # Strategy 1: Single
                single_metrics = metrics_n[0]
                single_dual_gt = compute_dual_gt_metrics(
                    single_metrics['coord'], power_gt_np, coverage_gt_np, single_metrics,
                    building_id, heatmap_base_dir, optimize_for, use_center, buildings_dir
                )
                
                if single_dual_gt:
                    single_dual_gt['building_id'] = building_id
                    results_by_n[n]['Single'].append(single_dual_gt)
                
                # Strategy 2: Best-Coverage
                best_cov_idx, best_cov_metrics = select_best_candidate(metrics_n, criterion='coverage')
                best_cov_dual_gt = compute_dual_gt_metrics(
                    best_cov_metrics['coord'], power_gt_np, coverage_gt_np, best_cov_metrics,
                    building_id, heatmap_base_dir, optimize_for, use_center, buildings_dir
                )
                
                if best_cov_dual_gt:
                    best_cov_dual_gt['building_id'] = building_id
                    best_cov_dual_gt['model_rank'] = best_cov_idx + 1
                    results_by_n[n]['Best-Coverage'].append(best_cov_dual_gt)
                
                # Strategy 3: Best-Power
                best_pow_idx, best_pow_metrics = select_best_candidate(metrics_n, criterion='power')
                best_pow_dual_gt = compute_dual_gt_metrics(
                    best_pow_metrics['coord'], power_gt_np, coverage_gt_np, best_pow_metrics,
                    building_id, heatmap_base_dir, optimize_for, use_center, buildings_dir
                )
                
                if best_pow_dual_gt:
                    best_pow_dual_gt['building_id'] = building_id
                    best_pow_dual_gt['model_rank'] = best_pow_idx + 1
                    results_by_n[n]['Best-Power'].append(best_pow_dual_gt)
    
    overall_time = time.time() - overall_start
    
    # Aggregate results
    final_results = {}
    
    for n in n_values:
        final_results[n] = {}
        
        for strategy in ['Single', 'Best-Coverage', 'Best-Power']:
            strategy_results = results_by_n[n][strategy]
            
            if len(strategy_results) > 0:
                aggregated = aggregate_results(strategy_results)
                
                # Add timing info
                aggregated['model_time_mean'] = float(np.mean(timing_data['model_times']))
                aggregated['radionet_time_mean'] = float(np.mean(timing_data['radionet_times']))
                aggregated['total_time_mean'] = float(np.mean(timing_data['total_times']))
                aggregated['total_evaluation_time'] = overall_time
                
                final_results[n][strategy] = aggregated
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate avgmap diffusion model on test set')
    
    # Model configuration
    parser.add_argument('--conditioning', type=str, required=True,
                       choices=['concat', 'film', 'cross_attn'],
                       help='Conditioning method')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    
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
    parser.add_argument('--optimize_for', type=str, required=True,
                       choices=['power', 'coverage'],
                       help='Optimization target')
    parser.add_argument('--heatmap_type', type=str, required=True,
                       help='Heatmap type')
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
    
    # Set tx_dir for each dataset (based on which GT coordinates they represent)
    if args.optimize_for == 'power':
        test_tx_dir = os.path.join(args.heatmap_base_dir, 'best_by_power')
    else:
        test_tx_dir = os.path.join(args.heatmap_base_dir, 'best_by_coverage')
    
    power_tx_dir = os.path.join(args.heatmap_base_dir, 'best_by_power')
    coverage_tx_dir = os.path.join(args.heatmap_base_dir, 'best_by_coverage')
    
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
    
    print(f"\nCreating test dataset (optimize_for={args.optimize_for}, heatmap_type={args.heatmap_type})")
    test_dataset = TxLocationDatasetLarge(
        eval_buildings,
        buildings_dir=args.buildings_dir,
        tx_dir=test_tx_dir,
        optimize_for=args.optimize_for,
        heatmap_type=args.heatmap_type,
        heatmap_base_dir=args.heatmap_base_dir,
        normalize_heatmap=False,
        augment=False,
        use_center=args.use_center
    )
    
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
    
    # Create diffusion model
    print(f"\nCreating diffusion model (conditioning={args.conditioning})")
    model = create_diffusion_model(
        conditioning=args.conditioning,
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
    
    # Create scheduler
    if args.use_ddim:
        scheduler = DDIMScheduler(num_train_timesteps=args.num_train_timesteps)
        print("Using DDIM scheduler")
    else:
        scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)
        print("Using DDPM scheduler")
    
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
    
    results = evaluate_avgmap_diffusion_multisample(
        model=model,
        scheduler=scheduler,
        test_dataset=test_dataset,
        power_dataset=power_dataset,
        coverage_dataset=coverage_dataset,
        device=device,
        optimize_for=args.optimize_for,
        use_center=args.use_center,
        buildings_dir=args.buildings_dir,
        radionet_model=radionet_model,
        heatmap_base_dir=args.heatmap_base_dir,
        num_inference_steps=args.num_inference_steps,
        use_ddim=args.use_ddim,
        n_values=args.n_values
    )
    
    # Prepare output
    output = {
        'model_info': {
            'model_type': 'diffusion',
            'conditioning': args.conditioning,
            'model_path': args.model_path,
            'optimize_for': args.optimize_for,
            'heatmap_type': args.heatmap_type,
            'num_inference_steps': args.num_inference_steps,
            'use_ddim': args.use_ddim,
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
