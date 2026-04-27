"""
Diffusion Model Training Script for Tx Location Prediction

Integrates diffusion models with existing validation infrastructure:
- Uses existing dataset from txLocationDatasetLargePL
- Uses existing RadioNet evaluation
- Adds diffusion-specific training and sampling
- Supports all 3 conditioning methods: concat, film, cross_attn
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

# Import existing dataset and validation functions
from data.dataset_heatmap import TxLocationDatasetLarge, get_building_splits

# Import diffusion models
from models.diffusion import create_diffusion_model

# Import existing modules
sys.path.append('.')
from models.saipp_net import RadioNet

# Import existing validation helpers (we'll use them!)
from training.train_heatmap import (
    evaluate_with_avg_maps,
    compute_building_density,
    load_topk_csv,
    find_closest_topk_rank,
    load_pl_map_metrics,
    compute_predicted_radio_metrics
)


# ============================================================================
# DIFFUSION UTILITIES
# ============================================================================

class DDPMScheduler:
    """
    Denoising Diffusion Probabilistic Models (DDPM) noise scheduler
    Simplified version for training and sampling
    """
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_train_timesteps = num_train_timesteps
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def add_noise(self, original_samples, noise, timesteps):
        """
        Add noise to clean samples according to noise schedule
        q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        device = original_samples.device
        sqrt_alpha_prod = self.sqrt_alphas_cumprod.to(device)[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod.to(device)[timesteps]
        
        # Reshape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = (
            sqrt_alpha_prod * original_samples +
            sqrt_one_minus_alpha_prod * noise
        )
        
        return noisy_samples
    
    def step(self, model_output, timestep, sample):
        """
        Predict x_{t-1} from x_t and predicted noise
        """
        device = sample.device
        
        # Get parameters for this timestep
        alpha = self.alphas.to(device)[timestep]
        alpha_cumprod = self.alphas_cumprod.to(device)[timestep]
        alpha_cumprod_prev = self.alphas_cumprod_prev.to(device)[timestep]
        beta = self.betas.to(device)[timestep]
        
        # Reshape for broadcasting
        while len(alpha.shape) < len(sample.shape):
            alpha = alpha.unsqueeze(-1)
            alpha_cumprod = alpha_cumprod.unsqueeze(-1)
            alpha_cumprod_prev = alpha_cumprod_prev.unsqueeze(-1)
            beta = beta.unsqueeze(-1)
        
        # Predict x_0
        pred_original_sample = (
            sample - torch.sqrt(1 - alpha_cumprod) * model_output
        ) / torch.sqrt(alpha_cumprod)
        
        # Clip for stability
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        
        # Compute x_{t-1}
        pred_sample_direction = torch.sqrt(1 - alpha_cumprod_prev) * model_output
        pred_prev_sample = (
            torch.sqrt(alpha_cumprod_prev) * pred_original_sample +
            pred_sample_direction
        )
        
        return pred_prev_sample


class DDIMScheduler:
    """
    Denoising Diffusion Implicit Models (DDIM) scheduler
    Faster sampling with fewer steps
    """
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_train_timesteps = num_train_timesteps
        
        # Same as DDPM
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def add_noise(self, original_samples, noise, timesteps):
        """Same as DDPM"""
        device = original_samples.device
        sqrt_alpha_prod = self.sqrt_alphas_cumprod.to(device)[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod.to(device)[timesteps]
        
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = (
            sqrt_alpha_prod * original_samples +
            sqrt_one_minus_alpha_prod * noise
        )
        
        return noisy_samples
    
    def step(self, model_output, timestep, sample, eta=0.0):
        """
        DDIM deterministic sampling (eta=0) or stochastic (eta=1 recovers DDPM)
        """
        device = sample.device
        
        # Get alpha values
        alpha_cumprod = self.alphas_cumprod.to(device)[timestep]
        
        # Next timestep (for strided sampling)
        prev_timestep = timestep - (self.num_train_timesteps // 50)  # Assuming 50 inference steps
        if prev_timestep < 0:
            prev_timestep = torch.tensor(0, device=device)
        
        alpha_cumprod_prev = self.alphas_cumprod.to(device)[prev_timestep]
        
        # Reshape for broadcasting
        while len(alpha_cumprod.shape) < len(sample.shape):
            alpha_cumprod = alpha_cumprod.unsqueeze(-1)
            alpha_cumprod_prev = alpha_cumprod_prev.unsqueeze(-1)
        
        # Predict x_0
        pred_original_sample = (
            sample - torch.sqrt(1 - alpha_cumprod) * model_output
        ) / torch.sqrt(alpha_cumprod)
        
        # Clip
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        
        # Direction pointing to x_t
        pred_sample_direction = torch.sqrt(1 - alpha_cumprod_prev) * model_output
        
        # x_{t-1}
        pred_prev_sample = (
            torch.sqrt(alpha_cumprod_prev) * pred_original_sample +
            pred_sample_direction
        )
        
        return pred_prev_sample


def sample_diffusion(model, building_maps, optimize_for, scheduler, 
                     num_inference_steps=50, use_ddim=True, device='cuda'):
    """
    Sample clean heatmaps from noise using trained diffusion model
    
    Args:
        model: Trained diffusion model
        building_maps: (B, 1, H, W) building layouts
        optimize_for: List of B strings ["power" or "coverage"]
        scheduler: DDPMScheduler or DDIMScheduler
        num_inference_steps: Number of denoising steps
        use_ddim: Use DDIM (faster) vs DDPM (slower but higher quality)
        device: Device to run on
    
    Returns:
        clean_heatmaps: (B, 1, H, W) predicted PL maps
    """
    model.eval()
    batch_size = building_maps.shape[0]
    img_size = building_maps.shape[2]
    
    # Start from pure noise
    heatmap = torch.randn(batch_size, 1, img_size, img_size, device=device)
    
    # Create sampling timesteps
    if use_ddim:
        # Strided timesteps for DDIM
        step_size = scheduler.num_train_timesteps // num_inference_steps
        timesteps = list(range(0, scheduler.num_train_timesteps, step_size))
        timesteps = timesteps[:num_inference_steps]
        timesteps = torch.tensor(timesteps, device=device).flip(0)
    else:
        # All timesteps for DDPM
        timesteps = torch.arange(scheduler.num_train_timesteps - 1, -1, -1, device=device)
        if num_inference_steps < scheduler.num_train_timesteps:
            step_size = scheduler.num_train_timesteps // num_inference_steps
            timesteps = timesteps[::step_size]
    
    # Iteratively denoise
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            # Predict noise
            timestep_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            pred_noise = model(building_maps, heatmap, timestep_batch, optimize_for)
            
            # Remove noise
            heatmap = scheduler.step(pred_noise, t, heatmap)
            if not isinstance(heatmap, torch.Tensor):
                heatmap = heatmap  # Already a tensor from our simplified scheduler
    
    return heatmap


def hard_argmax(logits, building_mask=None, img_size=150):
    """Extract coordinates from heatmap via hard argmax"""
    batch_size = logits.size(0)
    device = logits.device
    h, w = logits.shape[2], logits.shape[3]
    
    if building_mask is not None:
        if building_mask.shape[2] != logits.shape[2]:
            building_mask = F.interpolate(building_mask.float(), size=(h, w), mode='nearest').bool()
        logits_masked = logits.clone()
        logits_masked[building_mask] = -1e9
    else:
        logits_masked = logits
    
    flat_logits = logits_masked.view(batch_size, -1)
    max_indices = torch.argmax(flat_logits, dim=1)
    
    y_indices = max_indices // w
    x_indices = max_indices % w
    
    if h != img_size or w != img_size:
        scale_x = img_size / w
        scale_y = img_size / h
        x_coords = x_indices.float() * scale_x
        y_coords = y_indices.float() * scale_y
    else:
        x_coords = x_indices.float()
        y_coords = y_indices.float()
    
    return y_coords, x_coords


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_one_epoch_diffusion(model, train_loader, optimizer, device, 
                              noise_scheduler, optimize_for):
    """Train diffusion model for one epoch"""
    model.train()
    
    running_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch_data in enumerate(train_loader):
        # Unpack data
        building_maps, gt_coords, target_heatmap, building_ids = batch_data
        
        building_maps = building_maps.to(device)
        target_heatmap = target_heatmap.to(device)
        
        batch_size = building_maps.shape[0]
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, noise_scheduler.num_train_timesteps,
            (batch_size,), device=device
        ).long()
        
        # Sample noise
        noise = torch.randn_like(target_heatmap)
        
        # Add noise to clean heatmap
        noisy_heatmap = noise_scheduler.add_noise(target_heatmap, noise, timesteps)
        
        # Create optimize_for list for this batch
        optimize_for_batch = [optimize_for] * batch_size
        
        # Predict noise
        optimizer.zero_grad()
        pred_noise = model(building_maps, noisy_heatmap, timesteps, optimize_for_batch)
        
        # Simple MSE loss on noise prediction
        loss = F.mse_loss(pred_noise, noise)
        
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        num_batches += 1
    
    avg_loss = running_loss / num_batches
    
    return {
        'loss': avg_loss,
        'noise_loss': avg_loss  # For logging compatibility
    }


# ============================================================================
# VALIDATION FUNCTION
# ============================================================================

def validate_one_epoch_diffusion(model, val_loader, device, epoch, save_dir,
                                 noise_scheduler, optimize_for, use_center,
                                 radionet_model=None, buildings_dir_full=None,
                                 heatmap_base_dir=None, num_inference_steps=50,
                                 use_ddim=True, num_samples=1, max_val_samples=1000):
    """
    Validate diffusion model - generates clean heatmaps and evaluates
    
    Args:
        num_samples: Number of samples to generate per building (for multi-sample evaluation)
        max_val_samples: Maximum number of validation samples to process. 
                        None = full validation set, 1000 = fast validation.
                        Default 1000 for reasonable training speed.
    """
    model.eval()
    
    running_coord_error = 0.0
    num_batches = 0
    samples_processed = 0  # Track how many samples we've processed
    
    # Metrics storage
    pixel_errors = []
    
    if optimize_for == 'power':
        pl_pcts = []
        coverage_pcts_power = []
        coverage_pcts_gt = []
    else:  # coverage
        coverage_pcts = []
        pl_pcts_coverage = []
        pl_pcts_ref = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            # Early stopping when we've processed enough samples
            if max_val_samples is not None and samples_processed >= max_val_samples:
                break
            
            # Progress update every 50 batches
            if batch_idx % 50 == 0:
                progress_msg = f"  Val: {samples_processed}"
                if max_val_samples:
                    progress_msg += f"/{max_val_samples}"
                else:
                    progress_msg += f"/{len(val_loader.dataset)}"
                print(progress_msg, end='\r', flush=True)
            
            building_maps, gt_coords, target_heatmap, building_ids = batch_data
            
            building_maps = building_maps.to(device)
            gt_coords = gt_coords.to(device)
            batch_size = building_maps.shape[0]
            samples_processed += batch_size  # Increment counter
            
            # Create optimize_for list
            optimize_for_batch = [optimize_for] * batch_size
            
            if num_samples > 1:
                # Multi-sample evaluation: generate k samples and pick best
                best_pred_coords = []
                best_metrics = []
                
                for i in range(batch_size):
                    building_single = building_maps[i:i+1]
                    gt_coord_single = gt_coords[i]
                    building_id = building_ids[i]
                    optimize_for_single = [optimize_for]
                    
                    sample_coords = []
                    sample_metrics_list = []
                    
                    # Generate k samples
                    for k in range(num_samples):
                        pred_heatmap = sample_diffusion(
                            model, building_single, optimize_for_single,
                            noise_scheduler, num_inference_steps, use_ddim, device
                        )
                        
                        # Extract coordinates
                        building_mask = (building_single > 0.1)
                        y, x = hard_argmax(pred_heatmap, building_mask, img_size=150)
                        pred_coord = torch.stack([y, x], dim=1)[0]
                        
                        # Evaluate with RadioNet
                        pred_np = pred_coord.cpu().numpy()
                        gt_np = gt_coord_single.cpu().numpy()
                        
                        sample_coords.append(pred_np)
                        
                        if radionet_model and buildings_dir_full and heatmap_base_dir:
                            metrics = evaluate_with_avg_maps(
                                pred_np, gt_np, building_id, heatmap_base_dir,
                                optimize_for, use_center, buildings_dir_full, radionet_model
                            )
                            if metrics:
                                sample_metrics_list.append(metrics)
                    
                    # Pick best sample based on optimization criterion
                    if len(sample_metrics_list) > 0:
                        if optimize_for == 'power':
                            best_idx = np.argmax([m['pl_pct'] for m in sample_metrics_list])
                        else:
                            best_idx = np.argmax([m['coverage_pct'] for m in sample_metrics_list])
                        
                        best_pred_coords.append(sample_coords[best_idx])
                        best_metrics.append(sample_metrics_list[best_idx])
                    else:
                        best_pred_coords.append(sample_coords[0])
                
                # Collect metrics from best samples
                for i in range(batch_size):
                    if i < len(best_pred_coords):
                        pred_np = best_pred_coords[i]
                        gt_np = gt_coords[i].cpu().numpy()
                        
                        pixel_error = np.sqrt(np.sum((pred_np - gt_np)**2))
                        pixel_errors.append(pixel_error)
                        
                        if i < len(best_metrics):
                            if optimize_for == 'power':
                                pl_pcts.append(best_metrics[i]['pl_pct'])
                                coverage_pcts_power.append(best_metrics[i]['coverage_pct_power'])
                                coverage_pcts_gt.append(best_metrics[i]['coverage_pct_gt'])
                            else:
                                coverage_pcts.append(best_metrics[i]['coverage_pct'])
                                pl_pcts_coverage.append(best_metrics[i]['pl_pct_coverage'])
                                pl_pcts_ref.append(best_metrics[i]['pl_pct_ref'])
            
            else:
                # Single sample evaluation (faster)
                pred_heatmaps = sample_diffusion(
                    model, building_maps, optimize_for_batch,
                    noise_scheduler, num_inference_steps, use_ddim, device
                )
                
                # Extract coordinates
                building_mask = (building_maps > 0.1)
                y, x = hard_argmax(pred_heatmaps, building_mask, img_size=150)
                pred_coords = torch.stack([y, x], dim=1)
                
                # Compute coordinate error
                coord_error = torch.mean(torch.abs(pred_coords - gt_coords))
                running_coord_error += coord_error.item()
                num_batches += 1
                
                # Per-sample metrics
                for i in range(batch_size):
                    pred_np = pred_coords[i].cpu().numpy()
                    gt_np = gt_coords[i].cpu().numpy()
                    building_id = building_ids[i]
                    
                    pixel_error = np.sqrt(np.sum((pred_np - gt_np)**2))
                    pixel_errors.append(pixel_error)
                    
                    # Evaluate with RadioNet
                    if radionet_model and buildings_dir_full and heatmap_base_dir:
                        avg_metrics = evaluate_with_avg_maps(
                            pred_np, gt_np, building_id, heatmap_base_dir,
                            optimize_for, use_center, buildings_dir_full, radionet_model
                        )
                        
                        if avg_metrics:
                            if optimize_for == 'power':
                                pl_pcts.append(avg_metrics['pl_pct'])
                                coverage_pcts_power.append(avg_metrics['coverage_pct_power'])
                                coverage_pcts_gt.append(avg_metrics['coverage_pct_gt'])
                            else:
                                coverage_pcts.append(avg_metrics['coverage_pct'])
                                pl_pcts_coverage.append(avg_metrics['pl_pct_coverage'])
                                pl_pcts_ref.append(avg_metrics['pl_pct_ref'])
    
    # Clear progress line
    print(" " * 80, end='\r', flush=True)
    
    # Compute statistics
    metrics = {
        'coord_error': running_coord_error / num_batches if num_batches > 0 else 0.0,
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
    else:
        if len(coverage_pcts) > 0:
            metrics['avg_coverage_pct'] = float(np.mean(coverage_pcts))
            metrics['median_coverage_pct'] = float(np.median(coverage_pcts))
            metrics['std_coverage_pct'] = float(np.std(coverage_pcts))
            metrics['avg_pl_pct_coverage'] = float(np.mean(pl_pcts_coverage))
            metrics['avg_pl_pct_ref'] = float(np.mean(pl_pcts_ref))
    
    return metrics


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train_diffusion_model(model, train_loader, val_loader, optimizer, scheduler_opt,
                          noise_scheduler, num_epochs, device, save_dir, optimize_for,
                          use_center, radionet_model, buildings_dir_full, heatmap_base_dir,
                          num_inference_steps=50, use_ddim=True, val_interval=5,
                          num_samples_val=1):
    """Main training loop for diffusion model"""
    
    print(f"\n{'='*70}")
    print(f"DIFFUSION TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Epochs: {num_epochs}")
    print(f"Optimize for: {optimize_for}")
    print(f"Diffusion timesteps: {noise_scheduler.num_train_timesteps}")
    print(f"Inference steps: {num_inference_steps}")
    print(f"Sampler: {'DDIM' if use_ddim else 'DDPM'}")
    print(f"Multi-sampling (val): {num_samples_val} samples")
    print(f"{'='*70}\n")
    
    # Best model tracking
    if optimize_for == 'power':
        best_pl_pct = -float('inf')
        best_pl_epoch = -1
    else:
        best_coverage_pct = -float('inf')
        best_coverage_epoch = -1
        best_pl_from_cov_pct = -float('inf')
        best_pl_from_cov_epoch = -1
    
    best_coord_error = float('inf')
    best_coord_epoch = -1
    
    history = defaultdict(list)
    
    # CSV log
    csv_path = os.path.join(save_dir, 'training_log.csv')
    csv_file = open(csv_path, 'w', newline='')
    
    if optimize_for == 'power':
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['epoch', 'train_loss', 'coord_error', 'coord_error_median',
                            'avg_pl_pct', 'median_pl_pct', 'std_pl_pct',
                            'avg_coverage_pct_power', 'avg_coverage_pct_gt', 'lr'])
    else:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['epoch', 'train_loss', 'coord_error', 'coord_error_median',
                            'avg_coverage_pct', 'median_coverage_pct', 'std_coverage_pct',
                            'avg_pl_pct_coverage', 'avg_pl_pct_ref', 'lr'])
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training
        train_metrics = train_one_epoch_diffusion(
            model, train_loader, optimizer, device,
            noise_scheduler, optimize_for
        )
        
        # Validation (every val_interval epochs, starting from epoch 0)
        if epoch % val_interval == 0 or epoch == num_epochs - 1:
            print(f"[Epoch {epoch+1}] Running validation...", flush=True)
            val_metrics = validate_one_epoch_diffusion(
                model, val_loader, device, epoch, save_dir,
                noise_scheduler, optimize_for, use_center,
                radionet_model, buildings_dir_full, heatmap_base_dir,
                num_inference_steps, use_ddim, num_samples_val,
                max_val_samples=1000  # Fast validation: only 1000 samples (~1.5 min)
            )
            
            # Save metrics
            json_path = os.path.join(save_dir, f'metrics_epoch{epoch}.json')
            with open(json_path, 'w') as f:
                combined_metrics = {**train_metrics, **val_metrics}
                json.dump(combined_metrics, f, indent=2)
        else:
            # Use previous validation metrics
            val_metrics = history.get('val_coord_error', [0])
            val_metrics = {'coord_error': val_metrics[-1] if len(val_metrics) > 0 else 0}
        
        # Scheduler step
        if scheduler_opt is not None:
            scheduler_opt.step()
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print summary
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val Error: {val_metrics.get('coord_error', 0):.2f}px", end='')
        
        if 'coord_error_median' in val_metrics:
            print(f" (med: {val_metrics['coord_error_median']:.2f}px)", end='')
        
        print(f" | LR: {current_lr:.2e} | {epoch_time:.1f}s")
        
        # Print optimization-specific metrics
        if optimize_for == 'power':
            if 'avg_pl_pct' in val_metrics:
                print(f"  PL: {val_metrics['avg_pl_pct']:.1f}%", end='')
                if 'median_pl_pct' in val_metrics:
                    print(f" (med: {val_metrics['median_pl_pct']:.1f}%)", end='')
                if 'avg_coverage_pct_gt' in val_metrics:
                    print(f" | CovGT: {val_metrics['avg_coverage_pct_gt']:.1f}%", end='')
                print()
        else:
            if 'avg_coverage_pct' in val_metrics:
                print(f"  Cov: {val_metrics['avg_coverage_pct']:.1f}%", end='')
                if 'median_coverage_pct' in val_metrics:
                    print(f" (med: {val_metrics['median_coverage_pct']:.1f}%)", end='')
                if 'avg_pl_pct_coverage' in val_metrics:
                    print(f" | PL: {val_metrics['avg_pl_pct_coverage']:.1f}%", end='')
                if 'avg_pl_pct_ref' in val_metrics:
                    print(f" | PLref: {val_metrics['avg_pl_pct_ref']:.1f}%", end='')
                print()
        
        # Update history
        for k, v in train_metrics.items():
            history[f'train_{k}'].append(v)
        for k, v in val_metrics.items():
            history[f'val_{k}'].append(v)
        
        # Write CSV
        if optimize_for == 'power':
            csv_writer.writerow([
                epoch, train_metrics['loss'],
                val_metrics.get('coord_error', 0), val_metrics.get('coord_error_median', 0),
                val_metrics.get('avg_pl_pct', 0), val_metrics.get('median_pl_pct', 0),
                val_metrics.get('std_pl_pct', 0),
                val_metrics.get('avg_coverage_pct_power', 0),
                val_metrics.get('avg_coverage_pct_gt', 0),
                current_lr
            ])
        else:
            csv_writer.writerow([
                epoch, train_metrics['loss'],
                val_metrics.get('coord_error', 0), val_metrics.get('coord_error_median', 0),
                val_metrics.get('avg_coverage_pct', 0), val_metrics.get('median_coverage_pct', 0),
                val_metrics.get('std_coverage_pct', 0),
                val_metrics.get('avg_pl_pct_coverage', 0),
                val_metrics.get('avg_pl_pct_ref', 0),
                current_lr
            ])
        csv_file.flush()
        
        # Save best models
        if optimize_for == 'power':
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
        else:
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
        
        if 'coord_error' in val_metrics and val_metrics['coord_error'] < best_coord_error:
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
    
    # Final summary
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
    parser = argparse.ArgumentParser(description='Train Diffusion Model for Tx Location')
    
    # Model
    parser.add_argument('--conditioning', type=str, default='concat',
                       choices=['concat', 'film', 'cross_attn'],
                       help='Conditioning method (Phase 1/2/3)')
    
    # Diffusion parameters
    parser.add_argument('--num_diffusion_steps', type=int, default=1000,
                       help='Number of diffusion timesteps for training')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                       help='Number of sampling steps at inference')
    parser.add_argument('--use_ddim', action='store_true',
                       help='Use DDIM sampler (faster) instead of DDPM')
    parser.add_argument('--num_samples_val', type=int, default=1,
                       help='Number of samples to generate per building at validation (1=single, >1=pick best)')
    
    # Optimization
    parser.add_argument('--optimize_for', type=str, default='coverage',
                       choices=['coverage', 'power'],
                       help='Optimize for coverage or power')
    parser.add_argument('--heatmap_type', type=str, default='radio_power',
                       help='Type of heatmap (radio_power or radio_coverage)')
    parser.add_argument('--heatmap_base_dir', type=str,
                       default='./data/unified_results',
                       help='Base directory for heatmaps')
    
    # Data
    parser.add_argument('--use_center', action='store_true',
                       help='Use center 150x150 crop')
    parser.add_argument('--augment', action='store_true',
                       help='Enable D4 augmentation')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Training
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (lower than standard training)')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--val_interval', type=int, default=5,
                       help='Run validation every N epochs')
    
    # Paths
    parser.add_argument('--buildings_dir', type=str,
                       default='./data/buildings')
    parser.add_argument('--tx_dir', type=str, default=None,
                       help='Directory with Tx images (auto-set if None)')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--radionet_path', type=str,
                       default='./checkpoints/saipp_net.pt',
                       help='Path to RadioNet checkpoint')
    
    args = parser.parse_args()
    
    # Auto-set tx_dir
    if args.tx_dir is None:
        args.tx_dir = os.path.join(args.heatmap_base_dir, f'best_by_{args.optimize_for}')
        print(f"Auto-set tx_dir to: {args.tx_dir}")
    
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
    print(f"Conditioning: {args.conditioning.upper()}")
    print(f"Image size: {img_size}x{img_size}")
    print(f"Optimize for: {args.optimize_for}")
    print(f"Heatmap type: {args.heatmap_type}")
    print(f"Diffusion steps (train): {args.num_diffusion_steps}")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Sampler: {'DDIM' if args.use_ddim else 'DDPM'}")
    print(f"Multi-sampling (val): {args.num_samples_val}")
    print(f"Augmentation: {args.augment}")
    print(f"RadioNet: {'ENABLED' if radionet_model else 'DISABLED'}")
    print(f"{'='*70}\n")
    
    # Create diffusion model
    model = create_diffusion_model(
        conditioning=args.conditioning,
        img_size=img_size
    ).to(device)
    
    # Create noise scheduler
    if args.use_ddim:
        noise_scheduler = DDIMScheduler(num_train_timesteps=args.num_diffusion_steps)
    else:
        noise_scheduler = DDPMScheduler(num_train_timesteps=args.num_diffusion_steps)
    
    # Load data
    train_ids, val_ids, test_ids = get_building_splits(args.buildings_dir)
    
    train_dataset = TxLocationDatasetLarge(
        train_ids,
        buildings_dir=args.buildings_dir,
        tx_dir=args.tx_dir,
        optimize_for=args.optimize_for,
        heatmap_type=args.heatmap_type,
        heatmap_base_dir=args.heatmap_base_dir,
        normalize_heatmap=False,  # Keep in [0,1] range
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
        normalize_heatmap=False,
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
    model, history = train_diffusion_model(
        model, train_loader, val_loader, optimizer, scheduler,
        noise_scheduler, args.num_epochs, device, args.save_dir,
        args.optimize_for, args.use_center, radionet_model,
        args.buildings_dir, args.heatmap_base_dir,
        args.num_inference_steps, args.use_ddim, args.val_interval,
        args.num_samples_val
    )


if __name__ == '__main__':
    main()
