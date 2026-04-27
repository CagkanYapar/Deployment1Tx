"""
Advanced Loss Functions for Radio Map Prediction
Based on:
- DA-cGAN (Liu et al., CVPRW 2020)
- SIP2Net (Lu et al., ICASSP 2025)
"""

import torch
import torch.nn.functional as F
import numpy as np

def compute_coverage_binary_loss(pred, target, scale=10.0):
    """
    Coverage-aware loss: Emphasizes getting coverage structure right.
    
    Args:
        pred: Predicted PL map [B, 1, H, W]
        target: Target PL map [B, 1, H, W]
        scale: Steepness of sigmoid (higher = closer to hard threshold)
               - scale=1: Very soft, gradual transition
               - scale=10: Moderate (default)
               - scale=100: Very steep, almost hard threshold
    
    Returns:
        Loss value
    """
    # Binary coverage masks (which pixels should be covered)
    covered_gt = (target > 0).float()
    
    # Soft version of prediction for differentiability
    covered_pred = torch.sigmoid(pred * scale)
    
    # Binary cross-entropy on coverage structure
    coverage_loss = F.binary_cross_entropy(covered_pred, covered_gt)
    
    return coverage_loss


def compute_ssim_loss(pred, target, window_size=11, reduction='mean'):
    """
    Compute SSIM loss (1 - SSIM).
    Structural Similarity Index Loss from SIP2Net and DA-cGAN.

    Args:
        pred: predicted heatmap [B, C, H, W]
        target: target heatmap [B, C, H, W]
        window_size: size of Gaussian window (default: 11)
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Create Gaussian window
    sigma = 1.5
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2 / (2 * sigma**2)) 
                          for x in range(window_size)])
    gauss = gauss / gauss.sum()

    window_1d = gauss.unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = window_2d.expand(pred.size(1), 1, window_size, window_size).contiguous()
    window = window.to(pred.device)

    mu1 = F.conv2d(pred, window, padding=window_size//2, groups=pred.size(1))
    mu2 = F.conv2d(target, window, padding=window_size//2, groups=target.size(1))

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size//2, groups=pred.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size//2, groups=target.size(1)) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size//2, groups=pred.size(1)) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if reduction == 'mean':
        return 1 - ssim_map.mean()
    else:
        return 1 - ssim_map


def compute_ms_ssim_loss(pred, target, weights=None):
    """
    Multi-Scale SSIM loss from DA-cGAN.
    Computes SSIM at multiple scales.
    """
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    levels = min(len(weights), 3)
    mssim = []

    for i in range(levels):
        ssim_val = 1 - compute_ssim_loss(pred, target, window_size=11, reduction='mean')
        mssim.append(ssim_val)

        if i < levels - 1:
            pred = F.avg_pool2d(pred, kernel_size=2, stride=2)
            target = F.avg_pool2d(target, kernel_size=2, stride=2)

    ms_ssim = sum([weights[i] * mssim[i] for i in range(levels)])
    return 1 - ms_ssim


def compute_tv_loss(pred):
    """
    Total Variation loss from DA-cGAN.
    Encourages spatial smoothness.
    """
    diff_i = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    diff_j = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    tv_loss = torch.mean(diff_i) + torch.mean(diff_j)
    return tv_loss


def compute_gdl_loss(pred, target):
    """
    Gradient Difference Loss from SIP2Net.
    Emphasizes high-frequency details and sharpness.
    """
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]

    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]

    diff_dy = torch.abs(pred_dy - target_dy)
    diff_dx = torch.abs(pred_dx - target_dx)

    gdl = torch.mean(diff_dy) + torch.mean(diff_dx)
    return gdl


def compute_ms_gsim_loss(pred, target, scales=3):
    """
    Multi-Scale Gradient Similarity (MS-GSIM) loss from DA-cGAN.
    Custom loss for radio propagation patterns using gradient cosine similarity.
    """
    def compute_gradients(x):
        """Compute gradients using Sobel filters"""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=x.dtype, device=x.device).view(1, 1, 3, 3)

        sobel_x = sobel_x.repeat(x.size(1), 1, 1, 1)
        sobel_y = sobel_y.repeat(x.size(1), 1, 1, 1)

        grad_x = F.conv2d(x, sobel_x, padding=1, groups=x.size(1))
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=x.size(1))

        return grad_x, grad_y

    total_loss = 0.0

    for scale in range(scales):
        pred_gx, pred_gy = compute_gradients(pred)
        target_gx, target_gy = compute_gradients(target)

        pred_grad_mag = torch.sqrt(pred_gx**2 + pred_gy**2 + 1e-8)
        target_grad_mag = torch.sqrt(target_gx**2 + target_gy**2 + 1e-8)

        cos_sim_x = (pred_gx * target_gx) / (pred_grad_mag * target_grad_mag + 1e-8)
        cos_sim_y = (pred_gy * target_gy) / (pred_grad_mag * target_grad_mag + 1e-8)

        gsim = torch.mean(cos_sim_x) + torch.mean(cos_sim_y)
        total_loss += (1 - gsim / 2.0)

        if scale < scales - 1:
            pred = F.avg_pool2d(pred, kernel_size=2, stride=2)
            target = F.avg_pool2d(target, kernel_size=2, stride=2)

    return total_loss / scales


def compute_focal_l1_loss(pred, target, alpha=2.0):
    """
    Focal Shifted L1 loss from DA-cGAN.
    Assigns higher weights to pixels with stronger signal (higher values).
    """
    l1_dist = torch.abs(pred - target)
    weights = torch.pow(target + 1e-8, alpha)
    weights = weights / (torch.mean(weights) + 1e-8)
    focal_l1 = torch.mean(weights * l1_dist)
    return focal_l1


def compute_multi_loss(pred, target, loss_weights):
    """
    Compute multiple loss components and combine them.

    Args:
        pred: predicted heatmap
        target: target heatmap
        loss_weights: dict with keys 'l2', 'l1', 'ssim', 'ms_ssim', 'tv', 'gdl', 'ms_gsim', 'focal'

    Returns:
        total_loss: combined loss
        loss_dict: dictionary of individual loss components
    """
    total_loss = 0.0
    loss_dict = {}

    if loss_weights.get('l2', 0) > 0:
        l2_loss = F.mse_loss(pred, target)
        loss_dict['l2'] = l2_loss.item()
        total_loss += loss_weights['l2'] * l2_loss

    if loss_weights.get('l1', 0) > 0:
        l1_loss = F.l1_loss(pred, target)
        loss_dict['l1'] = l1_loss.item()
        total_loss += loss_weights['l1'] * l1_loss

    if loss_weights.get('ssim', 0) > 0:
        ssim_loss = compute_ssim_loss(pred, target)
        loss_dict['ssim'] = ssim_loss.item()
        total_loss += loss_weights['ssim'] * ssim_loss

    if loss_weights.get('ms_ssim', 0) > 0:
        ms_ssim_loss = compute_ms_ssim_loss(pred, target)
        loss_dict['ms_ssim'] = ms_ssim_loss.item()
        total_loss += loss_weights['ms_ssim'] * ms_ssim_loss

    if loss_weights.get('tv', 0) > 0:
        tv_loss = compute_tv_loss(pred)
        loss_dict['tv'] = tv_loss.item()
        total_loss += loss_weights['tv'] * tv_loss

    if loss_weights.get('gdl', 0) > 0:
        gdl_loss = compute_gdl_loss(pred, target)
        loss_dict['gdl'] = gdl_loss.item()
        total_loss += loss_weights['gdl'] * gdl_loss

    if loss_weights.get('ms_gsim', 0) > 0:
        ms_gsim_loss = compute_ms_gsim_loss(pred, target)
        loss_dict['ms_gsim'] = ms_gsim_loss.item()
        total_loss += loss_weights['ms_gsim'] * ms_gsim_loss

    if loss_weights.get('focal', 0) > 0:
        focal_loss = compute_focal_l1_loss(pred, target)
        loss_dict['focal'] = focal_loss.item()
        total_loss += loss_weights['focal'] * focal_loss
        
    if loss_weights.get('coverage_binary', 0) > 0:
        cov_loss = compute_coverage_binary_loss(pred, target)
        loss_dict['coverage_binary'] = cov_loss.item()
        total_loss += loss_weights['coverage_binary'] * cov_loss
        

    return total_loss, loss_dict
