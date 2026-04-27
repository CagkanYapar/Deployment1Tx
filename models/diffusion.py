"""
Diffusion Model for Radio Map Generation

Denoising Diffusion Probabilistic Model (DDPM) with UNet backbone for
generating received-power radio maps conditioned on building geometry.

Conditioning: Building map is concatenated with the noisy heatmap at
the input of each denoising step. Timestep information is injected into
encoder residual blocks.

This version includes size-mismatch guards in the decoder for robust
inference across different input sizes and batch configurations.

Note on CriterionEmbedding:
    The model includes a CriterionEmbedding module that encodes the
    optimization objective (power vs coverage) as a learnable vector.
    In all experiments reported in the paper, separate models were
    trained for each objective, making this embedding effectively
    constant within each training run. It is retained for architectural
    completeness and checkpoint compatibility.
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


# ============================================================================
# TIMESTEP EMBEDDING (Used by all phases)
# ============================================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for timesteps"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class TimestepEmbedding(nn.Module):
    """Convert timestep to embedding vector"""
    def __init__(self, dim=512):
        super().__init__()
        self.dim = dim
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(dim // 4),
            nn.Linear(dim // 4, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, timesteps):
        return self.time_embed(timesteps)


class CriterionEmbedding(nn.Module):
    """
    Learnable embeddings for 'power' vs 'coverage' optimization
    
    FIXED VERSION: Uses nn.Embedding for GPU parallelization!
    No more Python loops - processes entire batch on GPU at once.
    """
    def __init__(self, embed_dim=128):
        super().__init__()
        # Use nn.Embedding for efficient batched lookup
        # Index 0 = 'power', Index 1 = 'coverage'
        self.embedding = nn.Embedding(2, embed_dim)
        
        # Initialize with same distribution as before for compatibility
        nn.init.normal_(self.embedding.weight, mean=0, std=1)
    
    def forward(self, optimize_for_list):
        """
        Args:
            optimize_for_list: List of strings ["power", "coverage", ...]
        Returns:
            embeddings: (B, embed_dim) - processed in parallel on GPU!
        """
        # Convert string list to tensor indices (on CPU, done once)
        # 'power' -> 0, 'coverage' -> 1
        indices = torch.tensor(
            [0 if criterion == "power" else 1 for criterion in optimize_for_list],
            dtype=torch.long
        )
        
        # Move to same device as embedding weights
        indices = indices.to(self.embedding.weight.device)
        
        # Lookup embeddings (PARALLELIZED on GPU!)
        embeddings = self.embedding(indices)
        
        return embeddings


# ============================================================================
# CONCATENATION CONDITIONING 
# ============================================================================

def conv_block(in_ch, out_ch):
    """Standard conv block matching existing code"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True)
    )


class ResBlock(nn.Module):
    """Residual block with timestep injection"""
    def __init__(self, channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, channels),
            nn.GELU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x, time_emb):
        h = self.conv1(x)
        
        # Inject timestep
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        
        h = self.conv2(h)
        return x + h


class DiffusionUNet_DeepXL_Concat(nn.Module):
    """
    Phase 1: Concatenation conditioning
    Wraps DeepXL architecture for diffusion training
    """
    def __init__(self, img_size=150):
        super().__init__()
        self.img_size = img_size
        
        # Timestep embedding
        self.time_embed = TimestepEmbedding(512)
        
        # Criterion embedding (FIXED VERSION!)
        self.criterion_embed = CriterionEmbedding(128)
        
        # Encoder (input: building + noisy_heatmap = 2 channels)
        self.conv1 = conv_block(2, 64)  # Changed from 1 to 2 channels
        self.pool1 = nn.AvgPool2d(2)
        self.res1 = ResBlock(64, 512 + 128)
        
        self.conv2 = conv_block(64, 128)
        self.pool2 = nn.AvgPool2d(2)
        self.res2 = ResBlock(128, 512 + 128)
        
        self.conv3 = conv_block(128, 256)
        self.pool3 = nn.AvgPool2d(2)
        self.res3 = ResBlock(256, 512 + 128)
        
        self.conv4 = conv_block(256, 384)
        self.pool4 = nn.AvgPool2d(2)
        self.res4 = ResBlock(384, 512 + 128)
        
        self.conv5 = conv_block(384, 512)
        self.pool5 = nn.AvgPool2d(2)
        self.res5 = ResBlock(512, 512 + 128)
        
        self.conv6 = conv_block(512, 640)
        
        # Bottleneck
        self.bottleneck = conv_block(640, 768)
        
        # Decoder
        self.up6 = nn.ConvTranspose2d(768, 640, 2, stride=2)
        self.dec6 = conv_block(640 + 640, 640)
        
        self.up5 = nn.ConvTranspose2d(640, 512, 2, stride=2)
        self.dec5 = conv_block(512 + 512, 512)
        
        self.up4 = nn.ConvTranspose2d(512, 384, 2, stride=2)
        self.dec4 = conv_block(384 + 384, 384)
        
        self.up3 = nn.ConvTranspose2d(384, 256, 2, stride=2)
        self.dec3 = conv_block(256 + 256, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(128 + 128, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(64 + 64, 64)
        
        # Final output (predicts noise)
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, 1)
        )
    
    def forward(self, building, noisy_heatmap, timesteps, optimize_for):
        """
        Args:
            building: (B, 1, H, W) building layout
            noisy_heatmap: (B, 1, H, W) noisy PL map
            timesteps: (B,) timestep indices
            optimize_for: List of B strings ["power" or "coverage"]
        
        Returns:
            pred_noise: (B, 1, H, W) predicted noise
        """
        # Get embeddings (NOW FULLY PARALLELIZED!)
        t_emb = self.time_embed(timesteps)  # (B, 512)
        c_emb = self.criterion_embed(optimize_for)  # (B, 128) - GPU parallel!
        combined_emb = torch.cat([t_emb, c_emb], dim=1)  # (B, 640)
        
        # Concatenate inputs
        x = torch.cat([building, noisy_heatmap], dim=1)  # (B, 2, H, W)
        
        # Encoder with skip connections
        e1 = self.conv1(x)  # 150x150
        e1 = self.res1(e1, combined_emb)
        x = self.pool1(e1)  # 75x75
        
        e2 = self.conv2(x)
        e2 = self.res2(e2, combined_emb)
        x = self.pool2(e2)  # 37x37 -> 36x36 after pool
        
        e3 = self.conv3(x)
        e3 = self.res3(e3, combined_emb)
        x = self.pool3(e3)  # 18x18
        
        e4 = self.conv4(x)
        e4 = self.res4(e4, combined_emb)
        x = self.pool4(e4)  # 9x9
        
        e5 = self.conv5(x)
        e5 = self.res5(e5, combined_emb)
        x = self.pool5(e5)  # 4x4
        
        e6 = self.conv6(x)
        
        # Bottleneck
        x = self.bottleneck(e6)
        
        # Decoder with skip connections
        x = self.up6(x)
        # Handle size mismatch
        if x.shape[2:] != e6.shape[2:]:
            x = F.interpolate(x, size=e6.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, e6], dim=1)
        x = self.dec6(x)
        
        x = self.up5(x)
        if x.shape[2:] != e5.shape[2:]:
            x = F.interpolate(x, size=e5.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, e5], dim=1)
        x = self.dec5(x)
        
        x = self.up4(x)
        if x.shape[2:] != e4.shape[2:]:
            x = F.interpolate(x, size=e4.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, e4], dim=1)
        x = self.dec4(x)
        
        x = self.up3(x)
        if x.shape[2:] != e3.shape[2:]:
            x = F.interpolate(x, size=e3.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, e3], dim=1)
        x = self.dec3(x)
        
        x = self.up2(x)
        if x.shape[2:] != e2.shape[2:]:
            x = F.interpolate(x, size=e2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x)
        
        x = self.up1(x)
        if x.shape[2:] != e1.shape[2:]:
            x = F.interpolate(x, size=e1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x)
        
        # Predict noise
        pred_noise = self.final(x)
        
        return pred_noise
        x = self.up3(x)  # 36x36
        if x.shape[2:] != e3.shape[2:]:
            x = F.interpolate(x, size=e3.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, e3], dim=1)
        x = self.dec3(x)
        
        x = self.up2(x)  # 72x72 -> need 75x75
        if x.shape[2:] != e2.shape[2:]:
            x = F.interpolate(x, size=e2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x)
        
        x = self.up1(x)  # 150x150
        if x.shape[2:] != e1.shape[2:]:
            x = F.interpolate(x, size=e1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x)
        
        # Predict noise
        pred_noise = self.final(x)
        
        return pred_noise





def create_diffusion_model(conditioning="concat", img_size=150):
    """Create diffusion model.

    Args:
        conditioning: Only "concat" is supported (used in paper).
        img_size: Spatial dimension (default 150 for center crop).

    Returns:
        model: DiffusionUNet_DeepXL_Concat instance
    """
    if conditioning != "concat":
        raise ValueError(f"Only concat conditioning is supported. Got: {conditioning}")

    model = DiffusionUNet_DeepXL_Concat(img_size=img_size)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Diffusion] Concat model created (img_size={img_size}):")
    print(f"  Parameters: {n_params:,}")
    return model
