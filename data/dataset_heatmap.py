"""
Large-Scale Tx Location Dataset with Physical Heatmaps Support

NEW: Supports loading physical radio maps instead of just coordinates
- Radio coverage maps (best_by_coverage)
- Radio power maps (best_by_power)  
- Average coverage maps (normalized_by_free_pixels, normalized_by_total_center)
"""

import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
import json

class TxLocationDatasetLarge(Dataset):
    """Dataset with optional center crop, D4 augmentation, and physical heatmaps"""
    
    def __init__(self, building_ids,
                 buildings_dir="./data/buildings",
                 tx_dir="./data/unified_results/best_by_power",
                 optimize_for='coverage',
                 heatmap_type='gaussian',
                 heatmap_base_dir="./data/unified_results",
                 normalize_heatmap=False,
                 augment=False,
                 use_center=False):
        """
        Args:
            building_ids: List of building IDs
            buildings_dir: Directory with building images
            tx_dir: Directory with one-hot Tx images (for coordinates)
            heatmap_type: Type of heatmap to load
                - 'gaussian': Generate Gaussian on-the-fly (original behavior)
                - 'radio_coverage': Load from best_by_coverage/buildingID_PL.png
                - 'radio_power': Load from best_by_power/buildingID_PL.png
                - 'avg_coverage_free': Load from normalized_by_free_pixels/buildingID_avgCov.png
                - 'avg_coverage_total': Load from normalized_by_total_center/buildingID_avgCov.png
            heatmap_base_dir: Base directory for unified_resultsThr4
            normalize_heatmap: If True, normalize loaded heatmaps to [0,1]
            augment: Apply 8x D4 augmentation
            use_center: Use center 150x150 crop (True) or full 256x256 (False)
        """
        self.building_ids = building_ids
        self.buildings_dir = buildings_dir
        self.tx_dir = tx_dir
        self.heatmap_type = heatmap_type
        self.heatmap_base_dir = heatmap_base_dir
        self.normalize_heatmap = normalize_heatmap
        self.augment = augment
        self.use_center = use_center
        self.crop_offset = 53 if use_center else 0
        self.img_size = 150 if use_center else 256
        self.to_tensor = transforms.ToTensor()
        
        # Determine heatmap directory based on type
        self.heatmap_dir = None
        self.heatmap_suffix = None
        if heatmap_type == 'radio_coverage':
            self.heatmap_dir = os.path.join(heatmap_base_dir, 'best_by_coverage')
            self.heatmap_suffix = '_PL.png'
        elif heatmap_type == 'radio_power':
            self.heatmap_dir = os.path.join(heatmap_base_dir, 'best_by_power')
            self.heatmap_suffix = '_PL.png'
        elif heatmap_type == 'avg_coverage_free':
            self.heatmap_dir = os.path.join(heatmap_base_dir, 'normalized_by_free_pixels')
            self.heatmap_suffix = '_avgCov.png'
        elif heatmap_type == 'avg_power_free':
            self.heatmap_dir = os.path.join(heatmap_base_dir, 'normalized_by_free_pixels')
            self.heatmap_suffix = '_avgPL.png'
        elif heatmap_type == 'avg_coverage_total':
            self.heatmap_dir = os.path.join(heatmap_base_dir, 'normalized_by_total_center')
            self.heatmap_suffix = '_avgCov.png'
        elif heatmap_type == 'avg_power_total':
            self.heatmap_dir = os.path.join(heatmap_base_dir, 'normalized_by_total_center')
            self.heatmap_suffix = '_avgPL.png'
        elif heatmap_type == 'avg_coverage_per_map':
            self.heatmap_dir = os.path.join(heatmap_base_dir, 'normalized_per_map')
            self.heatmap_suffix = '_avgCov.png'
        elif heatmap_type == 'avg_power_per_map':
            self.heatmap_dir = os.path.join(heatmap_base_dir, 'normalized_per_map')
            self.heatmap_suffix = '_avgPL.png'
        # NONORM variants (skip denormalization, stay [0,1])
        elif heatmap_type == 'avg_coverage_per_map_nonorm':
            self.heatmap_dir = os.path.join(heatmap_base_dir, 'normalized_per_map')
            self.heatmap_suffix = '_avgCov.png'
        elif heatmap_type == 'avg_power_per_map_nonorm':
            self.heatmap_dir = os.path.join(heatmap_base_dir, 'normalized_per_map')
            self.heatmap_suffix = '_avgPL.png'
        # NORM11 variants (WITH denormalization, transform to [-1,1])
        elif heatmap_type == 'avg_coverage_per_map_norm11':
            self.heatmap_dir = os.path.join(heatmap_base_dir, 'normalized_per_map')
            self.heatmap_suffix = '_avgCov.png'
        elif heatmap_type == 'avg_power_per_map_norm11':
            self.heatmap_dir = os.path.join(heatmap_base_dir, 'normalized_per_map')
            self.heatmap_suffix = '_avgPL.png'
        # NORM11_NONORM variants (skip denormalization, transform to [-1,1])
        elif heatmap_type == 'avg_coverage_per_map_norm11_nonorm':
            self.heatmap_dir = os.path.join(heatmap_base_dir, 'normalized_per_map')
            self.heatmap_suffix = '_avgCov.png'
        elif heatmap_type == 'avg_power_per_map_norm11_nonorm':
            self.heatmap_dir = os.path.join(heatmap_base_dir, 'normalized_per_map')
            self.heatmap_suffix = '_avgPL.png'
        
        # Validate files
        self.valid_ids = []
        print(f"[Dataset] Validating {len(building_ids)} buildings...", flush=True)
        missing_count = 0
        missing_heatmap_count = 0
        
        for bld_id in building_ids:
            building_path = os.path.join(buildings_dir, f"{bld_id}.png")
            tx_path = os.path.join(tx_dir, f"{bld_id}_tx.png")
            
            # Check if heatmap exists (if not using gaussian)
            heatmap_exists = True
            if heatmap_type != 'gaussian':
                heatmap_path = os.path.join(self.heatmap_dir, f"{bld_id}{self.heatmap_suffix}")
                heatmap_exists = os.path.exists(heatmap_path)
                if not heatmap_exists:
                    missing_heatmap_count += 1
            
            if os.path.exists(building_path) and os.path.exists(tx_path) and heatmap_exists:
                self.valid_ids.append(bld_id)
            else:
                missing_count += 1
                if missing_count <= 5:
                    if not os.path.exists(building_path):
                        print(f"  Warning: Missing building for {bld_id}", flush=True)
                    elif not os.path.exists(tx_path):
                        print(f"  Warning: Missing tx for {bld_id}", flush=True)
                    elif not heatmap_exists:
                        print(f"  Warning: Missing heatmap for {bld_id}", flush=True)
        
        if missing_count > 5:
            print(f"  Warning: {missing_count - 5} more buildings missing...", flush=True)
        if missing_heatmap_count > 0:
            print(f"  Warning: {missing_heatmap_count} buildings missing heatmaps", flush=True)
        
        print(f"[Dataset] Loaded {len(self.valid_ids)} valid buildings", flush=True)
        print(f"[Dataset] Optimizing for: {optimize_for}", flush=True)
        print(f"[Dataset] GT Tx from: {self.tx_dir}", flush=True)
        print(f"[Dataset] Heatmap type: {heatmap_type}", flush=True)
        print(f"[Dataset] Normalize heatmap: {normalize_heatmap}", flush=True)
        if self.use_center:
            print(f"[Dataset] Mode: Center 150x150 crop", flush=True)
        else:
            print(f"[Dataset] Mode: Full 256x256", flush=True)
        
        if self.augment:
            print(f"[Dataset] Augmentation: ENABLED (8x D4 group)", flush=True)
    
    def _load_16bit_png(self, path):
        """Load 16-bit PNG and normalize to [0, 1]"""
        img = Image.open(path)
        # Keep original bit depth, don't convert to 8-bit
        arr = np.array(img, dtype=np.float32)
        # Normalize from [0, 65535] to [0, 1]
        arr = arr / 65535.0
        return arr
    
    def _apply_augmentation(self, building_img, tx_coords, heatmap_img=None):
        """Apply D4 augmentation with correct coordinate transforms"""
        angle = random.choice([0, 90, 180, 270])
        max_coord = self.img_size - 1
        
        if angle == 90:
            building_img = TF.rotate(building_img, 90)
            tx_coords = np.array([tx_coords[1], max_coord - tx_coords[0]], dtype=np.float32)
            if heatmap_img is not None:
                heatmap_img = TF.rotate(heatmap_img, 90)
        elif angle == 180:
            building_img = TF.rotate(building_img, 180)
            tx_coords = np.array([max_coord - tx_coords[0], max_coord - tx_coords[1]], dtype=np.float32)
            if heatmap_img is not None:
                heatmap_img = TF.rotate(heatmap_img, 180)
        elif angle == 270:
            building_img = TF.rotate(building_img, 270)
            tx_coords = np.array([max_coord - tx_coords[1], tx_coords[0]], dtype=np.float32)
            if heatmap_img is not None:
                heatmap_img = TF.rotate(heatmap_img, 270)
        
        if random.random() > 0.5:
            building_img = TF.hflip(building_img)
            tx_coords = np.array([tx_coords[0], max_coord - tx_coords[1]], dtype=np.float32)
            if heatmap_img is not None:
                heatmap_img = TF.hflip(heatmap_img)
        
        return building_img, tx_coords, heatmap_img
    
    def __len__(self):
        return len(self.valid_ids)
    
    def __getitem__(self, idx):
        building_id = self.valid_ids[idx]
        
        # Load full 256x256 building map
        building_path = os.path.join(self.buildings_dir, f"{building_id}.png")
        building_img = Image.open(building_path).convert('L')  # Always 256x256
        
        # Load Tx coordinates from one-hot image (always in 256x256 space)
        tx_path = os.path.join(self.tx_dir, f"{building_id}_tx.png")
        tx_img = np.array(Image.open(tx_path).convert('L'))
        
        # Find white pixel (255) - coordinates in 256x256 space
        tx_positions = np.where(tx_img == 255)
        if len(tx_positions[0]) == 0:
            tx_positions = np.where(tx_img == tx_img.max())
        
        tx_coords = np.array([tx_positions[0][0], tx_positions[1][0]], dtype=np.float32)
        
        # Load physical heatmap if not using gaussian (ALWAYS 256x256 from disk!)
        heatmap_img = None
        if self.heatmap_type != 'gaussian':
            heatmap_path = os.path.join(self.heatmap_dir, f"{building_id}{self.heatmap_suffix}")
            
            # Per_map types use 16-bit PNGs
            if 'per_map' in self.heatmap_type:
                # Load 16-bit PNG as numpy array (normalized to [0,1])
                heatmap_array = self._load_16bit_png(heatmap_path)
                
                # Determine if we need denormalization
                # Skip denorm for: *_nonorm and *_norm11_nonorm
                # Do denorm for: base per_map and *_norm11 (without nonorm)
                skip_denorm = 'nonorm' in self.heatmap_type
                
                if not skip_denorm:
                    # Load metadata and denormalize
                    metadata_path = os.path.join(self.heatmap_dir, f"{building_id}_metadata.json")
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Determine which map key to use
                    if 'avgCov' in self.heatmap_suffix:
                        map_key = 'avgCov'
                    else:
                        map_key = 'avgPL'
                    
                    max_val = metadata['maps'][map_key]['max']
                    min_val = metadata['maps'][map_key]['min']
                    
                    # Denormalize: value_original = value_normalized * (max - min) + min
                    # Metadata min/max are in [0, 1] range, so this keeps values in [0, 1]
                    heatmap_array = heatmap_array * (max_val - min_val) + min_val
                
                # Convert to PIL image for cropping/augmentation compatibility
                # Scale to 0-255 for PIL (works for both [0,1] and will be transformed later for [-1,1])
                heatmap_img = Image.fromarray((heatmap_array * 255).astype(np.uint8))
            else:
                # Standard 8-bit PNG loading
                heatmap_img = Image.open(heatmap_path).convert('L')  # Always 256x256
        
        # CRITICAL: If use_center=True, crop EVERYTHING to center 150x150 [53:203, 53:203]
        if self.use_center:
            # Crop building: 256x256 -> 150x150
            building_img = building_img.crop((53, 53, 203, 203))
            # Adjust TX coordinates: 256x256 space -> 150x150 space
            tx_coords = tx_coords - self.crop_offset
            # Crop physical heatmap: 256x256 -> 150x150 (SAME region as building!)
            if heatmap_img is not None:
                heatmap_img = heatmap_img.crop((53, 53, 203, 203))
        
        # Apply augmentation if enabled
        if self.augment:
            building_img, tx_coords, heatmap_img = self._apply_augmentation(
                building_img, tx_coords, heatmap_img)
        
        # Convert to tensor
        building_tensor = self.to_tensor(building_img)
        tx_coords_tensor = torch.from_numpy(tx_coords)
        
        # Process heatmap
        if self.heatmap_type == 'gaussian':
            # Return None, will be generated in training loop
            heatmap_tensor = None
        else:
            # Convert loaded heatmap to tensor
            heatmap_array = np.array(heatmap_img).astype(np.float32)
            
            # Normalize if requested
            if self.normalize_heatmap:
                max_val = heatmap_array.max()
                if max_val > 0:
                    heatmap_array = heatmap_array / max_val
            else:
                # Keep in 0-255 range but as float
                heatmap_array = heatmap_array / 255.0
            
            # Transform to [-1,1] if requested (for norm11 variants)
            if 'norm11' in self.heatmap_type:
                heatmap_array = heatmap_array * 2.0 - 1.0
            
            heatmap_tensor = torch.from_numpy(heatmap_array).unsqueeze(0)  # Add channel dim
        
        if heatmap_tensor is not None:
            return building_tensor, tx_coords_tensor, heatmap_tensor, building_id
        else:
            return building_tensor, tx_coords_tensor, building_id


def get_building_splits(buildings_dir="./data/buildings",
                       train_ratio=0.8, val_ratio=0.1, seed=42):
    """Get train/val/test splits"""
    building_ids = []
    for fname in os.listdir(buildings_dir):
        if fname.endswith('.png'):
            try:
                bld_id = int(fname.replace('.png', ''))
                building_ids.append(bld_id)
            except:
                continue
    
    building_ids = sorted(building_ids)
    n_total = len(building_ids)
    print(f"[Splits] Found {n_total} buildings", flush=True)
    
    # Shuffle with seed
    np.random.seed(seed)
    np.random.shuffle(building_ids)
    
    # Split
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_ids = building_ids[:n_train]
    val_ids = building_ids[n_train:n_train+n_val]
    test_ids = building_ids[n_train+n_val:]
    
    print(f"[Splits] Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}", flush=True)
    
    return train_ids, val_ids, test_ids
