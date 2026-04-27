"""
Dataset for Average Map Training - Proper 16-bit PNG Handling

This dataset extends TxLocationDatasetLarge to support training on average maps
with proper 16-bit precision preservation and metadata-based denormalization.

New heatmap types:
- avg_power_free: normalized_by_free_pixels/avgPL.png
- avg_power_total: normalized_by_total_center/avgPL.png  
- avg_power_per_map: normalized_per_map/avgPL.png (with metadata)
- avg_coverage_free: normalized_by_free_pixels/avgCov.png
- avg_coverage_total: normalized_by_total_center/avgCov.png
- avg_coverage_per_map: normalized_per_map/avgCov.png (with metadata)
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


class TxLocationDatasetAvgMap(Dataset):
    """Dataset for training on average maps with proper 16-bit handling"""
    
    def __init__(self, building_ids,
                 buildings_dir="./data/buildings",
                 tx_dir=None,
                 optimize_for='coverage',
                 heatmap_type='avg_power_per_map',
                 heatmap_base_dir="./data/unified_results",
                 augment=False,
                 use_center=True):
        """
        Args:
            building_ids: List of building IDs
            buildings_dir: Directory with building images
            tx_dir: Directory with GT Tx locations (determined from optimize_for)
            optimize_for: 'coverage' or 'power' (determines which GT to use)
            heatmap_type: Type of average map to load:
                - 'avg_power_free': normalized_by_free_pixels/avgPL.png
                - 'avg_power_total': normalized_by_total_center/avgPL.png
                - 'avg_power_per_map': normalized_per_map/avgPL.png (with metadata)
                - 'avg_coverage_free': normalized_by_free_pixels/avgCov.png
                - 'avg_coverage_total': normalized_by_total_center/avgCov.png
                - 'avg_coverage_per_map': normalized_per_map/avgCov.png (with metadata)
            heatmap_base_dir: Base directory for unified results
            augment: Apply D4 augmentation
            use_center: Use center 150x150 crop (should always be True for average maps)
        """
        self.building_ids = building_ids
        self.buildings_dir = buildings_dir
        self.optimize_for = optimize_for
        self.heatmap_type = heatmap_type
        self.heatmap_base_dir = heatmap_base_dir
        self.augment = augment
        self.use_center = use_center
        self.crop_offset = 53 if use_center else 0
        self.img_size = 150 if use_center else 256
        self.to_tensor = transforms.ToTensor()
        
        # Determine tx_dir based on optimize_for if not provided
        if tx_dir is None:
            if optimize_for == 'coverage':
                self.tx_dir = os.path.join(heatmap_base_dir, 'best_by_coverage')
            else:  # power
                self.tx_dir = os.path.join(heatmap_base_dir, 'best_by_power')
        else:
            self.tx_dir = tx_dir
        
        # Determine heatmap directory and suffix based on type
        self._setup_heatmap_paths()
        
        # Validate files
        self.valid_ids = []
        print(f"[AvgMapDataset] Validating {len(building_ids)} buildings...", flush=True)
        missing_count = 0
        missing_heatmap_count = 0
        missing_metadata_count = 0
        
        for bld_id in building_ids:
            building_path = os.path.join(buildings_dir, f"{bld_id}.png")
            tx_path = os.path.join(self.tx_dir, f"{bld_id}_tx.png")
            heatmap_path = os.path.join(self.heatmap_dir, f"{bld_id}{self.heatmap_suffix}")
            
            # Check metadata for per_map normalization
            metadata_exists = True
            if 'per_map' in self.heatmap_type:
                metadata_path = os.path.join(self.heatmap_dir, f"{bld_id}_metadata.json")
                metadata_exists = os.path.exists(metadata_path)
                if not metadata_exists:
                    missing_metadata_count += 1
            
            heatmap_exists = os.path.exists(heatmap_path)
            if not heatmap_exists:
                missing_heatmap_count += 1
            
            if os.path.exists(building_path) and os.path.exists(tx_path) and heatmap_exists and metadata_exists:
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
                    elif not metadata_exists:
                        print(f"  Warning: Missing metadata for {bld_id}", flush=True)
        
        if missing_count > 5:
            print(f"  Warning: {missing_count - 5} more buildings missing...", flush=True)
        if missing_heatmap_count > 0:
            print(f"  Warning: {missing_heatmap_count} buildings missing heatmaps", flush=True)
        if missing_metadata_count > 0:
            print(f"  Warning: {missing_metadata_count} buildings missing metadata", flush=True)
        
        print(f"[AvgMapDataset] Loaded {len(self.valid_ids)} valid buildings", flush=True)
        print(f"[AvgMapDataset] Optimizing for: {optimize_for}", flush=True)
        print(f"[AvgMapDataset] GT Tx from: {self.tx_dir}", flush=True)
        print(f"[AvgMapDataset] Heatmap type: {heatmap_type}", flush=True)
        print(f"[AvgMapDataset] Heatmap dir: {self.heatmap_dir}", flush=True)
        if self.use_center:
            print(f"[AvgMapDataset] Mode: Center 150x150 crop", flush=True)
        else:
            print(f"[AvgMapDataset] Mode: Full 256x256", flush=True)
        
        if self.augment:
            print(f"[AvgMapDataset] Augmentation: ENABLED (D4 group)", flush=True)
    
    def _setup_heatmap_paths(self):
        """Setup heatmap directory and suffix based on heatmap_type"""
        if self.heatmap_type == 'avg_power_free':
            self.heatmap_dir = os.path.join(self.heatmap_base_dir, 'normalized_by_free_pixels')
            self.heatmap_suffix = '_avgPL.png'
            self.map_key = 'avgPL'
        elif self.heatmap_type == 'avg_power_total':
            self.heatmap_dir = os.path.join(self.heatmap_base_dir, 'normalized_by_total_center')
            self.heatmap_suffix = '_avgPL.png'
            self.map_key = 'avgPL'
        elif self.heatmap_type == 'avg_power_per_map':
            self.heatmap_dir = os.path.join(self.heatmap_base_dir, 'normalized_per_map')
            self.heatmap_suffix = '_avgPL.png'
            self.map_key = 'avgPL'
        elif self.heatmap_type == 'avg_coverage_free':
            self.heatmap_dir = os.path.join(self.heatmap_base_dir, 'normalized_by_free_pixels')
            self.heatmap_suffix = '_avgCov.png'
            self.map_key = 'avgCov'
        elif self.heatmap_type == 'avg_coverage_total':
            self.heatmap_dir = os.path.join(self.heatmap_base_dir, 'normalized_by_total_center')
            self.heatmap_suffix = '_avgCov.png'
            self.map_key = 'avgCov'
        elif self.heatmap_type == 'avg_coverage_per_map':
            self.heatmap_dir = os.path.join(self.heatmap_base_dir, 'normalized_per_map')
            self.heatmap_suffix = '_avgCov.png'
            self.map_key = 'avgCov'
        else:
            raise ValueError(f"Unknown heatmap_type: {self.heatmap_type}")
    
    def _load_16bit_png(self, path):
        """Load 16-bit PNG properly without losing precision"""
        img = Image.open(path)
        # Don't use .convert('L') - it converts to 8-bit!
        # Keep original bit depth
        arr = np.array(img, dtype=np.float32)
        # Convert from [0, 65535] to [0, 1]
        arr = arr / 65535.0
        return arr
    
    def _apply_augmentation(self, building_img, tx_coords, heatmap_array):
        """Apply D4 augmentation with correct coordinate transforms"""
        angle = random.choice([0, 90, 180, 270])
        max_coord = self.img_size - 1
        
        # Convert numpy array to PIL for rotation
        heatmap_img = Image.fromarray((heatmap_array * 255).astype(np.uint8))
        
        if angle == 90:
            building_img = TF.rotate(building_img, 90)
            tx_coords = np.array([tx_coords[1], max_coord - tx_coords[0]], dtype=np.float32)
            heatmap_img = TF.rotate(heatmap_img, 90)
        elif angle == 180:
            building_img = TF.rotate(building_img, 180)
            tx_coords = np.array([max_coord - tx_coords[0], max_coord - tx_coords[1]], dtype=np.float32)
            heatmap_img = TF.rotate(heatmap_img, 180)
        elif angle == 270:
            building_img = TF.rotate(building_img, 270)
            tx_coords = np.array([max_coord - tx_coords[1], tx_coords[0]], dtype=np.float32)
            heatmap_img = TF.rotate(heatmap_img, 270)
        
        if random.random() > 0.5:
            building_img = TF.hflip(building_img)
            tx_coords = np.array([tx_coords[0], max_coord - tx_coords[1]], dtype=np.float32)
            heatmap_img = TF.hflip(heatmap_img)
        
        # Convert back to numpy
        heatmap_array = np.array(heatmap_img, dtype=np.float32) / 255.0
        
        return building_img, tx_coords, heatmap_array
    
    def __len__(self):
        return len(self.valid_ids)
    
    def __getitem__(self, idx):
        building_id = self.valid_ids[idx]
        
        # Load full 256x256 building map
        building_path = os.path.join(self.buildings_dir, f"{building_id}.png")
        building_img = Image.open(building_path).convert('L')
        
        # Load Tx coordinates from one-hot image (always in 256x256 space)
        tx_path = os.path.join(self.tx_dir, f"{building_id}_tx.png")
        tx_img = np.array(Image.open(tx_path).convert('L'))
        
        # Find white pixel (255) - coordinates in 256x256 space
        tx_positions = np.where(tx_img == 255)
        if len(tx_positions[0]) == 0:
            tx_positions = np.where(tx_img == tx_img.max())
        
        tx_coords = np.array([tx_positions[0][0], tx_positions[1][0]], dtype=np.float32)
        
        # Load 16-bit average map (always 256x256 from disk)
        heatmap_path = os.path.join(self.heatmap_dir, f"{building_id}{self.heatmap_suffix}")
        heatmap_array = self._load_16bit_png(heatmap_path)  # Now in [0,1] range
        
        # For per_map: denormalize using metadata to restore per-building range
        if 'per_map' in self.heatmap_type:
            metadata_path = os.path.join(self.heatmap_dir, f"{building_id}_metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            max_val = metadata['maps'][self.map_key]['max']
            min_val = metadata['maps'][self.map_key]['min']
            # Denormalize: value_original = value_normalized * (max - min) + min
            # Note: max and min are already in [0, 1] range, so this keeps values in [0, 1]
            heatmap_array = heatmap_array * (max_val - min_val) + min_val
        
        # CRITICAL: If use_center=True, crop EVERYTHING to center 150x150 [53:203, 53:203]
        if self.use_center:
            # Crop building: 256x256 -> 150x150
            building_img = building_img.crop((53, 53, 203, 203))
            # Adjust TX coordinates: 256x256 space -> 150x150 space
            tx_coords = tx_coords - self.crop_offset
            # Crop average heatmap: 256x256 -> 150x150 (SAME region as building!)
            heatmap_array = heatmap_array[53:203, 53:203]
        
        # Apply augmentation if enabled
        if self.augment:
            building_img, tx_coords, heatmap_array = self._apply_augmentation(
                building_img, tx_coords, heatmap_array)
        
        # Convert to tensors
        building_tensor = self.to_tensor(building_img)
        tx_coords_tensor = torch.from_numpy(tx_coords)
        heatmap_tensor = torch.from_numpy(heatmap_array).unsqueeze(0)  # Add channel dim
        
        return building_tensor, tx_coords_tensor, heatmap_tensor, building_id


# Import get_building_splits from original dataset
from data.dataset_heatmap import get_building_splits
