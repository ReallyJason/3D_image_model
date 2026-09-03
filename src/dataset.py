"""
PyTorch Dataset for 2D-to-3D Voxel Learning (Parts 13 & 15).

Loads training pairs from dataset splits (train.json, val.json, test.json):
  Input:  2D rendered image (128x128 RGB)
  Target: 3D binary voxel occupancy grid (32x32x32)

Includes automated mesh-to-voxel conversion with disk caching for high-speed training.
"""

import os
import glob
import json
import random
import numpy as np
import trimesh
from PIL import Image
from typing import Dict, Any, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASET_DIR = os.path.join(PROJECT_ROOT, "my_dataset")

def mesh_to_voxel_grid(mesh: trimesh.Trimesh, resolution: int = 32) -> np.ndarray:
    """
    Converts a 3D polygonal mesh into a centered binary voxel grid [resolution, resolution, resolution].
    """
    center = mesh.bounding_box.centroid
    scale = 0.88 / max(np.max(mesh.extents), 1e-6)
    normalized = mesh.copy()
    normalized.vertices = (normalized.vertices - center) * scale

    pitch = 1.0 / resolution
    vox = normalized.voxelized(pitch=pitch)
    mat = vox.matrix

    grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    d, h, w = mat.shape
    sd = max(0, (resolution - d) // 2)
    sh = max(0, (resolution - h) // 2)
    sw = max(0, (resolution - w) // 2)

    ed = min(resolution, sd + d)
    eh = min(resolution, sh + h)
    ew = min(resolution, sw + w)

    grid[sd:ed, sh:eh, sw:ew] = mat[:ed-sd, :eh-sh, :ew-sw].astype(np.float32)
    return grid

class Voxel3DDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str = DEFAULT_DATASET_DIR,
        split: str = "train",
        image_size: int = 128,
        voxel_res: int = 32,
        augment_image: bool = True
    ):
        self.dataset_dir = dataset_dir
        self.split = split
        self.image_size = image_size
        self.voxel_res = voxel_res
        self.augment_image = augment_image and (split == "train")

        split_file = os.path.join(dataset_dir, "splits", f"{split}.json")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}. Run 'python main.py --step split' first.")

        with open(split_file, "r", encoding="utf-8") as f:
            self.records: List[Dict[str, Any]] = json.load(f)

        # Filter to only records that have renders and a valid model
        self.valid_records = []
        for r in self.records:
            obj_dir = os.path.join(dataset_dir, r["renders_dir"])
            glb_file = os.path.join(dataset_dir, r["model_file"])
            if os.path.exists(obj_dir) and os.path.exists(glb_file):
                views = glob.glob(os.path.join(obj_dir, "view_*.png"))
                if views:
                    r_copy = dict(r)
                    r_copy["views"] = sorted(views)
                    self.valid_records.append(r_copy)

        # Image transforms
        if self.augment_image:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self) -> int:
        return len(self.valid_records)

    def _get_voxels(self, record: Dict[str, Any]) -> np.ndarray:
        """Retrieves cached voxels or generates and caches them on disk."""
        obj_dir = os.path.join(self.dataset_dir, record["renders_dir"])
        cache_file = os.path.join(obj_dir, f"voxel_{self.voxel_res}.npy")

        if os.path.exists(cache_file):
            try:
                return np.load(cache_file)
            except Exception:
                pass

        # Generate from GLB
        glb_path = os.path.join(self.dataset_dir, record["model_file"])
        try:
            scene = trimesh.load(glb_path, force='scene')
            geom = scene.to_geometry()
            if not isinstance(geom, trimesh.Trimesh):
                meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
                geom = trimesh.util.concatenate(meshes)
            voxels = mesh_to_voxel_grid(geom, resolution=self.voxel_res)
        except Exception:
            voxels = np.zeros((self.voxel_res, self.voxel_res, self.voxel_res), dtype=np.float32)

        # Cache on disk
        np.save(cache_file, voxels)
        return voxels

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        record = self.valid_records[idx]
        obj_id = record["id"]

        # Pick random view for training, first view for val/test
        if self.split == "train":
            img_path = random.choice(record["views"])
        else:
            img_path = record["views"][0]

        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        voxels_np = self._get_voxels(record)
        voxel_tensor = torch.from_numpy(voxels_np) # [32, 32, 32]

        return img_tensor, voxel_tensor, obj_id
