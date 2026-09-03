"""
Inference & Mesh Reconstruction Script (Part 16).

Transforms any single 2D image into a reconstructed 3D mesh:
  2D Image Input
       │
       ▼
  Trained Neural Net
       │
       ▼
  Predicted 3D Voxels (32x32x32)
       │
       ▼
  Marching Cubes Algorithm
       │
       ▼
  Reconstructed 3D Mesh (.obj / .glb)

Usage:
    python src/inference.py --image my_dataset/objects/toy_00002/view_000.png
    python src/inference.py --from-test-set
"""

import os
import sys
import argparse
import numpy as np
import trimesh
from PIL import Image
import torch
import torchvision.transforms as T

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import TinyImageToVoxelNet
from src.dataset import DEFAULT_DATASET_DIR

def reconstruct_3d(
    image_path: str,
    checkpoint_path: str = os.path.join(PROJECT_ROOT, "checkpoints", "tiny_voxel_model.pt"),
    output_dir: str = os.path.join(PROJECT_ROOT, "reconstructions"),
    threshold: float = 0.45,
    device_str: str = "auto"
) -> str:
    """
    Predicts 3D voxels from a 2D image and extracts a 3D mesh using Marching Cubes.
    """
    os.makedirs(output_dir, exist_ok=True)

    if device_str == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        device = torch.device(device_str)

    print(f"\n=======================================================")
    print(f"       🔮 2D-TO-3D NEURAL MESH RECONSTRUCTION")
    print(f"=======================================================")
    print(f"Input Image:     {image_path}")
    print(f"Checkpoint:      {checkpoint_path}")
    print(f"Device:          {device}")
    print(f"Voxel Threshold: {threshold}")
    print(f"=======================================================\n", flush=True)

    # 1. Load Model
    model = TinyImageToVoxelNet(latent_dim=512, voxel_res=32).to(device)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded trained model from Epoch {checkpoint.get('epoch', '?')} (Val IoU: {checkpoint.get('val_iou', 0)*100:.1f}%)")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Running with initialized weights.")

    model.eval()

    # 2. Preprocess Input Image
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0).to(device) # [1, 3, 128, 128]

    # 3. Predict 3D Voxels
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy() # [32, 32, 32]

    binary_grid = probs > threshold
    occupied_voxels = int(binary_grid.sum())
    print(f"Predicted Voxel Grid: 32x32x32")
    print(f"Active Occupied Cells: {occupied_voxels} / {32**3} ({occupied_voxels / (32**3) * 100:.2f}%)")

    # 4. Marching Cubes (Voxels to Triangular Mesh)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_obj_path = os.path.join(output_dir, f"{base_name}_reconstructed.obj")
    out_glb_path = os.path.join(output_dir, f"{base_name}_reconstructed.glb")

    if occupied_voxels < 8:
        print("Warning: Prediction has too few occupied voxels to extract mesh.")
        # Fallback to single box
        binary_grid[15:17, 15:17, 15:17] = True

    try:
        # Extract smooth isosurface via Marching Cubes
        mesh = trimesh.voxel.ops.matrix_to_marching_cubes(binary_grid, pitch=1.0/32.0)
        # Center mesh at origin
        mesh.vertices -= mesh.bounding_box.centroid

        # Save OBJ and GLB formats
        mesh.export(out_obj_path)
        mesh.export(out_glb_path)

        print(f"\n✅ 3D Mesh Reconstructed Successfully!")
        print(f"  Vertices: {len(mesh.vertices):,}")
        print(f"  Faces:    {len(mesh.faces):,}")
        print(f"  Saved OBJ: {out_obj_path}")
        print(f"  Saved GLB: {out_glb_path}")
        print(f"=======================================================\n")
        return out_obj_path

    except Exception as e:
        print(f"Marching cubes extraction error: {e}")
        # Fallback to cubic voxel boxes
        vg = trimesh.voxel.VoxelGrid(binary_grid)
        mesh = vg.as_boxes()
        mesh.export(out_obj_path)
        print(f"Saved cubic voxel mesh fallback to: {out_obj_path}")
        return out_obj_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict 3D mesh from single 2D image")
    parser.add_argument("--image", type=str, default=None, help="Path to input 2D image")
    parser.add_argument("--from-test-set", action="store_true", help="Pick a random image from test split")
    parser.add_argument("--threshold", type=float, default=0.45, help="Occupancy probability threshold")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(PROJECT_ROOT, "checkpoints", "tiny_voxel_model.pt"))
    args = parser.parse_args()

    img_path = args.image
    if img_path is None or args.from_test_set:
        test_split_path = os.path.join(DEFAULT_DATASET_DIR, "splits", "test.json")
        if os.path.exists(test_split_path):
            import json, glob
            with open(test_split_path) as f:
                test_records = json.load(f)
            # Pick first record with views
            for r in test_records:
                views = glob.glob(os.path.join(DEFAULT_DATASET_DIR, r["renders_dir"], "view_*.png"))
                if views:
                    img_path = views[0]
                    break

    if not img_path or not os.path.exists(img_path):
        # Default sample search
        import glob
        views = glob.glob(os.path.join(DEFAULT_DATASET_DIR, "objects", "*", "view_000.png"))
        if views:
            img_path = views[0]
        else:
            print("Error: No test images found. Please provide --image <path>.")
            sys.exit(1)

    reconstruct_3d(
        image_path=img_path,
        checkpoint_path=args.checkpoint,
        threshold=args.threshold
    )
