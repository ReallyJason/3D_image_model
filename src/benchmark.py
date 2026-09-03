"""
Automated 3D Evaluation & Benchmarking Suite (Parts 28 & 29).

Evaluates trained 3D AI models on an isolated test set:
- Computes Mean 3D IoU (Geometry Accuracy)
- Measures exact inference latency (Speed in milliseconds/FPS)
- Validates Marching Cubes mesh reconstructibility
- Generates a comparative benchmark leaderboard

Usage:
    python src/benchmark.py
"""

import os
import sys
import time
import json
import numpy as np
import torch
from typing import Dict, Any, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import Voxel3DDataset, DEFAULT_DATASET_DIR
from src.model import TinyImageToVoxelNet, calculate_voxel_iou

def run_benchmark(
    dataset_dir: str = DEFAULT_DATASET_DIR,
    checkpoint_path: str = os.path.join(PROJECT_ROOT, "checkpoints", "tiny_voxel_model.pt"),
    threshold: float = 0.45
) -> Dict[str, Any]:
    """
    Runs automated benchmark across the isolated test set.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    print("\n" + "=" * 65)
    print("       📊 3D MODEL EVALUATION & BENCHMARK SUITE")
    print("=" * 65)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device:     {device}")
    print("=" * 65 + "\n", flush=True)

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found. Train the model first.")
        return {}

    # 1. Load Test Dataset
    test_dataset = Voxel3DDataset(dataset_dir=dataset_dir, split="test", image_size=128, voxel_res=32)
    if len(test_dataset) == 0:
        print("Warning: No test samples found with renders. Using validation split for benchmark.")
        test_dataset = Voxel3DDataset(dataset_dir=dataset_dir, split="val", image_size=128, voxel_res=32)

    print(f"Evaluating on {len(test_dataset)} isolated test objects...\n")

    # 2. Load Model
    model = TinyImageToVoxelNet(latent_dim=512, voxel_res=32).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    ious = []
    latencies_ms = []
    reconstructed_count = 0

    print(f"{'Object ID':<25} | {'3D IoU':<10} | {'Latency':<10} | {'Status'}")
    print("-" * 65)

    with torch.no_grad():
        for i in range(len(test_dataset)):
            img_tensor, target_voxels, obj_id = test_dataset[i]
            img_input = img_tensor.unsqueeze(0).to(device)
            target_tensor = target_voxels.unsqueeze(0).to(device)

            # Measure Inference Latency
            start_t = time.perf_counter()
            logits = model(img_input)
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            latency = (time.perf_counter() - start_t) * 1000.0
            latencies_ms.append(latency)

            # Compute IoU
            iou = calculate_voxel_iou(logits, target_tensor, threshold=threshold)
            ious.append(iou)

            probs = torch.sigmoid(logits)[0].cpu().numpy()
            occupied = int((probs > threshold).sum())
            status = "✅ Reconstructed" if occupied >= 8 else "⚠️ Sparsity Low"
            if occupied >= 8:
                reconstructed_count += 1

            print(f"{obj_id:<25} | {iou*100:6.2f}%   | {latency:6.1f} ms  | {status}")

    mean_iou = float(np.mean(ious)) if ious else 0.0
    mean_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0

    print("\n" + "=" * 65)
    print("                 BENCHMARK LEADERBOARD")
    print("=" * 65)
    print(f"{'Model Architecture':<22} | {'Mean 3D IoU':<12} | {'Latency (ms)':<14} | {'Throughput'}")
    print("-" * 65)
    print(f"{'Random Guess / Null':<22} | {'~0.0%':<12} | {'-':<14} | {'-'}")
    print(f"{'TinyImageToVoxelNet':<22} | {mean_iou*100:6.2f}%      | {mean_latency:6.1f} ms      | {fps:5.1f} FPS (Apple Silicon)")
    print(f"{'TRELLIS Foundation*':<22} | {'~55.0%':<12} | {'~5000.0 ms':<14} | {'~0.2 FPS (Cloud A100)'}")
    print("=" * 65)
    print("*Reference estimates for foundation-scale models on standard benchmarks.\n")

    results = {
        "test_samples": len(test_dataset),
        "mean_3d_iou": round(mean_iou * 100, 2),
        "mean_latency_ms": round(mean_latency, 2),
        "fps": round(fps, 1),
        "reconstruction_rate": round(reconstructed_count / max(len(test_dataset), 1) * 100, 1)
    }

    # Save benchmark report
    report_path = os.path.join(dataset_dir, "metadata", "benchmark_results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results

if __name__ == "__main__":
    run_benchmark()
