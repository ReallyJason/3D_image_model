"""
Training Script for Tiny 2D-to-3D Voxel Model (Part 14 & 15).

Trains a lightweight neural network to learn the relationship between
a single 2D image and 3D voxel geometry.

Usage:
    python src/train.py --epochs 15 --batch-size 8 --lr 0.0005
"""

import os
import sys
import argparse
import time
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import Voxel3DDataset, DEFAULT_DATASET_DIR
from src.model import TinyImageToVoxelNet, VoxelLoss, calculate_voxel_iou

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def train_model(
    dataset_dir: str = DEFAULT_DATASET_DIR,
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 0.0005,
    checkpoint_dir: str = os.path.join(PROJECT_ROOT, "checkpoints")
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = get_device()

    print("\n" + "=" * 65)
    print("       🧠 TRAINING TINY 2D-TO-3D VOXEL AI MODEL")
    print("=" * 65)
    print(f"Device:       {device}")
    print(f"Epochs:       {epochs}")
    print(f"Batch Size:   {batch_size}")
    print(f"Learning Rate:{lr}")
    print("=" * 65 + "\n", flush=True)

    # 1. Datasets & Loaders
    print("Loading Training & Validation Splits...")
    train_dataset = Voxel3DDataset(dataset_dir=dataset_dir, split="train", image_size=128, voxel_res=32)
    val_dataset = Voxel3DDataset(dataset_dir=dataset_dir, split="val", image_size=128, voxel_res=32)

    if len(train_dataset) == 0:
        print("Error: No training samples found with renders. Run 'python main.py --step render' first.")
        return

    print(f"  🚂 Train Objects: {len(train_dataset)}")
    print(f"  🔍 Val Objects:   {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 2. Model, Loss, Optimizer
    model = TinyImageToVoxelNet(latent_dim=512, voxel_res=32).to(device)
    criterion = VoxelLoss(dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_iou = 0.0
    best_checkpoint_path = os.path.join(checkpoint_dir, "tiny_voxel_model.pt")

    # 3. Training Loop
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_iou = 0.0

        for images, targets, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_iou += calculate_voxel_iou(logits, targets) * images.size(0)

        scheduler.step()
        train_loss /= len(train_dataset)
        train_iou /= len(train_dataset)

        # Validation Step
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                logits = model(images)
                loss = criterion(logits, targets)

                val_loss += loss.item() * images.size(0)
                val_iou += calculate_voxel_iou(logits, targets) * images.size(0)

        if len(val_dataset) > 0:
            val_loss /= len(val_dataset)
            val_iou /= len(val_dataset)

        # Save Best Model
        saved_str = ""
        if val_iou >= best_val_iou:
            best_val_iou = val_iou
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_iou": val_iou,
                "val_loss": val_loss
            }, best_checkpoint_path)
            saved_str = "⭐ [Saved Best]"

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou*100:.1f}% | "
            f"Val Loss: {val_loss:.4f} | Val IoU: {val_iou*100:.1f}% {saved_str}",
            flush=True
        )

    total_time = time.time() - start_time
    print(f"\n=======================================================")
    print(f"                 TRAINING COMPLETE")
    print(f"=======================================================")
    print(f"  Total Training Time: {total_time:.1f}s")
    print(f"  Best Validation IoU: {best_val_iou * 100:.2f}%")
    print(f"  Checkpoint Saved:    {best_checkpoint_path}")
    print(f"=======================================================\n", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train tiny 2D to 3D voxel neural network")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR, help="Dataset directory")
    args = parser.parse_args()

    train_model(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
