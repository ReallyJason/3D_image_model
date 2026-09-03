"""
Tiny 2D-to-3D Neural Network Architecture (Part 14 & 15).

Architecture:
  2D Image (3, 128, 128)
       │
       ▼
  2D ConvNet Encoder
       │
       ▼
  512-dim 3D Latent Embedding
       │
       ▼
  3D Transposed Convolution Decoder
       │
       ▼
  Voxel Occupancy Probabilities (32, 32, 32)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyImageToVoxelNet(nn.Module):
    """
    Lightweight 2D-to-3D Voxel Predictor for rapid experimentation.
    """
    def __init__(self, latent_dim: int = 512, voxel_res: int = 32):
        super().__init__()
        self.voxel_res = voxel_res

        # 1. 2D Image Feature Extractor
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),   # 128 -> 64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 32 -> 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 16 -> 8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, latent_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 2. Latent to 3D seed feature volume
        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 2 * 2 * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 3. 3D Transposed Convolutional Decoder
        self.decoder = nn.Sequential(
            # 2x2x2 -> 4x4x4
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 4x4x4 -> 8x8x8
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8x8 -> 16x16x16
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16x16 -> 32x32x32
            nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  [Batch, 3, 128, 128]
        Output: Raw logits [Batch, 32, 32, 32]
        """
        feat = self.encoder(x)
        seed = self.fc_decoder(feat)
        seed_3d = seed.view(-1, 256, 2, 2, 2)
        voxels = self.decoder(seed_3d)
        return voxels.squeeze(1)

    def predict_occupancy(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Returns boolean binary occupancy grid [Batch, 32, 32, 32]."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            return probs > threshold

# -------------------------------------------------------------
# Loss Functions & 3D Metrics
# -------------------------------------------------------------

class VoxelLoss(nn.Module):
    """
    Combined BCE + Soft Dice Loss to handle extreme 3D voxel sparsity.
    """
    def __init__(self, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        cardinality = (probs + targets).sum(dim=(1, 2, 3))
        dice_loss = (1.0 - (2.0 * intersection + 1e-6) / (cardinality + 1e-6)).mean()

        return bce_loss + self.dice_weight * dice_loss

def calculate_voxel_iou(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Calculates 3D Voxel Intersection-over-Union (IoU) metric.
    """
    with torch.no_grad():
        if preds.dtype != torch.bool:
            preds_binary = (torch.sigmoid(preds) > threshold)
        else:
            preds_binary = preds
        targets_binary = (targets > threshold)

        intersection = (preds_binary & targets_binary).float().sum()
        union = (preds_binary | targets_binary).float().sum()

        if union == 0:
            return 1.0
        return float((intersection / union).item())
