# 3D Image Model Dataset Pipeline 🧊📸

An industrial-grade, end-to-end data pipeline for downloading, organizing, validating, deduplicating, filtering, augmenting, rendering, and partitioning 3D models for 3D generative AI and computer vision models (e.g., Multi-View Diffusion, NeRF, 3D Gaussian Splatting, Zero123, TripoSR).

---

## 🌟 Key Capabilities

1. **Standardized Dataset Organization (Part 1)**: Automatically structures 3D models into per-object directories while preserving all source provenance (Objaverse UIDs, categories, Creative Commons licenses, authors, and Sketchfab viewer links).
2. **Automated 3D Health Diagnostics (Part 2 & 11)**: Detects broken, corrupt, microscopic/gigantic, and non-polygonal (empty or spline-only) models before rendering.
3. **Realistic Data Augmentation (Part 10)**:
   - **Camera Randomization**: Azimuth/elevation jitter, variable zoom/distance, and random focal length (FOV).
   - **Lighting Randomization**: Key, fill, and rim lights with color temperature (warm tungsten, cool daylight, golden sunset, neutral) and intensity modulation.
   - **Background Randomization**: Studio white, neutral gray, soft indoor room gradient, outdoor sky-ground horizon, and transparent RGBA.
   - **Object Perturbation**: Random rotation jitter and slight off-center translation.
4. **Geometric Deduplication (Part 12)**: Uses invariant geometric fingerprinting (vertex/face signatures and aspect ratio hashes) to detect and flag duplicate re-uploads.
5. **License & Quality Filtering (Part 12)**: Automatically excludes corrupted/empty meshes, duplicates, and non-commercial licenses when needed.
6. **Dataset Partitioning (Part 12)**: Stratified generation of `train.json` (80%), `val.json` (10%), and `test.json` (10%).
7. **Hardware-Accelerated GPU Rendering**: Headless **ModernGL** offscreen rendering executing natively on Apple Silicon GPUs (M1/M2/M3/M4) in milliseconds per frame.

---

## 📁 Staged Dataset Architecture

```
my_dataset/
├── objects/                     # 3D assets and rendered views
│   ├── stair_00001/
│   │   ├── model.glb            # Original 3D mesh
│   │   ├── view_000.png         # Rendered camera angles
│   │   ├── view_001.png
│   │   └── metadata.json        # Camera poses, intrinsics, extrinsics, lighting
│   └── ...
├── metadata/                    # Catalogs and audit logs
│   ├── filtered_catalog.json    # Verified, deduplicated, clean models
│   └── deduplication.json       # Duplicate cluster mappings
├── splits/                      # ML Training partitions
│   ├── train.json               # 80% Training split
│   ├── val.json                 # 10% Validation split
│   ├── test.json                # 10% Test split
│   └── summary.json             # Stratification breakdown
├── metadata.json                # Master index of all objects
└── validation_report.json       # Full health check results
```

---

## 🚀 Quickstart

### 1. Installation

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Run the Full 6-Stage Pipeline

Executes download/organize ➔ validate ➔ deduplicate ➔ filter ➔ render (augmented) ➔ split:

```bash
python main.py --step all --limit 20 --num-views 12 --augment
```

---

## 🛠️ Pipeline Stages (Run Individually)

| Stage | Command | Description |
|---|---|---|
| **1. Organize** | `python main.py --step organize --limit 100` | Fetches UIDs & GLBs, normalizes directory structure, saves license metadata |
| **2. Validate** | `python main.py --step validate` | Tests file integrity, vertex/face count, scale, centering, and textures |
| **3. Deduplicate** | `python main.py --step deduplicate` | Scans geometric signatures and flags duplicate assets |
| **4. Filter** | `python main.py --step filter --commercial-only` | Filters out failed/duplicate models and applies license constraints |
| **5. Render** | `python main.py --step render --num-views 12 --augment` | GPU multi-view rendering with realistic lighting, backgrounds, and camera jitter |
| **6. Split** | `python main.py --step split` | Generates stratified `train.json`, `val.json`, and `test.json` splits |

---

## 📈 Scaling Progression (Part 11)

Do **not** download 500,000 models immediately:
1. **Phase 1 (Current)**: 20 – 100 objects (verify pipeline, catch parsing quirks and geometry anomalies).
2. **Phase 2**: 1,000 objects (profile deduplication, test training on a single GPU).
3. **Phase 3**: 10,000 objects (distributed rendering, filter for specific categories).
4. **Phase 4**: 100,000+ objects (full foundation model pretraining).

---

## 📜 Metadata & Camera Pose Schema

Each object's `metadata.json` embeds exact camera extrinsics and intrinsics for each view:

```json
{
  "index": 0,
  "file": "view_000.png",
  "azimuth_deg": 355.2,
  "elevation_deg": 18.4,
  "distance": 2.25,
  "fov_deg": 48.0,
  "camera_pos": [-0.19, 0.60, 2.06],
  "camera_lookat": [0.0, 0.0, 0.0],
  "lighting_preset": "warm_indoor",
  "background_preset": "room_gradient",
  "is_augmented": true,
  "world_to_camera_matrix": [
    [0.996, 0.0, 0.083, 0.02],
    [0.023, 0.96, -0.28, 0.01],
    [-0.08, 0.28, 0.956, -2.15],
    [0.0, 0.0, 0.0, 1.0]
  ],
  "camera_to_world_matrix": [...]
}
```

---

## 📄 License

The code in this repository is licensed under the MIT License. Downloaded 3D objects retain their respective Creative Commons licenses as documented in each object's `metadata.json`.
