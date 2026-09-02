# 3D Image Model Dataset Pipeline 🧊📸

An end-to-end data pipeline for downloading, organizing, validating, and rendering 3D models for 3D computer vision and generative 3D modeling (e.g., NeRF, 3D Gaussian Splatting, Multi-view Diffusion).

---

## 🌟 Features

- **Standardized Dataset Organization**: Automatically structures 3D models into clean per-object directories with full provenance tracking (Objaverse UIDs, categories, Creative Commons licensing, authors, Sketchfab source links).
- **Automated 3D Health Diagnostics**: Inspects geometry integrity, vertex/face counts, NaN coordinates, bounding box extents, and material/texture presence before rendering.
- **Hardware-Accelerated GPU Multi-View Rendering**: Uses headless **ModernGL** offscreen rendering natively on Apple Silicon GPUs (M1/M2/M3/M4) to generate 12–36+ orbit camera views per object in milliseconds.
- **Camera Pose Tracking**: Embeds exact camera extrinsics (world-to-camera, camera-to-world) and projection matrices directly into per-object `metadata.json`.

---

## 📁 Dataset Structure

```
my_dataset/
├── objects/
│   ├── stair_00001/
│   │   ├── model.glb             # 3D Mesh asset
│   │   ├── metadata.json         # Object metadata + camera poses
│   │   ├── view_000.png          # Rendered orbit views
│   │   ├── view_001.png
│   │   └── ...
│   ├── toy_00002/
│   │   ├── model.glb
│   │   └── ...
│   └── ...
├── metadata.json                 # Master dataset catalog & provenance
└── validation_report.json        # Automated health check results
```

---

## 🚀 Quickstart

### 1. Environment Setup

```bash
# Create virtual environment (Python 3.10+)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install objaverse trimesh moderngl pillow scipy certifi
```

### 2. Run the Full Pipeline

Run organization, health checks, and multi-view rendering in one command:

```bash
python run_pipeline.py --step all --limit 20 --num-views 12
```

---

## 🛠️ Pipeline Modules

### Step 1: Download & Organize (`dataset_manager.py`)
Downloads from Objaverse and populates `my_dataset/objects/<category>_<id>/model.glb` while retaining licensing and source metadata.

```bash
python run_pipeline.py --step organize --limit 50
```

### Step 2: 3D Model Health Check (`check_objects.py`)
Scans all downloaded models and flags corrupted files, non-polygonal splines, empty meshes, and extreme scale anomalies:

```bash
python check_objects.py
```

### Step 3: Multi-View Rendering (`render_objects.py`)
Renders camera orbit views around each object and computes camera matrices:

```bash
python render_objects.py --num-views 12 --resolution 512
```

---

## 📜 Metadata Schema

Each object's `metadata.json` includes:

```json
{
  "id": "stair_00001",
  "category": "stair",
  "source": "objaverse",
  "license": "CC-BY-4.0",
  "original_uid": "8476c4170df24cf5bbe6967222d1a42d",
  "file": "objects/stair_00001/model.glb",
  "validation": {
    "status": "PASS",
    "num_vertices": 19008,
    "num_faces": 14337
  },
  "views": [
    {
      "index": 0,
      "file": "view_000.png",
      "azimuth_deg": 0.0,
      "elevation_deg": 20.0,
      "camera_pos": [0.0, 0.82, 2.25],
      "world_to_camera_matrix": [...]
    }
  ]
}
```

---

## 📄 License

This repository code is licensed under the MIT License. 3D models downloaded from Objaverse retain their respective Creative Commons licenses as documented in `metadata.json`.
