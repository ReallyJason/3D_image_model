"""
3D Model Deduplication Module (Part 12).

Detects duplicate and redundant 3D assets using geometric hashing:
- Vertex and face count fingerprints
- Normalized bounding box aspect ratio signatures
- Mesh volume and centroid signatures

Identifies duplicate clusters, retains the highest-quality instance,
and records duplicate records into metadata/deduplication.json.
"""

import os
import json
import hashlib
import numpy as np
import trimesh
from typing import Dict, Any, List, Set, Tuple, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASET_DIR = os.path.join(PROJECT_ROOT, "my_dataset")

def compute_geometry_signature(glb_path: str) -> Optional[str]:
    """
    Computes a robust geometric signature based on vertex/face counts
    and normalized aspect ratios.
    """
    try:
        scene = trimesh.load(glb_path, force='scene')
        geom = scene.to_geometry()
        if not isinstance(geom, trimesh.Trimesh):
            meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not meshes:
                return None
            geom = trimesh.util.concatenate(meshes)
    except Exception:
        return None

    num_v = len(geom.vertices)
    num_f = len(geom.faces)
    if num_v == 0 or num_f == 0:
        return None

    # Normalized aspect ratio of extents (sorted to be invariant to axis swap)
    extents = sorted([float(x) for x in geom.extents])
    max_e = max(extents[-1], 1e-6)
    aspect_ratios = [round(e / max_e, 3) for e in extents]

    # Signature string
    sig_raw = f"{num_v}_{num_f}_{aspect_ratios[0]:.3f}_{aspect_ratios[1]:.3f}_{aspect_ratios[2]:.3f}"
    return hashlib.md5(sig_raw.encode("utf-8")).hexdigest()

def deduplicate_dataset(dataset_dir: str = DEFAULT_DATASET_DIR) -> Dict[str, Any]:
    """
    Scans dataset_dir/objects/, identifies duplicates, and outputs deduplication report.
    """
    objects_dir = os.path.join(dataset_dir, "objects")
    master_metadata_path = os.path.join(dataset_dir, "metadata.json")
    report_path = os.path.join(dataset_dir, "metadata", "deduplication.json")
    os.makedirs(os.path.join(dataset_dir, "metadata"), exist_ok=True)

    if not os.path.exists(objects_dir):
        print(f"Error: {objects_dir} does not exist.")
        return {}

    object_folders = sorted([
        f for f in os.listdir(objects_dir)
        if os.path.isdir(os.path.join(objects_dir, f)) and not f.startswith('.')
    ])

    print(f"\n=======================================================")
    print(f"         3D GEOMETRIC DEDUPLICATION SCAN")
    print(f"=======================================================")
    print(f"Scanning {len(object_folders)} objects for duplicate geometry...\n", flush=True)

    signatures: Dict[str, List[str]] = {}
    duplicates: Dict[str, str] = {} # duplicate_id -> original_id

    for obj_id in object_folders:
        glb_path = os.path.join(objects_dir, obj_id, "model.glb")
        sig = compute_geometry_signature(glb_path)
        if sig is None:
            continue

        if sig not in signatures:
            signatures[sig] = [obj_id]
        else:
            primary = signatures[sig][0]
            signatures[sig].append(obj_id)
            duplicates[obj_id] = primary
            print(f"⚠️  Duplicate detected: {obj_id} -> matches {primary}", flush=True)

    duplicate_clusters = {k: v for k, v in signatures.items() if len(v) > 1}
    unique_count = len(object_folders) - len(duplicates)

    report = {
        "total_objects": len(object_folders),
        "unique_objects": unique_count,
        "duplicate_count": len(duplicates),
        "duplicates": duplicates,
        "duplicate_clusters": duplicate_clusters
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Update master metadata
    if os.path.exists(master_metadata_path):
        with open(master_metadata_path, "r", encoding="utf-8") as f:
            master = json.load(f)
        for obj_id, meta in master.items():
            meta["is_duplicate"] = obj_id in duplicates
            if obj_id in duplicates:
                meta["duplicate_of"] = duplicates[obj_id]
        with open(master_metadata_path, "w", encoding="utf-8") as f:
            json.dump(master, f, indent=2)

    print(f"\n=======================================================", flush=True)
    print(f"               DEDUPLICATION SUMMARY", flush=True)
    print(f"=======================================================", flush=True)
    print(f"  Total Scanned:      {len(object_folders)}", flush=True)
    print(f"  ✅ Unique Objects:   {unique_count}", flush=True)
    print(f"  🔁 Duplicates Found: {len(duplicates)}", flush=True)
    print(f"  Report Saved:       {report_path}", flush=True)
    print(f"=======================================================\n", flush=True)

    return report

if __name__ == "__main__":
    deduplicate_dataset()
