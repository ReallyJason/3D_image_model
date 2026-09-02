"""
3D Model Health & Validation Diagnostics.

Detects corrupted files, missing geometry, NaN coordinates, and scale/texture anomalies.
"""

import os
import json
import argparse
import numpy as np
import trimesh
from typing import Dict, Any, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASET_DIR = os.path.join(PROJECT_ROOT, "my_dataset")

def validate_model(file_path: str) -> Dict[str, Any]:
    """
    Runs full diagnostics on a 3D model file.
    Returns a dictionary of metrics, issues, and validation status ('PASS', 'WARN', 'FAIL').
    """
    result: Dict[str, Any] = {
        "status": "FAIL",
        "file_path": file_path,
        "file_size_bytes": 0,
        "can_open": False,
        "is_watertight": False,
        "num_vertices": 0,
        "num_faces": 0,
        "extents": [0.0, 0.0, 0.0],
        "max_extent": 0.0,
        "center_offset": 0.0,
        "relative_center_offset": 0.0,
        "has_textures": False,
        "has_vertex_colors": False,
        "has_uv": False,
        "has_nan_inf": False,
        "degenerate_faces": 0,
        "issues": [],
        "warnings": []
    }

    if not os.path.exists(file_path):
        result["issues"].append("File not found")
        return result

    file_size = os.path.getsize(file_path)
    result["file_size_bytes"] = file_size
    if file_size == 0:
        result["issues"].append("File is empty (0 bytes)")
        return result

    try:
        scene = trimesh.load(file_path, force='scene')
        result["can_open"] = True
    except Exception as e:
        result["issues"].append(f"Failed to open model: {str(e)[:120]}")
        return result

    try:
        geom = scene.to_geometry()
        if not isinstance(geom, trimesh.Trimesh):
            mesh_list = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if mesh_list:
                geom = trimesh.util.concatenate(mesh_list)
            else:
                result["issues"].append("No 3D polygonal mesh found (only curves/paths/empty)")
                return result
    except Exception as e:
        result["issues"].append(f"Geometry extraction failed: {str(e)[:120]}")
        return result

    num_verts = len(geom.vertices)
    num_faces = len(geom.faces)
    result["num_vertices"] = num_verts
    result["num_faces"] = num_faces

    if num_verts == 0 or num_faces == 0:
        result["issues"].append(f"Empty geometry: {num_verts} vertices, {num_faces} faces")
        return result

    verts_arr = np.asarray(geom.vertices)
    if np.isnan(verts_arr).any() or np.isinf(verts_arr).any():
        result["has_nan_inf"] = True
        result["issues"].append("Model contains NaN or Infinite vertex coordinates")
        return result

    extents = [float(x) for x in geom.extents]
    max_extent = float(np.max(geom.extents))
    result["extents"] = extents
    result["max_extent"] = max_extent

    if max_extent < 1e-4:
        result["issues"].append(f"Model is microscopic / zero-scale (max extent {max_extent:.2e})")
        return result
    elif max_extent > 10000.0:
        result["warnings"].append(f"Model has unusually large dimensions (max extent {max_extent:.2e})")
    elif max_extent < 0.05:
        result["warnings"].append(f"Model is very small (max extent {max_extent:.4f})")

    center = geom.bounding_box.centroid
    center_offset = float(np.linalg.norm(center))
    result["center_offset"] = center_offset
    result["relative_center_offset"] = float(center_offset / max(max_extent, 1e-6))

    if result["relative_center_offset"] > 5.0:
        result["warnings"].append(f"Model is placed far from origin (center offset: {center_offset:.2f})")

    try:
        degenerate_count = int(np.sum(geom.area_faces <= 1e-9))
        result["degenerate_faces"] = degenerate_count
        if degenerate_count > num_faces * 0.2:
            result["warnings"].append(f"{degenerate_count} degenerate faces ({degenerate_count / num_faces * 100:.1f}%)")
    except Exception:
        pass

    try:
        result["is_watertight"] = bool(geom.is_watertight)
    except Exception:
        result["is_watertight"] = False

    visual = getattr(geom, 'visual', None)
    if visual is not None:
        uv = getattr(visual, 'uv', None)
        if uv is not None and len(uv) > 0:
            result["has_uv"] = True

        vc = getattr(visual, 'vertex_colors', None)
        if vc is not None and len(vc) > 0:
            result["has_vertex_colors"] = True

        mat = getattr(visual, 'material', None)
        if mat is not None:
            has_img = getattr(mat, 'baseColorTexture', None) is not None or getattr(mat, 'image', None) is not None
            if has_img:
                result["has_textures"] = True

    if not result["has_textures"] and not result["has_vertex_colors"]:
        result["warnings"].append("No textures or vertex colors detected (geometry only)")

    if len(result["issues"]) > 0:
        result["status"] = "FAIL"
    elif len(result["warnings"]) > 0:
        result["status"] = "WARN"
    else:
        result["status"] = "PASS"

    return result

def check_dataset(dataset_dir: str = DEFAULT_DATASET_DIR) -> Dict[str, Any]:
    """
    Validates all objects in dataset_dir/objects/ and updates metadata.json.
    """
    objects_dir = os.path.join(dataset_dir, "objects")
    master_metadata_path = os.path.join(dataset_dir, "metadata.json")
    report_path = os.path.join(dataset_dir, "validation_report.json")

    if not os.path.exists(objects_dir):
        print(f"Error: Objects directory {objects_dir} does not exist.")
        return {}

    master_metadata: Dict[str, Any] = {}
    if os.path.exists(master_metadata_path):
        with open(master_metadata_path, "r", encoding="utf-8") as f:
            master_metadata = json.load(f)

    object_folders = sorted([
        f for f in os.listdir(objects_dir)
        if os.path.isdir(os.path.join(objects_dir, f)) and not f.startswith('.')
    ])

    print(f"\n=======================================================")
    print(f"       3D MODEL HEALTH CHECK & VALIDATION")
    print(f"=======================================================")
    print(f"Inspecting {len(object_folders)} objects in {objects_dir}...\n", flush=True)

    summary = {"PASS": 0, "WARN": 0, "FAIL": 0, "total": len(object_folders)}
    results: Dict[str, Any] = {}

    for obj_id in object_folders:
        obj_dir = os.path.join(objects_dir, obj_id)
        glb_path = os.path.join(obj_dir, "model.glb")

        diag = validate_model(glb_path)
        status = diag["status"]
        summary[status] += 1
        results[obj_id] = diag

        obj_meta_path = os.path.join(obj_dir, "metadata.json")
        obj_meta = {}
        if os.path.exists(obj_meta_path):
            with open(obj_meta_path, "r", encoding="utf-8") as f:
                obj_meta = json.load(f)
        obj_meta["validation"] = {
            "status": status,
            "num_vertices": diag["num_vertices"],
            "num_faces": diag["num_faces"],
            "max_extent": round(diag["max_extent"], 4),
            "has_textures": diag["has_textures"],
            "issues": diag["issues"],
            "warnings": diag["warnings"]
        }
        with open(obj_meta_path, "w", encoding="utf-8") as f:
            json.dump(obj_meta, f, indent=2)

        if obj_id in master_metadata:
            master_metadata[obj_id]["validation"] = obj_meta["validation"]

        icon = "✅" if status == "PASS" else ("⚠️ " if status == "WARN" else "❌")
        info = f"verts: {diag['num_vertices']:,} | faces: {diag['num_faces']:,}"
        if status == "FAIL":
            detail = f"FAILED: {', '.join(diag['issues'])}"
        elif status == "WARN":
            detail = f"WARNING: {', '.join(diag['warnings'])}"
        else:
            detail = f"OK (extent: {diag['max_extent']:.2f})"

        print(f"[{icon} {status:4}] {obj_id[:25]:25} | {info:30} | {detail}", flush=True)

    with open(master_metadata_path, "w", encoding="utf-8") as f:
        json.dump(master_metadata, f, indent=2)

    report_data = {
        "summary": summary,
        "pass_rate_percent": round(summary["PASS"] / max(summary["total"], 1) * 100, 2),
        "usable_rate_percent": round((summary["PASS"] + summary["WARN"]) / max(summary["total"], 1) * 100, 2),
        "details": results
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n=======================================================", flush=True)
    print(f"                   VALIDATION SUMMARY", flush=True)
    print(f"=======================================================", flush=True)
    print(f"  Total Models Checked: {summary['total']}", flush=True)
    print(f"  ✅ Fully Valid (PASS): {summary['PASS']}", flush=True)
    print(f"  ⚠️  Usable with Warnings (WARN): {summary['WARN']}", flush=True)
    print(f"  ❌ Broken / Corrupted (FAIL): {summary['FAIL']}", flush=True)
    print(f"  Usable Rate (PASS + WARN): {report_data['usable_rate_percent']}%", flush=True)
    print(f"  Detailed Report: {report_path}", flush=True)
    print(f"=======================================================\n", flush=True)

    return report_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check health of 3D models in dataset")
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR, help="Dataset directory to check")
    args = parser.parse_args()

    check_dataset(dataset_dir=args.dataset_dir)
