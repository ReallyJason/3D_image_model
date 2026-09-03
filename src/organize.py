"""
Dataset Organizer: structures raw 3D assets into a standardized dataset directory.

Format:
my_dataset/
├── objects/
│   ├── <category>_<id>/
│   │   ├── model.glb
│   │   └── metadata.json
│   └── ...
└── metadata.json
"""

import os
import re
import json
import shutil
import ssl
import certifi
from typing import Dict, List, Optional, Any

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import objaverse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASET_DIR = os.path.join(PROJECT_ROOT, "my_dataset")

LICENSE_MAP = {
    "by": {"code": "CC-BY-4.0", "url": "https://creativecommons.org/licenses/by/4.0/"},
    "by-nc": {"code": "CC-BY-NC-4.0", "url": "https://creativecommons.org/licenses/by-nc/4.0/"},
    "by-sa": {"code": "CC-BY-SA-4.0", "url": "https://creativecommons.org/licenses/by-sa/4.0/"},
    "by-nc-sa": {"code": "CC-BY-NC-SA-4.0", "url": "https://creativecommons.org/licenses/by-nc-sa/4.0/"},
    "by-nd": {"code": "CC-BY-ND-4.0", "url": "https://creativecommons.org/licenses/by-nd/4.0/"},
    "by-nc-nd": {"code": "CC-BY-NC-ND-4.0", "url": "https://creativecommons.org/licenses/by-nc-nd/4.0/"},
    "cc0": {"code": "CC0-1.0", "url": "https://creativecommons.org/publicdomain/zero/1.0/"},
}

def sanitize_string(text: str, max_len: int = 30) -> str:
    """Convert text into a clean filename/identifier string."""
    if not text:
        return "object"
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    text = text.strip('_')
    if not text:
        return "object"
    return text[:max_len].rstrip('_')

def determine_category(annotation: Dict[str, Any]) -> str:
    """Extract the best category name from Objaverse annotation."""
    categories = annotation.get("categories", [])
    if categories and isinstance(categories, list):
        cat_name = categories[0].get("name") or categories[0].get("slug")
        if cat_name:
            return sanitize_string(cat_name)

    tags = annotation.get("tags", [])
    if tags and isinstance(tags, list):
        for tag in tags:
            tag_name = tag.get("name") if isinstance(tag, dict) else str(tag)
            if tag_name and not tag_name.isdigit() and len(tag_name) >= 3 and "draftpunk" not in tag_name:
                return sanitize_string(tag_name)

    name = annotation.get("name", "")
    if name:
        clean = sanitize_string(name)
        if clean and clean != "object":
            parts = [p for p in clean.split('_') if p and not p.isdigit()]
            if parts:
                return parts[0]
            return clean

    return "object"

def format_license_info(raw_license: Optional[str]) -> Dict[str, str]:
    """Map raw Objaverse license code to standard license info."""
    raw = (raw_license or "unknown").lower().strip()
    if raw in LICENSE_MAP:
        return {
            "license": LICENSE_MAP[raw]["code"],
            "license_url": LICENSE_MAP[raw]["url"],
            "raw_license": raw
        }
    return {
        "license": raw.upper() if raw != "unknown" else "Unknown",
        "license_url": "",
        "raw_license": raw
    }

def organize_dataset(
    uids: Optional[List[str]] = None,
    dataset_dir: str = DEFAULT_DATASET_DIR,
    category: Optional[str] = None,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Downloads (if needed) and organizes 3D models into dataset_dir.
    Optionally filters for a specific category (e.g. 'shoe', 'chair', 'car').
    Returns master metadata dictionary.
    """
    objects_dir = os.path.join(dataset_dir, "objects")
    os.makedirs(objects_dir, exist_ok=True)
    master_metadata_path = os.path.join(dataset_dir, "metadata.json")

    master_metadata: Dict[str, Any] = {}
    if os.path.exists(master_metadata_path):
        try:
            with open(master_metadata_path, "r", encoding="utf-8") as f:
                master_metadata = json.load(f)
        except Exception:
            master_metadata = {}

    existing_uids = {m.get("original_uid") for m in master_metadata.values() if m.get("original_uid")}
    print(f"Existing objects in dataset: {len(existing_uids)}")

    if uids is None:
        print("Fetching UIDs from Objaverse...")
        all_uids = objaverse.load_uids()
        new_uids = [u for u in all_uids if u not in existing_uids]

        if category:
            cat_clean = category.lower().strip()
            print(f"Filtering Objaverse models for category: '{cat_clean}' ...")
            # Batch scan annotations to find matching items
            matched_uids = []
            chunk_size = 500
            for start in range(0, min(len(new_uids), 10000), chunk_size):
                chunk = new_uids[start:start + chunk_size]
                chunk_annos = objaverse.load_annotations(chunk)
                for uid, anno in chunk_annos.items():
                    name = anno.get("name", "").lower()
                    tags = [t.get("name", "").lower() if isinstance(t, dict) else str(t).lower() for t in anno.get("tags", [])]
                    categories = [c.get("name", "").lower() if isinstance(c, dict) else str(c).lower() for c in anno.get("categories", [])]

                    if (cat_clean in name) or any(cat_clean in t for t in tags) or any(cat_clean in c for c in categories):
                        matched_uids.append(uid)
                        if limit and len(matched_uids) >= limit:
                            break
                if limit and len(matched_uids) >= limit:
                    break
            uids = matched_uids
            print(f"Found {len(uids)} models matching '{cat_clean}'.")
        else:
            uids = new_uids[:limit] if limit else new_uids[:50]
            print(f"Selected {len(uids)} NEW UIDs to download...")
    elif limit is not None:
        uids = uids[:limit]

    print(f"Preparing {len(uids)} objects for dataset organization...")
    print("Loading annotations...")
    annotations = objaverse.load_annotations(uids)

    print("Downloading/verifying GLB files (multiprocessed)...")
    cached_objects = objaverse.load_objects(uids=uids, download_processes=4)

    existing_ids = set(master_metadata.keys())
    start_index = len(existing_ids) + 1

    added_count = 0
    updated_count = 0

    for i, uid in enumerate(uids, start=start_index):
        glb_source_path = cached_objects.get(uid)
        if not glb_source_path or not os.path.exists(glb_source_path):
            print(f"Warning: GLB for UID {uid} not found. Skipping.")
            continue

        anno = annotations.get(uid, {})
        category = determine_category(anno)
        object_id = f"{category}_{i:05d}"

        # Reuse existing ID if UID was already indexed
        for ex_id, ex_meta in master_metadata.items():
            if ex_meta.get("original_uid") == uid:
                object_id = ex_id
                break

        object_dir = os.path.join(objects_dir, object_id)
        os.makedirs(object_dir, exist_ok=True)

        target_glb_path = os.path.join(object_dir, "model.glb")
        if not os.path.exists(target_glb_path) or os.path.getsize(target_glb_path) != os.path.getsize(glb_source_path):
            shutil.copy2(glb_source_path, target_glb_path)

        lic_info = format_license_info(anno.get("license"))
        user_info = anno.get("user", {})
        author = user_info.get("displayName") or user_info.get("username") or "Unknown"

        rel_file_path = f"objects/{object_id}/model.glb"

        obj_metadata = {
            "id": object_id,
            "category": category,
            "name": anno.get("name", object_id),
            "source": "objaverse",
            "license": lic_info["license"],
            "license_url": lic_info["license_url"],
            "raw_license": lic_info["raw_license"],
            "original_uid": uid,
            "author": author,
            "author_url": user_info.get("profileUrl", ""),
            "viewer_url": anno.get("viewerUrl", ""),
            "description": anno.get("description", ""),
            "tags": [t.get("name") if isinstance(t, dict) else str(t) for t in anno.get("tags", [])],
            "file": rel_file_path,
            "file_size_bytes": os.path.getsize(target_glb_path),
            "stats": {
                "face_count": anno.get("faceCount", 0),
                "vertex_count": anno.get("vertexCount", 0)
            }
        }

        # Preserve existing validation or views metadata
        if object_id in master_metadata:
            if "validation" in master_metadata[object_id]:
                obj_metadata["validation"] = master_metadata[object_id]["validation"]
            if "views" in master_metadata[object_id]:
                obj_metadata["views"] = master_metadata[object_id]["views"]
            updated_count += 1
        else:
            added_count += 1

        master_metadata[object_id] = obj_metadata

        local_meta_path = os.path.join(object_dir, "metadata.json")
        with open(local_meta_path, "w", encoding="utf-8") as f:
            json.dump(obj_metadata, f, indent=2)

    with open(master_metadata_path, "w", encoding="utf-8") as f:
        json.dump(master_metadata, f, indent=2)

    print(f"\nOrganization complete!")
    print(f"Dataset root: {dataset_dir}")
    print(f"Total objects in dataset: {len(master_metadata)} (Added: {added_count}, Updated: {updated_count})")
    print(f"Master metadata: {master_metadata_path}")

    return master_metadata

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Organize 3D models into standardized dataset structure")
    parser.add_argument("--limit", type=int, default=20, help="Number of objects to organize")
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR, help="Target dataset directory")
    args = parser.parse_args()

    organize_dataset(dataset_dir=args.dataset_dir, limit=args.limit)
