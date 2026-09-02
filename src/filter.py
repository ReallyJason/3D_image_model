"""
License & Quality Filtering Module (Part 12).

Filters dataset objects based on:
1. License compliance (e.g., CC0, CC-BY, commercial-safe vs non-commercial)
2. Health & geometry integrity (excluding FAIL models)
3. Deduplication (excluding duplicate assets)

Outputs metadata/filtered_catalog.json ready for model training.
"""

import os
import json
import argparse
from typing import Dict, Any, List, Set, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASET_DIR = os.path.join(PROJECT_ROOT, "my_dataset")

COMMERCIAL_SAFE_LICENSES = {"CC0-1.0", "CC-BY-4.0", "CC-BY-SA-4.0", "CC-BY-ND-4.0"}

def filter_dataset(
    dataset_dir: str = DEFAULT_DATASET_DIR,
    commercial_only: bool = False,
    exclude_duplicates: bool = True,
    only_passed_validation: bool = True
) -> Dict[str, Any]:
    """
    Filters master metadata catalog according to licensing and quality rules.
    """
    master_metadata_path = os.path.join(dataset_dir, "metadata.json")
    output_catalog_path = os.path.join(dataset_dir, "metadata", "filtered_catalog.json")
    os.makedirs(os.path.join(dataset_dir, "metadata"), exist_ok=True)

    if not os.path.exists(master_metadata_path):
        print(f"Error: {master_metadata_path} not found.")
        return {}

    with open(master_metadata_path, "r", encoding="utf-8") as f:
        master_metadata = json.load(f)

    print(f"\n=======================================================")
    print(f"         DATASET LICENSE & QUALITY FILTERING")
    print(f"=======================================================")
    print(f"Total objects in catalog:   {len(master_metadata)}")
    print(f"Commercial-only:            {commercial_only}")
    print(f"Exclude duplicates:         {exclude_duplicates}")
    print(f"Only passed validation:     {only_passed_validation}")
    print(f"=======================================================\n", flush=True)

    filtered: Dict[str, Any] = {}
    rejection_reasons = {
        "corrupted_or_failed": 0,
        "duplicate": 0,
        "non_commercial_license": 0
    }

    for obj_id, meta in master_metadata.items():
        # 1. Validation check
        val_status = meta.get("validation", {}).get("status")
        if only_passed_validation and val_status == "FAIL":
            rejection_reasons["corrupted_or_failed"] += 1
            continue

        # 2. Duplicate check
        if exclude_duplicates and meta.get("is_duplicate", False):
            rejection_reasons["duplicate"] += 1
            continue

        # 3. License check
        lic = meta.get("license", "Unknown")
        if commercial_only and lic not in COMMERCIAL_SAFE_LICENSES:
            rejection_reasons["non_commercial_license"] += 1
            continue

        filtered[obj_id] = meta

    with open(output_catalog_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2)

    print(f"  ✅ Kept for Training: {len(filtered)} / {len(master_metadata)}")
    print(f"  ❌ Excluded (Validation FAIL): {rejection_reasons['corrupted_or_failed']}")
    print(f"  ❌ Excluded (Duplicates):     {rejection_reasons['duplicate']}")
    if commercial_only:
        print(f"  ❌ Excluded (Non-commercial): {rejection_reasons['non_commercial_license']}")
    print(f"  Filtered Catalog: {output_catalog_path}\n", flush=True)

    return filtered

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter dataset by license and quality")
    parser.add_argument("--commercial-only", action="store_true", help="Keep only commercial-safe CC licenses")
    parser.add_argument("--include-duplicates", action="store_true", help="Keep duplicate models")
    args = parser.parse_args()

    filter_dataset(
        commercial_only=args.commercial_only,
        exclude_duplicates=not args.include_duplicates
    )
