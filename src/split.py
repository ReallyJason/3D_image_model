"""
Dataset Splitter Module (Part 12).

Partitions the verified dataset into training, validation, and test splits:
- training (80%)
- validation (10%)
- test (10%)

Stratified by category to ensure balanced representation across all splits.
Outputs my_dataset/splits/{train.json, val.json, test.json, summary.json}.
"""

import os
import json
import random
import argparse
from typing import Dict, Any, List
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASET_DIR = os.path.join(PROJECT_ROOT, "my_dataset")

def create_dataset_splits(
    dataset_dir: str = DEFAULT_DATASET_DIR,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, List[str]]:
    """
    Creates train/val/test splits stratified by object category.
    """
    random.seed(seed)
    splits_dir = os.path.join(dataset_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    # Prefer filtered catalog if exists, otherwise fallback to master metadata
    filtered_catalog_path = os.path.join(dataset_dir, "metadata", "filtered_catalog.json")
    master_metadata_path = os.path.join(dataset_dir, "metadata.json")

    catalog_path = filtered_catalog_path if os.path.exists(filtered_catalog_path) else master_metadata_path
    if not os.path.exists(catalog_path):
        print(f"Error: Catalog file not found at {catalog_path}")
        return {}

    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    # Exclude any FAIL models
    usable_ids = [
        obj_id for obj_id, meta in catalog.items()
        if meta.get("validation", {}).get("status") != "FAIL" and not meta.get("is_duplicate", False)
    ]

    # Group by category for stratified sampling
    category_groups: Dict[str, List[str]] = defaultdict(list)
    for obj_id in usable_ids:
        cat = catalog[obj_id].get("category", "object")
        category_groups[cat].append(obj_id)

    train_ids: List[str] = []
    val_ids: List[str] = []
    test_ids: List[str] = []

    for cat, ids in category_groups.items():
        random.shuffle(ids)
        n = len(ids)
        if n == 1:
            train_ids.append(ids[0])
        elif n == 2:
            train_ids.append(ids[0])
            val_ids.append(ids[1])
        else:
            n_train = max(1, int(n * train_ratio))
            n_val = max(1, int(n * val_ratio))
            train_ids.extend(ids[:n_train])
            val_ids.extend(ids[n_train:n_train + n_val])
            test_ids.extend(ids[n_train + n_val:])

    # If test_ids is empty due to small cluster sizes, balance from train
    if not test_ids and len(train_ids) > 5:
        test_ids.append(train_ids.pop())

    def format_split_records(id_list: List[str]) -> List[Dict[str, Any]]:
        return [
            {
                "id": obj_id,
                "category": catalog[obj_id].get("category"),
                "name": catalog[obj_id].get("name"),
                "model_file": catalog[obj_id].get("file"),
                "renders_dir": f"objects/{obj_id}",
                "license": catalog[obj_id].get("license"),
                "original_uid": catalog[obj_id].get("original_uid"),
                "stats": catalog[obj_id].get("stats")
            }
            for obj_id in id_list
        ]

    train_records = format_split_records(train_ids)
    val_records = format_split_records(val_ids)
    test_records = format_split_records(test_ids)

    # Save split JSONs
    with open(os.path.join(splits_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_records, f, indent=2)

    with open(os.path.join(splits_dir, "val.json"), "w", encoding="utf-8") as f:
        json.dump(val_records, f, indent=2)

    with open(os.path.join(splits_dir, "test.json"), "w", encoding="utf-8") as f:
        json.dump(test_records, f, indent=2)

    summary = {
        "total_verified": len(usable_ids),
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "test_count": len(test_ids),
        "train_percent": round(len(train_ids) / len(usable_ids) * 100, 1),
        "val_percent": round(len(val_ids) / len(usable_ids) * 100, 1),
        "test_percent": round(len(test_ids) / len(usable_ids) * 100, 1),
        "categories_count": len(category_groups)
    }

    with open(os.path.join(splits_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=======================================================")
    print(f"               DATASET SPLITS GENERATED")
    print(f"=======================================================")
    print(f"  Total Usable Models: {len(usable_ids)}")
    print(f"  🚂 Training (80%):    {len(train_ids)} ({summary['train_percent']}%)")
    print(f"  🔍 Validation (10%):  {len(val_ids)} ({summary['val_percent']}%)")
    print(f"  🧪 Test (10%):        {len(test_ids)} ({summary['test_percent']}%)")
    print(f"  Splits Output:       {splits_dir}")
    print(f"=======================================================\n", flush=True)

    return {"train": train_ids, "val": val_ids, "test": test_ids}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR, help="Dataset directory")
    args = parser.parse_args()

    create_dataset_splits(dataset_dir=args.dataset_dir)
