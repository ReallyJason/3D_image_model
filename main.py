"""
End-to-End 3D Dataset Pipeline Orchestrator (Parts 10, 11, 12).

Workflow:
  Download & Organize ➔ 3D Health Check ➔ Deduplication ➔ License Filter ➔ Multi-View Render ➔ Train/Val/Test Split

Usage:
    python main.py --step all --limit 50 --augment
    python main.py --step organize --limit 100
    python main.py --step validate
    python main.py --step deduplicate
    python main.py --step filter --commercial-only
    python main.py --step render --num-views 12 --augment
    python main.py --step split
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import (
    organize_dataset,
    check_dataset,
    deduplicate_dataset,
    filter_dataset,
    render_dataset,
    create_dataset_splits
)
from src.organize import DEFAULT_DATASET_DIR

def main():
    parser = argparse.ArgumentParser(
        description="3D Generative Dataset Pipeline (Organize, Validate, Deduplicate, Render, Split)"
    )
    parser.add_argument(
        "--step",
        type=str,
        choices=["all", "organize", "validate", "deduplicate", "filter", "render", "split"],
        default="all",
        help="Which pipeline stage to execute (default: all)"
    )
    parser.add_argument("--limit", type=int, default=20, help="Number of objects to process (default: 20)")
    parser.add_argument("--num-views", type=int, default=12, help="Views per object for rendering (default: 12)")
    parser.add_argument("--resolution", type=int, default=512, help="Render resolution in px (default: 512)")
    parser.add_argument("--augment", action="store_true", help="Enable realistic data augmentation (lighting, camera, bg)")
    parser.add_argument("--commercial-only", action="store_true", help="Filter for only commercial-safe CC licenses")
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR, help="Dataset output directory")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("        🧊 3D IMAGE MODEL DATASET PIPELINE")
    print("=" * 65)
    print(f"Stage:        {args.step.upper()}")
    print(f"Dataset Dir:  {args.dataset_dir}")
    print(f"Object Limit: {args.limit}")
    print(f"Augmentation: {'ENABLED (Realistic)' if args.augment else 'CLEAN (Studio)'}")
    print("=" * 65 + "\n", flush=True)

    # 1. Organize
    if args.step in ["all", "organize"]:
        print("\n--- [STAGE 1] ORGANIZING 3D OBJECTS & PROVENANCE ---", flush=True)
        organize_dataset(dataset_dir=args.dataset_dir, limit=args.limit)

    # 2. Validate & Health Check
    if args.step in ["all", "validate"]:
        print("\n--- [STAGE 2] VALIDATING 3D MODEL INTEGRITY & GEOMETRY ---", flush=True)
        check_dataset(dataset_dir=args.dataset_dir)

    # 3. Deduplicate
    if args.step in ["all", "deduplicate"]:
        print("\n--- [STAGE 3] GEOMETRIC DEDUPLICATION SCAN ---", flush=True)
        deduplicate_dataset(dataset_dir=args.dataset_dir)

    # 4. Filter (License & Quality)
    if args.step in ["all", "filter"]:
        print("\n--- [STAGE 4] LICENSE & QUALITY FILTERING ---", flush=True)
        filter_dataset(
            dataset_dir=args.dataset_dir,
            commercial_only=args.commercial_only,
            exclude_duplicates=True,
            only_passed_validation=True
        )

    # 5. Multi-View Render
    if args.step in ["all", "render"]:
        print("\n--- [STAGE 5] MULTI-VIEW RENDERING (GPU) ---", flush=True)
        render_dataset(
            dataset_dir=args.dataset_dir,
            num_views=args.num_views,
            resolution=args.resolution,
            only_valid=True,
            augment=args.augment,
            limit=args.limit
        )

    # 6. Train/Val/Test Split
    if args.step in ["all", "split"]:
        print("\n--- [STAGE 6] GENERATING TRAIN / VAL / TEST SPLITS ---", flush=True)
        create_dataset_splits(dataset_dir=args.dataset_dir)

    print("\n" + "=" * 65)
    print("        🎉 PIPELINE EXECUTION FINISHED")
    print("=" * 65 + "\n", flush=True)

if __name__ == "__main__":
    main()
