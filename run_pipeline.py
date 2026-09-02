"""
Complete End-to-End Pipeline for 3D Image Model Dataset:
1. Organize: Downloads/structures objects into my_dataset/objects/<id>/model.glb & metadata.json
2. Check: Validates 3D models (detects broken, empty, uncentered, or corrupted objects)
3. Render: Generates multi-angle rendered images (view_000.png...) with exact camera poses

Usage:
    python run_pipeline.py --step all --limit 10
    python run_pipeline.py --step organize --limit 20
    python run_pipeline.py --step check
    python run_pipeline.py --step render --limit 10 --num-views 12
"""

import argparse
import sys
import os

from dataset_manager import organize_dataset, DEFAULT_DATASET_DIR
from check_objects import check_dataset
from render_objects import render_dataset

def main():
    parser = argparse.ArgumentParser(description="3D Dataset Preparation & Rendering Pipeline")
    parser.add_argument(
        "--step",
        type=str,
        choices=["all", "organize", "check", "render"],
        default="all",
        help="Which pipeline step to execute (default: all)"
    )
    parser.add_argument("--limit", type=int, default=10, help="Number of objects to process (default: 10)")
    parser.add_argument("--num-views", type=int, default=12, help="Views per object for rendering (default: 12)")
    parser.add_argument("--resolution", type=int, default=512, help="Render resolution (default: 512)")
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR, help="Dataset output path")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("        🚀 3D IMAGE MODEL DATASET PIPELINE")
    print("=" * 65)
    print(f"Step:        {args.step.upper()}")
    print(f"Dataset Dir: {args.dataset_dir}")
    print(f"Object Limit:{args.limit}")
    print("=" * 65 + "\n")

    # Step 1: Organize
    if args.step in ["all", "organize"]:
        print("\n--- [STEP 1/3] ORGANIZING OBJECTS & METADATA ---")
        organize_dataset(dataset_dir=args.dataset_dir, limit=args.limit)

    # Step 2: Check / Validate
    if args.step in ["all", "check"]:
        print("\n--- [STEP 2/3] CHECKING & VALIDATING 3D OBJECTS ---")
        check_dataset(dataset_dir=args.dataset_dir)

    # Step 3: Render
    if args.step in ["all", "render"]:
        print("\n--- [STEP 3/3] RENDERING MULTI-VIEW IMAGES ---")
        render_dataset(
            dataset_dir=args.dataset_dir,
            num_views=args.num_views,
            resolution=args.resolution,
            only_valid=True,
            limit=args.limit
        )

    print("\n" + "=" * 65)
    print("        🎉 PIPELINE EXECUTION FINISHED")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
