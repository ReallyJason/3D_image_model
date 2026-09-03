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
    create_dataset_splits,
    train_model,
    reconstruct_3d,
    run_benchmark
)
from src.organize import DEFAULT_DATASET_DIR

def main():
    parser = argparse.ArgumentParser(
        description="3D Generative Dataset Pipeline (Organize, Validate, Deduplicate, Render, Split, Train, Benchmark)"
    )
    parser.add_argument(
        "--step",
        type=str,
        choices=["all", "organize", "validate", "deduplicate", "filter", "render", "split", "train", "reconstruct", "benchmark", "serve"],
        default="all",
        help="Which pipeline stage to execute (default: all)"
    )
    parser.add_argument("--limit", type=int, default=None, help="Number of objects to process (default: all)")
    parser.add_argument("--num-views", type=int, default=12, help="Views per object for rendering (default: 12)")
    parser.add_argument("--resolution", type=int, default=512, help="Render resolution in px (default: 512)")
    parser.add_argument("--augment", action="store_true", help="Enable realistic data augmentation (lighting, camera, bg)")
    parser.add_argument("--category", type=str, default=None, help="Filter Objaverse models by category/tag (e.g. shoe, chair, car)")
    parser.add_argument("--commercial-only", action="store_true", help="Filter for only commercial-safe CC licenses")
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR, help="Dataset output directory")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--image", type=str, default=None, help="Input image path for 3D reconstruction")
    parser.add_argument("--port", type=int, default=8080, help="Web viewer port (default: 8080)")
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
        organize_dataset(dataset_dir=args.dataset_dir, category=args.category, limit=args.limit)

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
            category=args.category,
            limit=args.limit
        )

    # 6. Train/Val/Test Split
    if args.step in ["all", "split"]:
        print("\n--- [STAGE 6] GENERATING TRAIN / VAL / TEST SPLITS ---", flush=True)
        create_dataset_splits(dataset_dir=args.dataset_dir)

    # 7. Train AI Model (Part 14 & 15)
    if args.step == "train":
        print("\n--- [STAGE 7] TRAINING 2D-TO-3D VOXEL NEURAL NETWORK ---", flush=True)
        train_model(
            dataset_dir=args.dataset_dir,
            epochs=args.epochs,
            batch_size=8,
            lr=0.0005
        )

    # 8. Reconstruct 3D Mesh from Single 2D Image (Part 16)
    if args.step == "reconstruct":
        print("\n--- [STAGE 8] RECONSTRUCTING 3D MESH FROM 2D IMAGE ---", flush=True)
        img_path = args.image
        if not img_path:
            import glob
            views = glob.glob(os.path.join(args.dataset_dir, "objects", "*", "view_000.png"))
            if views:
                img_path = views[0]
            else:
                print("Error: No images found. Provide --image <path/to/img.png>")
                return
        reconstruct_3d(image_path=img_path)

    # 9. Automated Evaluation Benchmark (Part 28 & 29)
    if args.step == "benchmark":
        print("\n--- [STAGE 9] RUNNING EVALUATION BENCHMARK & PROFILING ---", flush=True)
        run_benchmark(dataset_dir=args.dataset_dir)

    # 10. Interactive 3D Web Viewer (Option 2)
    if args.step == "serve":
        print("\n--- [STAGE 10] LAUNCHING INTERACTIVE 3D WEB VIEWER ---", flush=True)
        from app import run_server
        run_server(port=args.port)

    print("\n" + "=" * 65)
    print("        🎉 PIPELINE EXECUTION FINISHED")
    print("=" * 65 + "\n", flush=True)

if __name__ == "__main__":
    main()
