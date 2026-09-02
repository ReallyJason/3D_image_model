"""
Download & Organize 3D Objects from Objaverse.

Populates my_dataset/ in the standardized format:
my_dataset/
├── objects/
│   ├── <object_id>/
│   │   ├── model.glb
│   │   └── metadata.json
│   └── ...
└── metadata.json
"""

import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import objaverse
from dataset_manager import organize_dataset, DEFAULT_DATASET_DIR

if __name__ == '__main__':
    # 1. Fetch UIDs from Objaverse
    uids = objaverse.load_uids()
    print(f"Total objects available in Objaverse: {len(uids)}")

    # 2. Select initial test batch (e.g. 10 or 20 objects)
    test_uids = uids[:20]

    # 3. Organize into standardized dataset directory
    organize_dataset(uids=test_uids, dataset_dir=DEFAULT_DATASET_DIR)