"""
Download and organize script: fetches 3D assets from Objaverse into my_dataset.
"""
from src.organize import organize_dataset

if __name__ == "__main__":
    organize_dataset(limit=20)