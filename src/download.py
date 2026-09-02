"""
Download module: fetches UIDs and 3D GLB assets from Objaverse.
"""

import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import objaverse
from typing import List, Dict, Optional

def get_available_uids() -> List[str]:
    """Returns the full list of available UIDs from Objaverse."""
    return objaverse.load_uids()

def download_uids(uids: List[str], download_processes: int = 4) -> Dict[str, str]:
    """
    Downloads objects corresponding to the given UIDs into the local cache.
    Returns mapping from UID to local file path.
    """
    return objaverse.load_objects(uids=uids, download_processes=download_processes)

def load_annotations(uids: List[str]) -> Dict[str, dict]:
    """Fetches annotation and licensing metadata for the given UIDs."""
    return objaverse.load_annotations(uids=uids)
