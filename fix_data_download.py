#!/usr/bin/env python3
"""
Script to fix data download issues by manually downloading and renaming files.
"""

import os
import sys
from pathlib import Path
import requests
import gdown
from tqdm import tqdm
import zipfile
import shutil

def download_and_fix_data():
    """Download data and rename files to match expected names."""
    
    # Get data directory
    from tariff_search.utils import get_default_data_dir
    data_dir = Path(get_default_data_dir())
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Data directory: {data_dir}")
    
    # Check if files already exist with correct names
    required_files = ['embeddings.npy', 'metadata.pkl', 'info.pkl']
    if all((data_dir / f).exists() for f in required_files):
        print("Data files already exist with correct names.")
        return True
    
    # Try to download the zip file
    temp_zip = data_dir / "temp_download.zip"
    
    print("Downloading data file (this may take a while)...")
    url = "https://drive.google.com/uc?export=download&id=1JROLp3BqFMYeuLDeiam3gzdRgpj0yCqc"
    
    try:
        gdown.download(url, str(temp_zip), quiet=False)
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False
    
    # Extract files
    print("Extracting files...")
    try:
        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            # List contents
            print("Zip contents:")
            for name in zip_ref.namelist():
                print(f"  - {name}")
            
            # Extract all
            zip_ref.extractall(data_dir)
        
        # Remove temp file
        temp_zip.unlink()
        
    except Exception as e:
        print(f"Error extracting files: {e}")
        if temp_zip.exists():
            temp_zip.unlink()
        return False
    
    # Now rename files to match expected names
    print("\nChecking and renaming files...")
    
    # Possible file mappings
    file_mappings = [
        # (possible names, target name)
        (['all_tariffs_emb_vectors.npy', 'embeddings.npy', 'all_tariffs_embeddings.npy'], 'embeddings.npy'),
        (['all_tariffs_emb_metadata.pkl', 'metadata.pkl', 'all_tariffs_metadata.pkl'], 'metadata.pkl'),
        (['info.pkl', 'all_tariffs_info.pkl'], 'info.pkl'),
    ]
    
    for possible_names, target_name in file_mappings:
        target_path = data_dir / target_name
        if not target_path.exists():
            for possible_name in possible_names:
                source_path = data_dir / possible_name
                if source_path.exists():
                    print(f"Renaming {possible_name} -> {target_name}")
                    source_path.rename(target_path)
                    break
            else:
                print(f"Warning: Could not find source file for {target_name}")
    
    # Final check
    missing_files = [f for f in required_files if not (data_dir / f).exists()]
    if missing_files:
        print(f"\nError: Still missing files: {missing_files}")
        print("\nFiles in data directory:")
        for f in data_dir.iterdir():
            print(f"  - {f.name}")
        return False
    
    print("\nAll files successfully downloaded and renamed!")
    return True

if __name__ == "__main__":
    success = download_and_fix_data()
    sys.exit(0 if success else 1)