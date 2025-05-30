"""
Upload tariff search data to Hugging Face Hub
"""

def upload_to_huggingface():
    """
    Upload data files to Hugging Face Hub
    
    Prerequisites:
    1. pip install huggingface-hub
    2. huggingface-cli login (or set HF_TOKEN environment variable)
    """
    
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Please install huggingface-hub: pip install huggingface-hub")
        return
    
    import os
    from pathlib import Path
    
    # Configuration
    repo_id = "your-username/tariff-search-data"  # Change this!
    
    print(f"Uploading to Hugging Face Hub: {repo_id}")
    
    # Initialize API
    api = HfApi()
    
    # Create repository if it doesn't exist
    try:
        create_repo(repo_id, repo_type="dataset", exist_ok=True)
        print(f"Repository ready: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"Note: {e}")
    
    # Upload data files
    data_dir = Path("tariff_search/data")
    
    if data_dir.exists():
        for file_path in data_dir.glob("*.pkl"):
            print(f"Uploading {file_path.name}...")
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=f"data/{file_path.name}",
                repo_id=repo_id,
                repo_type="dataset"
            )
        
        for file_path in data_dir.glob("*.npy"):
            print(f"Uploading {file_path.name}...")
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=f"data/{file_path.name}",
                repo_id=repo_id,
                repo_type="dataset"
            )
    
    # Upload zip file if exists
    zip_file = Path("tariff_search_data_v1.0.0.zip")
    if zip_file.exists():
        print(f"Uploading {zip_file.name}...")
        api.upload_file(
            path_or_fileobj=str(zip_file),
            path_in_repo=zip_file.name,
            repo_id=repo_id,
            repo_type="dataset"
        )
    
    print("\nUpload complete!")
    print(f"Dataset URL: https://huggingface.co/datasets/{repo_id}")
    
    # Show how to use in download.py
    print("\nUpdate download.py to use Hugging Face:")
    print(f'''
from huggingface_hub import hf_hub_download, snapshot_download

def download_from_huggingface(data_dir: str):
    """Download from Hugging Face Hub"""
    
    # Download all files in data/ directory
    snapshot_download(
        repo_id="{repo_id}",
        repo_type="dataset",
        local_dir=data_dir,
        allow_patterns="data/*"
    )
    
    # Or download specific file:
    # file_path = hf_hub_download(
    #     repo_id="{repo_id}",
    #     repo_type="dataset",
    #     filename="tariff_search_data_v1.0.0.zip"
    # )
''')

if __name__ == "__main__":
    print("=== Hugging Face Upload Script ===")
    print("\nBefore running:")
    print("1. Install: pip install huggingface-hub")
    print("2. Login: huggingface-cli login")
    print("3. Update repo_id in this script")
    print("\nReady? (y/n): ", end="")
    
    if input().lower() == 'y':
        upload_to_huggingface()