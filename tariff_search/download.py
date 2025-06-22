"""
Download pre-prepared tariff search data files.
"""

import logging
import requests
from pathlib import Path
from tqdm import tqdm
import hashlib
import zipfile
import shutil
from .utils import get_default_data_dir

logger = logging.getLogger(__name__)

# Data file URLs and checksums
DATA_FILES = {
    "prepared_data.zip": {
        "url": "https://drive.google.com/uc?export=download&id=1JROLp3BqFMYeuLDeiam3gzdRgpj0yCqc",
        "sha256": "TO_BE_CALCULATED",  # Will be calculated after first download
        "size_mb": 1500  # Approximate size for the pickle file
    }
}

def download_file(url: str, dest_path: Path, expected_size_mb: float = None) -> bool:
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from
        dest_path: Destination file path
        expected_size_mb: Expected file size in MB (for progress bar)
        
    Returns:
        bool: True if successful
    """
    try:
        # Check if it's a Google Drive URL
        if "drive.google.com" in url:
            return download_from_google_drive(url, dest_path, expected_size_mb)
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as file:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
                    pbar.update(len(chunk))
                    
        return True
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False

def download_from_google_drive(url: str, dest_path: Path, expected_size_mb: float = None) -> bool:
    """
    Download file from Google Drive, handling virus scan warnings.
    
    Args:
        url: Google Drive URL
        dest_path: Destination file path
        expected_size_mb: Expected file size in MB
        
    Returns:
        bool: True if successful
    """
    try:
        # Try using gdown if available
        try:
            import gdown
            logger.info("Using gdown for Google Drive download...")
            gdown.download(url, str(dest_path), quiet=False)
            return dest_path.exists()
        except ImportError:
            logger.info("gdown not available, using requests...")
        
        # Extract file ID from URL if needed
        import re
        file_id_match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
        if file_id_match:
            file_id = file_id_match.group(1)
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        session = requests.Session()
        response = session.get(url, stream=True)
        
        # Check for virus scan warning
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                # Get confirmation URL
                params = {'id': file_id, 'confirm': value}
                url = "https://drive.google.com/uc?export=download"
                response = session.get(url, params=params, stream=True)
                break
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as file:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=32768):
                    if chunk:
                        file.write(chunk)
                        pbar.update(len(chunk))
                        
        return True
    except Exception as e:
        logger.error(f"Error downloading from Google Drive: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False

def verify_checksum(file_path: Path, expected_sha256: str) -> bool:
    """Verify file checksum."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    actual_checksum = sha256_hash.hexdigest()
    return actual_checksum == expected_sha256

def download_prepared_data(data_dir: str = None, force: bool = False) -> bool:
    """
    Download pre-prepared tariff search data.
    
    Args:
        data_dir: Directory to save data (default: package data directory)
        force: Force re-download even if files exist
        
    Returns:
        bool: True if successful
    """
    if data_dir is None:
        data_dir = get_default_data_dir()
    
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if data already exists
    required_files = ['embeddings.npy', 'metadata.pkl', 'info.pkl']
    if not force and all((data_dir / f).exists() for f in required_files):
        logger.info("Data files already exist. Use force=True to re-download.")
        return True
    
    logger.info("Downloading pre-prepared tariff search data...")
    logger.info("This is a one-time download of approximately 500MB.")
    
    # Download zip file
    temp_zip = data_dir / "temp_download.zip"
    
    file_info = DATA_FILES["prepared_data.zip"]
    success = download_file(file_info["url"], temp_zip, file_info["size_mb"])
    
    if not success:
        logger.error("Failed to download data files.")
        return False
    
    # Verify checksum
    logger.info("Verifying download...")
    if not verify_checksum(temp_zip, file_info["sha256"]):
        logger.error("Checksum verification failed. The download may be corrupted.")
        temp_zip.unlink()
        return False
    
    # Extract files
    logger.info("Extracting data files...")
    try:
        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        temp_zip.unlink()
        logger.info("Data files successfully downloaded and extracted!")
        return True
    except Exception as e:
        logger.error(f"Error extracting files: {e}")
        temp_zip.unlink()
        return False

def download_raw_pickle(dest_path: str = "df_with_embeddings.pkl") -> bool:
    """
    Download the raw pickle file with embeddings.
    
    Args:
        dest_path: Destination file path
        
    Returns:
        bool: True if successful
    """
    logger.info("Downloading raw DataFrame with embeddings...")
    logger.info("This is a large file (>1GB) and may take some time.")
    
    # Using the Google Drive file
    url = "https://drive.google.com/uc?export=download&id=1JROLp3BqFMYeuLDeiam3gzdRgpj0yCqc"
    
    return download_file(url, Path(dest_path), expected_size_mb=1500)