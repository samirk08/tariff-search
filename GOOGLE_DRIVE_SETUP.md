# Google Drive Setup for Tariff Search Data

## Step 1: Prepare Your Data

First, create the data file to upload:

```bash
# If you have the actual df_with_embeddings.pkl:
python -c "
from tariff_search import prepare_data_files
prepare_data_files(
    '../path/to/df_with_embeddings.pkl',
    'tariff_search/data'
)
"

# Create a zip file
python scripts/prepare_release_data.py
```

## Step 2: Upload to Google Drive

1. Go to [Google Drive](https://drive.google.com)
2. Upload `tariff_search_data_v1.0.0.zip`
3. Right-click the file → "Get link"
4. Change from "Restricted" to "Anyone with the link"
5. Copy the link

## Step 3: Extract File ID

Your link will look like:
```
https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/view?usp=sharing
```

The FILE_ID is the part between `/d/` and `/view`:
```
1AbCdEfGhIjKlMnOpQrStUvWxYz
```

## Step 4: Update download.py

Replace the FILE_ID in the example below with your actual ID:

```python
# In tariff_search/download.py
DATA_FILES = {
    "prepared_data.zip": {
        "url": "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID_HERE",
        "sha256": "calculate_after_download",  # Run sha256sum on your file
        "size_mb": 500  # Update with actual size
    }
}
```

## Alternative: Direct Download URL for Raw Pickle

If you want to host the raw `df_with_embeddings.pkl`:

```python
def download_raw_pickle(dest_path: str = "df_with_embeddings.pkl") -> bool:
    """Download the raw pickle file with embeddings."""
    
    # Replace with your file ID
    file_id = "YOUR_RAW_PICKLE_FILE_ID"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    return download_file(url, Path(dest_path), expected_size_mb=1500)
```

## Handling Large Files

For files over 100MB, Google Drive shows a virus scan warning. The package should handle this automatically, but if issues arise, you can use the `gdown` library:

```bash
pip install gdown
```

Then update the download function to use gdown for more reliable downloads.