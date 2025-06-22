# Manual Setup Instructions

If you're experiencing issues with automatic data download, follow these steps:

## Option 1: Manual Download from Google Drive

1. Download the data file manually from Google Drive:
   - URL: https://drive.google.com/file/d/1JROLp3BqFMYeuLDeiam3gzdRgpj0yCqc/view
   - File size: ~5.6GB

2. Create the data directory:
   ```bash
   mkdir -p ~/.tariff_search/data
   ```

3. Move the downloaded file to the data directory and extract it:
   ```bash
   # Replace ~/Downloads/file.zip with your actual download path
   mv ~/Downloads/[downloaded-file].zip ~/.tariff_search/data/
   cd ~/.tariff_search/data/
   unzip [downloaded-file].zip
   ```

4. Check that you have the required files:
   ```bash
   ls ~/.tariff_search/data/
   ```
   
   You should see files like:
   - `all_tariffs_emb_metadata.pkl` and `all_tariffs_emb_vectors.npy`, OR
   - `metadata.pkl` and `embeddings.npy`

## Option 2: Use Custom Data Directory

If you have the data files in a different location:

```python
from tariff_search import TariffSearch

# Specify your custom data directory
searcher = TariffSearch(data_dir="/path/to/your/data")
```

## Option 3: Disable Auto-Download

To prevent automatic download attempts:

```python
from tariff_search import TariffSearch

# Disable auto-download
searcher = TariffSearch(auto_download=False, data_dir="/path/to/your/data")
```

## Troubleshooting

### Issue: "Failed to download data files"
- The automatic download may fail due to Google Drive's download limits
- Use Option 1 above to manually download

### Issue: "No metadata file found"
- Ensure the files are extracted properly
- Check that file names match one of the expected patterns
- The package now supports multiple naming conventions

### Issue: Installation from wrong directory
- Make sure to run `pip install` from the package directory:
  ```bash
  cd /Users/samirkadariya/Desktop/UROP-Fall24/tradelab/tariff-search-package
  pip install -e .
  ```

## File Structure

The package expects one of these file structures:

### Structure 1 (Standard):
```
~/.tariff_search/data/
├── embeddings.npy
├── metadata.pkl
└── info.pkl
```

### Structure 2 (Full names):
```
~/.tariff_search/data/
├── all_tariffs_emb_vectors.npy
├── all_tariffs_emb_metadata.pkl
└── (other files)
```

### Structure 3 (Census):
```
~/.tariff_search/data/
├── census_selected_df_vectors.npy
├── census_selected_df_metadata.pkl
└── (other files)
```

The latest version of the package (after the fixes) will automatically detect and work with any of these structures.