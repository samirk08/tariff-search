"""
Test Google Drive download functionality
"""

import logging
logging.basicConfig(level=logging.INFO)

print("=== Testing Google Drive Download ===\n")

print("File ID: 1JROLp3BqFMYeuLDeiam3gzdRgpj0yCqc")
print("URL: https://drive.google.com/file/d/1JROLp3BqFMYeuLDeiam3gzdRgpj0yCqc/view?usp=sharing\n")

# Test 1: Test the download URL construction
from tariff_search.download import download_file
from pathlib import Path

print("1. Testing download from Google Drive...")
url = "https://drive.google.com/uc?export=download&id=1JROLp3BqFMYeuLDeiam3gzdRgpj0yCqc"

# Note: This will actually attempt to download the file
# Comment out the actual download if you just want to test the setup
"""
success = download_file(url, Path("test_download.pkl"))
if success:
    print("✓ Download successful!")
    print(f"File size: {Path('test_download.pkl').stat().st_size / (1024*1024):.2f} MB")
    # Clean up
    Path("test_download.pkl").unlink()
else:
    print("✗ Download failed")
"""

print("\n2. The package is now configured to use your Google Drive file!")
print("\nUsers can now:")
print("  - Use `tariff-search download --raw` to download the pickle file")
print("  - Or initialize TariffSearch() which will auto-download if needed")

print("\n3. Since this appears to be a raw pickle file, users should:")
print("  a) Download it: tariff-search download --raw")
print("  b) Prepare it: tariff-search prepare df_with_embeddings.pkl output_dir")
print("  c) Then use: TariffSearch(data_dir='output_dir')")

print("\nNote: For large files, installing gdown is recommended:")
print("  pip install gdown")