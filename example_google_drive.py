"""
Example: Using tariff-search with Google Drive hosted data
"""

print("=== Google Drive Integration Example ===\n")

print("1. First, upload your data to Google Drive and get the file ID")
print("   Example link: https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/view")
print("   File ID: 1AbCdEfGhIjKlMnOpQrStUvWxYz\n")

print("2. Update tariff_search/download.py:")
print("""
DATA_FILES = {
    "prepared_data.zip": {
        "url": "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID",
        "sha256": "calculate_this_after_download",
        "size_mb": 500
    }
}
""")

print("3. For better reliability with large files, install gdown:")
print("   pip install gdown\n")

print("4. Then use the package normally:")
print("""
from tariff_search import TariffSearch

# Will auto-download from Google Drive on first use
searcher = TariffSearch()
results = searcher.search("Cotton cloth")
""")

print("\n5. To manually specify a Google Drive URL:")
print("""
from tariff_search import download_prepared_data

# Using direct Google Drive link
google_drive_url = "https://drive.google.com/file/d/YOUR_FILE_ID/view"
download_prepared_data(data_dir="my_data", url=google_drive_url)
""")

print("\n=== Quick Test ===")
print("You can test the download functionality with a small file:")

# Create a test file to upload
import pickle
test_data = {"test": "This is a test file for Google Drive"}
with open("test_gdrive.pkl", "wb") as f:
    pickle.dump(test_data, f)

print("\n1. Upload 'test_gdrive.pkl' to Google Drive")
print("2. Share it and get the link")
print("3. Test download with:")
print("""
from tariff_search.download import download_file
from pathlib import Path

url = "https://drive.google.com/file/d/YOUR_TEST_FILE_ID/view"
success = download_file(url, Path("downloaded_test.pkl"))
print(f"Download successful: {success}")
""")

print("\n=== Tips ===")
print("- Make sure your Google Drive file is set to 'Anyone with the link'")
print("- For files >100MB, gdown is more reliable")
print("- You can also use Google Drive folder IDs to host multiple files")