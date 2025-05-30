"""
Download the large file using gdown
"""

import gdown
import os

# Google Drive file ID
file_id = "1JROLp3BqFMYeuLDeiam3gzdRgpj0yCqc"
url = f"https://drive.google.com/uc?id={file_id}"

# Output file
output = "df_with_embeddings.pkl"

# Check if file already partially downloaded
if os.path.exists(output):
    print(f"File {output} already exists. Size: {os.path.getsize(output) / (1024**3):.2f} GB")
    print("Remove it if you want to re-download.")
else:
    print(f"Downloading from Google Drive...")
    print(f"File ID: {file_id}")
    print(f"This is a large file (5.6GB) and will take some time...")
    
    # Download with gdown
    gdown.download(url, output, quiet=False)
    
    print(f"\nDownload complete!")
    print(f"File size: {os.path.getsize(output) / (1024**3):.2f} GB")