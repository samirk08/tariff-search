"""
Helper to set up Google Drive hosting
"""

print("=== Google Drive Setup Instructions ===\n")

print("1. Upload your file to Google Drive")
print("2. Right-click → 'Get link' → 'Anyone with the link'")
print("3. Copy the sharing link, it will look like:")
print("   https://drive.google.com/file/d/FILE_ID/view?usp=sharing\n")

print("4. Extract the FILE_ID from the URL")
print("5. Update download.py with:\n")

file_id = input("Enter your Google Drive FILE_ID: ").strip()

if file_id:
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    print(f"\nYour download URL is:")
    print(f"{download_url}")
    
    print(f"\nUpdate download.py DATA_FILES with:")
    print(f'''
    "prepared_data.zip": {{
        "url": "{download_url}",
        "sha256": "UPDATE_AFTER_DOWNLOAD",  # Calculate this after downloading
        "size_mb": 500  # Update with actual size
    }}
    ''')
    
    # For large files, Google Drive requires confirmation
    print("\nNote: For files >100MB, you may need to handle Google Drive's virus scan warning.")
    print("Consider using gdown library for more reliable downloads:")
    print("pip install gdown")
    print(f"gdown.download('{download_url}', 'prepared_data.zip', quiet=False)")