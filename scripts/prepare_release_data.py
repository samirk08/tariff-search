"""
Prepare data files for GitHub release
"""

import os
import zipfile
from pathlib import Path
import hashlib

def create_release_zip():
    """Create a zip file of prepared data for release"""
    
    # Ensure data exists
    data_dir = Path("tariff_search/data")
    if not data_dir.exists():
        print("Error: Run create_mock_data.py first to create data files")
        return
    
    # Create zip file
    output_file = "tariff_search_data_v1.0.0.zip"
    
    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in data_dir.glob('*'):
            if file.is_file():
                zipf.write(file, file.relative_to(data_dir.parent))
    
    # Calculate checksum
    sha256_hash = hashlib.sha256()
    with open(output_file, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    checksum = sha256_hash.hexdigest()
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    
    print(f"Created: {output_file}")
    print(f"Size: {size_mb:.2f} MB")
    print(f"SHA256: {checksum}")
    
    # Generate download.py update
    print("\nUpdate download.py with:")
    print(f'''
DATA_FILES = {{
    "prepared_data.zip": {{
        "url": "https://github.com/YOUR_USERNAME/YOUR_REPO/releases/download/v1.0.0/{output_file}",
        "sha256": "{checksum}",
        "size_mb": {size_mb:.1f}
    }}
}}
''')
    
    # Create release notes
    with open("RELEASE_NOTES.md", "w") as f:
        f.write(f"""# Release v1.0.0

## Data Files

- **File**: {output_file}
- **Size**: {size_mb:.2f} MB
- **SHA256**: {checksum}

## Installation

```bash
pip install tariff-search
```

## Usage

```python
from tariff_search import TariffSearch

# Auto-downloads data on first use
searcher = TariffSearch()
results = searcher.search("Cotton cloth")
```
""")
    
    print("\nRelease notes saved to RELEASE_NOTES.md")
    print("\nNext steps:")
    print("1. Create a GitHub release")
    print("2. Upload the zip file as a release asset")
    print("3. Update download.py with the actual URL")

if __name__ == "__main__":
    create_release_zip()