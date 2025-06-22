# Persistent Storage for Tariff Search

## Overview

The tariff search package now stores downloaded data in a persistent location that survives terminal restarts and virtual environment changes.

## Default Storage Location

By default, data is stored in:
- **macOS/Linux**: `~/.tariff_search/data/`
- **Windows**: `C:\Users\<username>\.tariff_search\data\`

This ensures that the 5GB data file is downloaded only once and persists across:
- Terminal restarts
- Virtual environment recreations
- Python interpreter sessions
- System reboots

## Custom Storage Location

You can specify a custom storage location in three ways:

### 1. Environment Variable (Recommended)
Set the `TARIFF_SEARCH_DATA_DIR` environment variable:

```bash
export TARIFF_SEARCH_DATA_DIR="/path/to/your/data"
```

Add this to your `.bashrc`, `.zshrc`, or shell configuration file to make it permanent.

### 2. Python Code
Specify the directory when initializing:

```python
from tariff_search import TariffSearch

searcher = TariffSearch(data_dir="/path/to/your/data")
```

### 3. Command Line
Use the `--data-dir` flag:

```bash
python -m tariff_search.cli search "Cotton cloth" --data-dir /path/to/your/data
```

## Checking Storage Location

To see where data will be stored:

```python
from tariff_search import get_default_data_dir
print(get_default_data_dir())
```

## Managing Storage

### Check if data is downloaded:
```bash
ls ~/.tariff_search/data/
```

Expected files:
- `embeddings.npy` - Vector embeddings
- `metadata.pkl` - Tariff metadata
- `info.pkl` - Data structure information

### Remove downloaded data:
```bash
rm -rf ~/.tariff_search/data/
```

### Force re-download:
```python
from tariff_search import download_prepared_data
download_prepared_data(force=True)
```

## Benefits

1. **One-time Download**: The 5GB file is downloaded only once
2. **Persistent Storage**: Data survives across sessions
3. **Shared Access**: Multiple projects can use the same data
4. **Easy Management**: Clear location for data storage
5. **Customizable**: Use environment variables or code to change location

## Troubleshooting

If you're still experiencing repeated downloads:

1. Check write permissions:
   ```bash
   ls -la ~/.tariff_search/
   ```

2. Verify data files exist:
   ```bash
   ls -la ~/.tariff_search/data/
   ```

3. Check environment variable:
   ```bash
   echo $TARIFF_SEARCH_DATA_DIR
   ```

4. Use verbose logging:
   ```python
   import logging
   logging.basicConfig(level=logging.INFO)
   searcher = TariffSearch()
   ```