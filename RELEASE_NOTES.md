# Release v1.0.0

## Data Files

- **File**: tariff_search_data_v1.0.0.zip
- **Size**: 0.27 MB
- **SHA256**: e311d9a6c88e447d61bc77d8e366ed3cf4fb49c539f57331316217536b63c362

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
