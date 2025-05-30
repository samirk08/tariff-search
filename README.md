# Tariff Search

Fast similarity search for US tariff descriptions (1789-2023).

## Installation

```bash
pip install git+https://github.com/samirk08/tariff-search.git
```

## Quick Start

```python
from tariff_search import TariffSearch

# Initialize (downloads data on first use)
searcher = TariffSearch()

# Search for similar tariff descriptions
results = searcher.search("Cotton cloth", top_k=5)

# Display results
for idx, row in results.iterrows():
    print(f"Similarity: {row['combined_similarity']:.3f}")
    print(f"Description: {row['Description_N']}")
    print(f"HS Code: {row['HS_N']}")
```

## Command Line Usage

```bash
# Download data manually
python -m tariff_search.cli download --raw

# Prepare data for fast searching
python -m tariff_search.cli prepare df_with_embeddings.pkl tariff_data

# Search from command line
python -m tariff_search.cli search "Steel pipes" --top-k 10
```

## Features

- Fast vector similarity search
- Multiple similarity metrics (cosine, Jaccard, Levenshtein)
- 750,000+ historical tariff descriptions
- Automatic data download from Google Drive (5.6GB)

## License

MIT