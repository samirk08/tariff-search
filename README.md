# Tariff Search Package

A specialized search engine for finding similar US tariff descriptions across 234 years of historical data (1789-2023). This package uses state-of-the-art NLP techniques to enable semantic search across over 750,000 tariff descriptions, helping researchers and analysts understand how tariff classifications have evolved over time.

## Key Features

- 🔍 **Semantic Search**: Uses transformer-based embeddings for understanding meaning beyond keywords
- 📊 **Multi-Metric Matching**: Combines cosine similarity, Jaccard coefficient, and Levenshtein distance
- 🚀 **Fast Performance**: FAISS integration for efficient large-scale similarity search
- 📚 **Historical Coverage**: Complete US tariff data from 1789 to 2023
- 💾 **Smart Storage**: One-time download with persistent data management
- 🔧 **Flexible Interface**: Both Python API and command-line interface

## Installation

```bash
# Install from PyPI (coming soon)
pip install tariff-search

# Install from GitHub
pip install git+https://github.com/samirk08/tariff-search.git

# Install with development dependencies
pip install -e ".[dev]"
```

## Quick Start

### Python API

```python
from tariff_search import TariffSearch

# Initialize the search engine (downloads data on first use)
searcher = TariffSearch()

# Search for similar tariff descriptions
results = searcher.search("cotton shirts", top_k=5)

# Display results
for result in results:
    print(f"Score: {result['combined_similarity']:.3f}")
    print(f"Description: {result['concord_name']}")
    print(f"HS Code: {result['HS']}")
    print(f"Year: {result['year']}")
    print("---")
```

### Command Line Interface

```bash
# Basic search
tariff-search search "leather shoes" --top-k 5

# Search with year filter
tariff-search search "automobiles" --year 2023 --top-k 10

# Output as CSV
tariff-search search "steel pipes" --output-format csv > results.csv

# Download data files
tariff-search download --prepared
```

## Documentation

- [Full Documentation](DOCUMENTATION.md) - Comprehensive guide to the package
- [Data Pipeline](DATA_PIPELINE.md) - How the historical database was created
- [Architecture](ARCHITECTURE.md) - Technical design and implementation details
- [Code Examples](EXAMPLES.md) - Extensive usage examples and patterns
- [Persistent Storage](PERSISTENT_STORAGE.md) - Data storage and management

## How It Works

The package combines three similarity metrics to find the best matches:

1. **Semantic Similarity** (50%): Uses sentence-transformers to understand meaning
2. **Word Overlap** (25%): Jaccard similarity for common terminology
3. **String Similarity** (25%): Levenshtein distance for spelling variations

This multi-metric approach ensures accurate matches even when historical terminology differs significantly from modern descriptions.

## Use Cases

- **Trade Policy Research**: Track how specific products have been classified over time
- **Economic History**: Understand the evolution of trade categories
- **Customs Analysis**: Find historical precedents for modern classifications
- **Academic Research**: Analyze long-term trends in trade policy

## Data Sources

The package includes two main datasets:

1. **Full Dataset** (`all_tariffs`): Complete historical records (750,000+ entries)
2. **Census-Enhanced** (`census_selected`): Subset with additional Census Bureau classifications

Data is automatically downloaded on first use (~5.6GB) and stored in `~/.tariff_search/data/`.

## Performance

- **Search Speed**: <100ms for top-10 results with FAISS enabled
- **Memory Usage**: 2-3GB for full dataset
- **Initial Download**: One-time download of 5.6GB

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

```bash
# Clone the repository
git clone https://github.com/samirk08/tariff-search.git

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## Citation

If you use this package in your research, please cite:

```bibtex
@software{tariff_search,
  title = {Tariff Search: A Search Engine for Historical US Tariff Data},
  author = {Kadariya, Samir},
  year = {2024},
  url = {https://github.com/samirk08/tariff-search}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

This project was developed as part of the Undergraduate Research Opportunities Program (UROP) at MIT, in collaboration with the Trade Lab.