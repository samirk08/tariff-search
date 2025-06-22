# Tariff Search Package Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Data Pipeline](#data-pipeline)
4. [API Reference](#api-reference)
5. [Implementation Details](#implementation-details)
6. [Performance Considerations](#performance-considerations)
7. [Development Guide](#development-guide)
8. [Additional Resources](#additional-resources)

## Project Overview

The Tariff Search Package is a specialized search engine designed to find similar US tariff descriptions across historical data from 1789 to 2023. It uses state-of-the-art natural language processing techniques combined with traditional string similarity metrics to provide accurate and fast similarity search across over 750,000 historical tariff descriptions.

### Key Features
- **Semantic Search**: Uses transformer-based embeddings (all-mpnet-base-v2) for understanding meaning
- **Multi-metric Matching**: Combines cosine similarity, Jaccard similarity, and Levenshtein distance
- **Fast Search**: Leverages FAISS for efficient nearest neighbor search
- **Historical Coverage**: Complete US tariff data from 1789 to 2023
- **Persistent Storage**: One-time download with automatic data management
- **Flexible Interface**: Both Python API and command-line interface

## System Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
├──────────────────────┬──────────────────────────────────────┤
│     CLI (cli.py)     │         Python API (searcher.py)     │
├──────────────────────┴──────────────────────────────────────┤
│                     Core Search Engine                       │
│                      (TariffSearch)                          │
├──────────────────────────────────────────────────────────────┤
│                     Similarity Metrics                       │
│  ┌────────────┐  ┌────────────┐  ┌─────────────────────┐   │
│  │   Cosine   │  │  Jaccard   │  │    Levenshtein      │   │
│  │ Similarity │  │ Similarity │  │     Distance        │   │
│  └────────────┘  └────────────┘  └─────────────────────┘   │
├──────────────────────────────────────────────────────────────┤
│                    Data Layer                                │
│  ┌────────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │   Embeddings   │  │   Metadata   │  │  FAISS Index   │  │
│  │  (NPY files)   │  │ (PKL files)  │  │   (Optional)   │  │
│  └────────────────┘  └──────────────┘  └────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### Data Storage Structure

The package uses a persistent storage model with the following structure:

```
~/.tariff_search/data/
├── all_tariffs_emb_metadata.pkl    # Metadata for all tariff descriptions
├── all_tariffs_emb_vectors.npy     # Pre-computed embeddings
├── census_selected_df_metadata.pkl  # Census-processed items metadata
└── census_selected_df_vectors.npy   # Census-processed embeddings
```

## Data Pipeline

The creation of the pickle database involves several stages:

### 1. Data Collection and Preparation
- **Source**: Historical US tariff data from 1789-2023
- **Format**: CSV files containing tariff descriptions, HS codes, and metadata
- **Processing**: Clean and standardize descriptions across different historical periods

### 2. Embedding Generation
- **Model**: sentence-transformers/all-mpnet-base-v2
- **Process**: 
  - Load tariff descriptions for each year
  - Generate 768-dimensional embeddings for each description
  - Store embeddings separately for efficient loading

### 3. Year-to-Year Mapping
- **Approach**: Map tariff items between consecutive years
- **Metrics**: Combined similarity using:
  - Cosine similarity (semantic meaning)
  - Jaccard similarity (word overlap)
  - Levenshtein distance (string similarity)
- **Output**: Mapping files showing how tariff items evolve year-to-year

### 4. Consolidation to 2023
- **Goal**: Trace all historical items to their 2023 HS code equivalents
- **Method**: Follow mapping chains from historical years to 2023
- **Result**: Complete mapping database with modern HS code resolutions

### 5. Census Algorithm Enhancement
- **Purpose**: Handle low-confidence mappings
- **Process**:
  - Identify items with similarity scores below threshold
  - Query US Census HS classification website
  - Use GPT-4 to assist in classification selection
  - Update mapping database with enhanced classifications

## API Reference

### TariffSearch Class

```python
from tariff_search import TariffSearch

# Initialize the search engine
searcher = TariffSearch(
    data_source='combined',  # 'all_tariffs', 'census_selected', or 'combined'
    device='cpu',           # 'cpu' or 'cuda'
    use_faiss=True,        # Enable FAISS for faster search
    data_dir=None          # Custom data directory (optional)
)
```

#### Methods

##### search(query, year=None, top_k=5, similarity_weights=None)
Find similar tariff descriptions.

**Parameters:**
- `query` (str): Search query text
- `year` (int, optional): Filter results by specific year
- `top_k` (int): Number of results to return
- `similarity_weights` (dict, optional): Custom weights for similarity metrics

**Returns:**
- List of dictionaries containing matched items with scores

**Example:**
```python
results = searcher.search(
    "cotton shirts", 
    year=2023, 
    top_k=10,
    similarity_weights={'cosine': 0.5, 'jaccard': 0.3, 'levenshtein': 0.2}
)
```

##### prepare_all_data()
Download and prepare all data files if not present.

```python
searcher.prepare_all_data()
```

### Command-Line Interface

```bash
# Basic search
tariff-search search "leather shoes" --year 2023 --top-k 5

# Download data files
tariff-search download --prepared

# Prepare data from raw pickle files
tariff-search prepare --input-dir ./raw_data --output-dir ./prepared_data
```

## Implementation Details

### Similarity Calculation

The package uses a weighted combination of three similarity metrics:

1. **Cosine Similarity**: Measures semantic similarity using embeddings
   ```python
   cosine_sim = np.dot(query_embedding, doc_embedding) / (norm(query_embedding) * norm(doc_embedding))
   ```

2. **Jaccard Similarity**: Measures word overlap
   ```python
   jaccard_sim = len(set1.intersection(set2)) / len(set1.union(set2))
   ```

3. **Levenshtein Similarity**: Normalized edit distance
   ```python
   lev_sim = 1 - (levenshtein_distance(str1, str2) / max(len(str1), len(str2)))
   ```

### Default Weights
- Cosine: 50%
- Jaccard: 25%
- Levenshtein: 25%

### FAISS Integration

When enabled, FAISS provides approximate nearest neighbor search:
- Index type: IndexFlatIP (Inner Product)
- Preprocessing: L2 normalization of embeddings
- Performance: ~100x faster for large-scale searches

## Performance Considerations

### Memory Usage
- Full dataset: ~5.6GB on disk, ~2-3GB in memory
- Census-selected only: ~1GB in memory
- Embeddings are loaded as memory-mapped arrays when possible

### Search Performance
- Without FAISS: O(n) linear search
- With FAISS: O(log n) approximate search
- Typical query time: <100ms for top-10 results

### Optimization Tips
1. Use FAISS for production deployments
2. Filter by year when possible to reduce search space
3. Use census-selected dataset for faster loads if full coverage not needed
4. Implement caching for repeated queries

## Development Guide

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/tariff-search-package.git
cd tariff-search-package

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=tariff_search

# Run specific test file
pytest tests/test_searcher.py
```

### Adding New Features

1. **New Similarity Metrics**: Extend `_calculate_combined_similarity()` in `searcher.py`
2. **New Data Sources**: Add loading logic in `_load_data()` method
3. **New CLI Commands**: Add command handler in `cli.py`

### Code Style

The project follows PEP 8 guidelines:
```bash
# Format code
black tariff_search/

# Check linting
flake8 tariff_search/

# Type checking
mypy tariff_search/
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Ensure all tests pass
5. Submit a pull request

### Release Process

1. Update version in `setup.py` and `pyproject.toml`
2. Update CHANGELOG.md
3. Create git tag: `git tag v0.x.x`
4. Build distribution: `python -m build`
5. Upload to PyPI: `twine upload dist/*`

## Additional Resources

### Documentation Files
- [Data Pipeline Details](DATA_PIPELINE.md) - Comprehensive guide to how the database was created
- [Architecture Guide](ARCHITECTURE.md) - Technical design and implementation details
- [Code Examples](EXAMPLES.md) - Extensive usage examples and patterns
- [Persistent Storage Guide](PERSISTENT_STORAGE.md) - Data storage and management

### External Resources
- [GitHub Repository](https://github.com/samirk08/tariff-search) - Source code and issue tracking
- [PyPI Package](https://pypi.org/project/tariff-search/) - Python package index listing
- [MIT Trade Lab](https://tradelab.mit.edu/) - Research group behind the project

### Related Projects
- [US Census HS Search](https://uscensus.prod.3ceonline.com/) - Official HS classification search
- [Sentence Transformers](https://www.sbert.net/) - The embedding model used
- [FAISS](https://github.com/facebookresearch/faiss) - Similarity search library