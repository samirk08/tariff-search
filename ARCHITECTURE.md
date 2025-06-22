# Architecture Documentation

This document provides a detailed technical overview of the Tariff Search Package architecture, including design decisions, component interactions, and implementation details.

## Table of Contents
1. [System Overview](#system-overview)
2. [Core Components](#core-components)
3. [Data Flow Architecture](#data-flow-architecture)
4. [Search Algorithm Design](#search-algorithm-design)
5. [Performance Optimizations](#performance-optimizations)
6. [Storage Architecture](#storage-architecture)
7. [API Design Patterns](#api-design-patterns)
8. [Extension Points](#extension-points)

## System Overview

The Tariff Search Package is designed as a modular, scalable search engine with the following architectural principles:

- **Separation of Concerns**: Clear boundaries between data loading, search logic, and user interfaces
- **Lazy Loading**: Data is loaded only when needed to minimize memory footprint
- **Pluggable Components**: Easy to extend with new similarity metrics or data sources
- **Performance First**: Optimized for sub-second search responses

### High-Level Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        Application Layer                        │
├─────────────────────────┬───────────────────────────────────────┤
│      CLI Interface      │           Python API                  │
│       (cli.py)          │         (TariffSearch)               │
├─────────────────────────┴───────────────────────────────────────┤
│                      Search Engine Core                         │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │  Query Parser   │  │ Similarity Engine │  │ Result Ranker │ │
│  └─────────────────┘  └──────────────────┘  └───────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                       Data Access Layer                         │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │  Data Loader    │  │  Cache Manager   │  │ Index Manager │ │
│  └─────────────────┘  └──────────────────┘  └───────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                      Storage Layer                              │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │   Embeddings    │  │    Metadata      │  │  FAISS Index  │ │
│  │   (.npy files)  │  │  (.pkl files)    │  │  (optional)   │ │
│  └─────────────────┘  └──────────────────┘  └───────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. TariffSearch Class (`searcher.py`)

The main search engine class that orchestrates all operations.

```python
class TariffSearch:
    def __init__(self, data_source='combined', device='cpu', 
                 use_faiss=True, data_dir=None):
        self.data_source = data_source
        self.device = device
        self.use_faiss = use_faiss
        self.data_dir = data_dir or get_default_data_dir()
        self._load_model()
        self._load_data()
```

**Responsibilities:**
- Initialize transformer model for encoding queries
- Load pre-computed embeddings and metadata
- Build FAISS index if enabled
- Execute searches and rank results

### 2. Data Loader Module

Handles efficient loading of large embedding files and metadata.

```python
class DataLoader:
    def load_embeddings(self, path):
        # Memory-mapped loading for large files
        return np.load(path, mmap_mode='r')
    
    def load_metadata(self, path):
        # Lazy loading with pickle
        return pd.read_pickle(path)
```

**Key Features:**
- Memory-mapped numpy arrays for embeddings
- Lazy DataFrame loading for metadata
- Automatic fallback for missing files

### 3. Similarity Engine

Implements the multi-metric similarity calculation.

```python
class SimilarityEngine:
    def calculate_similarities(self, query_embedding, doc_embeddings, 
                             query_text, doc_texts, weights):
        # Vectorized cosine similarity
        cosine_sims = self._cosine_similarity_batch(
            query_embedding, doc_embeddings
        )
        
        # Parallel Jaccard calculation
        jaccard_sims = self._jaccard_similarity_parallel(
            query_text, doc_texts
        )
        
        # Optimized Levenshtein with caching
        lev_sims = self._levenshtein_similarity_cached(
            query_text, doc_texts
        )
        
        return self._combine_scores(
            cosine_sims, jaccard_sims, lev_sims, weights
        )
```

### 4. FAISS Integration

Optional component for accelerated similarity search.

```python
class FAISSIndexManager:
    def build_index(self, embeddings):
        # Normalize embeddings for inner product
        normalized = embeddings / np.linalg.norm(
            embeddings, axis=1, keepdims=True
        )
        
        # Create flat index for exact search
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(normalized)
        
        return index
    
    def search(self, query_embedding, k):
        # Fast approximate nearest neighbor search
        scores, indices = self.index.search(query_embedding, k)
        return indices, scores
```

## Data Flow Architecture

### Search Request Flow

```
1. User Query
   ↓
2. Query Preprocessing
   - Lowercase conversion
   - Special character handling
   - Tokenization
   ↓
3. Query Encoding
   - Transform to embedding using sentence-transformers
   - Cache encoded query
   ↓
4. Similarity Search
   - FAISS search (if enabled) for top candidates
   - Calculate multi-metric similarities
   - Apply year filtering (if specified)
   ↓
5. Result Ranking
   - Sort by combined similarity score
   - Apply top-k selection
   - Format results
   ↓
6. Response Generation
   - Return DataFrame or list of results
   - Include metadata and scores
```

### Data Loading Flow

```
1. Initialization
   ↓
2. Check Data Directory
   - Verify data files exist
   - Trigger download if missing
   ↓
3. Load Embeddings
   - Memory-map numpy arrays
   - Validate dimensions
   ↓
4. Load Metadata
   - Read pickle files
   - Index by year if needed
   ↓
5. Build Indices
   - Create FAISS index (optional)
   - Build year lookup tables
   ↓
6. Ready for Search
```

## Search Algorithm Design

### Multi-Metric Similarity

The search algorithm combines three complementary similarity metrics:

```python
def combined_similarity(query, document, weights):
    # Semantic similarity via embeddings
    semantic = cosine_similarity(
        query_embedding, 
        document_embedding
    )
    
    # Token overlap via Jaccard
    tokens_q = set(query.lower().split())
    tokens_d = set(document.lower().split())
    jaccard = len(tokens_q & tokens_d) / len(tokens_q | tokens_d)
    
    # String similarity via Levenshtein
    max_len = max(len(query), len(document))
    levenshtein = 1 - (edit_distance(query, document) / max_len)
    
    # Weighted combination
    return (weights['cosine'] * semantic +
            weights['jaccard'] * jaccard +
            weights['levenshtein'] * levenshtein)
```

### Optimization Strategies

1. **Vectorized Operations**: Use NumPy for batch calculations
2. **Caching**: Cache frequently used calculations
3. **Early Termination**: Stop calculating when threshold met
4. **Parallel Processing**: Use multiprocessing for CPU-intensive operations

## Performance Optimizations

### 1. Memory Management

```python
class MemoryOptimizedLoader:
    def __init__(self):
        self.embedding_cache = {}
        self.cache_size = 0
        self.max_cache_size = 1_000_000_000  # 1GB
    
    def load_embeddings_chunked(self, path, chunk_size=10000):
        # Load embeddings in chunks to avoid memory spikes
        embeddings = np.load(path, mmap_mode='r')
        for i in range(0, len(embeddings), chunk_size):
            yield embeddings[i:i+chunk_size]
```

### 2. Search Optimization

```python
class OptimizedSearch:
    def search_with_pruning(self, query, threshold=0.5):
        # Use FAISS for initial candidate selection
        candidates_idx, _ = self.faiss_index.search(
            query_embedding, k=1000
        )
        
        # Apply more expensive metrics only to candidates
        candidates = self.embeddings[candidates_idx]
        similarities = self.calculate_similarities(
            query, candidates
        )
        
        # Return only results above threshold
        return candidates[similarities > threshold]
```

### 3. Caching Strategy

```python
class SearchCache:
    def __init__(self, max_size=1000):
        self.cache = OrderedDict()
        self.max_size = max_size
    
    def get(self, query):
        if query in self.cache:
            # Move to end (LRU)
            self.cache.move_to_end(query)
            return self.cache[query]
        return None
    
    def put(self, query, results):
        self.cache[query] = results
        if len(self.cache) > self.max_size:
            # Remove least recently used
            self.cache.popitem(last=False)
```

## Storage Architecture

### File Organization

```
~/.tariff_search/data/
├── prepared/                          # Optimized data files
│   ├── all_tariffs_emb_metadata.pkl  # Metadata DataFrame
│   ├── all_tariffs_emb_vectors.npy   # Embedding vectors
│   ├── census_selected_df_metadata.pkl
│   └── census_selected_df_vectors.npy
├── raw/                              # Original pickle files
│   ├── df_with_embeddings.pkl
│   └── census_selected_df.pkl
└── indices/                          # Optional indices
    └── faiss_index.bin
```

### Data Format Specifications

#### Embedding Files (.npy)
- **Format**: NumPy array, float32
- **Shape**: (n_items, 768)
- **Compression**: None (for memory-mapping)
- **Byte Order**: Native

#### Metadata Files (.pkl)
- **Format**: Pandas DataFrame, pickle protocol 4
- **Columns**: Standard set defined in DATA_PIPELINE.md
- **Index**: Integer index for fast lookup
- **Compression**: None

## API Design Patterns

### 1. Builder Pattern for Configuration

```python
class SearchBuilder:
    def __init__(self):
        self.config = {}
    
    def with_data_source(self, source):
        self.config['data_source'] = source
        return self
    
    def with_device(self, device):
        self.config['device'] = device
        return self
    
    def with_faiss(self, use_faiss):
        self.config['use_faiss'] = use_faiss
        return self
    
    def build(self):
        return TariffSearch(**self.config)

# Usage
searcher = (SearchBuilder()
    .with_data_source('combined')
    .with_device('cuda')
    .with_faiss(True)
    .build())
```

### 2. Strategy Pattern for Similarity Metrics

```python
class SimilarityStrategy(ABC):
    @abstractmethod
    def calculate(self, query, document):
        pass

class CosineSimilarity(SimilarityStrategy):
    def calculate(self, query, document):
        return np.dot(query, document) / (
            np.linalg.norm(query) * np.linalg.norm(document)
        )

class JaccardSimilarity(SimilarityStrategy):
    def calculate(self, query, document):
        q_tokens = set(query.lower().split())
        d_tokens = set(document.lower().split())
        return len(q_tokens & d_tokens) / len(q_tokens | d_tokens)
```

### 3. Facade Pattern for Simple API

```python
class TariffSearchFacade:
    def __init__(self):
        self.searcher = TariffSearch()
    
    def find_similar(self, query, limit=5):
        # Simple interface hiding complexity
        return self.searcher.search(query, top_k=limit)
    
    def find_by_year(self, query, year):
        return self.searcher.search(query, year=year)
```

## Extension Points

### 1. Adding New Similarity Metrics

```python
# 1. Create new similarity class
class BM25Similarity(SimilarityStrategy):
    def calculate(self, query, document):
        # Implement BM25 algorithm
        pass

# 2. Register in similarity engine
similarity_engine.register_metric('bm25', BM25Similarity())

# 3. Use in search
searcher.search(query, similarity_weights={'bm25': 0.3, ...})
```

### 2. Adding New Data Sources

```python
# 1. Create data loader
class CustomDataLoader:
    def load(self, path):
        # Load custom format
        pass

# 2. Register loader
data_manager.register_loader('custom', CustomDataLoader())

# 3. Use in initialization
searcher = TariffSearch(data_source='custom')
```

### 3. Custom Result Ranking

```python
# 1. Create ranking strategy
class PopularityRanker:
    def rank(self, results, query_metadata):
        # Boost popular items
        return results.sort_values(
            by=['popularity', 'similarity'], 
            ascending=[False, False]
        )

# 2. Set custom ranker
searcher.set_ranker(PopularityRanker())
```

## Security Considerations

### Input Validation
- Sanitize query inputs to prevent injection
- Validate file paths for data loading
- Limit query length to prevent DoS

### Data Protection
- Use read-only file access where possible
- Validate downloaded files with checksums
- Implement rate limiting for API access

## Monitoring and Debugging

### Logging Strategy

```python
import logging

logger = logging.getLogger('tariff_search')

class TariffSearch:
    def search(self, query, **kwargs):
        logger.info(f"Search query: {query}")
        logger.debug(f"Search params: {kwargs}")
        
        try:
            results = self._execute_search(query, **kwargs)
            logger.info(f"Found {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
```

### Performance Metrics

```python
class PerformanceMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
    
    @contextmanager
    def measure(self, operation):
        start = time.time()
        yield
        duration = time.time() - start
        self.metrics[operation].append(duration)
        
        if duration > 1.0:  # Log slow operations
            logger.warning(f"{operation} took {duration:.2f}s")
```

## Future Enhancements

1. **Distributed Search**: Shard data across multiple nodes
2. **GPU Acceleration**: Use CUDA for similarity calculations
3. **Incremental Updates**: Support adding new data without full rebuild
4. **Multi-language Support**: Extend to non-English descriptions
5. **Streaming API**: Support real-time data ingestion