# Code Examples and Usage Patterns

This document provides comprehensive examples of using the Tariff Search Package for various use cases, from basic searches to advanced analysis.

## Table of Contents
1. [Basic Usage](#basic-usage)
2. [Advanced Search Options](#advanced-search-options)
3. [Working with Results](#working-with-results)
4. [Historical Analysis](#historical-analysis)
5. [Batch Processing](#batch-processing)
6. [Integration Examples](#integration-examples)
7. [Performance Optimization](#performance-optimization)
8. [Custom Workflows](#custom-workflows)

## Basic Usage

### Simple Search

```python
from tariff_search import TariffSearch

# Initialize the searcher
searcher = TariffSearch()

# Basic search
results = searcher.search("leather handbags", top_k=5)

# Print results
for idx, row in results.iterrows():
    print(f"Score: {row['combined_similarity']:.3f}")
    print(f"Description: {row['concord_name']}")
    print(f"HS Code: {row['HS']}")
    print(f"Year: {row['year']}")
    print("-" * 50)
```

### Search with Year Filter

```python
# Search for items in a specific year
results_2023 = searcher.search("electric vehicles", year=2023, top_k=10)

# Search for historical items
results_1950 = searcher.search("automobiles", year=1950, top_k=5)

# Compare descriptions across time
print("1950 Automobile Classifications:")
for _, row in results_1950.iterrows():
    print(f"- {row['concord_name']}")

print("\n2023 Electric Vehicle Classifications:")
for _, row in results_2023.iterrows():
    print(f"- {row['concord_name']}")
```

## Advanced Search Options

### Custom Similarity Weights

```python
# Emphasize exact word matches
word_focused_weights = {
    'cosine': 0.3,      # Reduce semantic weight
    'jaccard': 0.5,     # Increase word overlap weight
    'levenshtein': 0.2  # Keep string similarity
}

results = searcher.search(
    "cotton shirts for men",
    similarity_weights=word_focused_weights,
    top_k=10
)

# Emphasize semantic meaning
semantic_weights = {
    'cosine': 0.8,      # High semantic weight
    'jaccard': 0.1,     # Low word overlap
    'levenshtein': 0.1  # Low string similarity
}

results = searcher.search(
    "apparel made from natural fibers",
    similarity_weights=semantic_weights,
    top_k=10
)
```

### Using Different Data Sources

```python
# Use only census-selected data for faster searches
census_searcher = TariffSearch(data_source='census_selected')
results = census_searcher.search("smartphones", top_k=5)

# Use all tariffs for comprehensive coverage
full_searcher = TariffSearch(data_source='all_tariffs')
results = full_searcher.search("rare earth metals", top_k=10)

# Use combined dataset (default)
combined_searcher = TariffSearch(data_source='combined')
results = combined_searcher.search("solar panels", top_k=5)
```

## Working with Results

### Extracting Specific Information

```python
results = searcher.search("coffee beans", top_k=20)

# Extract unique HS codes
unique_hs_codes = results['HS'].unique()
print(f"Found {len(unique_hs_codes)} unique HS codes:")
for code in unique_hs_codes[:5]:
    print(f"  - {code}")

# Group by year
year_groups = results.groupby('year').agg({
    'combined_similarity': 'mean',
    'HS': 'count'
}).rename(columns={'HS': 'count'})

print("\nResults by year:")
print(year_groups)

# Find best match per decade
results['decade'] = (results['year'] // 10) * 10
best_by_decade = results.loc[results.groupby('decade')['combined_similarity'].idxmax()]
```

### Filtering and Sorting Results

```python
# Get high-confidence matches only
high_confidence = results[results['combined_similarity'] > 0.8]

# Sort by multiple criteria
sorted_results = results.sort_values(
    by=['year', 'combined_similarity'],
    ascending=[False, False]
)

# Filter by HS code prefix
hs_filtered = results[results['HS'].str.startswith('8471')]

# Complex filtering
complex_filter = results[
    (results['combined_similarity'] > 0.7) &
    (results['year'] >= 2000) &
    (results['concord_name'].str.contains('computer', case=False))
]
```

## Historical Analysis

### Tracking Product Evolution

```python
def track_product_evolution(searcher, product_name, start_year=1850, end_year=2023):
    """Track how a product classification has evolved over time."""
    evolution = []
    
    for year in range(start_year, end_year + 1, 10):  # Check every 10 years
        results = searcher.search(product_name, year=year, top_k=1)
        
        if not results.empty:
            evolution.append({
                'year': year,
                'description': results.iloc[0]['concord_name'],
                'hs_code': results.iloc[0]['HS'],
                'similarity': results.iloc[0]['combined_similarity']
            })
    
    return pd.DataFrame(evolution)

# Track the evolution of "telephone" classifications
telephone_evolution = track_product_evolution(searcher, "telephone")
print(telephone_evolution)

# Visualize the evolution
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(telephone_evolution['year'], telephone_evolution['similarity'], marker='o')
plt.xlabel('Year')
plt.ylabel('Similarity Score')
plt.title('Classification Similarity for "Telephone" Over Time')
plt.grid(True)
plt.show()
```

### Finding Historical Equivalents

```python
def find_historical_equivalent(searcher, modern_description, target_year):
    """Find the historical equivalent of a modern product."""
    # First, find the modern item
    modern_results = searcher.search(modern_description, year=2023, top_k=1)
    
    if modern_results.empty:
        return None
    
    modern_hs = modern_results.iloc[0]['HS']
    
    # Load the mapping database
    mappings = pd.read_pickle('FINAL_MAPPING_1789_2023.pkl')
    
    # Find items that map to this modern HS code
    historical = mappings[
        (mappings['Mapped_2023_HS'] == modern_hs) &
        (mappings['from_year'] == target_year)
    ]
    
    if not historical.empty:
        return historical.iloc[0]
    return None

# Find 1850 equivalent of modern "smartphones"
historical_phone = find_historical_equivalent(searcher, "smartphones", 1850)
if historical_phone is not None:
    print(f"1850 equivalent: {historical_phone['Description_N']}")
```

## Batch Processing

### Processing Multiple Queries

```python
def batch_search(searcher, queries, **kwargs):
    """Process multiple search queries efficiently."""
    results = {}
    
    for query in queries:
        try:
            results[query] = searcher.search(query, **kwargs)
        except Exception as e:
            print(f"Error searching for '{query}': {e}")
            results[query] = pd.DataFrame()
    
    return results

# Batch search example
queries = [
    "cotton fabric",
    "steel pipes",
    "electronic circuits",
    "leather shoes",
    "plastic bottles"
]

batch_results = batch_search(searcher, queries, top_k=3, year=2023)

# Display summary
for query, results in batch_results.items():
    if not results.empty:
        best_match = results.iloc[0]
        print(f"{query}: {best_match['HS']} - {best_match['concord_name'][:50]}...")
```

### Parallel Processing for Large Batches

```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def parallel_search(queries, num_workers=None):
    """Perform searches in parallel for better performance."""
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    # Create a searcher for each worker
    def search_worker(query):
        searcher = TariffSearch()
        return query, searcher.search(query, top_k=5)
    
    # Process queries in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = dict(executor.map(search_worker, queries))
    
    return results

# Large batch processing
large_query_list = [f"product type {i}" for i in range(100)]
parallel_results = parallel_search(large_query_list)
```

## Integration Examples

### Flask Web API

```python
from flask import Flask, request, jsonify
from tariff_search import TariffSearch

app = Flask(__name__)
searcher = TariffSearch()

@app.route('/search', methods=['POST'])
def search_api():
    data = request.json
    query = data.get('query', '')
    year = data.get('year')
    top_k = data.get('top_k', 5)
    
    try:
        results = searcher.search(query, year=year, top_k=top_k)
        
        # Convert to JSON-serializable format
        response = {
            'query': query,
            'results': results.to_dict('records')
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

### Streamlit Dashboard

```python
import streamlit as st
from tariff_search import TariffSearch
import pandas as pd

@st.cache_resource
def load_searcher():
    return TariffSearch()

st.title("Tariff Search Dashboard")

# Initialize searcher
searcher = load_searcher()

# Search interface
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input("Search Query", "Enter product description...")
with col2:
    year = st.number_input("Year (optional)", min_value=1789, max_value=2023, value=2023)

top_k = st.slider("Number of results", min_value=1, max_value=50, value=10)

if st.button("Search"):
    with st.spinner("Searching..."):
        results = searcher.search(query, year=year if year else None, top_k=top_k)
    
    if not results.empty:
        st.success(f"Found {len(results)} results")
        
        # Display results
        for idx, row in results.iterrows():
            with st.expander(f"{row['HS']} - Score: {row['combined_similarity']:.3f}"):
                st.write(f"**Description:** {row['concord_name']}")
                st.write(f"**Year:** {row['year']}")
                st.write(f"**Similarity Breakdown:**")
                col1, col2, col3 = st.columns(3)
                col1.metric("Cosine", f"{row.get('cosine_similarity', 0):.3f}")
                col2.metric("Jaccard", f"{row.get('jaccard_similarity', 0):.3f}")
                col3.metric("Levenshtein", f"{row.get('levenshtein_similarity', 0):.3f}")
    else:
        st.warning("No results found")
```

## Performance Optimization

### Caching Search Results

```python
from functools import lru_cache
import hashlib

class CachedTariffSearch(TariffSearch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = {}
    
    def _get_cache_key(self, query, **kwargs):
        """Generate cache key from search parameters."""
        key_str = f"{query}_{kwargs}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def search(self, query, **kwargs):
        """Search with caching."""
        cache_key = self._get_cache_key(query, **kwargs)
        
        if cache_key in self._cache:
            return self._cache[cache_key].copy()
        
        results = super().search(query, **kwargs)
        self._cache[cache_key] = results.copy()
        
        # Limit cache size
        if len(self._cache) > 1000:
            # Remove oldest entry
            self._cache.pop(next(iter(self._cache)))
        
        return results

# Use cached searcher
cached_searcher = CachedTariffSearch()

# First search is slow
results1 = cached_searcher.search("cotton shirts")

# Subsequent identical searches are instant
results2 = cached_searcher.search("cotton shirts")
```

### Preloading for Production

```python
class PreloadedSearcher:
    def __init__(self):
        """Initialize and preload all data for production use."""
        print("Initializing searcher...")
        self.searcher = TariffSearch(use_faiss=True)
        
        print("Warming up model...")
        # Warm up the model with a dummy search
        _ = self.searcher.search("test", top_k=1)
        
        print("Building indices...")
        # Pre-build any additional indices
        self._build_year_index()
        
        print("Ready for production use!")
    
    def _build_year_index(self):
        """Build index for fast year-based filtering."""
        self.year_index = {}
        if hasattr(self.searcher, 'df'):
            for year in self.searcher.df['year'].unique():
                self.year_index[year] = self.searcher.df[
                    self.searcher.df['year'] == year
                ].index.tolist()

# Initialize once at startup
production_searcher = PreloadedSearcher()
```

## Custom Workflows

### Finding Related Products

```python
def find_related_products(searcher, hs_code, top_k=10):
    """Find products related to a given HS code."""
    # Get the description for this HS code
    base_results = searcher.df[searcher.df['HS'] == hs_code]
    
    if base_results.empty:
        return pd.DataFrame()
    
    base_description = base_results.iloc[0]['concord_name']
    
    # Search for similar items
    related = searcher.search(base_description, top_k=top_k+1)
    
    # Remove the original item
    related = related[related['HS'] != hs_code]
    
    return related.head(top_k)

# Find products related to HS code 6109.10
related = find_related_products(searcher, "6109.10")
print("Related products:")
for _, row in related.iterrows():
    print(f"- {row['HS']}: {row['concord_name']}")
```

### Building a Recommendation System

```python
class TariffRecommender:
    def __init__(self, searcher):
        self.searcher = searcher
        self.user_history = []
    
    def add_to_history(self, hs_code):
        """Add an HS code to user history."""
        self.user_history.append(hs_code)
    
    def get_recommendations(self, top_k=5):
        """Get recommendations based on user history."""
        if not self.user_history:
            return pd.DataFrame()
        
        # Aggregate descriptions from history
        historical_items = self.searcher.df[
            self.searcher.df['HS'].isin(self.user_history)
        ]
        
        # Create a combined query from historical descriptions
        combined_desc = " ".join(
            historical_items['concord_name'].tolist()
        )
        
        # Search for similar items
        recommendations = self.searcher.search(
            combined_desc, 
            top_k=top_k + len(self.user_history)
        )
        
        # Filter out items already in history
        recommendations = recommendations[
            ~recommendations['HS'].isin(self.user_history)
        ]
        
        return recommendations.head(top_k)

# Example usage
recommender = TariffRecommender(searcher)
recommender.add_to_history("6109.10")  # T-shirts
recommender.add_to_history("6110.20")  # Sweaters
recommendations = recommender.get_recommendations()
```

### Export Functionality

```python
def export_search_results(results, format='csv', filename=None):
    """Export search results in various formats."""
    if filename is None:
        filename = f"search_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    
    if format == 'csv':
        results.to_csv(f"{filename}.csv", index=False)
        print(f"Results saved to {filename}.csv")
    
    elif format == 'excel':
        with pd.ExcelWriter(f"{filename}.xlsx") as writer:
            results.to_excel(writer, index=False, sheet_name='Search Results')
            
            # Add metadata sheet
            metadata = pd.DataFrame({
                'Property': ['Search Date', 'Number of Results', 'Top Score'],
                'Value': [
                    pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    len(results),
                    results['combined_similarity'].max() if not results.empty else 0
                ]
            })
            metadata.to_excel(writer, index=False, sheet_name='Metadata')
        print(f"Results saved to {filename}.xlsx")
    
    elif format == 'json':
        results.to_json(f"{filename}.json", orient='records', indent=2)
        print(f"Results saved to {filename}.json")
    
    else:
        raise ValueError(f"Unsupported format: {format}")

# Export example
results = searcher.search("automotive parts", top_k=20)
export_search_results(results, format='excel', filename='auto_parts_search')
```

## Error Handling and Debugging

### Robust Search Wrapper

```python
def safe_search(searcher, query, **kwargs):
    """Perform search with comprehensive error handling."""
    try:
        # Validate inputs
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        
        if 'year' in kwargs and kwargs['year']:
            year = kwargs['year']
            if not (1789 <= year <= 2023):
                raise ValueError(f"Year must be between 1789 and 2023, got {year}")
        
        if 'top_k' in kwargs:
            top_k = kwargs['top_k']
            if not (1 <= top_k <= 1000):
                raise ValueError(f"top_k must be between 1 and 1000, got {top_k}")
        
        # Perform search
        results = searcher.search(query, **kwargs)
        
        # Log search
        print(f"Search completed: '{query}' returned {len(results)} results")
        
        return results
        
    except Exception as e:
        print(f"Search error: {e}")
        # Return empty DataFrame on error
        return pd.DataFrame()

# Example with error handling
results = safe_search(searcher, "electronics", year=2025)  # Invalid year
results = safe_search(searcher, "", top_k=5)  # Empty query
results = safe_search(searcher, "valid query", top_k=10)  # Valid search
```

These examples demonstrate the flexibility and power of the Tariff Search Package for various use cases, from simple searches to complex analytical workflows.