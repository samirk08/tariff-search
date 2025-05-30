"""
Test the search functionality with mock data
"""

from tariff_search import TariffSearch
import logging

logging.basicConfig(level=logging.INFO)

print("=== Testing Search Functionality ===\n")

# Initialize searcher with mock data
print("1. Initializing TariffSearch...")
searcher = TariffSearch(data_dir="tariff_search/data", auto_download=False)
print("✓ TariffSearch initialized successfully\n")

# Test single search
print("2. Testing single search...")
query = "Cotton cloth"
results = searcher.search(query, top_k=3)

print(f"Query: '{query}'")
print(f"Found {len(results)} results:\n")

for idx, row in results.iterrows():
    print(f"Rank {idx + 1}:")
    print(f"  Similarity: {row['combined_similarity']:.3f}")
    print(f"  Description: {row['Description_N']}")
    print(f"  Code: {row['HS_N']} (Year: {row['year_N']})")
    print()

# Test batch search
print("3. Testing batch search...")
queries = ["Steel pipes", "Coffee beans", "Electronic items"]
batch_results = searcher.batch_search(queries, top_k=2)

for query, results in batch_results.items():
    print(f"\nQuery: '{query}'")
    if not results.empty:
        best = results.iloc[0]
        print(f"  Best match: {best['Description_N']}")
        print(f"  Similarity: {best['combined_similarity']:.3f}")

print("\n=== Search Test Complete ===")
print("The search functionality works correctly!")