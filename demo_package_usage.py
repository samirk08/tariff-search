"""
Demonstrate how the tariff-search package would be used
"""

print("=== Tariff Search Package Usage Demo ===\n")

print("1. Installation:")
print("   pip install tariff-search")
print("   # or with GPU support:")
print("   pip install tariff-search[faiss-gpu]\n")

print("2. First-time Usage (auto-downloads data):")
print("   >>> from tariff_search import TariffSearch")
print("   >>> searcher = TariffSearch()  # Auto-downloads ~500MB on first use")
print("   >>> results = searcher.search('Cotton cloth', top_k=5)\n")

print("3. Manual Data Download:")
print("   # Download prepared data")
print("   tariff-search download")
print("   ")
print("   # Or download raw pickle file")
print("   tariff-search download --raw\n")

print("4. Using Your Own Data:")
print("   # Convert existing DataFrame with embeddings")
print("   tariff-search prepare my_data.pkl output_dir --embedding-column description_embedding")
print("   ")
print("   # Or in Python:")
print("   >>> from tariff_search import prepare_data_files")
print("   >>> prepare_data_files('my_data.pkl', 'output_dir')\n")

print("5. Command Line Search:")
print("   # Simple search")
print("   tariff-search search 'Steel pipes' --top-k 10")
print("   ")
print("   # Export as CSV")
print("   tariff-search search 'Coffee beans' --format csv > results.csv")
print("   ")
print("   # Custom weights")
print("   tariff-search search 'Electronic items' --alpha 0.8 --beta 0.1 --gamma 0.1\n")

print("6. Python API:")
print("""
from tariff_search import TariffSearch

# Initialize
searcher = TariffSearch(data_dir='my_data')

# Single search
results = searcher.search(
    query='Leather shoes',
    top_k=5,
    alpha=0.7,  # cosine similarity weight
    beta=0.2,   # Jaccard similarity weight  
    gamma=0.1   # Levenshtein similarity weight
)

# Display results
for idx, row in results.iterrows():
    print(f"Similarity: {row['combined_similarity']:.3f}")
    print(f"Description: {row['Description_N']}")
    print(f"HS Code: {row['HS_N']}")

# Batch search
queries = ['Steel', 'Cotton', 'Electronics']
batch_results = searcher.batch_search(queries, top_k=3)
""")

print("\n=== Key Features ===")
print("✓ Fast vector similarity search (with optional Faiss)")
print("✓ Multiple similarity metrics (cosine, Jaccard, Levenshtein)")
print("✓ Efficient data storage (embeddings separated from metadata)")
print("✓ Auto-download capability for easy setup")
print("✓ Both CLI and Python API")
print("✓ Batch processing support")