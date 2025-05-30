"""
Create mock data files to demonstrate the package functionality
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

# Create data directory
data_dir = Path("tariff_search/data")
data_dir.mkdir(parents=True, exist_ok=True)

# Create mock data
n_items = 100
embedding_dim = 768

# Create mock embeddings
embeddings = np.random.randn(n_items, embedding_dim).astype(np.float32)

# Create mock metadata
metadata = pd.DataFrame({
    'Description_N': [f"Tariff item {i}: {desc}" for i, desc in enumerate([
        "Cotton cloth, bleached",
        "Leather shoes for men",
        "Electronic calculators",
        "Fresh apples",
        "Wooden furniture",
        "Steel pipes",
        "Plastic containers",
        "Coffee beans",
        "Silk fabric",
        "Rubber tires",
    ] * 10)],
    'HS_N': [f"HS{i:06d}" for i in range(n_items)],
    'year_N': [1900 + i % 124 for i in range(n_items)],
    'best_match_hscode': [f"HS2023_{i%100:04d}" for i in range(n_items)],
    'best_match_description': [f"Modern description for item {i}" for i in range(n_items)]
})

# Save files
np.save(data_dir / 'embeddings.npy', embeddings)

with open(data_dir / 'metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

info = {
    'n_items': n_items,
    'embedding_dim': embedding_dim,
    'columns': list(metadata.columns),
    'embedding_column': 'description_embedding'
}

with open(data_dir / 'info.pkl', 'wb') as f:
    pickle.dump(info, f)

print(f"Created mock data files in {data_dir}")
print(f"- {n_items} items with {embedding_dim}-dim embeddings")
print(f"- Metadata columns: {', '.join(metadata.columns)}")