#!/usr/bin/env python3
"""
Test script to demonstrate persistent storage functionality.
"""

from tariff_search import TariffSearch, get_default_data_dir
import os

def main():
    print("Testing Tariff Search Persistent Storage")
    print("=" * 50)
    
    # Show where data will be stored
    data_dir = get_default_data_dir()
    print(f"Default data directory: {data_dir}")
    
    # Check if data already exists
    data_exists = os.path.exists(data_dir) and len(os.listdir(data_dir)) > 0 if os.path.exists(data_dir) else False
    print(f"Data already downloaded: {data_exists}")
    
    print("\nInitializing TariffSearch...")
    try:
        # This will download data on first run, but not on subsequent runs
        searcher = TariffSearch()
        
        print("\nSearching for 'Cotton cloth'...")
        results = searcher.search("Cotton cloth", top_k=3)
        
        print("\nResults:")
        for idx, row in results.iterrows():
            print(f"\nSimilarity: {row['combined_similarity']:.3f}")
            print(f"Description: {row['Description_N']}")
            print(f"HS Code: {row['HS_N']}")
            
        print("\n" + "=" * 50)
        print("SUCCESS! Data is stored persistently at:")
        print(f"{data_dir}")
        print("\nFuture runs will NOT re-download the data.")
        print("To use a custom location, initialize with:")
        print('  searcher = TariffSearch(data_dir="/your/custom/path")')
        print("\nTo set a global custom location, use environment variable:")
        print('  export TARIFF_SEARCH_DATA_DIR="/your/custom/path"')
        
    except Exception as e:
        print(f"\nError: {e}")
        print("If download fails, you may need to manually download the data.")

if __name__ == "__main__":
    main()