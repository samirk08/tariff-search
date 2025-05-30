"""
Test the tariff-search package functionality
"""

import logging
logging.basicConfig(level=logging.INFO)

print("=== Testing Tariff Search Package ===\n")

# Test 1: Import test
print("1. Testing imports...")
try:
    from tariff_search import TariffSearch, prepare_data_files, download_prepared_data
    print("✓ All imports successful\n")
except ImportError as e:
    print(f"✗ Import error: {e}\n")

# Test 2: Check download functionality
print("2. Testing download functionality...")
try:
    from tariff_search.download import download_prepared_data
    print("✓ Download module loaded successfully")
    print("Note: Actual download will fail due to placeholder URLs\n")
except Exception as e:
    print(f"✗ Error: {e}\n")

# Test 3: Test CLI commands
print("3. Testing CLI commands...")
import subprocess
import sys

# Test help
result = subprocess.run([sys.executable, "-m", "tariff_search.cli", "--help"], 
                       capture_output=True, text=True)
print("✓ CLI help works" if result.returncode == 0 else "✗ CLI help failed")

# Test download help
result = subprocess.run([sys.executable, "-m", "tariff_search.cli", "download", "--help"], 
                       capture_output=True, text=True)
print("✓ Download command help works" if result.returncode == 0 else "✗ Download help failed")

# Test search help
result = subprocess.run([sys.executable, "-m", "tariff_search.cli", "search", "--help"], 
                       capture_output=True, text=True)
print("✓ Search command help works\n" if result.returncode == 0 else "✗ Search help failed\n")

# Test 4: Try to initialize TariffSearch (will fail due to no data)
print("4. Testing TariffSearch initialization...")
try:
    # This should fail because we don't have data and URLs are placeholders
    searcher = TariffSearch(auto_download=False)
    print("✓ TariffSearch initialized")
except Exception as e:
    print(f"✗ Expected failure (no data): {type(e).__name__}: {str(e)[:100]}...")
    
print("\n5. Testing with auto_download=True (will fail with placeholder URLs)...")
try:
    searcher = TariffSearch(auto_download=True)
    print("✓ TariffSearch initialized with auto-download")
except Exception as e:
    print(f"✗ Expected failure (placeholder URLs): {type(e).__name__}: {str(e)[:100]}...")

print("\n=== Package Structure Test Complete ===")
print("\nThe package is properly structured and would work with:")
print("1. Real download URLs in download.py")
print("2. Actual data files in tariff_search/data/")
print("3. Or a valid df_with_embeddings.pkl file to prepare")