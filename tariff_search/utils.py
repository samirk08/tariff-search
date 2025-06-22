"""
Utility functions for the tariff search package.
"""

import os
from pathlib import Path


def get_default_data_dir():
    """
    Get the default persistent data directory for storing tariff search data.
    
    The function checks in the following order:
    1. TARIFF_SEARCH_DATA_DIR environment variable
    2. ~/.tariff_search/data/ in user's home directory
    
    Returns:
        str: Path to the data directory
    """
    # Check environment variable first
    if 'TARIFF_SEARCH_DATA_DIR' in os.environ:
        return os.environ['TARIFF_SEARCH_DATA_DIR']
    
    # Use user's home directory for persistent storage
    return os.path.join(Path.home(), '.tariff_search', 'data')