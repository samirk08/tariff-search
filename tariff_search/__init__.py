"""
Tariff Search Module

A fast and efficient search tool for finding similar tariff descriptions
across historical US tariff data from 1789-2023.
"""

from .searcher import TariffSearch
from .data_preparation import prepare_data_files
from .download import download_prepared_data, download_raw_pickle
from .utils import get_default_data_dir

__version__ = "0.1.0"
__all__ = ["TariffSearch", "prepare_data_files", "download_prepared_data", "download_raw_pickle", "get_default_data_dir"]