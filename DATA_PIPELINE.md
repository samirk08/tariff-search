# Data Pipeline Documentation

This document explains how the historical tariff database was created, including the data processing pipeline, algorithms used, and the structure of the final pickle database.

## Table of Contents
1. [Overview](#overview)
2. [Data Sources](#data-sources)
3. [Processing Pipeline](#processing-pipeline)
4. [Database Structure](#database-structure)
5. [Column Descriptions](#column-descriptions)
6. [Algorithm Details](#algorithm-details)
7. [Quality Assurance](#quality-assurance)

## Overview

The tariff search database maps historical US tariff descriptions from 1789 to their modern 2023 Harmonized System (HS) code equivalents. This mapping enables researchers to track how specific products have been classified over 234 years of US trade policy.

### Key Challenges Addressed
- **Terminology Evolution**: Product names and descriptions change significantly over time
- **Classification Changes**: The structure of tariff systems has evolved from simple lists to complex hierarchical codes
- **Missing Links**: Some years have incomplete data or classification changes that break direct mappings
- **Scale**: Processing over 750,000 tariff descriptions across 234 years

## Data Sources

### Primary Data
- **Historical Tariff Schedules**: Digitized US tariff schedules from 1789-2023
- **Format**: CSV files with columns for year, tariff description, and various classification codes
- **Source File**: `all_tariffs.csv` containing the complete historical record

### Reference Data
- **2023 HS Codes**: Current Harmonized System codes as the target classification
- **US Census Bureau**: HS classification lookup for validation
- **Trade Concordances**: Various concordance tables for different classification systems

## Processing Pipeline

### Stage 1: Data Preparation and Cleaning
```
Input: all_tariffs.csv
Process:
  1. Load and validate historical tariff data
  2. Standardize text formatting (lowercase, remove special characters)
  3. Handle missing values and data inconsistencies
  4. Group by year for processing
Output: Cleaned DataFrame ready for embedding generation
```

### Stage 2: Embedding Generation
```
Script: make_embeddings.ipynb
Model: sentence-transformers/all-mpnet-base-v2
Process:
  1. Load cleaned tariff descriptions for each year
  2. Generate 768-dimensional embeddings for each description
  3. Save embeddings as pickle files by year
Output: embeddings/embeddings_[year].pkl files
```

### Stage 3: Year-to-Year Mapping
```
Scripts: process_years.py, process_years_N_N1.py
Process:
  1. For each consecutive year pair (N, N+1):
     a. Load embeddings for both years
     b. Calculate similarity metrics
     c. Find best matches based on combined similarity
  2. Create bidirectional mappings:
     - N → N+1 (forward mapping)
     - N+1 → N (backward mapping)
Output: CSV files in N_N1_mappings/ and N1_N_mappings/ directories
```

### Stage 4: Consolidation to 2023
```
Script: combine_N_N1.ipynb
Process:
  1. Load all year-to-year mappings
  2. Trace mapping chains from historical years to 2023
  3. Handle mapping conflicts and ambiguities
  4. Assign 2023 HS codes to all historical items
Output: Consolidated mapping with 2023 resolutions
```

### Stage 5: Census Enhancement
```
Script: census_algo.py
Process:
  1. Identify low-confidence mappings (similarity < threshold)
  2. Query US Census HS search with Selenium
  3. Use GPT-4 to select best classification from results
  4. Update mapping database with enhanced classifications
Output: Enhanced mappings with Census validations
```

### Stage 6: Final Database Creation
```
Script: final_mapping_analysis.ipynb
Process:
  1. Combine all mappings and enhancements
  2. Add metadata and quality indicators
  3. Calculate final similarity scores
  4. Create analysis visualizations
Output: FINAL_MAPPING_1789_2023.pkl
```

## Database Structure

The final pickle database (`FINAL_MAPPING_1789_2023.pkl`) is a pandas DataFrame with the following structure:

```python
DataFrame with 750,000+ rows and 21 columns:
Index(['from_year', 'to_year', 'year_N', 'year_N1', 'HS_N', 'HS_N1',
       'Cosine_Similarity', 'Jaccard_Similarity', 'Levenshtein_Similarity',
       'Combined_Similarity', 'Description_N', 'Description_N1', 'desc_N',
       'desc_N1', 'Mapped_2023_HS', 'Mapped_2023_Description',
       'embedding_similarity', 'best_match_hscode', 'best_match_similarity',
       'best_match_description', 'best_match_source'],
      dtype='object')
```

## Column Descriptions

### Year and Period Columns
- **`from_year`**: Starting year of the mapping period (e.g., 1789)
- **`to_year`**: Ending year of the mapping period (e.g., 1790)
- **`year_N`**: The earlier year in a year-to-year mapping
- **`year_N1`**: The later year in a year-to-year mapping (N+1)

### Classification Codes
- **`HS_N`**: Harmonized System code for year N
- **`HS_N1`**: Harmonized System code for year N+1
- **`Mapped_2023_HS`**: The final 2023 HS code this historical item maps to
- **`best_match_hscode`**: Alternative HS code from Census matching (if applicable)

### Similarity Metrics
- **`Cosine_Similarity`**: Semantic similarity between descriptions (0-1 scale)
  - Based on transformer embeddings
  - Captures meaning regardless of exact wording
- **`Jaccard_Similarity`**: Word overlap between descriptions (0-1 scale)
  - Measures common vocabulary
  - Useful for technical terms that persist over time
- **`Levenshtein_Similarity`**: String similarity between descriptions (0-1 scale)
  - Based on edit distance
  - Captures spelling variations and minor changes
- **`Combined_Similarity`**: Weighted average of all three metrics
  - Default weights: Cosine (0.5), Jaccard (0.25), Levenshtein (0.25)
- **`embedding_similarity`**: Direct cosine similarity of embeddings
- **`best_match_similarity`**: Similarity score from Census matching

### Description Fields
- **`Description_N`**: Full tariff description for year N
- **`Description_N1`**: Full tariff description for year N+1
- **`desc_N`**: Processed/cleaned description for year N
- **`desc_N1`**: Processed/cleaned description for year N+1
- **`Mapped_2023_Description`**: The 2023 description this maps to
- **`best_match_description`**: Description from Census matching

### Metadata
- **`best_match_source`**: Source of the match (e.g., "census", "algorithm", "manual")

## Algorithm Details

### Similarity Calculation

The combined similarity score is calculated as:

```python
def calculate_combined_similarity(desc1, desc2, embedding1, embedding2, weights):
    # Cosine similarity from embeddings
    cosine_sim = cosine_similarity(embedding1, embedding2)
    
    # Jaccard similarity
    words1 = set(desc1.lower().split())
    words2 = set(desc2.lower().split())
    jaccard_sim = len(words1 & words2) / len(words1 | words2)
    
    # Levenshtein similarity
    lev_distance = levenshtein(desc1, desc2)
    lev_sim = 1 - (lev_distance / max(len(desc1), len(desc2)))
    
    # Weighted combination
    combined = (weights['cosine'] * cosine_sim +
                weights['jaccard'] * jaccard_sim +
                weights['levenshtein'] * lev_sim)
    
    return combined
```

### Mapping Chain Resolution

To map a historical item to 2023:

```python
def trace_to_2023(item_year, item_hs, mappings_dict):
    current_year = item_year
    current_hs = item_hs
    
    while current_year < 2023:
        # Find mapping from current_year to current_year + 1
        mapping = mappings_dict[(current_year, current_year + 1)]
        match = mapping[mapping['HS_N'] == current_hs]
        
        if match.empty:
            # No direct mapping found, use similarity search
            match = find_best_match(current_hs, mapping)
        
        current_hs = match['HS_N1'].iloc[0]
        current_year += 1
    
    return current_hs
```

### Census Enhancement Algorithm

For low-confidence mappings:

```python
def enhance_with_census(description, year, low_confidence_threshold=0.7):
    if combined_similarity < low_confidence_threshold:
        # Query Census website
        census_results = query_census_hs_search(description)
        
        # Use GPT-4 to select best match
        prompt = f"Given the historical description '{description}' from {year}, "
                f"which of these modern HS codes is the best match?"
        
        best_match = gpt4_select(prompt, census_results)
        return best_match
```

## Quality Assurance

### Validation Metrics
- **Coverage**: Percentage of historical items successfully mapped to 2023
- **Confidence Distribution**: Distribution of similarity scores
- **Chain Length**: Average number of year-to-year mappings to reach 2023
- **Manual Validation**: Spot checks of mappings for accuracy

### Known Limitations
1. **Early Years (1789-1850)**: Lower confidence due to vastly different terminology
2. **Major Reclassifications**: Some years have systematic changes that affect mapping quality
3. **Ambiguous Items**: Generic descriptions that could map to multiple modern codes
4. **Missing Years**: Some years have incomplete data

### Quality Indicators in Database
- Items with `Combined_Similarity` < 0.7 should be reviewed carefully
- Items with `best_match_source` = "census" have been enhanced but may need validation
- Long mapping chains (many intermediate years) may accumulate errors

## Usage Examples

### Loading the Database
```python
import pandas as pd

# Load the complete mapping database
mappings = pd.read_pickle('FINAL_MAPPING_1789_2023.pkl')

# Filter for a specific year
year_1850 = mappings[mappings['from_year'] == 1850]

# Find high-confidence mappings
high_conf = mappings[mappings['Combined_Similarity'] > 0.8]
```

### Tracing a Historical Item
```python
# Find how "cotton cloth" was classified in 1850
cotton_1850 = mappings[
    (mappings['from_year'] == 1850) & 
    (mappings['Description_N'].str.contains('cotton cloth', case=False))
]

# See its modern classification
print(f"1850: {cotton_1850['Description_N'].iloc[0]}")
print(f"2023: {cotton_1850['Mapped_2023_Description'].iloc[0]}")
print(f"HS Code: {cotton_1850['Mapped_2023_HS'].iloc[0]}")
```

### Analyzing Mapping Quality
```python
# Distribution of similarity scores by era
mappings['era'] = pd.cut(mappings['from_year'], 
                         bins=[1789, 1850, 1900, 1950, 2000, 2023],
                         labels=['Early', 'Industrial', 'Modern', 'Contemporary', 'Recent'])

quality_by_era = mappings.groupby('era')['Combined_Similarity'].agg(['mean', 'std', 'count'])
print(quality_by_era)
```

## File Outputs

The pipeline generates several intermediate and final files:

### Embeddings
- Location: `embeddings/embeddings_[year].pkl`
- Format: Pickle files containing numpy arrays
- Size: ~50-100MB per year

### Year-to-Year Mappings
- Location: `N_N1_mappings/[year]_to_[year+1].csv`
- Format: CSV files with similarity scores
- Columns: HS codes, descriptions, similarity metrics

### Final Database
- File: `FINAL_MAPPING_1789_2023.pkl`
- Format: Pandas DataFrame pickle
- Size: ~2-3GB
- Rows: 750,000+

### Analysis Outputs
- Summary statistics by era
- Visualization plots
- Quality metrics reports