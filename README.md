## Project Objective

This project analyzes the Netflix titles dataset to uncover insights about:

- Content distribution by country
- Movie vs TV Show trends
- Release year trends
- Genre patterns
- Rating distributions
- Duration analysis

The goal is to perform exploratory data analysis (EDA) and prepare a clean dataset for further analysis and visualization.

## Milestone 1 (Week 1-2): Requirements and Dataset Preparation

### Scope
- Define project scope and success metrics
- Load the Netflix dataset from `data/raw/netflix_titles.csv`
- Clean data by handling missing values and removing duplicates
- Normalize categorical features (`country`, `rating`, `listed_in`)
- Save cleaned data to `data/processed/netflix_titles_cleaned.csv`

### Success Metrics
- Raw dataset loads without errors
- Duplicate rows removed from the dataset
- Missing values in key columns are handled with explicit defaults (`Unknown`)
- Multi-valued categorical columns are normalized to a consistent `|`-separated format
- Cleaned dataset is successfully exported for analysis

### Run Milestone 1 Pipeline
```bash
python src/cleaning.py
```
