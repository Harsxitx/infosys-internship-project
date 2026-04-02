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

## Milestone 2 (Week 3-4): Exploratory Analysis + Feature Engineering

### Scope
- Growth over time: visualize year-by-year content additions
- Distribution analysis: genres and ratings split by Movies vs TV Shows
- Country-level contribution: identify top content-producing countries
- Feature engineering: duration categories and original vs licensed content

### Run Feature Engineering
```bash
python src/feature_engineering.py
```

### Milestone 2 Notebook
- `notebooks/02_milestone2_analysis.ipynb`

## Milestone 3 (Week 5-6): Modeling & Advanced Analysis

### Scope
- Cluster Netflix titles using genre, rating, and duration signals
- Classify content type (`Movie` vs `TV Show`) using engineered features
- Analyze key drivers of content availability across top countries and genres
- Interpret model behavior with feature-importance methods

### Run Milestone 3 Modeling Pipeline
```bash
python src/modeling.py
```

### Generate Milestone 3 Notebook
```bash
python src/generate_milestone3_notebook.py
```

### Milestone 3 Notebook
- `notebooks/03_milestone3_modeling.ipynb`

## Milestone 4 (Week 7-8): Interactive Dashboard

### Scope
- Build an interactive Streamlit dashboard using the cleaned Netflix dataset
- Add sidebar filters for release year, genre, country, and content type
- Surface key insights such as top genres by year, country-level content distribution, and rating analysis
- Finalize dashboard testing and prepare it for deployment

### Run the Dashboard
```bash
streamlit run app.py
```

### Dashboard Features
- KPI cards for total titles, movies, TV shows, and median release year
- Year-over-year release trend and content-type mix
- Top genre per release year
- Country-wise content distribution
- Rating distribution and rating analysis by content type
- CSV download for the filtered dataset
