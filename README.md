# Scandinavian Hotel Price Estimation Using Machine Learning

A machine learning project that analyzes hotel listings in Scandinavia and predicts hotel prices based on location, ratings, reviews, distance to the city center, season, and amenities. It also includes a **Streamlit web application** where users can enter hotel characteristics and obtain a price estimate and corresponding price category.

## Project Overview

The project includes two complementary tasks:

1. **Hotel price category classification:** Evaluate and compare machine learning models for categorical price prediction.
2. **Numerical hotel price estimation:** Use a trained regression model to predict a numerical hotel price, then display its corresponding category.

According to the project's existing documentation, the original data collection comprised **13,958 hotel records and 40 columns**; **10,214 usable records** remained after cleaning.

## Features

- Data preprocessing and exploratory analysis of hotel listings.
- Hotel price classification experiments across multiple machine learning models.
- Comparisons of model performance, training time, and inference time.
- Feature importance and feature selection analysis.
- Saved models and preprocessing pipelines for inference.
- Interactive Streamlit UI for numerical price estimation.
- Automatic categorization of predicted prices into Economy, Standard, Premium, and Luxury segments.
- Research reports, charts, and result tables included in the repository.

## Machine Learning Results

The published classification comparison reports the following results:

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Random Forest | 60.06% | 0.8366 |
| Support Vector Machine | 54.63% | 0.8068 |
| K-Nearest Neighbors | 52.47% | 0.7702 |
| Gradient Boosting | 52.18% | 0.7713 |
| Logistic Regression | 42.58% | 0.7280 |

*These metrics apply to the documented price-category classification experiments; they are not accuracy metrics for numerical price regression.*

## Tech Stack

- **Language:** Python
- **Machine learning / Data processing:** scikit-learn, pandas, NumPy, joblib
- **Web app:** Streamlit
- **Experiment environment:** Jupyter Notebook
- **Research assets:** CSV datasets, model artifacts, figures, and reports

## Run the Web App

### Prerequisites

- Python 3
- pip

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/Cgdscn/ML-Hotel-Price-Estimation.git
   cd ML-Hotel-Price-Estimation
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   # macOS/Linux: source .venv/bin/activate
   ```

3. Install the libraries required by the Streamlit application:

   ```bash
   pip install streamlit pandas numpy scikit-learn joblib
   ```

   **Note:** The repository currently contains an empty file named `requirements.txt.txt`, rather than a populated `requirements.txt`. For reproducible installation, a pinned dependency file should be created from the working environment. Serialized scikit-learn models may also require a compatible scikit-learn version.

4. Start the application:

   ```bash
   python -m streamlit run app.py
   ```

5. Open the URL printed by Streamlit (usually `http://localhost:8501`).

The application loads pre-trained models and preprocessing artifacts from `saved_model/`; training is not required just to open the prediction interface.

## Input Features

The web app accepts country, city, season, hotel score, review count, distance to the city center, star rating, and selectable amenities. It displays the estimated hotel price in Turkish lira and a derived price category.

## Repository Contents

```text
├── app.py                        # Streamlit prediction interface
├── Iskandinavya_Otel_Verisi.csv  # Hotel dataset
├── MLProject_Notebook.ipynb.ipynb# Research notebook
├── saved_model/                  # Trained models and preprocessors
├── figures/                      # Visualizations and evaluation figures
├── tables/                       # Performance and analysis tables
├── MLProject.pdf                 # Project report
└── README.txt.txt                # Original text-based project notes
```

## Limitations

- Predicted values are **estimates**, not real-time hotel quotations.
- Results depend on the training dataset and its coverage.
- Hotel prices can change with availability, booking date, promotions, and market conditions.
- The saved regression model and the classification experiments are distinct outputs; their metrics should not be conflated.

## Author

[GitHub — Cgdscn](https://github.com/Cgdscn)
