# Housing Price Prediction Decomposition

This project decomposes the original notebook `housing-price-prediction-with-regression-models.ipynb` into a structured machine learning project. The selected source notebook is a publicly available housing price prediction regression workflow based on the Kaggle-style dataset `housing_price_dataset.csv`.

## Project Structure

```text
housing_price_prediction_decomposed/
├── main.py
├── config.py
├── requirements.txt
├── data/
│   ├── data_loader.py
│   └── raw/housing_price_dataset.csv
├── preprocessing/
│   └── preprocessing.py
├── models/
│   ├── ml_models.py
│   └── deep_learning.py
├── evaluation/
│   └── evaluation.py
├── utils/
│   └── visualization.py
├── tests/
│   ├── conftest.py
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_evaluation.py
├── notebooks/
│   └── original_housing_price_regression.ipynb
└── outputs/
    ├── plots/
    └── reports/
```

## Module Documentation

`config.py` stores central project settings, including dataset location, output folders, target column, categorical columns, test size, random seed, and cross-validation settings.

`data/data_loader.py` handles data loading and basic dataset exploration. The `load_housing_data()` function reads the CSV file, while `get_dataset_overview()` summarizes rows, columns, missing values, and numeric statistics.

`preprocessing/preprocessing.py` contains preprocessing and feature preparation. It removes invalid records, handles missing values, detects `SquareFeet` outliers using IQR, creates interaction features, one-hot encodes `Neighborhood`, and splits the dataset into training and testing sets.

`models/ml_models.py` defines traditional regression models from the original notebook: Linear Regression, Decision Tree, Random Forest, and Gradient Boosting.

`models/deep_learning.py` defines an MLP Regressor pipeline. It includes feature scaling with `StandardScaler` because neural networks are sensitive to feature scale.

`evaluation/evaluation.py` evaluates trained models using Mean Absolute Error and R2 score. It also provides K-fold cross-validation so models can be compared more fairly than using only one train-test split.

`utils/visualization.py` saves visualization outputs. It creates feature-correlation plots and actual-versus-predicted scatter plots.

`main.py` runs the full workflow from data loading to report and plot generation.

## Machine Learning Workflow

1. Load `housing_price_dataset.csv`.
2. Clean invalid rows such as zero square feet, invalid bedroom or bathroom counts, invalid year built values, and non-positive prices.
3. Encode the categorical `Neighborhood` feature with one-hot encoding.
4. Split the data into training and testing sets.
5. Train several regression models.
6. Evaluate models using MAE and R2.
7. Validate models with K-fold cross-validation.
8. Save model comparison reports and visualizations.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python main.py
```

Run a faster version without cross-validation:

```bash
python main.py --no-cv
```

Run tests:

```bash
pytest
```

## Outputs

The pipeline saves these files after execution:

```text
outputs/reports/holdout_model_results.csv
outputs/reports/cross_validation_results.csv
outputs/plots/feature_correlation.png
outputs/plots/actual_vs_predicted_best_model.png
```

## Source Code Selection

The original notebook already contained the required assignment components:

- Data processing: CSV loading and dataset inspection
- Preprocessing: invalid row filtering and missing value removal
- Feature engineering: one-hot encoding and interaction features
- Model training: Linear Regression, Decision Tree, Random Forest, Gradient Boosting, and MLP Regressor
- Evaluation: MAE, R2, and cross-validation
- Visualization: correlation chart and actual-versus-predicted plots

This decomposed version keeps the same workflow but organizes it into maintainable Python scripts.
