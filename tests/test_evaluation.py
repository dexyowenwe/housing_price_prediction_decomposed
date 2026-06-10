import pandas as pd
from sklearn.linear_model import LinearRegression

from evaluation.evaluation import evaluate_regression_model
from preprocessing.preprocessing import clean_housing_data, encode_features, split_dataset


def test_evaluate_regression_model_returns_metrics():
    housing_df = pd.DataFrame(
        {
            "SquareFeet": [1500, 1800, 1650, 2400, 2100, 1300],
            "Bedrooms": [3, 4, 3, 5, 4, 2],
            "Bathrooms": [2, 3, 2, 4, 3, 1],
            "YearBuilt": [2001, 2010, 2005, 2018, 2014, 1996],
            "Neighborhood": ["Rural", "Urban", "Rural", "Urban", "Suburb", "Suburb"],
            "Price": [250000, 320000, 270000, 410000, 350000, 210000],
        }
    )
    cleaned = clean_housing_data(housing_df)
    X, y = encode_features(cleaned)
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.5)
    model = LinearRegression().fit(X_train, y_train)
    metrics = evaluate_regression_model(model, X_test, y_test)
    assert "MAE" in metrics
    assert "R2" in metrics
