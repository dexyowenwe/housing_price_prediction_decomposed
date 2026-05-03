from sklearn.linear_model import LinearRegression

from evaluation.evaluation import evaluate_regression_model
from preprocessing.preprocessing import clean_housing_data, encode_features, split_dataset


def test_evaluate_regression_model_returns_metrics(sample_housing_df):
    cleaned = clean_housing_data(sample_housing_df)
    X, y = encode_features(cleaned)
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.5)
    model = LinearRegression().fit(X_train, y_train)
    metrics = evaluate_regression_model(model, X_test, y_test)
    assert "MAE" in metrics
    assert "R2" in metrics
