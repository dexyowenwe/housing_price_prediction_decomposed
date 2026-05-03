from preprocessing.preprocessing import clean_housing_data, encode_features


def test_clean_housing_data_removes_invalid_rows(sample_housing_df):
    cleaned = clean_housing_data(sample_housing_df)
    assert len(cleaned) == 2
    assert cleaned["Price"].min() > 0


def test_encode_features_creates_target_and_dummy_columns(sample_housing_df):
    cleaned = clean_housing_data(sample_housing_df)
    X, y = encode_features(cleaned)
    assert "Price" not in X.columns
    assert len(y) == len(cleaned)
    assert any(column.startswith("Neighborhood_") for column in X.columns)
