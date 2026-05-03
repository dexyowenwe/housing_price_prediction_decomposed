from data.data_loader import get_dataset_overview


def test_get_dataset_overview(sample_housing_df):
    overview = get_dataset_overview(sample_housing_df)
    assert overview["rows"] == 3
    assert "Price" in overview["column_names"]
