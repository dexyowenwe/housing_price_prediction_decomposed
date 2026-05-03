import pandas as pd
import pytest


@pytest.fixture
def sample_housing_df():
    return pd.DataFrame(
        {
            "SquareFeet": [1500, 1800, 0],
            "Bedrooms": [3, 4, 2],
            "Bathrooms": [2, 3, 1],
            "YearBuilt": [2001, 2010, 1999],
            "Neighborhood": ["Rural", "Urban", "Suburb"],
            "Price": [250000, 320000, -10],
        }
    )
