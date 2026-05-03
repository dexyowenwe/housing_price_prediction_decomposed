import pandas as pd
from sklearn.model_selection import train_test_split


def clean_housing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove invalid rows and missing values from the housing dataset."""
    cleaned = df.copy()
    cleaned = cleaned[
        (cleaned["SquareFeet"] > 0)
        & (cleaned["Bedrooms"] > 0)
        & (cleaned["Bathrooms"] > 0)
        & (cleaned["YearBuilt"] > 1800)
        & (cleaned["Price"] > 0)
    ].reset_index(drop=True)
    return cleaned.dropna().reset_index(drop=True)


def detect_squarefeet_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Detect SquareFeet outliers using the IQR rule."""
    q1 = df["SquareFeet"].quantile(0.25)
    q3 = df["SquareFeet"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return df[(df["SquareFeet"] < lower) | (df["SquareFeet"] > upper)]


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features used in the original notebook."""
    featured = df.copy()
    featured["sqft_bedroom"] = featured["SquareFeet"] * featured["Bedrooms"]
    featured["sqft_bathroom"] = featured["SquareFeet"] * featured["Bathrooms"]
    featured["sqft_year"] = featured["SquareFeet"] * featured["YearBuilt"]
    return featured


def encode_features(
    df: pd.DataFrame,
    target_column: str = "Price",
    categorical_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """One-hot encode categorical columns and separate features from target."""
    categorical_columns = categorical_columns or ["Neighborhood"]
    X = pd.get_dummies(
        df.drop(columns=[target_column]),
        columns=categorical_columns,
        drop_first=True,
    )
    y = df[target_column]
    return X, y


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split features and target into train and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
