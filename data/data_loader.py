from pathlib import Path

import pandas as pd


def load_housing_data(data_path: str | Path) -> pd.DataFrame:
    """Load the housing price CSV dataset."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset was not found: {data_path}")
    return pd.read_csv(data_path)


def get_dataset_overview(df: pd.DataFrame) -> dict:
    """Return basic dataset information for reporting."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_summary": df.describe().to_dict(),
    }
