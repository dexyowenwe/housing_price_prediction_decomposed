from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "housing_price_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORTS_DIR = OUTPUT_DIR / "reports"

TARGET_COLUMN = "Price"
CATEGORICAL_COLUMNS = ["Neighborhood"]
NUMERIC_COLUMNS = ["SquareFeet", "Bedrooms", "Bathrooms", "YearBuilt"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5
CV_EXCLUDED_MODELS = ["MLP Regressor"]
