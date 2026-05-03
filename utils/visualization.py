from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_correlation_plot(df: pd.DataFrame, target_column: str, output_path: str | Path):
    """Save a bar chart of numeric correlation against the target column."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    correlations = df.corr(numeric_only=True)[target_column].sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    correlations.drop(target_column).plot(kind="bar")
    plt.ylabel("Correlation")
    plt.title(f"Feature Correlation with {target_column}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_actual_vs_predicted_plot(
    y_test,
    y_pred,
    title: str,
    output_path: str | Path,
):
    """Save an actual-versus-predicted scatter plot."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred, alpha=0.45)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="red",
        linewidth=2,
    )
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
