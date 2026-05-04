from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _prepare_output_path(output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def save_correlation_plot(df: pd.DataFrame, target_column: str, output_path: str | Path):
    """Save a bar chart of numeric correlation against the target column."""
    output_path = _prepare_output_path(output_path)
    correlations = df.corr(numeric_only=True)[target_column].sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    ax = correlations.drop(target_column).plot(kind="bar", color="#2f6f9f")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Correlation Coefficient")
    ax.set_title(f"Numeric Feature Correlation with {target_column}")
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_price_distribution_plot(
    df: pd.DataFrame,
    target_column: str,
    output_path: str | Path,
):
    """Save a histogram showing the target price distribution."""
    output_path = _prepare_output_path(output_path)

    plt.figure(figsize=(8, 5))
    ax = df[target_column].plot(kind="hist", bins=30, color="#3f8f6b", edgecolor="white")
    ax.set_xlabel("House Price")
    ax.set_ylabel("Number of Houses")
    ax.set_title("Distribution of House Prices")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_price_by_neighborhood_plot(
    df: pd.DataFrame,
    target_column: str,
    neighborhood_column: str,
    output_path: str | Path,
):
    """Save a bar chart of average price by neighborhood."""
    output_path = _prepare_output_path(output_path)
    average_prices = (
        df.groupby(neighborhood_column)[target_column].mean().sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 5))
    ax = average_prices.plot(kind="bar", color="#c96f3d")
    ax.set_xlabel("Neighborhood")
    ax.set_ylabel("Average House Price")
    ax.set_title("Average House Price by Neighborhood")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_squarefeet_price_scatter_plot(
    df: pd.DataFrame,
    target_column: str,
    squarefeet_column: str,
    neighborhood_column: str,
    output_path: str | Path,
):
    """Save a scatter plot of house size against price by neighborhood."""
    output_path = _prepare_output_path(output_path)

    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    for neighborhood, group in df.groupby(neighborhood_column):
        ax.scatter(
            group[squarefeet_column],
            group[target_column],
            alpha=0.55,
            s=24,
            label=neighborhood,
        )
    ax.set_xlabel("Square Feet")
    ax.set_ylabel("House Price")
    ax.set_title("House Price vs Square Feet by Neighborhood")
    ax.legend(title="Neighborhood")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_model_mae_comparison_plot(
    results: pd.DataFrame,
    output_path: str | Path,
):
    """Save a model comparison bar chart using holdout MAE."""
    output_path = _prepare_output_path(output_path)
    sorted_results = results.sort_values("MAE", ascending=True)

    plt.figure(figsize=(9, 5))
    ax = plt.barh(sorted_results["Model"], sorted_results["MAE"], color="#6d6aa8")
    plt.xlabel("Mean Absolute Error")
    plt.ylabel("Model")
    plt.title("Holdout Model Comparison by MAE")
    plt.grid(axis="x", alpha=0.25)
    plt.bar_label(ax, fmt="%.0f", padding=4)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_model_r2_comparison_plot(
    results: pd.DataFrame,
    output_path: str | Path,
):
    """Save a model comparison bar chart using holdout R2."""
    output_path = _prepare_output_path(output_path)
    sorted_results = results.sort_values("R2", ascending=False)

    plt.figure(figsize=(9, 5))
    ax = plt.barh(sorted_results["Model"], sorted_results["R2"], color="#4d908e")
    plt.xlabel("R2 Score")
    plt.ylabel("Model")
    plt.title("Holdout Model Comparison by R2")
    plt.grid(axis="x", alpha=0.25)
    plt.bar_label(ax, fmt="%.3f", padding=4)
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
    output_path = _prepare_output_path(output_path)

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred, alpha=0.45, color="#2f6f9f")
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="red",
        linewidth=2,
        label="Perfect Prediction",
    )
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
