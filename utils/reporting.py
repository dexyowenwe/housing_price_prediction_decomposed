import os
from pathlib import Path

import pandas as pd


def _money(value: float) -> str:
    return f"${value:,.0f}"


def _relative_link(path: Path, base_dir: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base_dir.resolve())).as_posix()


def _markdown_table(df: pd.DataFrame, float_format: str = ".3f") -> str:
    def format_value(value):
        if isinstance(value, float):
            return format(value, float_format)
        return str(value)

    headers = [str(column) for column in df.columns]
    rows = [[format_value(value) for value in row] for row in df.to_numpy()]
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    table.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(table)


def build_housing_analysis_markdown(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    holdout_results: pd.DataFrame,
    cv_results: pd.DataFrame | None,
    plot_paths: dict[str, Path],
    output_path: str | Path,
    target_column: str = "Price",
    neighborhood_column: str = "Neighborhood",
) -> Path:
    """Write a Markdown analysis report with generated graph references."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir = output_path.parent

    price_summary = clean_df[target_column].describe()
    neighborhood_summary = (
        clean_df.groupby(neighborhood_column)[target_column]
        .agg(["count", "mean", "median", "min", "max"])
        .sort_values("mean", ascending=False)
        .reset_index()
    )
    correlations = (
        clean_df.corr(numeric_only=True)[target_column]
        .drop(target_column)
        .sort_values(ascending=False)
        .reset_index()
    )
    correlations.columns = ["Feature", "Correlation with Price"]

    best_holdout = holdout_results.iloc[0]
    strongest_feature = correlations.iloc[0]
    top_neighborhood = neighborhood_summary.iloc[0]

    lines = [
        "# Housing Price Graphs and Analysis",
        "",
        "This report is generated from `data/raw/housing_price_dataset.csv` by the project pipeline.",
        "",
        "## Dataset Summary",
        "",
        f"- Raw rows: {len(raw_df):,}",
        f"- Clean rows used for analysis: {len(clean_df):,}",
        f"- Columns: {', '.join(clean_df.columns)}",
        f"- Price range: {_money(price_summary['min'])} to {_money(price_summary['max'])}",
        f"- Median price: {_money(price_summary['50%'])}",
        f"- Average price: {_money(price_summary['mean'])}",
        "",
        "## Key Findings",
        "",
        (
            f"- The best holdout model is **{best_holdout['Model']}**, with "
            f"MAE of {_money(best_holdout['MAE'])} and R2 of {best_holdout['R2']:.3f}."
        ),
        (
            f"- The strongest numeric relationship with price is **{strongest_feature['Feature']}** "
            f"with correlation {strongest_feature['Correlation with Price']:.3f}."
        ),
        (
            f"- The highest average price neighborhood is **{top_neighborhood[neighborhood_column]}** "
            f"at {_money(top_neighborhood['mean'])}."
        ),
        "",
        "## Generated Graphs",
        "",
    ]

    for title, path in plot_paths.items():
        lines.extend(
            [
                f"### {title}",
                "",
                f"![{title}]({_relative_link(path, report_dir)})",
                "",
            ]
        )

    lines.extend(
        [
            "## Feature Correlation",
            "",
            _markdown_table(correlations),
            "",
            "## Average Price by Neighborhood",
            "",
            _markdown_table(neighborhood_summary, float_format=".2f"),
            "",
            "## Holdout Model Results",
            "",
            _markdown_table(holdout_results, float_format=".4f"),
            "",
        ]
    )

    if cv_results is not None:
        lines.extend(
            [
                "## Cross-Validation Results",
                "",
                _markdown_table(cv_results, float_format=".4f"),
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation",
            "",
            (
                "The graphs show how price is distributed, how average price differs by "
                "neighborhood, and how square footage relates to price. The model comparison "
                "plots make it easier to compare error and explained variance across models. "
                "Lower MAE is better because it means smaller average pricing error, while "
                "higher R2 is better because it means the model explains more variation in prices."
            ),
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
