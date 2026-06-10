import os
from html import escape
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


def _format_value(value, float_format: str = ".3f") -> str:
    if isinstance(value, float):
        return format(value, float_format)
    return str(value)


def _html_table(df: pd.DataFrame, float_format: str = ".3f") -> str:
    header_cells = "".join(f"<th>{escape(str(column))}</th>" for column in df.columns)
    rows = []
    for row in df.to_numpy():
        cells = "".join(
            f"<td>{escape(_format_value(value, float_format))}</td>" for value in row
        )
        rows.append(f"<tr>{cells}</tr>")
    return (
        '<div class="table-wrap"><table>'
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table></div>"
    )


def _analysis_data(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    holdout_results: pd.DataFrame,
    target_column: str,
    neighborhood_column: str,
):
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

    return {
        "price_summary": price_summary,
        "neighborhood_summary": neighborhood_summary,
        "correlations": correlations,
        "best_holdout": holdout_results.iloc[0],
        "strongest_feature": correlations.iloc[0],
        "top_neighborhood": neighborhood_summary.iloc[0],
        "raw_rows": len(raw_df),
        "clean_rows": len(clean_df),
        "columns": ", ".join(clean_df.columns),
    }


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

    data = _analysis_data(
        raw_df,
        clean_df,
        holdout_results,
        target_column,
        neighborhood_column,
    )
    price_summary = data["price_summary"]
    neighborhood_summary = data["neighborhood_summary"]
    correlations = data["correlations"]
    best_holdout = data["best_holdout"]
    strongest_feature = data["strongest_feature"]
    top_neighborhood = data["top_neighborhood"]

    lines = [
        "# Housing Price Graphs and Analysis",
        "",
        "This report is generated from `data/raw/housing_price_dataset.csv` by the project pipeline.",
        "",
        "## Dataset Summary",
        "",
        f"- Raw rows: {data['raw_rows']:,}",
        f"- Clean rows used for analysis: {data['clean_rows']:,}",
        f"- Columns: {data['columns']}",
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


def build_housing_analysis_html(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    holdout_results: pd.DataFrame,
    cv_results: pd.DataFrame | None,
    plot_paths: dict[str, Path],
    output_path: str | Path,
    target_column: str = "Price",
    neighborhood_column: str = "Neighborhood",
) -> Path:
    """Write a browser-friendly HTML report for the IDE Preview button."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir = output_path.parent

    data = _analysis_data(
        raw_df,
        clean_df,
        holdout_results,
        target_column,
        neighborhood_column,
    )
    price_summary = data["price_summary"]
    best_holdout = data["best_holdout"]
    strongest_feature = data["strongest_feature"]
    top_neighborhood = data["top_neighborhood"]

    graph_cards = []
    for title, path in plot_paths.items():
        graph_cards.append(
            '<section class="graph-card">'
            f"<h3>{escape(title)}</h3>"
            f'<img src="{escape(_relative_link(path, output_dir))}" alt="{escape(title)}">'
            "</section>"
        )

    cv_section = ""
    if cv_results is not None:
        cv_section = (
            '<section class="section">'
            "<h2>Cross-Validation Results</h2>"
            f"{_html_table(cv_results, float_format='.4f')}"
            "</section>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Housing Price Prediction Preview</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8f5;
      --panel: #ffffff;
      --text: #1d2526;
      --muted: #5f6c6d;
      --line: #dde3df;
      --accent: #2f6f9f;
      --accent-2: #3f8f6b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }}
    header {{
      padding: 32px 24px 24px;
      background: #223235;
      color: #ffffff;
    }}
    main {{
      width: min(1120px, calc(100% - 32px));
      margin: 24px auto 48px;
    }}
    h1, h2, h3 {{ margin: 0; line-height: 1.2; }}
    h1 {{ font-size: clamp(28px, 4vw, 42px); }}
    h2 {{ font-size: 24px; margin-bottom: 16px; }}
    h3 {{ font-size: 18px; margin-bottom: 12px; }}
    p {{ margin: 10px 0 0; color: var(--muted); }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 24px;
    }}
    .metric, .section, .graph-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }}
    .metric strong {{
      display: block;
      font-size: 22px;
      margin-top: 4px;
    }}
    .metric span {{ color: var(--muted); font-size: 14px; }}
    .section {{ margin-bottom: 24px; }}
    .findings {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 12px;
    }}
    .finding {{
      border-left: 4px solid var(--accent);
      padding: 10px 12px;
      background: #f2f7f8;
      border-radius: 6px;
    }}
    .graphs {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    img {{
      display: block;
      width: 100%;
      height: auto;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #ffffff;
    }}
    .table-wrap {{ overflow-x: auto; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 520px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      white-space: nowrap;
    }}
    th {{ background: #edf3f0; }}
    footer {{
      color: var(--muted);
      font-size: 14px;
      text-align: center;
      padding-bottom: 32px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Housing Price Prediction Preview</h1>
    <p>Generated from data/raw/housing_price_dataset.csv by the Python pipeline.</p>
  </header>
  <main>
    <section class="summary" aria-label="Dataset summary">
      <div class="metric"><span>Raw Rows</span><strong>{data['raw_rows']:,}</strong></div>
      <div class="metric"><span>Clean Rows</span><strong>{data['clean_rows']:,}</strong></div>
      <div class="metric"><span>Median Price</span><strong>{_money(price_summary['50%'])}</strong></div>
      <div class="metric"><span>Average Price</span><strong>{_money(price_summary['mean'])}</strong></div>
    </section>

    <section class="section">
      <h2>Key Findings</h2>
      <div class="findings">
        <div class="finding">Best holdout model: <strong>{escape(str(best_holdout['Model']))}</strong><br>MAE {_money(best_holdout['MAE'])}, R2 {best_holdout['R2']:.3f}</div>
        <div class="finding">Strongest numeric relationship: <strong>{escape(str(strongest_feature['Feature']))}</strong><br>Correlation {strongest_feature['Correlation with Price']:.3f}</div>
        <div class="finding">Highest average price: <strong>{escape(str(top_neighborhood[neighborhood_column]))}</strong><br>{_money(top_neighborhood['mean'])}</div>
      </div>
    </section>

    <section>
      <h2>Generated Graphs</h2>
      <div class="graphs">
        {''.join(graph_cards)}
      </div>
    </section>

    <section class="section">
      <h2>Feature Correlation</h2>
      {_html_table(data['correlations'])}
    </section>

    <section class="section">
      <h2>Average Price by Neighborhood</h2>
      {_html_table(data['neighborhood_summary'], float_format='.2f')}
    </section>

    <section class="section">
      <h2>Holdout Model Results</h2>
      {_html_table(holdout_results, float_format='.4f')}
    </section>

    {cv_section}
  </main>
  <footer>Run python main.py again to refresh this preview.</footer>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path
