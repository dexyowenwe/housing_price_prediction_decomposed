import csv
import math
from pathlib import Path
from statistics import mean

from config import DATA_PATH, PLOTS_DIR, TARGET_COLUMN


WIDTH = 900
HEIGHT = 560
MARGIN = {"top": 54, "right": 36, "bottom": 86, "left": 94}
COLORS = ["#2f6f9f", "#c96f3d", "#3f8f6b", "#6d6aa8", "#4d908e"]


def load_rows():
    with DATA_PATH.open(newline="") as csv_file:
        rows = []
        for row in csv.DictReader(csv_file):
            rows.append(
                {
                    "SquareFeet": float(row["SquareFeet"]),
                    "Bedrooms": float(row["Bedrooms"]),
                    "Bathrooms": float(row["Bathrooms"]),
                    "YearBuilt": float(row["YearBuilt"]),
                    "Neighborhood": row["Neighborhood"],
                    TARGET_COLUMN: float(row[TARGET_COLUMN]),
                }
            )
    return rows


def fmt_money(value):
    return f"${value:,.0f}"


def scale(value, source_min, source_max, target_min, target_max):
    if source_max == source_min:
        return (target_min + target_max) / 2
    ratio = (value - source_min) / (source_max - source_min)
    return target_min + ratio * (target_max - target_min)


def write_svg(path, title, body, x_label=None, y_label=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ""
    if x_label:
        labels += (
            f'<text x="{WIDTH / 2}" y="{HEIGHT - 24}" text-anchor="middle" '
            f'font-size="16" fill="#222">{x_label}</text>'
        )
    if y_label:
        labels += (
            f'<text x="24" y="{HEIGHT / 2}" text-anchor="middle" '
            f'transform="rotate(-90 24 {HEIGHT / 2})" font-size="16" '
            f'fill="#222">{y_label}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{WIDTH / 2}" y="30" text-anchor="middle" font-size="22" font-weight="700" fill="#111">{title}</text>
  {body}
  {labels}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def axis_lines():
    left = MARGIN["left"]
    top = MARGIN["top"]
    right = WIDTH - MARGIN["right"]
    bottom = HEIGHT - MARGIN["bottom"]
    return (
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#333"/>'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333"/>'
    )


def y_grid_ticks(min_value, max_value, formatter=str, tick_count=5):
    left = MARGIN["left"]
    right = WIDTH - MARGIN["right"]
    top = MARGIN["top"]
    bottom = HEIGHT - MARGIN["bottom"]
    parts = []
    for index in range(tick_count + 1):
        value = min_value + (max_value - min_value) * index / tick_count
        y = scale(value, min_value, max_value, bottom, top)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="#444">{formatter(value)}</text>'
        )
    return "".join(parts)


def save_price_distribution(rows):
    prices = [row[TARGET_COLUMN] for row in rows]
    bins = 12
    minimum = min(prices)
    maximum = max(prices)
    bin_width = (maximum - minimum) / bins
    counts = [0] * bins
    for price in prices:
        index = min(int((price - minimum) / bin_width), bins - 1)
        counts[index] += 1

    left = MARGIN["left"]
    right = WIDTH - MARGIN["right"]
    top = MARGIN["top"]
    bottom = HEIGHT - MARGIN["bottom"]
    plot_width = right - left
    bar_gap = 4
    bar_width = plot_width / bins - bar_gap
    max_count = max(counts)
    body = [y_grid_ticks(0, max_count, lambda value: f"{value:.0f}"), axis_lines()]
    for index, count in enumerate(counts):
        x = left + index * plot_width / bins + bar_gap / 2
        y = scale(count, 0, max_count, bottom, top)
        height = bottom - y
        body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{height:.1f}" fill="#3f8f6b"/>')
        if index % 2 == 0:
            label = fmt_money(minimum + index * bin_width)
            body.append(
                f'<text x="{x + bar_width / 2:.1f}" y="{bottom + 18}" text-anchor="middle" font-size="11" fill="#444">{label}</text>'
            )
    write_svg(
        PLOTS_DIR / "price_distribution.svg",
        "Distribution of House Prices",
        "".join(body),
        "House Price Range",
        "Number of Houses",
    )


def save_average_price_by_neighborhood(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["Neighborhood"], []).append(row[TARGET_COLUMN])
    averages = sorted(((key, mean(values)) for key, values in grouped.items()), key=lambda item: item[1], reverse=True)
    max_average = max(value for _, value in averages)

    left = MARGIN["left"]
    right = WIDTH - MARGIN["right"]
    top = MARGIN["top"]
    bottom = HEIGHT - MARGIN["bottom"]
    plot_width = right - left
    bar_width = plot_width / len(averages) * 0.58
    body = [y_grid_ticks(0, max_average, fmt_money), axis_lines()]
    for index, (name, value) in enumerate(averages):
        center = left + (index + 0.5) * plot_width / len(averages)
        y = scale(value, 0, max_average, bottom, top)
        body.append(f'<rect x="{center - bar_width / 2:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bottom - y:.1f}" fill="#c96f3d"/>')
        body.append(f'<text x="{center:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-size="12" fill="#333">{fmt_money(value)}</text>')
        body.append(f'<text x="{center:.1f}" y="{bottom + 22}" text-anchor="middle" font-size="13" fill="#333">{name}</text>')
    write_svg(
        PLOTS_DIR / "average_price_by_neighborhood.svg",
        "Average House Price by Neighborhood",
        "".join(body),
        "Neighborhood",
        "Average House Price",
    )


def save_price_vs_squarefeet(rows):
    squarefeet = [row["SquareFeet"] for row in rows]
    prices = [row[TARGET_COLUMN] for row in rows]
    min_squarefeet = min(squarefeet)
    max_squarefeet = max(squarefeet)
    min_price = min(prices)
    max_price = max(prices)
    neighborhoods = sorted({row["Neighborhood"] for row in rows})
    color_map = {name: COLORS[index % len(COLORS)] for index, name in enumerate(neighborhoods)}

    left = MARGIN["left"]
    right = WIDTH - MARGIN["right"] - 130
    top = MARGIN["top"]
    bottom = HEIGHT - MARGIN["bottom"]
    body = [y_grid_ticks(min_price, max_price, fmt_money), axis_lines()]
    for row in rows:
        x = scale(row["SquareFeet"], min_squarefeet, max_squarefeet, left, right)
        y = scale(row[TARGET_COLUMN], min_price, max_price, bottom, top)
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color_map[row["Neighborhood"]]}" opacity="0.55"/>')
    for index in range(6):
        value = min_squarefeet + (max_squarefeet - min_squarefeet) * index / 5
        x = scale(value, min_squarefeet, max_squarefeet, left, right)
        body.append(f'<text x="{x:.1f}" y="{bottom + 18}" text-anchor="middle" font-size="12" fill="#444">{value:.0f}</text>')
    for index, name in enumerate(neighborhoods):
        y = top + 18 + index * 24
        body.append(f'<rect x="{right + 38}" y="{y - 10}" width="14" height="14" fill="{color_map[name]}"/>')
        body.append(f'<text x="{right + 58}" y="{y + 2}" font-size="13" fill="#333">{name}</text>')
    body.append(f'<text x="{right + 38}" y="{top}" font-size="13" font-weight="700" fill="#222">Neighborhood</text>')
    write_svg(
        PLOTS_DIR / "price_vs_squarefeet_by_neighborhood.svg",
        "House Price vs Square Feet by Neighborhood",
        "".join(body),
        "Square Feet",
        "House Price",
    )


def pearson(values_a, values_b):
    avg_a = mean(values_a)
    avg_b = mean(values_b)
    numerator = sum((a - avg_a) * (b - avg_b) for a, b in zip(values_a, values_b))
    denominator_a = math.sqrt(sum((a - avg_a) ** 2 for a in values_a))
    denominator_b = math.sqrt(sum((b - avg_b) ** 2 for b in values_b))
    if denominator_a == 0 or denominator_b == 0:
        return 0
    return numerator / (denominator_a * denominator_b)


def save_feature_correlation(rows):
    features = ["SquareFeet", "Bedrooms", "Bathrooms", "YearBuilt"]
    prices = [row[TARGET_COLUMN] for row in rows]
    correlations = [(feature, pearson([row[feature] for row in rows], prices)) for feature in features]
    correlations.sort(key=lambda item: item[1], reverse=True)

    left = MARGIN["left"]
    right = WIDTH - MARGIN["right"]
    top = MARGIN["top"]
    bottom = HEIGHT - MARGIN["bottom"]
    plot_width = right - left
    bar_width = plot_width / len(correlations) * 0.55
    body = [y_grid_ticks(-1, 1, lambda value: f"{value:.1f}"), axis_lines()]
    zero_y = scale(0, -1, 1, bottom, top)
    body.append(f'<line x1="{left}" y1="{zero_y:.1f}" x2="{right}" y2="{zero_y:.1f}" stroke="#555" stroke-width="1.2"/>')
    for index, (feature, value) in enumerate(correlations):
        center = left + (index + 0.5) * plot_width / len(correlations)
        y = scale(value, -1, 1, bottom, top)
        body.append(f'<rect x="{center - bar_width / 2:.1f}" y="{min(y, zero_y):.1f}" width="{bar_width:.1f}" height="{abs(zero_y - y):.1f}" fill="#2f6f9f"/>')
        body.append(f'<text x="{center:.1f}" y="{min(y, zero_y) - 8:.1f}" text-anchor="middle" font-size="12" fill="#333">{value:.3f}</text>')
        body.append(f'<text x="{center:.1f}" y="{bottom + 22}" text-anchor="middle" font-size="13" fill="#333">{feature}</text>')
    write_svg(
        PLOTS_DIR / "feature_correlation.svg",
        f"Numeric Feature Correlation with {TARGET_COLUMN}",
        "".join(body),
        "Feature",
        "Correlation Coefficient",
    )


def main():
    rows = load_rows()
    save_price_distribution(rows)
    save_average_price_by_neighborhood(rows)
    save_price_vs_squarefeet(rows)
    save_feature_correlation(rows)
    print(f"Generated 4 SVG graphs in {PLOTS_DIR}")


if __name__ == "__main__":
    main()
