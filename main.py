import argparse

from config import (
    CATEGORICAL_COLUMNS,
    CV_SPLITS,
    DATA_PATH,
    PLOTS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
)
from data.data_loader import get_dataset_overview, load_housing_data
from evaluation.evaluation import cross_validate_model_set, evaluate_models
from models.deep_learning import build_mlp_model
from models.ml_models import build_ml_models
from preprocessing.preprocessing import clean_housing_data, encode_features, split_dataset
from utils.reporting import build_housing_analysis_markdown
from utils.visualization import (
    save_actual_vs_predicted_plot,
    save_correlation_plot,
    save_model_mae_comparison_plot,
    save_model_r2_comparison_plot,
    save_price_by_neighborhood_plot,
    save_price_distribution_plot,
    save_squarefeet_price_scatter_plot,
)


def run_pipeline(include_cv: bool = True):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_housing_data(DATA_PATH)
    raw_overview = get_dataset_overview(raw_df)

    clean_df = clean_housing_data(raw_df)
    X, y = encode_features(clean_df, TARGET_COLUMN, CATEGORICAL_COLUMNS)
    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    models = build_ml_models(RANDOM_STATE)
    models["MLP Regressor"] = build_mlp_model(RANDOM_STATE)

    holdout_results = evaluate_models(models, X_train, X_test, y_train, y_test)
    holdout_results.to_csv(REPORTS_DIR / "holdout_model_results.csv", index=False)

    plot_paths = {
        "Feature Correlation": PLOTS_DIR / "feature_correlation.png",
        "Price Distribution": PLOTS_DIR / "price_distribution.png",
        "Average Price by Neighborhood": PLOTS_DIR / "average_price_by_neighborhood.png",
        "Price vs Square Feet by Neighborhood": PLOTS_DIR / "price_vs_squarefeet_by_neighborhood.png",
        "Holdout Model MAE Comparison": PLOTS_DIR / "holdout_model_mae_comparison.png",
        "Holdout Model R2 Comparison": PLOTS_DIR / "holdout_model_r2_comparison.png",
    }

    save_correlation_plot(clean_df, TARGET_COLUMN, plot_paths["Feature Correlation"])
    save_price_distribution_plot(
        clean_df,
        TARGET_COLUMN,
        plot_paths["Price Distribution"],
    )
    save_price_by_neighborhood_plot(
        clean_df,
        TARGET_COLUMN,
        "Neighborhood",
        plot_paths["Average Price by Neighborhood"],
    )
    save_squarefeet_price_scatter_plot(
        clean_df,
        TARGET_COLUMN,
        "SquareFeet",
        "Neighborhood",
        plot_paths["Price vs Square Feet by Neighborhood"],
    )
    save_model_mae_comparison_plot(
        holdout_results,
        plot_paths["Holdout Model MAE Comparison"],
    )
    save_model_r2_comparison_plot(
        holdout_results,
        plot_paths["Holdout Model R2 Comparison"],
    )

    best_model_name = holdout_results.iloc[0]["Model"]
    best_model = models[best_model_name]
    best_predictions = best_model.predict(X_test)
    plot_paths[f"Actual vs Predicted ({best_model_name})"] = (
        PLOTS_DIR / "actual_vs_predicted_best_model.png"
    )
    save_actual_vs_predicted_plot(
        y_test,
        best_predictions,
        f"Actual vs Predicted ({best_model_name})",
        plot_paths[f"Actual vs Predicted ({best_model_name})"],
    )

    cv_results = None
    if include_cv:
        cv_results = cross_validate_model_set(models, X, y, CV_SPLITS, RANDOM_STATE)
        cv_results.to_csv(REPORTS_DIR / "cross_validation_results.csv", index=False)

    analysis_path = build_housing_analysis_markdown(
        raw_df=raw_df,
        clean_df=clean_df,
        holdout_results=holdout_results,
        cv_results=cv_results,
        plot_paths=plot_paths,
        output_path=REPORTS_DIR / "housing_price_analysis.md",
        target_column=TARGET_COLUMN,
        neighborhood_column="Neighborhood",
    )

    print("Housing price prediction pipeline completed.")
    print(f"Original dataset rows: {raw_overview['rows']}")
    print(f"Cleaned dataset rows: {len(clean_df)}")
    print(f"Best holdout model: {best_model_name}")
    print(f"Markdown analysis: {analysis_path}")
    print(holdout_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run housing price prediction pipeline.")
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="Skip cross-validation for a faster run.",
    )
    args = parser.parse_args()
    run_pipeline(include_cv=not args.no_cv)
