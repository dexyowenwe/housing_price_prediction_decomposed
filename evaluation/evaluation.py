import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_validate


def evaluate_regression_model(model, X_test, y_test) -> dict:
    """Calculate MAE and R2 for a trained regression model."""
    predictions = model.predict(X_test)
    return {
        "MAE": mean_absolute_error(y_test, predictions),
        "R2": r2_score(y_test, predictions),
    }


def evaluate_models(models: dict, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """Train and evaluate several models on the same holdout split."""
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_regression_model(model, X_test, y_test)
        rows.append({"Model": name, **metrics})
    return pd.DataFrame(rows).sort_values("MAE")


def cross_validate_model_set(
    models: dict,
    X,
    y,
    cv_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run K-fold cross-validation and return MAE/R2 comparison."""
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    rows = []

    for name, model in models.items():
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring={"mae": "neg_mean_absolute_error", "r2": "r2"},
            n_jobs=-1,
        )
        rows.append(
            {
                "Model": name,
                "CV MAE Mean": -scores["test_mae"].mean(),
                "CV MAE Std": scores["test_mae"].std(),
                "CV R2 Mean": scores["test_r2"].mean(),
                "CV R2 Std": scores["test_r2"].std(),
            }
        )

    return pd.DataFrame(rows).sort_values("CV MAE Mean")
