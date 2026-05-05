from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_mlp_model(random_state: int = 42) -> Pipeline:
    """Create a neural network regression pipeline for the housing dataset."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=0.0005,
                    batch_size=256,
                    learning_rate_init=0.001,
                    max_iter=80,
                    random_state=random_state,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=8,
                    tol=0.001,
                ),
            ),
        ]
    )
