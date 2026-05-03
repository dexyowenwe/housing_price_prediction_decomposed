from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_mlp_model(random_state: int = 42) -> Pipeline:
    """Create the neural network regression pipeline from the notebook."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    solver="adam",
                    alpha=0.0005,
                    batch_size=64,
                    learning_rate_init=0.001,
                    max_iter=300,
                    random_state=random_state,
                    early_stopping=True,
                    validation_fraction=0.1,
                ),
            ),
        ]
    )
