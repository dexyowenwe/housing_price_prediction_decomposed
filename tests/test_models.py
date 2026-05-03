from models.deep_learning import build_mlp_model
from models.ml_models import build_ml_models


def test_build_ml_models_contains_expected_models():
    models = build_ml_models()
    assert "Linear Regression" in models
    assert "Random Forest" in models


def test_build_mlp_model_returns_pipeline():
    model = build_mlp_model()
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
