import pandas as pd

from sklearn.metrics import r2_score, mean_squared_error

from src.utils.model_utils import load_model
from src.utils.config_utils import ensure
from src.pipeline.build_dataset import build_evaluation_dataset
from src.config.model_metadata import ModelMetadata
from src.config.evaluate_config import EvaluateConfig
from src.models.evaluate_models.validate import validate_models_compatible
from src.models.apply_targets import apply_target_to_evaluation_dataset

def evaluate_models(evaluate_config: EvaluateConfig):
    evaluate_config = ensure(evaluate_config, EvaluateConfig)
    validate_models_compatible(evaluate_config)

    results = {}
    for m in evaluate_config.models:
        bundle = load_model(m)
        metadata = ModelMetadata.from_name(m)
        scaler = bundle.get("scaler")
        model = bundle.get("model")

        df = build_evaluation_dataset(evaluate_config, metadata)

        df, target_cols = apply_target_to_evaluation_dataset(df, metadata)
        df = df.dropna()

        X = df.drop(columns=target_cols)
        X = X[metadata.selected_features]
        y = df[target_cols]

        if scaler: X = scaler.transform(X.values)

        preds = model.predict(X)
        results[m] = {
            "r2": r2_score(y, preds),
            "mse": mean_squared_error(y, preds)}    

    return results

def get_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        return (
            pd.Series(model.feature_importances_, index=feature_names)
            .sort_values(ascending=False))
    else: return None