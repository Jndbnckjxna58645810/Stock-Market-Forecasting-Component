from src.utils.config_utils import ensure

from src.config.model_metadata import ModelMetadata
from src.config.evaluate_config import EvaluateConfig

from src.pipeline.apply_targets import apply_target_to_evaluation_dataset
from src.pipeline.build_dataset import build_evaluation_dataset

from src.models.shared.metrics import compute_metrics

from src.models.predict_models.predict_tabular import predict_tabular_by_parameters

def evaluate_model_tabular(evaluate_config: EvaluateConfig, model_name):
    evaluate_config = ensure(evaluate_config, EvaluateConfig)
    model_metadata = ModelMetadata.from_name(model_name)

    df = build_evaluation_dataset(evaluate_config, model_metadata)

    df, target_cols = apply_target_to_evaluation_dataset(df, model_metadata)
    df = df.dropna()

    X = df.drop(columns=target_cols)
    X = X[model_metadata.selected_features]
    y = df[target_cols]

    from src.models.registry import load_model
    preds_bundle = predict_tabular_by_parameters(load_model(model_name), X)

    preds, dates = preds_bundle["preds"], preds_bundle["dates"]

    return {"metrics": compute_metrics(y, preds),
            "y_true": y.values.ravel().tolist(),
            "y_pred": preds, "dates": dates}
