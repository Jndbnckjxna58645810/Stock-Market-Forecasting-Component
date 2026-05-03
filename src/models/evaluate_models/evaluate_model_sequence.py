import pandas as pd

from src.utils.config_utils import ensure
from src.utils.logging_utils import get_logger

from src.config.model_metadata import ModelMetadata
from src.config.evaluate_config import EvaluateConfig

from src.pipeline.apply_targets import apply_target_to_evaluation_dataset
from src.pipeline.build_dataset import build_evaluation_dataset

from src.models.shared.metrics import compute_multi_target_metrics

from src.models.predict_models.predict_sequence import predict_sequence_by_parameters

logger = get_logger("models.evaluate_models.evaluate_model_sequence")

def evaluate_model_sequence(evaluate_config: EvaluateConfig, model_name):
    evaluate_config = ensure(evaluate_config, EvaluateConfig)
    model_metadata = ModelMetadata.from_name(model_name)

    logger.info(f"Evaluating model: {model_name}")

    df = build_evaluation_dataset(evaluate_config, model_metadata)

    df, target_cols = apply_target_to_evaluation_dataset(df, model_metadata)
    df = df.dropna()

    X = df.drop(columns=target_cols)
    X = X[model_metadata.selected_features]
    y = df[target_cols]

    seq_len = model_metadata.hyperparameters["seq_len"]

    y_aligned = y.iloc[seq_len:]

    from src.models.registry import load_model
    preds_bundle = predict_sequence_by_parameters(
        load_model(model_name), X, seq_len, target_cols)

    y_pred_df = pd.DataFrame(
        preds_bundle["preds"], 
        index=preds_bundle["dates"], 
        columns=target_cols)

    y_aligned.index = pd.to_datetime(y_aligned.index)
    y_pred_df.index = pd.to_datetime(y_pred_df.index)

    common_idx = y_aligned.index.intersection(y_pred_df.index)
    y_true = y_aligned.loc[common_idx]
    y_pred = y_pred_df.loc[common_idx]

    metrics_report = compute_multi_target_metrics(y_true, y_pred)

    logger.info(f"Model evaluation completed: {model_name}")
    
    return {
        "metrics": metrics_report,
        "y_true": y_true, 
        "y_pred": y_pred,
        "dates": common_idx.strftime('%Y-%m-%d').tolist()
    }
