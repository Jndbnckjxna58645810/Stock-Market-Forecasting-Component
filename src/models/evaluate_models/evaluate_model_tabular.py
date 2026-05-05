import pandas as pd

from src.utils.config_utils import ensure
from src.utils.logging_utils import get_logger

from src.config.model_metadata import ModelMetadata
from src.config.evaluate_config import EvaluateConfig

from src.pipeline.prepare_evaluation_data import prepare_evaluation_data

from src.models.shared.metrics import compute_multi_target_metrics

from src.models.predict_models.predict_tabular import predict_tabular_by_parameters

logger = get_logger("models.evaluate_models.evaluate_model_tabular")

def evaluate_model_tabular(evaluate_config: EvaluateConfig, model_name):
    evaluate_config = ensure(evaluate_config, EvaluateConfig)
    model_metadata = ModelMetadata.from_name(model_name)

    logger.info(f"Evaluating model: {model_name}")

    df, target_cols = prepare_evaluation_data(evaluate_config, model_metadata)
    df = df.dropna()

    X = df[model_metadata.selected_features]
    y_true_df = df[target_cols]

    from src.models.registry import load_model
    preds_bundle = predict_tabular_by_parameters(load_model(model_name), X, target_cols)

    y_pred_df = pd.DataFrame(
        preds_bundle["preds"], 
        index=preds_bundle["dates"], 
        columns=target_cols)

    y_true_df.index = pd.to_datetime(y_true_df.index)
    y_pred_df.index = pd.to_datetime(y_pred_df.index)

    common_idx = y_true_df.index.intersection(y_pred_df.index)
    valid_eval_idx = common_idx[(common_idx >= evaluate_config.start_date) & (common_idx <= evaluate_config.end_date)]

    y_true = y_true_df.loc[valid_eval_idx]
    y_pred = y_pred_df.loc[valid_eval_idx]

    if y_true.empty:
        raise ValueError(
            f"No overlapping target data found within the requested evaluation window "
            f"({evaluate_config.start_date} to {evaluate_config.end_date}). Check data alignment.")

    metrics_report = compute_multi_target_metrics(y_true, y_pred)

    logger.info(f"Model evaluation completed: {model_name}")
    
    return {
        "metrics": metrics_report,
        "y_true": y_true, 
        "y_pred": y_pred,
        "dates": common_idx.strftime('%Y-%m-%d').tolist()
    }
