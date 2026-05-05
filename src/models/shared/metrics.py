import numpy as np

from sklearn.metrics import r2_score, mean_squared_error

from src.utils.logging_utils import get_logger

logger = get_logger("models.shared.compute_metrics")

def compute_metrics(y_true, preds):
    y_true = np.asarray(y_true).flatten()
    preds = np.asarray(preds).flatten()

    mse = mean_squared_error(y_true, preds)
    baseline = np.zeros_like(y_true)
    baseline_mse = mean_squared_error(y_true, baseline)

    directional_acc = np.mean(np.sign(y_true) == np.sign(preds))

    logger.info("Metrics calculated")
    return {
        "mse": mse,
        "rmse": np.sqrt(mse),
        "r2": r2_score(y_true, preds),
        "baseline_mse": baseline_mse,
        "mse_ratio": mse / baseline_mse if baseline_mse > 1e-12 else None,
        "mean_target": np.mean(y_true),
        "std_target": np.std(y_true),
        "directional_accuracy": directional_acc
    }

def compute_multi_target_metrics(y_true_df, y_pred_df):
    breakdown = {}
    for col in y_true_df.columns:
        breakdown[col] = compute_metrics(y_true_df[col], y_pred_df[col])

    overall = compute_metrics(
        y_true_df.values.ravel(), 
        y_pred_df.values.ravel())

    return {"overall": overall, "breakdown": breakdown}