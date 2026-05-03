from src.utils.config_utils import ensure
from src.utils.logging_utils import get_logger

from src.pipeline.build_dataset import build_input

from src.config.predict_config import PredictConfig
from src.config.model_metadata import ModelMetadata

logger = get_logger("models.predict_models.predict_tabular")

def predict_tabular_by_parameters(model_bundle, X, target_cols):
    model = model_bundle.get("model")
    x_scaler = model_bundle.get("x_scaler")
    y_scaler = model_bundle.get("y_scaler")

    dates = X.index.strftime('%Y-%m-%d').tolist()

    if x_scaler:
        X = x_scaler.transform(X)

        logger.info("Scaling performed on X")

    preds_scaled = model.predict(X)

    if y_scaler:
        preds_final = y_scaler.inverse_transform(preds_scaled)

        logger.info("Descaling performed on y")
    else:
        preds_final = preds_scaled
    
    logger.info("Prediction completed")

    preds_dict = {}
    for i, col_name in enumerate(target_cols):
        preds_dict[col_name] = preds_final[:, i].tolist()
    return {"dates": dates, "preds": preds_final.tolist()}

def predict_tabular(predict_config: PredictConfig):
    predict_config = ensure(predict_config, PredictConfig)
    metadata = ModelMetadata.from_name(predict_config.model_path)

    logger.info(f"Predicting using {predict_config.model_path}" +
                f" | Period: {predict_config.start_date} to {predict_config.end_date}")

    from src.models.registry import load_model
    return predict_tabular_by_parameters(
        load_model(predict_config.model_path),
        build_input(predict_config)[metadata.selected_features],
        metadata.target_cols)