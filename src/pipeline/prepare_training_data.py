from src.pipeline.apply_targets import apply_targets_by_parameters
from src.pipeline.build_dataset import build_dataset_by_parameters

from src.utils.config_utils import ensure
from src.utils.logging_utils import get_logger
from src.utils.data_manager import save_data, load_data

from src.config.train_config import TrainConfig
from src.config.model_config import ModelConfig

logger = get_logger("pipeline.prepare_training_data")

def prepare_training_data(run: TrainConfig, model_config: ModelConfig):
    run = ensure(run, TrainConfig)
    if model_config == None:
        if run.model_config_path == None: raise ValueError("Training requires model_config")
        model_config = ModelConfig.from_name(run.model_config_path)

    df = load_data(
        "processed" ,run.data, ticker=run.ticker,
        start_date=run.start_date, end_date=run.end_date, interval=run.interval,
        features=model_config.features, macro_features=model_config.macro_features)
    
    if df.empty:
        logger.info("Processed dataset cache miss. Triggering engine build pipeline...")
        df = build_dataset_by_parameters(
            run.ticker,
            run.start_date, run.end_date,
            run.interval,
            model_config.features, model_config.macro_features,
            model_config.targets,
            model_config.model.hyperparameters, run.data)
    
    save_data(df, "processed", run.data, ticker=run.ticker,
              start_date=run.start_date, end_date=run.end_date, interval=run.interval,
              features=model_config.features, macro_features=model_config.macro_features)
    
    logger.info(f"Processed data for {run.ticker}" +
                f" | Period: {run.start_date} to {run.end_date}" +
                f" | Interval: {run.interval}" + (
                    f" | Features from configuration file: {run.model_config_path}"
                    if run.model_config_path else ""))

    df, target_cols = apply_targets_by_parameters(df, model_config.targets)
    df = df.dropna()

    X = df.drop(columns=target_cols)
    y = df[target_cols]

    train_end = run.split.train_end
    val_end = run.split.val_end or train_end

    if not run.split.val_end:
        logger.warning(f"val_end not specified | val_end set to train_end: {val_end}")

    X_train = X.loc[:train_end]
    y_train = y.loc[:train_end]

    X_val = X[(X.index > train_end) & (X.index <= val_end)]
    y_val = y[(y.index > train_end) & (y.index <= val_end)]

    X_test = X[X.index > val_end]
    y_test = y[y.index > val_end]

    logger.info(f"Dataset split for train_date={train_end} and val_end={val_end}")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "target_cols": target_cols,
        "df": df
    }