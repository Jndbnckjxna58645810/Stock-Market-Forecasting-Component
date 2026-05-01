from src.pipeline.build_dataset import build_dataset

from src.pipeline.apply_targets import apply_target_to_dataset

from src.utils.config_utils import ensure

from src.config.train_config import TrainConfig
from src.config.model_config import ModelConfig

def prepare_training_data(run: TrainConfig, model_config: ModelConfig):
    run = ensure(run, TrainConfig)
    if model_config == None:
        if run.model_config_path == None: raise ValueError("Training requires model_config")
        model_config = ModelConfig.from_name(run.model_config_path)
    
    df = build_dataset(run, model_config)

    df, target_cols = apply_target_to_dataset(df, run, model_config)
    df = df.dropna()

    X = df.drop(columns=target_cols)
    y = df[target_cols]

    train_end = run.split["train_end"]
    val_end = run.split.get("val_end") or train_end

    X_train = X.loc[:train_end]
    y_train = y.loc[:train_end]

    X_val = X[(X.index > train_end) & (X.index <= val_end)]
    y_val = y[(y.index > train_end) & (y.index <= val_end)]

    X_test = X[X.index > val_end]
    y_test = y[y.index > val_end]

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