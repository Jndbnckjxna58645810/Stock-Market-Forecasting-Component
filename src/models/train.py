import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

from src.pipeline.build_dataset import build_dataset

from src.models.build_model import build_model
from src.models.apply_targets import apply_target

from src.utils.model_utils import save_model
from src.utils.config_utils import load_run_config, load_model_config

from src.config.run_config import RunConfig

def train(run : RunConfig):
    model_config = load_model_config(run.model_config_path)

    df = build_dataset(run)

    df, target_cols = apply_target(df, run)
    df = df.dropna()

    X = df.drop(columns=target_cols)
    y = df[target_cols]

    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    df = df.drop(columns=to_drop)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 1. Prepare data (Scaling with column recovery)
    scaler = StandardScaler()
    X_train_df = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test_df = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

    # 2. Build and train baseline
    model = build_model(run)
    model.fit(X_train_df, y_train)

    # 3. Feature Selection
    importances = pd.Series(model.feature_importances_, index=X_train_df.columns).sort_values(ascending=False)
    k = int(len(importances) * 0.8)
    selected = importances.iloc[:k].index.tolist()

    # 4. Final Train
    model.fit(X_train_df[selected], y_train)
    preds = model.predict(X_test_df[selected])

    print("R2:", r2_score(y_test, preds))
    print("MSE:", mean_squared_error(y_test, preds))
    print(X_test_df[selected].columns)
    importances = pd.Series(model.feature_importances_, index=X_train_df[selected].columns).sort_values(ascending=False)
    print(importances)

    save_model(model, model_config, X_test_df[selected].columns)