import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

from src.pipeline.build_dataset import build_dataset

from src.models.build_model import build_model
from src.models.apply_targets import apply_target

from src.utils.model_utils import save_model
from src.utils.config_utils import load_run_config, load_model_config

def train(run):
    run_config = load_run_config(run)
    model_config = load_model_config(run_config["model_config"])

    df = build_dataset(run)

    df, target_cols = apply_target(df, model_config["target"])
    df = df.dropna()

    X = df.drop(columns=target_cols)
    y = df[target_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = build_model(model_config)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("R2:", r2_score(y_test, preds))
    print("MSE:", mean_squared_error(y_test, preds))

    importances = model.feature_importances_

    feat_importance = pd.Series(importances, index=X.columns)
    feat_importance = feat_importance.sort_values(ascending=False)
    print(feat_importance)

    save_model(model, model_config, X.columns)