from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

from src.pipeline.build_dataset import build_dataset_from_run
from src.models.build_model import build_model

from src.utils.model_utils import save_model
from src.utils.config_utils import load_run_config, load_model_config

def add_target(df, target_config):
    h = target_config["horizon"]
    df["target"] = df["close"].pct_change(h).shift(-h)
    return df

def train(run_config_path):
    run_config = load_run_config(run_config_path)
    model_config = load_model_config(run_config["model_config"])

    df = build_dataset_from_run(run_config_path)

    df["target"] = df["close"].pct_change().shift(-1)
    df = df.dropna()

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = build_model(model_config)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("R2:", r2_score(y_test, preds))
    print("MSE:", mean_squared_error(y_test, preds))

    save_model(model, model_config, X.columns)