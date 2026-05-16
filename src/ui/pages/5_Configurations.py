import streamlit as st
import os

from src.pipeline.apply_features import FEATURE_FUNCTIONS
from src.pipeline.apply_targets import TARGET_FUNCTIONS
from src.settings.config import MODELS_CONFIG_DIR

from src.utils.io_utils import list_contents, save_json, delete_file

from src.config.model_config import ModelConfig
from src.config.common import ModelSettings, TargetConfig

st.set_page_config(layout="wide", page_title="Financial AI Lab")

NEED_WINDOW = ["sma", "ema", "momentum", "volatility", "rsi", "bband_upper", "bband_lower", "volume_sma", "dist_sma", "zscore_close", "rolling_max", "rolling_min"]
NEED_N = ["lag", "return_lag", "log_return"]
MACRO_OPTIONS = {
    "Interest Rate": "FEDFUNDS",
    "Inflation (CPI)": "CPIAUCSL",
    "Unemployment": "UNRATE",
    "GDP": "GDP",
    "S&P 500 (VIX)": "VIXCLS"
}
DEFAULT_PARAMS = {
    "xgb": {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.03
    },
    "rf": {
        "n_estimators": 300,
        "max_depth": 8
    },
    "lstm": {}
}
DEFAULT_HYPERPARAMETERS = {
    "xgb": {},
    "rf": {},
    "lstm": {
        "seq_len": 20,
        "units": 64,
        "epochs": 20,
        "batch_size": 32,
        "dropout": 0.2,
        "learning_rate": 0.001
    }
}

def get_default_config():
    """Returns a fresh ModelConfig instance with defaults."""
    return ModelConfig(
        features=[],
        targets=[],
        macro_features=[],
        model=ModelSettings(name="xgb", params={}, hyperparameters={}))

if "draft_model" not in st.session_state:
    st.session_state.draft_model = get_default_config()

def add_feature_to_draft(draft, name, params):
    for feat in draft.features:
        if feat["name"] == name and feat["params"] == params:
            st.error("🔄 This exact feature/parameter pair already exists!")
            return
    draft.features.append({"name": name, "params": params})

def add_target_to_draft(draft, name, params):

    for t in draft.targets:
        if t.name == name and t.params == params:
            st.error("🔄 This exact target already exists!")
            return

    draft.targets.append(TargetConfig(name=name, params=params))

col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("📂 Blueprints")
    all_configs = list_contents(MODELS_CONFIG_DIR)
    selected = st.selectbox("Select to Edit", ["New..."] + all_configs, key="blueprint_selector")

    if selected != "New...":
        if st.button("📂 Load Configuration", use_container_width=True):
            st.session_state.draft_model = ModelConfig.from_name(selected)
            st.rerun()
    
    if st.button("🔄 Reset to Blank", use_container_width=True):
        st.session_state.draft_model = get_default_config()
        st.rerun()

    if selected != "New...":
        if st.button("🗑️ Delete Configuration", use_container_width=True):
            delete_file(selected, MODELS_CONFIG_DIR)
            st.success(f"Deleted {selected}")
            st.rerun()

with col2:
    draft = st.session_state.draft_model

    model_list = ["xgb", "rf", "lstm"]

    # current_model_name = draft.model.name
    # model_type = st.selectbox(
    #     "Model Type", 
    #     model_list, 
    #     index=model_list.index(current_model_name) if current_model_name in model_list else 0
    # )
    # draft.model.name = model_type
    selected_model_type = st.selectbox(
        "Model",
        ["xgb", "rf", "lstm"],
        index=["xgb", "rf", "lstm"].index(draft.model.name)
    )

    if selected_model_type != draft.model.name:
        draft.model.name = selected_model_type
        draft.model.params = DEFAULT_PARAMS[selected_model_type].copy()
        draft.model.hyperparameters = DEFAULT_HYPERPARAMETERS[selected_model_type].copy()

    st.header("📈 Technical Features")
    with st.container(border=True):
        col_feat, col_param, col_btn = st.columns([2, 1, 1])
        with col_feat:
            f_name = st.selectbox("🔍 Select Feature", list(FEATURE_FUNCTIONS.keys()))
        with col_param:
            f_params = {}
            if f_name in NEED_WINDOW:
                f_params["window"] = st.number_input("Window", min_value=1, value=20)
            elif f_name in NEED_N:
                f_params["n"] = st.number_input("N (Lags)", min_value=1, value=1)
        with col_btn:
            if st.button("➕ Add Feature"):
                add_feature_to_draft(draft, f_name, f_params)

    for i, feat in enumerate(draft.features):
        cols = st.columns([3, 1])
        cols[0].write(f"**{feat['name']}** — `{feat['params']}`")
        if cols[1].button("🗑️ Delete", key=f"del_{i}"):
            draft.features.pop(i)
            st.rerun()

    st.header("🌍 Macroeconomic Indicators")
    with st.container(border=True):
        m_col1, m_col2 = st.columns([2, 1])
        with m_col1:
            m_label = st.selectbox("🔍 Select Macro Indicator", list(MACRO_OPTIONS.keys()))
        with m_col2:
            if st.button("➕ Add Macro"):
                new_macro = {"name": m_label.lower().replace(" ", "_"), "source": MACRO_OPTIONS[m_label]}
                if new_macro not in draft.macro_features:
                    draft.macro_features.append(new_macro)

    for i, m in enumerate(draft.macro_features):
        cols = st.columns([3, 1])
        cols[0].write(f"{m['name']} (`{m['source']}`)")
        if cols[1].button("🗑️ Delete", key=f"del_m_{i}"):
            draft.macro_features.pop(i)
            st.rerun()

    st.header("🎯 Target Variables")
    with st.container(border=True):
        t_col_func, t_col_btn = st.columns([3, 1])
        
        target_list = list(TARGET_FUNCTIONS.keys())
        t_func = t_col_func.selectbox("➕ Add Target Function", target_list)

        p_col1, p_col2, p_col3 = st.columns(3)
        new_t_params = {}
        
        if t_func == "direction":
            new_t_params["threshold"] = p_col1.number_input("🛠️ Threshold", value=0.0, format="%.4f", key="t_thresh")
            new_t_params["horizon"] = p_col2.number_input("🛠️ Horizon", min_value=1, value=1, key="t_hor_dir")
        elif t_func == "multi_return":
            hor_input = p_col1.text_input("🛠️ Horizons (comma separated)", value="1, 5, 10")
            new_t_params["horizons"] = [int(x.strip()) for x in hor_input.split(",") if x.strip()]
        elif t_func == "price":
            new_t_params["horizon"] = p_col1.number_input("🛠️ Horizon", min_value=1, value=1)
        else:
            new_t_params["horizon"] = p_col1.number_input("🛠️ Horizon", min_value=1, value=1)
            new_t_params["smoothing"] = p_col2.number_input("🛠️ Smoothing", min_value=1, value=1)
            new_t_params["log"] = p_col3.checkbox("🛠️ Log Returns", value=False)

        if st.button("➕ Add Target to Blueprint", use_container_width=True):
            add_target_to_draft(draft, t_func, new_t_params)

    for i, t in enumerate(draft.targets):
        cols = st.columns([3, 1])

        p_str = ", ".join([f"{k}: {v}" for k, v in t.params.items()])
        cols[0].write(f"🎯 **{t.name.upper()}** — `{p_str}`")
        if cols[1].button("🗑️ Remove", key=f"del_t_{i}"):
            draft.targets.pop(i)
            st.rerun()

    st.header("🧠 Model & Training Parameters")
    
    m_p = draft.model.params

    with st.container(border=True):
        if selected_model_type == "xgb":
            m_p["n_estimators"] = st.number_input("🛠️ N Estimators", value=int(m_p.get("n_estimators", 100)))
            m_p["max_depth"] = st.slider("🛠️ Max Depth", 1, 15, int(m_p.get("max_depth", 3)))
            m_p["learning_rate"] = st.number_input("🛠️ Learning Rate", value=float(m_p.get("learning_rate", 0.1)))
        elif selected_model_type == "rf":
            m_p["n_estimators"] = st.number_input("🛠️ N Estimators", value=int(m_p.get("n_estimators", 100)))
            m_p["max_depth"] = st.slider("🛠️ Max Depth", 1, 30, int(m_p.get("max_depth", 10)))
        elif selected_model_type == "lstm":
            st.info("⚙️ Configuration moves to Hyperparameters section below.")

    if selected_model_type == "lstm":
        st.subheader("⚙️ LSTM Hyperparameters")
        hp = draft.model.hyperparameters
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            hp["seq_len"] = c1.number_input("🛠️ Sequence Length", value=int(hp.get("seq_len", 20)))
            hp["epochs"] = c2.number_input("🛠️ Epochs", value=int(hp.get("epochs", 10)))
            hp["units"] = c3.number_input("🛠️ Hidden Units", value=int(hp.get("units", 64)))
            
            c4, c5, c6 = st.columns(3)
            batch_list = [16, 32, 64, 128]
            curr_batch = hp.get("batch_size", 32)
            hp["batch_size"] = c4.selectbox("🛠️ Batch Size", batch_list, index=batch_list.index(curr_batch) if curr_batch in batch_list else 1)
            hp["dropout"] = c5.slider("🛠️ Dropout", 0.0, 0.5, float(hp.get("dropout", 0.2)))
            hp["learning_rate"] = c6.number_input("🛠️ Learning Rate (HP)", value=float(hp.get("learning_rate", 0.001)), format="%.4f")
            hp["seed"] = st.number_input("🛠️ Random Seed", value=int(hp.get("seed", 42)))
    else:
        st.subheader("⚙️ Hyperparameters")
        hp = draft.model.hyperparameters
        hp["seed"] = st.number_input("🛠️ Random Seed", value=int(hp.get("seed", 42)))

    st.divider()

    default_name = selected if selected != "New..." else ""
    final_name = st.text_input("📂 Configuration Filename", value=default_name)
    
    if st.button("💾 Save Blueprint", use_container_width=True):
        if not final_name:
            st.error("🔄 Please provide a name!")
        else:
            if not final_name.endswith(".json"):
                final_name += ".json"

            save_json(draft.to_dict(), MODELS_CONFIG_DIR / final_name)
            st.success(f"📂 Blueprint '{final_name}' saved successfully.")
            st.rerun()