import streamlit as st
import os

from src.pipeline.apply_features import FEATURE_FUNCTIONS
from src.pipeline.apply_targets import TARGET_FUNCTIONS
from src.settings.config import MODELS_CONFIG_DIR
from src.utils.io_utils import list_contents, save_json
from src.config.model_config import ModelConfig

# --- Constants & Helpers ---
NEED_WINDOW = ["sma", "ema", "momentum", "volatility", "rsi", "bband_upper", "bband_lower", "volume_sma", "dist_sma", "zscore_close", "rolling_max", "rolling_min"]
NEED_N = ["lag", "return_lag", "log_return"]
MACRO_OPTIONS = {
    "Interest Rate": "FEDFUNDS",
    "Inflation (CPI)": "CPIAUCSL",
    "Unemployment": "UNRATE",
    "GDP": "GDP",
    "S&P 500 (VIX)": "VIXCLS"
}

def get_default_config():
    """Returns a fresh ModelConfig instance with defaults."""
    return ModelConfig({
        "features": [],
        "target": {
            "name": "return",
            "params": {"horizon": 1, "smoothing": 1, "log": False}
        },
        "macro_features": [],
        "hyperparameters": {},
        "model": {"name": "xgb", "params": {}}
    })

# --- Initialize Session State ---
if "draft_model" not in st.session_state:
    st.session_state.draft_model = get_default_config()

# --- Functions using the draft object ---
def add_feature_to_draft(draft, name, params):
    for feat in draft.features:
        if feat["name"] == name and feat["params"] == params:
            st.error("This exact feature/parameter pair already exists!")
            return
    draft.features.append({"name": name, "params": params})

# --- Layout ---
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Blueprints")
    all_configs = list_contents(MODELS_CONFIG_DIR)
    selected = st.selectbox("Select to Edit", ["New..."] + all_configs, key="blueprint_selector")

    if selected != "New...":
        if st.button("Load Configuration", use_container_width=True):
            st.session_state.draft_model = ModelConfig.from_name(selected)
            st.rerun()
    
    if st.button("Reset to Blank", use_container_width=True):
        st.session_state.draft_model = get_default_config()
        st.rerun()

with col2:
    # Use the draft reference for the rest of the page
    draft = st.session_state.draft_model

    # 1. Model Type
    model_list = ["xgb", "rf", "lstm"]
    # We access the dict inside ModelConfig via .model
    current_model_name = draft.model.get("name", "xgb")
    model_type = st.selectbox(
        "Model Type", 
        model_list, 
        index=model_list.index(current_model_name) if current_model_name in model_list else 0
    )
    draft.model["name"] = model_type

    # 2. Technical Features
    st.header("Technical Features")
    with st.container(border=True):
        col_feat, col_param, col_btn = st.columns([2, 1, 1])
        with col_feat:
            # Assumes FEATURE_FUNCTIONS is imported/defined globally
            f_name = st.selectbox("Select Feature", list(FEATURE_FUNCTIONS.keys()))
        with col_param:
            f_params = {}
            if f_name in NEED_WINDOW:
                f_params["window"] = st.number_input("Window", min_value=1, value=20)
            elif f_name in NEED_N:
                f_params["n"] = st.number_input("N (Lags)", min_value=1, value=1)
        with col_btn:
            st.write("##")
            if st.button("Add Feature"):
                add_feature_to_draft(draft, f_name, f_params)

    # Display Features from draft.features
    for i, feat in enumerate(draft.features):
        cols = st.columns([3, 1])
        cols[0].write(f"**{feat['name']}** — `{feat['params']}`")
        if cols[1].button("Delete", key=f"del_{i}"):
            draft.features.pop(i)
            st.rerun()

    # 3. Macro Features
    st.header("Macroeconomic Indicators")
    with st.container(border=True):
        m_col1, m_col2 = st.columns([2, 1])
        with m_col1:
            m_label = st.selectbox("Select Macro Indicator", list(MACRO_OPTIONS.keys()))
        with m_col2:
            if st.button("Add Macro"):
                new_macro = {"name": m_label.lower().replace(" ", "_"), "source": MACRO_OPTIONS[m_label]}
                if new_macro not in draft.macro_features:
                    draft.macro_features.append(new_macro)

    for i, m in enumerate(draft.macro_features):
        cols = st.columns([3, 1])
        cols[0].write(f"{m['name']} (`{m['source']}`)")
        if cols[1].button("Delete", key=f"del_m_{i}"):
            draft.macro_features.pop(i)
            st.rerun()

    # 4. Target Variable
    st.header("Target Variable")
    with st.expander("Configure Target", expanded=True):
        # Default to current target name in draft
        current_target_name = draft.target.get("name", "return")
        target_list = list(TARGET_FUNCTIONS.keys())
        t_name = st.selectbox("Target Function", target_list, index=target_list.index(current_target_name))

        t_params = draft.target.get("params", {})
        
        if t_name == "direction":
            t_params["threshold"] = st.number_input("Threshold", value=float(t_params.get("threshold", 0.0)), format="%.4f")
            t_params["horizon"] = st.number_input("Horizon", value=int(t_params.get("horizon", 1)))
        elif t_name == "multi_return":
            existing_hors = ", ".join(map(str, t_params.get("horizons", [1, 2, 3, 5])))
            hor_input = st.text_input("Horizons (comma separated)", value=existing_hors)
            t_params["horizons"] = [int(x.strip()) for x in hor_input.split(",") if x.strip()]
            t_params["smoothing"] = st.number_input("Smoothing", value=int(t_params.get("smoothing", 1)))
        elif t_name == "price":
            t_params["horizon"] = st.number_input("Horizon", value=int(t_params.get("horizon", 1)))
        else:
            t_params["horizon"] = st.number_input("Horizon", value=int(t_params.get("horizon", 1)))
            t_params["smoothing"] = st.number_input("Smoothing", value=int(t_params.get("smoothing", 1)))
            t_params["log"] = st.checkbox("Use Log Returns", value=bool(t_params.get("log", False)))

        # Update draft target
        draft.target["name"] = t_name
        draft.target["params"] = t_params

    # 5. Model & Hyperparameters
    st.header("Model & Training Parameters")
    
    # Initialize empty params if switching
    if "params" not in draft.model: draft.model["params"] = {}
    m_p = draft.model["params"]

    with st.container(border=True):
        if model_type == "xgb":
            m_p["n_estimators"] = st.number_input("N Estimators", value=int(m_p.get("n_estimators", 100)))
            m_p["max_depth"] = st.slider("Max Depth", 1, 15, int(m_p.get("max_depth", 3)))
            m_p["learning_rate"] = st.number_input("Learning Rate", value=float(m_p.get("learning_rate", 0.1)))
        elif model_type == "rf":
            m_p["n_estimators"] = st.number_input("N Estimators", value=int(m_p.get("n_estimators", 100)))
            m_p["max_depth"] = st.slider("Max Depth", 1, 30, int(m_p.get("max_depth", 10)))
        elif model_type == "lstm":
            st.info("Configuration moves to Hyperparameters section below.")

    if model_type == "lstm":
        st.subheader("LSTM Hyperparameters")
        hp = draft.hyperparameters # Reference
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            hp["seq_len"] = c1.number_input("Sequence Length", value=int(hp.get("seq_len", 20)))
            hp["epochs"] = c2.number_input("Epochs", value=int(hp.get("epochs", 10)))
            hp["units"] = c3.number_input("Hidden Units", value=int(hp.get("units", 64)))
            
            c4, c5, c6 = st.columns(3)
            batch_list = [16, 32, 64, 128]
            curr_batch = hp.get("batch_size", 32)
            hp["batch_size"] = c4.selectbox("Batch Size", batch_list, index=batch_list.index(curr_batch) if curr_batch in batch_list else 1)
            hp["dropout"] = c5.slider("Dropout", 0.0, 0.5, float(hp.get("dropout", 0.2)))
            hp["learning_rate"] = c6.number_input("Learning Rate (HP)", value=float(hp.get("learning_rate", 0.001)), format="%.4f")
            hp["seed"] = st.number_input("Random Seed", value=int(hp.get("seed", 42)))

    # --- Save Section ---
    st.divider()
    # Default name to the current selection if it's not "New..."
    default_name = selected if selected != "New..." else ""
    final_name = st.text_input("Configuration Filename", value=default_name)
    
    if st.button("💾 Save Blueprint", use_container_width=True):
        if not final_name:
            st.error("Please provide a name!")
        else:
            if not final_name.endswith(".json"):
                final_name += ".json"
            # Call to_dict() if your ModelConfig class has it, else pass draft.__dict__
            save_json(draft.to_dict(), MODELS_CONFIG_DIR / final_name)
            st.success(f"Blueprint '{final_name}' saved successfully.")