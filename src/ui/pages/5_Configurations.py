import streamlit as st
import os

from src.settings.config import MODELS_CONFIG_DIR

from src.utils.io_utils import list_contents, save_json

from src.config.model_config import ModelConfig

st.title("Model Registry")
    
col1, col2 = st.columns([1, 3])
    
with col1:
    st.subheader("Blueprints")
    all_configs = list_contents(MODELS_CONFIG_DIR)
    selected = st.selectbox("Select to Edit", ["New..."] + all_configs)

with col2:
    with st.form("model_config_form"):
        st.subheader(f"Editing: {selected}")

        save_dict = ModelConfig.from_name(selected).to_dict() if selected != "New..." else {}
        save_name = st.text_input("Configuration Name", value=selected if selected != "New..." else "")
                
        if st.form_submit_button("Save Blueprint"):
            save_json(save_dict, MODELS_CONFIG_DIR / save_name)
            st.success("Blueprint Saved!")
            st.rerun()

    if selected != "New...":
        if st.button("Delete Blueprint", type="secondary"):
            os.remove(MODELS_CONFIG_DIR / selected)
            st.warning(f"Deleted {selected}")
            st.rerun()

        if st.button("Train this Blueprint"):
            st.session_state['selected_blueprint'] = selected
            st.switch_page("pages/6_Training.py")