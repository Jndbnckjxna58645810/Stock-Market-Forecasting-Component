import streamlit as st
import pandas as pd

from src.settings.config import *
from src.utils.io_utils import list_contents, delete_directory

from src.models.registry import predict

from src.config.predict_config import PredictConfig
from src.config.model_metadata import ModelMetadata

from src.utils.gui_utils import get_feature_table

st.title("Model Registry")
    
models = list_contents(MODELS_DIR)
    
for m_name in models:
    with st.expander(f"Model: {m_name}"):
        col_a, col_b, col_c = st.columns([4, 1, 1])
            
        with col_a:
            metadata = ModelMetadata.from_name(m_name)
            st.markdown(f"### {metadata.model['name'].upper()}: {m_name}")
            
        with col_b:
            if st.button("Predict", key=f"pred_{m_name}", use_container_width=True):
                st.session_state['active_model'] = m_name
                st.session_state['show_predict_widget'] = True

        with col_c:
            if st.button("Delete Model", key=f"del_{m_name}", use_container_width=True):
                delete_directory(m_name, base_dir=MODELS_DIR)
                st.success(f"Deleted model folder: {m_name}")
                st.rerun()

        col_info, col_metrics, col_features = st.columns(3)
        with col_info:
            st.write(f"**Ticker:** {metadata.ticker}")
            st.write(f"**Interval:** {metadata.interval}")

            train_end = pd.to_datetime(metadata.split["train_end"] or metadata.end_date)
            val_end = pd.to_datetime(metadata.split["val_end"] or train_end)

            st.write(f"**Train:** {pd.to_datetime(metadata.start_date)} to {train_end}")
            if train_end != val_end:
                st.write(f"**Validate:** {train_end + pd.DateOffset(days=1)} to {val_end}")
            st.write(f"**Test:** {val_end} to {pd.to_datetime(metadata.end_date)}")

            st.write(f"**Rows:** {metadata.n_rows}")

            target = metadata.target
            t_name = target['name']

            t_params = ", ".join([f"{k}={v}" for k, v in target['params'].items()])
            target_display = f"{t_name}({t_params})"

            st.markdown(f"**Target:** {target_display}")

            if metadata.model.get("params", False):
                model_params = metadata.model
                model_params_name = model_params['name']

                model_params_params = ", ".join([f"{k}={v}" for k, v in metadata.model['params'].items()])
                model_params_display = f"{model_params_name.upper()}({model_params_params})"

                st.markdown(f"**Model:** {model_params_display}")

            if metadata.model_path_config:
                st.write(f"**Configuration:** {metadata.model_path_config}")

            st.write(f"**Created at:** {metadata.created_at}")

        with col_metrics:
            st.write("**Performance:**")
            st.json(metadata.metrics)

            if metadata.hyperparameters:
                st.write("**Hyperparameters:**")
                st.json(metadata.hyperparameters)

        with col_features:
            feat_df = get_feature_table(metadata)
                
            if not feat_df.empty:
                st.dataframe(
                    feat_df, 
                    height=(600 if metadata.hyperparameters else 400), 
                    use_container_width=True,
                    hide_index=True)

if st.session_state.get('show_predict_widget'):
    st.divider()
    st.subheader(f"Inference: {st.session_state['active_model']}")
        
    c1, c2 = st.columns(2)
    start_date = c1.date_input("Start Date")
    end_date = c2.date_input("End Date")

    if st.button("Generate Forecast"):
        with st.spinner("Calculating..."):
            config = PredictConfig({
                "model_path": st.session_state['active_model'],
                "start_date": start_date, "end_date": end_date
            })
                
            result = predict(config)
                
            if result.size >= 5:
                st.line_chart(result)
                
            st.table(pd.DataFrame(result))