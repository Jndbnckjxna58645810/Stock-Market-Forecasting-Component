import streamlit as st
import pandas as pd

from src.config.model_metadata import ModelMetadata

from src.ui.components.feature_table import get_feature_table

def render_model_card(m_name, is_selected):
    with st.expander(f"Model: {m_name}", expanded=is_selected):
        col_a, col_b = st.columns([5, 1])
        metadata = ModelMetadata.from_name(m_name)
        
        with col_a:
            st.markdown(f"### {metadata.model['name'].upper()}: {m_name}")
        
        with col_b:
            btn_label = "Remove" if is_selected else "Select"
            if st.button(btn_label, key=f"toggle_{m_name}", use_container_width=True):
                if is_selected:
                    st.session_state['selected_model_names'].remove(m_name)
                else:
                    st.session_state['selected_model_names'].append(m_name)
                st.rerun()
        col_info, col_metrics, col_features = st.columns(3)

        col_info, col_metrics, col_features = st.columns(3)
        with col_info:
            train_end = pd.to_datetime(metadata.split["train_end"] or metadata.end_date)
            val_end = pd.to_datetime(metadata.split["val_end"] or train_end)

            st.write(f"**Train:** {pd.to_datetime(metadata.start_date)} to {train_end}")
            if train_end != val_end:
                st.write(f"**Validate:** {train_end + pd.DateOffset(days=1)} to {val_end}")
            st.write(f"**Test:** {val_end} to {pd.to_datetime(metadata.end_date)}")

            st.write(f"**Rows:** {metadata.n_rows}")

            if metadata.model.get("params", False):
                model_params = metadata.model
                model_params_name = model_params['name']

                model_params_params = ", ".join([f"{k}={v}" for k, v in metadata.model['params'].items()])
                model_params_display = f"{model_params_name.upper()}({model_params_params})"

                st.markdown(f"**Model:** {model_params_display}")

            if metadata.model_path_config:
                st.write(f"**Configuration:** {metadata.model_path_config}")

            st.write(f"**Created at:** {metadata.created_at}")

            if metadata.hyperparameters:
                st.write("**Hyperparameters:**")
                st.json(metadata.hyperparameters)

        with col_metrics:
            st.write("**Performance:**")
            with st.container(height=300):
                st.json(metadata.metrics)

        with col_features:
            st.write("**Features:**")
            feat_df = get_feature_table(metadata)
                
            if not feat_df.empty:
                st.dataframe(
                    feat_df, 
                    height=300, 
                    use_container_width=True,
                    hide_index=True)
