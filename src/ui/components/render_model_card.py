import streamlit as st
import pandas as pd

from src.settings.config import *

from src.config.model_metadata import ModelMetadata

from src.ui.components.feature_table import get_feature_table
from src.utils.io_utils import delete_directory

def render_model_card(m_name, mode="view", is_selected=False):
    with st.expander(f"Model: {m_name}"):
        # Load metadata once
        metadata = ModelMetadata.from_name(m_name)
        
        # --- Top Row: Header and Actions ---
        if mode == "evaluate":
            col_a, col_b = st.columns([5, 1])
            with col_a:
                st.subheader(f"{metadata.model.name.upper()}: {m_name}")
            with col_b:
                btn_label = "Remove" if is_selected else "Select"
                if st.button(btn_label, key=f"toggle_{m_name}", use_container_width=True):
                    if is_selected:
                        st.session_state['selected_models'].remove(m_name)
                    else:
                        st.session_state['selected_models'].append(m_name)
                    st.rerun()
        else:
            col_a, col_b, col_c = st.columns([4, 1, 1])
            with col_a:
                st.subheader(f"{metadata.model.name.upper()}: {m_name}")
            with col_b:
                if st.button("Predict", key=f"pred_{m_name}", use_container_width=True):
                    st.session_state.update({'active_model': m_name, 'show_predict_widget': True})
                    st.switch_page("pages/3_Inference.py")
            with col_c:
                if st.button("Delete", key=f"del_{m_name}", type="secondary", use_container_width=True):
                    delete_directory(m_name, base_dir=MODELS_DIR)
                    st.success(f"Deleted model folder: {m_name}")

                    if 'valid_models' in st.session_state:
                        if m_name in st.session_state['valid_models']:
                            st.session_state['valid_models'].remove(m_name)
                            
                    if 'selected_model_names' in st.session_state:
                        if m_name in st.session_state['selected_model_names']:
                            st.session_state['selected_model_names'].remove(m_name)

                    st.success(f"Model {m_name} deleted successfully.")
                    st.rerun()

        # --- Second Row: The "Property Grid" ---
        # Setting a fixed height here ensures all columns look aligned
        GRID_HEIGHT = 450 
        
        col_info, col_metrics, col_features = st.columns([2, 2, 2])

        with col_info:
            with st.container(height=GRID_HEIGHT, border=True):
                st.markdown("##### 📊 Model Info")
                st.caption(f"Created: {metadata.created_at}")
                
                # Use st.columns inside for a "key-value" look
                c1, c2 = st.columns(2)
                c1.metric("Ticker", metadata.ticker)
                c2.metric("Interval", metadata.interval)
                
                st.markdown("---")
                st.write(f"**Dataset Rows:** {metadata.n_rows}")
                st.write(f"**Date Range:**")
                st.caption(f"{metadata.start_date} → {metadata.end_date}")
                
                if metadata.model.hyperparameters:
                    st.write("**Hyperparameters:**")
                    st.json(metadata.model.hyperparameters, expanded=False)

                if metadata.model.params:
                    st.write("**Parameters:**")
                    st.json(metadata.model.params, expanded=False)

        with col_metrics:
            with st.container(height=GRID_HEIGHT, border=True):
                st.markdown("##### 🎯 Targets & Performance")
                
                # BETTER TARGET DISPLAY
                st.write("**Targets Configuration:**")
                for i, t in enumerate(metadata.targets):
                    # Display as a nice bold string
                    param_desc = ", ".join([f"{k}: {v}" for k, v in t.params.items()])
                    st.markdown(f"`{i}` **{t.name}** ({param_desc})")
                
                st.markdown("---")
                if isinstance(metadata.metrics, dict) and "overall" in metadata.metrics:
                    st.write("**Overall Performance**")
                    overall_df = pd.DataFrame([metadata.metrics["overall"]])
                    st.dataframe(overall_df, use_container_width=True, hide_index=True)

                    # --- Breakdown Table ---
                    if "breakdown" in metadata.metrics:
                        st.write("**Target Breakdown**")
                        # This turns the dict {target_name: {metric: val}} into a nice table
                        breakdown_df = pd.DataFrame(metadata.metrics["breakdown"]).T
                        # Format to 4 decimal places for the UI
                        st.dataframe(
                            breakdown_df.style.format("{:.4f}"), 
                            use_container_width=True
                        )
                else:
                    st.json(metadata.metrics)

        with col_features:
            with st.container(height=GRID_HEIGHT, border=True):
                st.markdown("##### 🛠️ Features")
                feat_df = get_feature_table(metadata)
                if not feat_df.empty:
                    st.dataframe(
                        feat_df, 
                        use_container_width=True,
                        hide_index=True)
                else:
                    st.info("No feature data recorded.")