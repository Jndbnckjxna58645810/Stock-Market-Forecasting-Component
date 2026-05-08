import streamlit as st
import pandas as pd

from src.settings.config import *
from src.utils.io_utils import list_contents, load_csv, delete_file

st.title("Data Explorer")

processed_data_files = list_contents(PROCESSED_DATA_DIR, pattern="*.csv")
raw_data_files = list_contents(RAW_DATA_DIR, pattern="*.csv")
macro_data_files = [f for f in raw_data_files if f.startswith("macro")]
technical_data_files = [f for f in raw_data_files if not f.startswith("macro")]

if 'view_df' not in st.session_state:
    st.session_state.view_df = None
if 'view_name' not in st.session_state:
    st.session_state.view_name = ""

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Processed")
    selected_processed = st.selectbox("Select Processed", processed_data_files)
    
    btn_col_spacer, btn_col1, btn_col2 = st.columns([1, 1, 1])

    with btn_col1:
        if st.button("Load Data", key=f"load_{selected_processed}", use_container_width=True):
            st.session_state.view_df = load_csv(selected_processed, PROCESSED_DATA_DIR)
            st.session_state.view_name = selected_processed

    with btn_col2:
        if st.button("Delete", key=f"del_{selected_processed}", use_container_width=True):
            delete_file(selected_processed, base_dir=PROCESSED_DATA_DIR)
            st.success(f"Deleted {selected_processed}")
            st.rerun()

with col2:
    st.subheader("Technical")
    selected_raw = st.selectbox("Select Technical", technical_data_files)

    btn_col_spacer, btn_col1, btn_col2 = st.columns([1, 1, 1])

    with btn_col1:
        if st.button("Load Data", key=f"load_{selected_raw}", use_container_width=True):
            st.session_state.view_df = load_csv(selected_raw, RAW_DATA_DIR)
            st.session_state.view_name = selected_raw

    with btn_col2:
        if st.button("Delete", key=f"del_{selected_raw}", use_container_width=True):
            delete_file(selected_raw, base_dir=RAW_DATA_DIR)
            st.success(f"Deleted {selected_raw}")
            st.rerun()

with col3:
    st.subheader("Macro")
    selected_macro = st.selectbox("Select Macroeconomic", macro_data_files)
    btn_col_spacer, btn_col1, btn_col2 = st.columns([1, 1, 1])

    with btn_col1:
        if st.button("Load Data", key=f"load_{selected_macro}", use_container_width=True):
            st.session_state.view_df = load_csv(selected_macro, RAW_DATA_DIR)
            st.session_state.view_name = selected_macro

    with btn_col2:
        if st.button("Delete", key=f"del_{selected_macro}", use_container_width=True):
            delete_file(selected_macro, base_dir=RAW_DATA_DIR)
            st.success(f"Deleted {selected_macro}")
            st.rerun()

st.divider() 

if st.session_state.view_df is not None:
    st.write(f"### Viewing: {st.session_state.view_name}")
        
    st.dataframe(st.session_state.view_df, use_container_width=True)
        
    if st.button("Close Viewer"):
        st.session_state.view_df = None
        st.rerun()
