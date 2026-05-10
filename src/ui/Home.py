import streamlit as st

from src.utils.logging_utils import setup_logger

setup_logger()

st.set_page_config(layout="wide", page_title="Financial AI Lab")

st.title("🏠 Welcome to the Financial AI Lab")
st.write("Use the sidebar to navigate between Data and Models.")