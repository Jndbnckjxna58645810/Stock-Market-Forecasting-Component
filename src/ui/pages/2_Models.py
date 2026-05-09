import streamlit as st
import pandas as pd

from src.settings.config import *
from src.utils.io_utils import list_contents

from src.config.model_metadata import ModelMetadata

from src.ui.components.render_model_card import render_model_card

st.title("Model Registry")
    
for m_name in list_contents(MODELS_DIR): render_model_card(m_name)