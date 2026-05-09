import streamlit as st
from datetime import datetime, date

from src.settings.config import MODELS_CONFIG_DIR

from src.utils.io_utils import list_contents

from src.config.train_config import TrainConfig
from src.config.common import SplitConfig, DataStrategy

from src.models.registry import train

st.title("Training Lab")

all_blueprints = list_contents(MODELS_CONFIG_DIR)

default_ix = 0
if 'selected_blueprint' in st.session_state:
    if st.session_state['selected_blueprint'] in all_blueprints:
        default_ix = all_blueprints.index(st.session_state['selected_blueprint'])

selected_blueprint = st.selectbox("Select Model Blueprint", all_blueprints, index=default_ix)

with st.form("training_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ticker = st.text_input("Ticker Symbol", value="AAPL")
        interval = st.selectbox("Interval", ["1d", "1wk"], index=0)
    
    with col2:
        start_date = st.date_input("Start Date", value=datetime(2000, 1, 1))
        end_date = st.date_input("End Date", value=datetime(2025, 1, 1))
        
    with col3:
        train_end = st.date_input("Training End", value=datetime(2020, 1, 1))
        val_end = st.date_input("Validation End", value=datetime(2022, 1, 1))

    st.markdown("---")
    st.subheader("Data Strategy")
    
    d_col1, d_col2, d_col3 = st.columns(3)
    
    with d_col1:
        st.write("**Technical**")
        t_save = st.checkbox("Save Tech", value=True)
        t_force = st.checkbox("Force Tech Download", value=False)
        
    with d_col2:
        st.write("**Macro**")
        m_save = st.checkbox("Save Macro", value=True)
        m_force = st.checkbox("Force Macro Download", value=False)
        
    with d_col3:
        st.write("**Processed**")
        p_save = st.checkbox("Save Processed", value=True)
        p_force = st.checkbox("Force Rebuild", value=False)

    submit = st.form_submit_button("Start Training Session")

    if submit:
        if not (start_date < train_end < val_end < end_date):
            st.error("Timeline error: Ensure Start < Train End < Val End < End Date.")
            st.stop()

        if end_date > date.today():
            st.error(f"End Date cannot be in the future. Max available date is {date.today()}.")
            st.stop()

        delta_days = (train_end - start_date).days
        if delta_days < 30:
            st.warning(f"The training period looks very short: {delta_days}.")

        with st.spinner("Calculating..."):
            model_path = train(TrainConfig(
                ticker=ticker,
                start_date=str(start_date),
                end_date=str(end_date),
                interval=interval,
                split=SplitConfig(
                    type="date",
                    train_end=str(train_end),
                    val_end=str(val_end)),
                data=DataStrategy(
                    technical={"save": t_save, "force_download": t_force},
                    macro={"save": m_save, "force_download": m_force},
                    processed={"save": p_save, "force_download": p_force}),
                model_config_path=selected_blueprint))
        
        st.success(f"Trained model saved to {model_path}")        