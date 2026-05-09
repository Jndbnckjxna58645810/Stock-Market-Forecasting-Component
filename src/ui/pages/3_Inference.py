import streamlit as st
import pandas as pd

from src.config.predict_config import PredictConfig

from src.models.registry import predict

if st.session_state.get('show_predict_widget'):
    st.subheader(f"Inference: {st.session_state['active_model']}")
        
    c1, c2 = st.columns(2)
    start_date = c1.date_input("Start Date")
    end_date = c2.date_input("End Date")

    if st.button("Generate Forecast"):
        with st.spinner("Calculating..."):
            config = PredictConfig(
                model_path=st.session_state['active_model'],
                start_date=start_date.strftime('%Y-%m-%d'), 
                end_date=end_date.strftime('%Y-%m-%d')
            )
                
            try:
                result = predict(config)[config.start_date:config.end_date]
                
                if result.size >= 5:
                    st.line_chart(result)
                    
                st.table(pd.DataFrame(result))
                
            except ValueError as ve:
                st.error(ve)
            except Exception as e:
                st.error(f"A critical error occurred: {e}")
                st.exception(e)

    if st.button("Cancel"):
        st.session_state['show_predict_widget'] = False
        st.rerun()

else:
    st.info("Please select a model from the Models page first.")