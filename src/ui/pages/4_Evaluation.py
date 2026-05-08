import streamlit as st
import pandas as pd
from datetime import date

from src.ui.components.render_model_card import render_model_card
from src.ui.components.model_filter import get_filter_options, get_target_string, get_valid_models

from src.models.registry import evaluate_models

from src.config.evaluate_config import EvaluateConfig

if 'valid_models' not in st.session_state:
    st.session_state['valid_models'] = []
if 'selected_model_names' not in st.session_state:
    st.session_state['selected_model_names'] = []

tickers, intervals, targets = get_filter_options()

c1, c2 = st.columns(2)
ticker = c1.selectbox("Ticker", tickers)
interval = c1.selectbox("Interval", intervals)

eval_period = c2.date_input("Evaluation Period", [])
target = c2.selectbox("Target",
                            options=targets.values(),
                            format_func=lambda x: get_target_string(x))

if st.button("Filter Models"):
    if len(eval_period) == 2:
        st.session_state['selected_model_names'] = []

        if 'eval_results' in st.session_state:
            del st.session_state['eval_results']
            
        start_str = eval_period[0].strftime('%Y-%m-%d')
        end_str = eval_period[1].strftime('%Y-%m-%d')

        if eval_period[1] > date.today():
            st.error(f"End Date cannot be in the future. Max available date is {date.today()}.")
            st.stop()

        delta_days = (eval_period[1] - eval_period[0]).days
        if delta_days < 30:
            st.warning(f"The training period looks very short: {delta_days}.")
            st.stop()

        st.session_state['valid_models'] = get_valid_models(ticker, interval, start_str, end_str, target)

        st.rerun()

valid_names = st.session_state['valid_models']
selected_set = st.session_state['selected_model_names']

if selected_set:
    st.subheader("Selected for Evaluation")
    for m_name in selected_set:
        render_model_card(m_name, is_selected=True)

st.divider()

if len(eval_period) == 2:
    start_str = eval_period[0].strftime('%Y-%m-%d')
    end_str = eval_period[1].strftime('%Y-%m-%d')

if st.button("Evaluate Models", type="primary", use_container_width=True):
    if not st.session_state['selected_model_names']:
        st.error("Please select at least one model first!")
    elif len(eval_period) != 2:
        st.error("Please select a valid Start and End date.")
    else:
        with st.spinner("Calculating..."):
            start_str = eval_period[0].strftime('%Y-%m-%d')
            end_str = eval_period[1].strftime('%Y-%m-%d')
            
            results = evaluate_models(EvaluateConfig({
                "models": selected_set,
                "ticker": ticker,
                "start_date": start_str,
                "end_date": end_str,
                "interval": interval,
                "target": target
            }))
            
            st.session_state['eval_results'] = results

if 'eval_results' in st.session_state:
    results = st.session_state['eval_results']
    
    st.divider()
    st.header("Comparative Metrics")

    overall_metrics = {}
    for m_name, data in results.items():
        overall_metrics[m_name] = data['metrics']['overall']

    metrics_df = pd.DataFrame(overall_metrics)
    st.subheader("Global Performance Summary")
    st.dataframe(metrics_df, use_container_width=True)

    first_model = list(results.keys())[0]
    target_cols = results[first_model]['y_true'].columns

    for col in target_cols:
        st.markdown(f"---")
        st.subheader(f"Target Feature: **{col}**")

        col_metrics = {}
        for m_name in results:
            if 'breakdown' in results[m_name]['metrics'] and col in results[m_name]['metrics']['breakdown']:
                col_metrics[m_name] = results[m_name]['metrics']['breakdown'][col]
                
        if col_metrics:
            st.markdown("**Feature Specific Errors:**")
            st.dataframe(pd.DataFrame(col_metrics), use_container_width=True)

        plot_df = pd.DataFrame(index=results[first_model]['y_true'].index)
        plot_df['Actual'] = results[first_model]['y_true'][col]

        for m_name in results:
            plot_df[m_name] = results[m_name]['y_pred'][col]

        plot_df = plot_df.dropna()
        
        st.markdown("**Visualizing Alignment:**")
        st.line_chart(plot_df)

if valid_names:
    st.divider()
    st.subheader("Available Models")
    available = [m for m in valid_names if m not in selected_set]
    if not available and not selected_set:
        st.info("No models match the filters.")
    for m_name in available:
        render_model_card(m_name, is_selected=False)