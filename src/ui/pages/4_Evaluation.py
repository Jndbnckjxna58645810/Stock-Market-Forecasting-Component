import streamlit as st
import pandas as pd

from src.models.registry import evaluate_models

from src.config.evaluate_config import EvaluateConfig
from src.config.model_metadata  import ModelMetadata

from src.ui.components.model_filter import get_filter_options, get_valid_models
from src.ui.components.render_model_card import render_model_card

st.set_page_config(layout="wide", page_title="Financial AI Lab")

# --- STEP 1: INITIALIZATION ---
if 'valid_models' not in st.session_state: st.session_state['valid_models'] = []
if 'selected_models' not in st.session_state: st.session_state['selected_models'] = []

# --- STEP 2: THE SELECTORS ---
st.title("🔬 Model Evaluation")

tickers, intervals, target_map = get_filter_options()

c1, c2, c3 = st.columns(3)
ticker = c1.selectbox("⚙️ Ticker", tickers)
interval = c2.selectbox("🛠️ Interval", intervals)

# target_map keys are the "Pretty Strings" of the whole target list
selected_target_str = c3.selectbox("🎯 Target Configuration", options=list(target_map.keys()))
selected_target_list = target_map[selected_target_str] # This is our List[TargetConfig]

eval_period = st.date_input("📅 Evaluation Period", [])

# --- STEP 3: FILTERING ---
if st.button("🔍 Find Compatible Models"):
    if len(eval_period) == 2:
        # We pass the WHOLE LIST to filter models
        st.session_state['valid_models'] = get_valid_models(
            ticker, interval, 
            eval_period[0].strftime('%Y-%m-%d'), 
            eval_period[1].strftime('%Y-%m-%d'), 
            selected_target_list
        )
        st.rerun()

# --- STEP 4: SELECTION & CARDS ---
valid_names = st.session_state['valid_models']
selected_set = st.session_state['selected_models']

if selected_set:
    st.subheader("🧠 Selected Models")
    for model in selected_set:
        render_model_card(model, mode="evaluate", is_selected=True)

# (Render your model cards here using st.session_state['valid_models'])
# (User adds names to st.session_state['selected_models'])

# --- STEP 5: EVALUATION ---
if st.button("🚀 Run Evaluation", type="primary"):
    config = EvaluateConfig(
        models=st.session_state['selected_models'],
        ticker=ticker,
        interval=interval,
        start_date=eval_period[0].strftime('%Y-%m-%d'),
        end_date=eval_period[1].strftime('%Y-%m-%d'),
        targets=selected_target_list # Pass the list here
    )
    st.session_state['eval_results'] = evaluate_models(config)

if valid_names:
    st.divider()
    st.subheader("🧠 Available Models")
    available = [m for m in valid_names if m not in selected_set]
    if not available and not selected_set:
        st.info("🔄 No models match the filters.")
    for m_name in available:
        render_model_card(m_name, mode="evaluate", is_selected=False)

# --- STEP 6: VISUALIZATION ---
if 'eval_results' in st.session_state:
    results = st.session_state['eval_results']
    
    st.divider()
    st.header("📊 Comparative Metrics")

    overall_metrics = {}
    for m_name, data in results.items():
        overall_metrics[m_name] = data['metrics']['overall']

    metrics_df = pd.DataFrame(overall_metrics)
    st.subheader("📊 Global Performance Summary")
    st.dataframe(metrics_df, use_container_width=True)

    first_model = list(results.keys())[0]
    target_cols = results[first_model]['y_true'].columns

    for col in target_cols:
        st.markdown(f"---")
        st.subheader(f"🎯 Target Feature: **{col}**")

        col_metrics = {}
        for m_name in results:
            if 'breakdown' in results[m_name]['metrics'] and col in results[m_name]['metrics']['breakdown']:
                col_metrics[m_name] = results[m_name]['metrics']['breakdown'][col]
                
        if col_metrics:
            st.markdown("**📊 Feature Specific Errors:**")
            st.dataframe(pd.DataFrame(col_metrics), use_container_width=True)

        plot_df = pd.DataFrame(index=results[first_model]['y_true'].index)
        plot_df['Actual'] = results[first_model]['y_true'][col]

        for m_name in results:
            plot_df[m_name] = results[m_name]['y_pred'][col]

        plot_df = plot_df.dropna()
        
        st.markdown("**📊 Visualizing Alignment:**")
        st.line_chart(plot_df)