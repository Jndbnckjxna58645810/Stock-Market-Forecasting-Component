import streamlit as st
import pandas as pd

from src.utils.gui_utils import get_filter_options, get_target_string, get_valid_models, get_feature_table

from src.models.registry import evaluate_models

from src.config.model_metadata import ModelMetadata
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

        st.session_state['valid_models'] = get_valid_models(ticker, interval, start_str, end_str, target)

        st.rerun()

valid_names = st.session_state['valid_models']
selected_set = st.session_state['selected_model_names']

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
    st.dataframe(metrics_df, use_container_width=True)

    first_model = list(results.keys())[0]
    target_cols = results[first_model]['y_true'].columns

    for col in target_cols:
        st.subheader(f"Target: {col}")

        common_dates = set(results[first_model]['dates'])
        for m_name in results:
            common_dates = common_dates.intersection(set(results[m_name]['dates']))
        
        common_dates = sorted(list(common_dates))
        
        plot_df = pd.DataFrame(index=pd.to_datetime(common_dates))

        y_true_all = results[first_model]['y_true']
        plot_df['Actual'] = y_true_all.reindex(plot_df.index)[col]
        
        for m_name in results:
            y_pred_m = results[m_name]['y_pred']
            plot_df[m_name] = y_pred_m.reindex(plot_df.index)[col]
        
        st.line_chart(plot_df)

if valid_names:
    st.divider()
    st.subheader("Available Models")
    available = [m for m in valid_names if m not in selected_set]
    if not available and not selected_set:
        st.info("No models match the filters.")
    for m_name in available:
        render_model_card(m_name, is_selected=False)