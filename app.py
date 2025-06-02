# src/app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import logging
import datetime
import yfinance as yf
from scipy.optimize import minimize
import time # Import time if needed for any delays (though not used here)

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR): # Smart pathing for local dev vs potential deployment
    MODEL_DIR = os.path.join("..", "models")
    if not os.path.exists(MODEL_DIR): # If../models also doesn't exist, default to current dir 'models'
        MODEL_DIR = "models"

MODEL_FILE = os.path.join(MODEL_DIR, "optimized_portfolio.joblib")
DEFAULT_BETA = 0.95
RISK_FREE_RATE = 0.0
DEFAULT_OPTIMIZATION_YEARS = 5 # Define years for custom optimization data fetch

# --- Helper Functions (Existing & Modified) ---

@st.cache_data
def load_optimization_results(filepath):
    """Loads pre-calculated optimization results from a joblib file."""
    if os.path.exists(filepath):
        try:
            results = joblib.load(filepath)
            logging.info(f"Results loaded successfully from {filepath}")
            if isinstance(results, dict) and "weights" in results and "performance" in results:
                # Ensure CVaR from loaded file is treated as a positive value representing potential loss
                cvar_key_to_check = None
                if isinstance(results.get('performance'), dict):
                    for k in results['performance'].keys():
                        if "Conditional Value at Risk (CVaR" in k:
                            cvar_key_to_check = k
                            break
                    if cvar_key_to_check and isinstance(results['performance'][cvar_key_to_check], (int, float)):
                        results['performance'][cvar_key_to_check] = abs(results['performance'][cvar_key_to_check])
                return results
            else:
                logging.error("Loaded file does not contain expected 'weights' and 'performance' keys.")
                st.error("Loaded results file has an unexpected format.")
                return None
        except Exception as e:
            logging.error(f"Error loading results file '{filepath}': {e}", exc_info=True)
            st.error(f"Error loading results file: {e}")
            return None
    else:
        logging.error(f"Results file not found at {filepath}.")
        st.error(f"Results file not found at `{filepath}`. Please run the training/optimization script first.")
        return None

def create_sunburst_chart(weights_dict):
    """Creates a sunburst chart for portfolio allocation."""
    if not weights_dict or not isinstance(weights_dict, dict):
        logging.warning("Invalid or empty weights data provided for sunburst chart.")
        return go.Figure().update_layout(title_text="No valid weights data available for chart")

    if not any(abs(weight) > 1e-9 for weight in weights_dict.values()):
        logging.warning("All weights are zero or extremely close to zero and will not be displayed.")
        return go.Figure().update_layout(title_text="All weights are zero or negligible")

    labels = list(weights_dict.keys())
    values = [abs(w) for w in weights_dict.values()]
    parents = ["Portfolio"] * len(labels)

    labels.insert(0, "Portfolio")
    current_sum_of_child_values = sum(values) 
    values.insert(0, current_sum_of_child_values)
    parents.insert(0, "")

    try:
        fig = px.sunburst(
            names=labels,
            parents=parents,
            values=values,
            title="Optimized Portfolio Allocation",
            branchvalues="total"
        )
        fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
        fig.update_traces(textinfo='label+percent parent')
        return fig
    except Exception as e:
        logging.error(f"Error creating sunburst chart: {e}", exc_info=True)
        st.error(f"Error creating sunburst chart: {e}")
        return go.Figure().update_layout(title_text="Error generating allocation chart")

def display_performance_metrics(performance_dict, beta_val_for_label=DEFAULT_BETA):
    """Displays portfolio performance metrics using st.metric."""
    if not performance_dict or not isinstance(performance_dict, dict):
        logging.warning("Invalid or empty performance data provided.")
        st.warning("No valid performance data available.")
        return

    st.subheader("Portfolio Performance Metrics")
    num_metrics = len(performance_dict)
    num_cols = min(num_metrics, 4)
    cols = st.columns(num_cols)
    metric_items = list(performance_dict.items())

    for i, (key, value) in enumerate(metric_items):
        with cols[i % num_cols]:
            formatted_value = value
            label_key = key
            help_text = None
            if isinstance(value, (int, float, np.number)):
                if "return" in key.lower():
                    formatted_value = f"{value:.2%}"
                elif "Conditional Value at Risk (CVaR" in key:
                    formatted_value = f"{abs(value):.2%}"
                    label_key = f"CVaR (Worst {int(beta_val_for_label*100)}% Losses)"
                    help_text = f"Average loss in the worst {100*(1-beta_val_for_label):.0f}% of scenarios, based on historical data."
                elif "sharpe" in key.lower():
                    formatted_value = f"{value:.2f}"
                elif "beta" in key.lower() and key != "CVaR Beta":
                    formatted_value = f"{value:.2f}"
                elif key == "CVaR Beta":
                    formatted_value = f"{value:.2f}"
                    label_key = "Optimization Beta for CVaR"
                elif "volatility" in key.lower():
                    formatted_value = f"{value:.3f}"
                else:
                    formatted_value = f"{value:.3f}"
            else:
                formatted_value = str(value)
            st.metric(label=label_key, value=formatted_value, help=help_text)

def display_metric_gauges(performance_dict, beta_val_for_gauge=DEFAULT_BETA):
    """Displays gauges for Expected Return and CVaR."""
    st.subheader("Key Metric Gauges")
    expected_return_key = "Expected Annual Return"
    cvar_key_pattern = "Conditional Value at Risk (CVaR"
    actual_cvar_key = None

    if isinstance(performance_dict, dict):
        for k in performance_dict.keys():
            if cvar_key_pattern in k:
                actual_cvar_key = k
                break

    if expected_return_key in performance_dict and actual_cvar_key and \
       isinstance(performance_dict.get(expected_return_key), (int, float, np.number)) and \
       isinstance(performance_dict.get(actual_cvar_key), (int, float, np.number)) and \
       not pd.isna(performance_dict[expected_return_key]) and \
       not pd.isna(performance_dict[actual_cvar_key]):

        col1, col2 = st.columns(2)
        return_val = performance_dict[expected_return_key]
        cvar_val = performance_dict[actual_cvar_key]

        with col1:
            return_gauge_max = max(0.50, abs(return_val) * 1.2)
            return_gauge_max = min(return_gauge_max, 2.0)
            fig_return = go.Figure(go.Indicator(
                mode="number+gauge", value=return_val * 100,
                number={'valueformat': ".2f", 'suffix': "%"},
                domain={'x': [0, 1], 'y': [0, 1]},
                title={"text": "Expected Return<br><sub>Annualized</sub>", "align": "center"},
                gauge={'axis': {'range': [None, return_gauge_max * 100]}, 'bar': {'color': "green"},
                       'bgcolor': "rgba(255, 255, 255, 0.7)", 'borderwidth': 1, 'bordercolor': "gray",
                       'steps': [{'range': [0, return_gauge_max * 0.4 * 100], 'color': 'rgba(211, 211, 211, 0.7)'},
                                 {'range': [return_gauge_max * 0.4 * 100, return_gauge_max * 0.8 * 100], 'color': 'rgba(160, 160, 160, 0.7)'}]}))
            fig_return.update_layout(height=250, margin=dict(t=50, b=40, l=30, r=30), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_return, use_container_width=True)

        with col2:
            abs_cvar_val = abs(cvar_val)
            cvar_gauge_max = max(0.50, abs_cvar_val * 1.2)
            cvar_gauge_max = min(cvar_gauge_max, 2.0)
            fig_cvar = go.Figure(go.Indicator(
                mode="number+gauge", value=abs_cvar_val * 100,
                number={'valueformat': ".2f", 'suffix': "%"},
                domain={'x': [0, 1], 'y': [0, 1]},
                title={"text": f"CVaR ({int(beta_val_for_gauge*100)}% Confidence)<br><sub>Worst {100*(1-beta_val_for_gauge):.0f}% Potential Loss</sub>", "align": "center"},
                gauge={'axis': {'range': [None, cvar_gauge_max * 100]}, 'bar': {'color': "red"},
                       'bgcolor': "rgba(255, 255, 255, 0.7)", 'borderwidth': 1, 'bordercolor': "gray",
                       'steps': [{'range': [0, cvar_gauge_max * 0.4 * 100], 'color': 'rgba(255, 192, 203, 0.7)'},
                                 {'range': [cvar_gauge_max * 0.4 * 100, cvar_gauge_max * 0.8 * 100], 'color': 'rgba(240, 128, 128, 0.7)'}]}))
            fig_cvar.update_layout(height=250, margin=dict(t=50, b=40, l=30, r=30), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_cvar, use_container_width=True)
    else:
        st.caption("Gauge charts cannot be displayed as key metrics (Return/CVaR) are missing or invalid.")

@st.cache_data
def fetch_stock_data_app(tickers_list_str, years=DEFAULT_OPTIMIZATION_YEARS):
    if not tickers_list_str:
        return None, "Ticker list is empty."
    tickers = [ticker.strip().upper() for ticker in tickers_list_str.split(',') if ticker.strip()]
    if not tickers:
        return None, "No valid tickers provided after parsing."

    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=years*365)
    status_messages = [f"Fetching data for: {', '.join(tickers)} from {start_date} to {end_date}"]

    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False, group_by='ticker', timeout=30)
        if data.empty:
            return None, "No data fetched. Please check tickers and date range."

        price_data_frames = []
        valid_tickers_downloaded = list(data.columns.get_level_values(0).unique())
        for ticker_symbol in tickers:
            if ticker_symbol not in valid_tickers_downloaded:
                status_messages.append(f"Warning: Data for requested {ticker_symbol} not found in download.")
                continue
            try:
                ticker_specific_data = data[ticker_symbol]
                price_series = None
                if 'Adj Close' in ticker_specific_data.columns and not ticker_specific_data['Adj Close'].isnull().all():
                    price_series = ticker_specific_data['Adj Close'].rename(ticker_symbol)
                elif 'Close' in ticker_specific_data.columns and not ticker_specific_data['Close'].isnull().all():
                    price_series = ticker_specific_data['Close'].rename(ticker_symbol)
                    status_messages.append(f"Info: Using 'Close' for {ticker_symbol}.")
                else:
                    status_messages.append(f"Warning: No valid price data ('Adj Close' or 'Close') for {ticker_symbol}.")
                    continue
                if price_series.count() < 20:
                    status_messages.append(f"Warning: Ticker {ticker_symbol} has too few valid data points ({price_series.count()}). Excluding.")
                    continue
                price_data_frames.append(price_series)
            except Exception as e_ticker_proc:
                status_messages.append(f"Warning: Error processing {ticker_symbol}: {str(e_ticker_proc)[:50]}")
                continue
        
        if not price_data_frames:
            return None, "No valid price series extracted. " + " ".join(status_messages)
        
        adj_close_combined = pd.concat(price_data_frames, axis=1)
        adj_close_combined.dropna(axis=1, how='all', inplace=True)
        final_price_data = adj_close_combined.dropna(axis=0, how='any')

        if final_price_data.empty or len(final_price_data) < 20:
            return None, "Price data is insufficient after cleaning. " + " ".join(status_messages)
        
        daily_returns = final_price_data.pct_change().dropna()
        if daily_returns.empty:
            return None, "Could not calculate returns. " + " ".join(status_messages)
        
        status_messages.append(f"Successfully processed data for: {list(daily_returns.columns)}")
        return daily_returns, " ".join(status_messages)
    except Exception as e:
        logging.error(f"Error in fetch_stock_data_app: {e}", exc_info=True)
        return None, f"Error fetching stock data: {e}. " + " ".join(status_messages)

def cvar_objective_function_app(x, returns_matrix, beta, num_assets, num_observations):
    weights = x[:num_assets]
    alpha_var = x[num_assets]
    u_slacks = x[num_assets+1:]
    cvar = alpha_var + (1.0 / (num_observations * (1.0 - beta))) * np.sum(u_slacks)
    return cvar

@st.cache_data
def optimize_portfolio_cvar_app(_daily_returns_df, beta_opt=DEFAULT_BETA):
    if _daily_returns_df is None or _daily_returns_df.empty:
        return None, None, "Input daily returns data is empty."

    returns_matrix = _daily_returns_df.values
    num_assets = _daily_returns_df.shape[1]
    num_observations = len(returns_matrix)

    initial_weights = np.ones(num_assets) / num_assets
    initial_portfolio_returns = np.dot(returns_matrix, initial_weights)
    initial_alpha_guess = np.percentile(-initial_portfolio_returns, beta_opt * 100)
    if initial_alpha_guess <= 0: initial_alpha_guess = 0.001
    initial_slacks = np.maximum(0, -initial_portfolio_returns - initial_alpha_guess)
    x0 = np.concatenate([initial_weights, [initial_alpha_guess], initial_slacks])
    bounds = [(0, 1)] * num_assets + [(1e-6, None)] + [(0, None)] * num_observations
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x[:num_assets]) - 1.0}]
    for t in range(num_observations):
        constraints.append({'type': 'ineq', 'fun': lambda x, t_idx=t: x[num_assets + 1 + t_idx] + np.dot(returns_matrix[t_idx, :], x[:num_assets]) + x[num_assets]})

    try:
        result = minimize(cvar_objective_function_app, x0, args=(returns_matrix, beta_opt, num_assets, num_observations),
                          method='SLSQP', bounds=bounds, constraints=constraints, options={'disp': False, 'maxiter': 1000, 'ftol': 1e-9})
        if result.success:
            optimized_weights_array = result.x[:num_assets]
            optimized_weights_array = np.maximum(0, optimized_weights_array)
            if np.sum(optimized_weights_array) > 1e-9:
                optimized_weights_array /= np.sum(optimized_weights_array)
            else:
                optimized_weights_array = np.zeros_like(optimized_weights_array)
            optimized_weights_dict = dict(zip(_daily_returns_df.columns, optimized_weights_array))

            optimized_portfolio_daily_returns = np.dot(returns_matrix, optimized_weights_array)
            expected_annual_return = np.mean(optimized_portfolio_daily_returns) * 252
            daily_cvar = result.fun
            annual_cvar = daily_cvar * np.sqrt(252)
            portfolio_std_dev_annual = np.std(optimized_portfolio_daily_returns) * np.sqrt(252)
            sharpe_ratio = (expected_annual_return - RISK_FREE_RATE) / portfolio_std_dev_annual if portfolio_std_dev_annual > 1e-9 else 0.0
            
            performance_metrics = {
                'Expected Annual Return': expected_annual_return,
                f'Conditional Value at Risk (CVaR, beta={beta_opt:.2f})': annual_cvar,
                'Annual Volatility': portfolio_std_dev_annual,
                'Sharpe Ratio': sharpe_ratio,
                'CVaR Beta': beta_opt}
            return optimized_weights_dict, performance_metrics, "Optimization successful."
        else:
            logging.error(f"Optimization failed: {result.message}")
            return None, None, f"Optimization failed: {result.message}"
    except Exception as e:
        logging.error(f"Optimization error: {e}", exc_info=True)
        return None, None, f"An error occurred during optimization: {e}"

def display_risk_return_profile(daily_returns_df):
    if daily_returns_df is None or daily_returns_df.empty or not isinstance(daily_returns_df, pd.DataFrame):
        st.caption("Risk-return profile cannot be displayed as daily returns data is unavailable or invalid.")
        return
    st.subheader("Individual Asset Risk-Return Profile")
    try:
        annual_returns = daily_returns_df.mean() * 252
        annual_volatility = daily_returns_df.std() * np.sqrt(252)
        if not annual_returns.index.equals(annual_volatility.index):
            common_index = annual_returns.index.intersection(annual_volatility.index)
            annual_returns = annual_returns[common_index]
            annual_volatility = annual_volatility[common_index]
            if common_index.empty:
                st.caption("Could not align returns and volatility data for plotting.")
                return
        plot_df = pd.DataFrame({'Ticker': annual_returns.index, 'Annualized Return': annual_returns.values, 'Annualized Volatility': annual_volatility.values})
        plot_df['Sharpe Ratio'] = np.where(plot_df['Annualized Volatility'].abs() > 1e-9, (plot_df['Annualized Return'] - RISK_FREE_RATE) / plot_df['Annualized Volatility'], 0.0)
        plot_df['Sharpe Ratio'] = plot_df['Sharpe Ratio'].fillna(0)
        
        fig = px.scatter(plot_df, x='Annualized Volatility', y='Annualized Return', text='Ticker',
                         labels={'Annualized Volatility': 'Annualized Volatility (Risk)', 'Annualized Return': 'Annualized Return'},
                         color='Sharpe Ratio', color_continuous_scale='RdYlGn', hover_name='Ticker', custom_data=['Sharpe Ratio'])
        fig.update_traces(textposition='top center', marker=dict(size=10),
                          hovertemplate="<b>%{hovertext}</b><br><br>Volatility: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe Ratio: %{customdata[0]:.2f}<extra></extra>")
        fig.update_layout(coloraxis_colorbar=dict(title="Sharpe Ratio"), height=500, xaxis_title="Annualized Volatility (Risk)",
                          yaxis_title="Annualized Return", xaxis_tickformat=".0%", yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error generating risk-return profile: {e}")
        logging.error(f"Error in display_risk_return_profile: {e}", exc_info=True)

# --- Streamlit App Layout ---
st.set_page_config(page_title="Portfolio Optimization Dashboard", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
st.title("📈 Portfolio Optimization Dashboard")

if 'view_selection' not in st.session_state:
    st.session_state.view_selection = 'View Pre-calculated Portfolio'
if 'custom_tickers_input' not in st.session_state:
    st.session_state.custom_tickers_input = ", ".join(['PIDILITIND.NS', 'DMART.NS', 'TITAN.NS', 'ABB.NS', 'CUMMINSIND.NS'])
if 'custom_optimization_results' not in st.session_state:
    st.session_state.custom_optimization_results = None
if 'run_custom_optimization' not in st.session_state:
    st.session_state.run_custom_optimization = False
if 'last_custom_run_tickers' not in st.session_state:
    st.session_state.last_custom_run_tickers = ""
if 'custom_daily_returns_df' not in st.session_state:
    st.session_state.custom_daily_returns_df = None

st.sidebar.header("Dashboard Controls")
def view_change_callback():
    st.session_state.run_custom_optimization = False
view_choice = st.sidebar.radio("Select Analysis Type:", ('View Pre-calculated Portfolio', 'Optimize Custom Portfolio'), key='view_selection', on_change=view_change_callback)
st.sidebar.markdown("---")
st.sidebar.caption("""
    **Note:** Portfolio optimization can be computationally intensive.
    Custom optimization uses historical data and may take a moment.
    Pre-calculated results are loaded for speed.
""")

if st.session_state.view_selection == 'View Pre-calculated Portfolio':
    st.header("Pre-calculated Optimized Portfolio")
    st.write(f"Displays the results of a portfolio optimized to minimize Conditional Value at Risk (CVaR), loaded from `{MODEL_FILE}`.")
    st.markdown("---")
    results_data = load_optimization_results(MODEL_FILE)
    if results_data:
        st.sidebar.subheader("Pre-calculated Portfolio Info")
        st.sidebar.success("Results loaded successfully.")
        update_time = results_data.get('last_updated', 'N/A')
        if isinstance(update_time, (datetime.datetime, pd.Timestamp)):
            st.sidebar.write(f"**Last Updated:** {pd.to_datetime(update_time).strftime('%Y-%m-%d %H:%M:%S %Z')}")
        else:
            st.sidebar.write(f"**Last Updated:** {update_time}")
        tickers_in_portfolio = results_data.get("tickers_used", list(results_data.get("weights", {}).keys()))
        st.sidebar.write(f"**Optimized Tickers ({len(tickers_in_portfolio)}):**")
        st.sidebar.caption(", ".join(tickers_in_portfolio) if tickers_in_portfolio else "None")
        perf_data = results_data.get("performance", {})
        beta_for_precalc_display = perf_data.get('CVaR Beta', DEFAULT_BETA)
        st.sidebar.write(f"**Optimization Metric:** Minimize CVaR")
        st.sidebar.caption(f"(Beta = {beta_for_precalc_display:.0%})")

        weights = results_data.get("weights")
        performance = results_data.get("performance")
        if weights is not None and performance is not None:
            display_performance_metrics(performance, beta_val_for_label=beta_for_precalc_display)
            display_metric_gauges(performance, beta_val_for_gauge=beta_for_precalc_display)
            st.markdown("---")
            st.subheader("Optimized Portfolio Allocation")
            if weights:
                sunburst_fig = create_sunburst_chart(weights)
                st.plotly_chart(sunburst_fig, use_container_width=True)
            else:
                st.info("Portfolio contains no assets (all weights are zero).")
            
            with st.expander("View Detailed Weights (Pre-calculated)"):
                if weights:
                    display_weights_detail = {ticker: f"{weight:.6f}" for ticker, weight in weights.items()}
                    if display_weights_detail:
                        weights_df = pd.DataFrame(list(display_weights_detail.items()), columns=['Ticker', 'Weight (Raw)'])
                        # CORRECTED LINE BELOW
                        weights_df['Weight %'] = weights_df['Weight (Raw)'].astype(float).map('{:.2%}'.format)
                        df_to_display = weights_df.sort_values(by='Weight (Raw)', key=pd.to_numeric, ascending=False)
                        st.dataframe(df_to_display[['Ticker', 'Weight %', 'Weight (Raw)']], use_container_width=True, hide_index=True)
                    else:
                        st.write("All calculated weights are effectively zero.")
                else:
                    st.write("No weights data available to display.")
        else:
            st.error("Loaded results file is missing valid 'weights' or 'performance' data.")

elif st.session_state.view_selection == 'Optimize Custom Portfolio':
    st.header("Optimize Your Custom Portfolio")
    st.write(f"Enter comma-separated stock tickers (e.g., from Yahoo Finance like AAPL, MSFT, GOOG) to optimize a portfolio aimed at minimizing Conditional Value at Risk (CVaR) with beta={DEFAULT_BETA:.0%}.")
    current_ticker_input = st.text_input("📌 Enter comma-separated tickers:", value=st.session_state.custom_tickers_input,
                                         placeholder="e.g., AAPL, GOOG, MSFT, RELIANCE.NS, TCS.NS", key="custom_tickers_widget").upper()
    if current_ticker_input != st.session_state.custom_tickers_input:
        st.session_state.custom_tickers_input = current_ticker_input
        st.session_state.custom_optimization_results = None
        st.session_state.custom_daily_returns_df = None
        st.session_state.run_custom_optimization = False

    invalid_chars_list = []
    cleaned_tickers = ""
    if st.session_state.custom_tickers_input:
        cleaned_tickers = "".join(c for c in st.session_state.custom_tickers_input if c.isalnum() or c in {',', '.', '-'})
        invalid_chars_list = [c for c in st.session_state.custom_tickers_input if c not in cleaned_tickers]
        if invalid_chars_list:
            st.warning(f"Invalid characters detected and will be ignored: `{' '.join(list(set(invalid_chars_list)))}`")

    if st.button("Run Custom Optimization", key="run_custom_opt_button"):
        final_tickers_to_run = ",".join([t for t in cleaned_tickers.split(',') if t.strip()])
        if not final_tickers_to_run:
            st.warning("Please enter some valid tickers to optimize.")
        else:
            st.session_state.run_custom_optimization = True
            st.session_state.custom_optimization_results = None
            st.session_state.custom_daily_returns_df = None
            st.session_state.last_custom_run_tickers = final_tickers_to_run
    
    if st.session_state.run_custom_optimization and st.session_state.last_custom_run_tickers == ",".join([t for t in cleaned_tickers.split(',') if t.strip()]):
        with st.spinner("Fetching data and optimizing portfolio... This may take a moment."):
            daily_returns, fetch_status_msg = fetch_stock_data_app(st.session_state.last_custom_run_tickers, years=DEFAULT_OPTIMIZATION_YEARS)
            st.info(fetch_status_msg)
            if daily_returns is not None and not daily_returns.empty:
                st.session_state.custom_daily_returns_df = daily_returns
                opt_weights, opt_performance, opt_status_msg = optimize_portfolio_cvar_app(daily_returns, beta_opt=DEFAULT_BETA)
                st.info(opt_status_msg)
                if opt_weights and opt_performance:
                    st.session_state.custom_optimization_results = {"weights": opt_weights, "performance": opt_performance, "tickers": list(daily_returns.columns)}
                    st.success("Custom portfolio optimized successfully!")
                else:
                    st.session_state.custom_optimization_results = None
            else:
                st.session_state.custom_optimization_results = None
        st.session_state.run_custom_optimization = False

    if st.session_state.custom_optimization_results:
        st.markdown("---")
        st.subheader("Custom Optimized Portfolio Results")
        custom_results = st.session_state.custom_optimization_results
        display_performance_metrics(custom_results["performance"], beta_val_for_label=DEFAULT_BETA)
        display_metric_gauges(custom_results["performance"], beta_val_for_gauge=DEFAULT_BETA)
        st.markdown("---")
        st.subheader("Custom Portfolio Allocation")
        if custom_results["weights"]:
            sunburst_fig_custom = create_sunburst_chart(custom_results["weights"])
            st.plotly_chart(sunburst_fig_custom, use_container_width=True)
        else:
            st.info("Custom portfolio optimization resulted in no asset allocation.")
        
        with st.expander("View Detailed Weights (Custom Portfolio)"):
            if custom_results["weights"]:
                display_weights_detail_custom = {ticker: f"{weight:.6f}" for ticker, weight in custom_results["weights"].items()}
                if display_weights_detail_custom:
                    weights_df_custom = pd.DataFrame(list(display_weights_detail_custom.items()), columns=['Ticker', 'Weight (Raw)'])
                    # CORRECTED LINE BELOW
                    weights_df_custom['Weight %'] = weights_df_custom['Weight (Raw)'].astype(float).map('{:.2%}'.format)
                    df_to_display_custom = weights_df_custom.sort_values(by='Weight (Raw)', key=pd.to_numeric, ascending=False)
                    st.dataframe(df_to_display_custom[['Ticker', 'Weight %', 'Weight (Raw)']], use_container_width=True, hide_index=True)
                else:
                    st.write("All calculated weights for the custom portfolio are effectively zero.")
            else:
                st.write("No weights data available for the custom portfolio.")
        
        if st.session_state.custom_daily_returns_df is not None:
            display_risk_return_profile(st.session_state.custom_daily_returns_df)