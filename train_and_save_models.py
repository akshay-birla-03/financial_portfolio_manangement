# train_and_save_models.py
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import joblib
import os
import logging
import datetime
import time # For sleep in retries

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default tickers for the pre-calculated portfolio
DEFAULT_TICKERS = [
    'RELIANCE.NS',
    'TCS.NS',
    'ICICIBANK.NS',
    'BAJFINANCE.NS',
]
# Note: Ensure these tickers are valid and available on Yahoo Finance
# You can add more tickers to this list as needed.

# Date configuration
YEARS_OF_DATA = 5 # More descriptive variable name
end_date_dt = pd.Timestamp.now()
start_date_dt = end_date_dt - pd.DateOffset(years=YEARS_OF_DATA)
START_DATE = start_date_dt.strftime('%Y-%m-%d') # Dynamic start date
END_DATE = end_date_dt.strftime('%Y-%m-%d') # Dynamic end date

DEFAULT_BETA_CVAR = 0.95
RISK_FREE_RATE = 0.0

# Output directory and file
MODEL_DIR = "models" # Assumes 'models' is a subdirectory relative to script location
MODEL_FILE = os.path.join(MODEL_DIR, "optimized_portfolio.joblib")

# Proxy configuration (set to your proxy if needed, otherwise None)
PROXY_SETTINGS = None
# Example: PROXY_SETTINGS = {"http": "http://user:pass@proxy_server:port", "https": "https://user:pass@proxy_server:port"}

# --- Helper Functions ---

def fetch_stock_data(tickers, start_date_str, end_date_str, retries=3, delay=5, proxy=None, timeout=30):
    """Fetches historical stock data using yfinance with retries and validation."""
    if not tickers:
        logging.error("No tickers provided for data fetching.")
        return None

    logging.info(f"Attempting to fetch data for tickers: {tickers} from {start_date_str} to {end_date_str}")

    for attempt in range(retries):
        try:
            data = yf.download(
                tickers,
                start=start_date_str,
                end=end_date_str,
                progress=False,
                group_by='ticker', # Crucial for consistent multi-ticker handling
                timeout=timeout,
                proxy=proxy
            )

            if data.empty:
                logging.warning(f"Attempt {attempt + 1}/{retries}: No data returned from yfinance for tickers: {tickers}.")
                if attempt < retries - 1:
                    logging.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logging.error("Failed to fetch data after multiple retries: Data empty.")
                    return None
                continue # Go to next retry attempt

            # --- Data Extraction and Validation ---
            price_data_frames = []
            valid_tickers_downloaded = list(data.columns.get_level_values(0).unique()) # Tickers actually returned

            for ticker_symbol in tickers: # Check against originally requested tickers
                if ticker_symbol not in valid_tickers_downloaded:
                    logging.warning(f"Data for requested ticker {ticker_symbol} was not found in the download result.")
                    continue

                try:
                    ticker_specific_data = data[ticker_symbol]
                    price_series = None
                    if 'Adj Close' in ticker_specific_data.columns and not ticker_specific_data['Adj Close'].isnull().all():
                        price_series = ticker_specific_data['Adj Close'].rename(ticker_symbol)
                    elif 'Close' in ticker_specific_data.columns and not ticker_specific_data['Close'].isnull().all():
                        price_series = ticker_specific_data['Close'].rename(ticker_symbol)
                        logging.info(f"Using 'Close' prices for {ticker_symbol} as 'Adj Close' not available/valid.")
                    else:
                        logging.warning(f"No valid 'Adj Close' or 'Close' price data found for {ticker_symbol}.")
                        continue

                    # Basic validation on the series itself
                    if price_series.count() < 20: # Need at least ~1 month of data points
                         logging.warning(f"Ticker {ticker_symbol} has too few valid data points ({price_series.count()}). Excluding.")
                         continue

                    price_data_frames.append(price_series)

                except Exception as e_ticker:
                    logging.warning(f"Could not process data for ticker {ticker_symbol}: {e_ticker}")
                    continue

            if not price_data_frames:
                logging.error("No valid price data could be extracted for any requested ticker.")
                if attempt < retries - 1: time.sleep(delay); continue
                else: return None

            # --- Combine and Final Clean ---
            adj_close_combined = pd.concat(price_data_frames, axis=1)

            # Drop columns that are *still* all NaN after concat (should be rare now)
            adj_close_combined.dropna(axis=1, how='all', inplace=True)

            # Drop rows where *any* remaining ticker has NaN (ensures consistent matrix for optimization)
            final_price_data = adj_close_combined.dropna(axis=0, how='any')

            if final_price_data.empty or len(final_price_data) < 20:
                logging.error(f"Price data is insufficient after cleaning (Rows: {len(final_price_data)}). Tickers available before dropna: {list(adj_close_combined.columns)}")
                if attempt < retries - 1: time.sleep(delay); continue
                else: return None

            # Calculate daily returns
            daily_returns = final_price_data.pct_change().dropna() # Drop first NaN row from pct_change

            if daily_returns.empty:
                logging.error("Could not calculate returns from cleaned price data (result is empty).")
                if attempt < retries - 1: time.sleep(delay); continue
                else: return None

            logging.info(f"Data fetched and processed successfully for: {list(daily_returns.columns)}")
            return daily_returns # Success! Exit retry loop

        except Exception as e:
            logging.error(f"Attempt {attempt + 1}/{retries}: Error during stock data fetching/processing: {e}", exc_info=False) # Set exc_info=True for full traceback
            if attempt < retries - 1:
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logging.error("Failed to fetch/process data after multiple retries due to errors.")
                return None
    # If loop finishes without success
    return None


def cvar_objective_function_train(x, returns_matrix, beta, num_assets, num_observations):
    """Objective function for CVaR minimization."""
    weights = x[:num_assets]
    alpha_var = x[num_assets] # This is Value-at-Risk (VaR)
    u_slacks = x[num_assets+1:] # Slacks for losses exceeding VaR

    portfolio_losses = -np.dot(returns_matrix, weights) # Calculate portfolio losses (negative returns)

    # CVaR = VaR + (1 / (T * (1 - beta))) * sum(u_t)
    # where u_t = max(0, L_t - VaR) and L_t is portfolio loss at time t
    cvar = alpha_var + (1.0 / (num_observations * (1.0 - beta))) * np.sum(u_slacks)
    return cvar


def optimize_portfolio_for_cvar(daily_returns_df, beta_cvar=DEFAULT_BETA_CVAR):
    """Optimizes portfolio weights to minimize CVaR using SciPy."""
    if daily_returns_df is None or daily_returns_df.empty:
        logging.error("Cannot optimize: Input daily returns data is empty.")
        return None, None

    returns_matrix = daily_returns_df.values
    num_assets = daily_returns_df.shape[1]
    num_observations = len(returns_matrix)
    logging.info(f"Optimizing portfolio for {num_assets} assets with {num_observations} observations, Beta={beta_cvar:.2f}")

    # --- Setup Optimization Problem ---
    # Initial guess
    initial_weights = np.ones(num_assets) / num_assets
    initial_portfolio_returns = np.dot(returns_matrix, initial_weights)
    # Initial VaR (alpha) guess: (1-beta)th percentile of portfolio *losses*
    initial_alpha_guess = np.percentile(-initial_portfolio_returns, beta_cvar * 100) # VaR is positive loss value
    # Ensure initial VaR guess is reasonable
    if initial_alpha_guess <= 0: initial_alpha_guess = 0.001
    # Initial slacks (u_t = max(0, Loss_t - VaR))
    initial_slacks = np.maximum(0, -initial_portfolio_returns - initial_alpha_guess)
    # Combine into initial guess vector x0 = [weights, alpha_VaR, slacks_u]
    x0 = np.concatenate([initial_weights, [initial_alpha_guess], initial_slacks])

    # Bounds: weights [0,1], VaR (alpha_var) >= small_positive, slacks >= 0
    bounds = [(0, 1)] * num_assets + [(1e-6, None)] + [(0, None)] * num_observations

    # Constraints
    constraints = []
    # 1. Sum of weights = 1
    constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x[:num_assets]) - 1.0})
    # 2. Slack constraints: u_t >= L_t - VaR  => u_t - L_t + VaR >= 0
    #    Substitute L_t = -portfolio_return = -dot(returns_row, weights)
    #    => u_t + dot(returns_row, weights) + VaR >= 0
    for t in range(num_observations):
        constraints.append({
            'type': 'ineq', # Constraint is >= 0
            'fun': lambda x, t_idx=t: x[num_assets + 1 + t_idx] + np.dot(returns_matrix[t_idx, :], x[:num_assets]) + x[num_assets]
        })

    # --- Run Optimization ---
    try:
        result = minimize(cvar_objective_function_train, x0,
                          args=(returns_matrix, beta_cvar, num_assets, num_observations),
                          method='SLSQP', bounds=bounds, constraints=constraints,
                          options={'disp': False, 'maxiter': 1000, 'ftol': 1e-9}) # Adjust options if needed

        if result.success:
            optimized_weights_array = result.x[:num_assets]
            # Post-process weights: ensure non-negativity and normalization due to potential solver precision issues
            optimized_weights_array = np.maximum(0, optimized_weights_array)
            optimized_weights_array /= np.sum(optimized_weights_array) # Ensure sum is exactly 1
            optimized_weights_dict = dict(zip(daily_returns_df.columns, optimized_weights_array))

            # --- Calculate Performance Metrics ---
            optimized_portfolio_daily_returns = np.dot(returns_matrix, optimized_weights_array)
            expected_annual_return = np.mean(optimized_portfolio_daily_returns) * 252

            # CVaR value is the minimized objective function result (average daily tail loss magnitude)
            daily_cvar_value = result.fun
            # Annualize CVaR: Multiply average daily shortfall by number of trading days
            # Note: Other methods exist (e.g., *sqrt(252)), but this gives an annualized total expected shortfall perspective.
            annual_cvar_value = daily_cvar_value * 252

            portfolio_std_dev_annual = np.std(optimized_portfolio_daily_returns) * np.sqrt(252)
            sharpe_ratio = (expected_annual_return - RISK_FREE_RATE) / portfolio_std_dev_annual if portfolio_std_dev_annual != 0 else 0

            performance_metrics = {
                'Expected Annual Return': expected_annual_return,
                 # Store CVaR as a positive value representing the magnitude of the expected tail loss
                f'Conditional Value at Risk (CVaR, beta={beta_cvar:.2f})': annual_cvar_value,
                'Annual Volatility': portfolio_std_dev_annual,
                'Sharpe Ratio': sharpe_ratio,
                'CVaR Beta': beta_cvar # Store the beta used for this optimization for reference
            }
            logging.info("Portfolio optimized successfully.")
            return optimized_weights_dict, performance_metrics
        else:
            logging.error(f"Optimization failed: {result.message}")
            return None, None
    except Exception as e:
        logging.error(f"An error occurred during optimization: {e}", exc_info=True)
        return None, None


# --- Main Execution ---
if __name__ == "__main__":
    start_time_script = time.time()
    logging.info(f"Starting portfolio optimization process at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Ensure the models directory exists
    if not os.path.exists(MODEL_DIR):
        try:
            os.makedirs(MODEL_DIR)
            logging.info(f"Created directory: {MODEL_DIR}")
        except OSError as e:
             logging.error(f"Failed to create directory {MODEL_DIR}: {e}")
             # Decide if you want to exit or try saving elsewhere
             exit() # Exit if we can't create the save directory


    # Step 1: Fetch Data
    daily_returns = fetch_stock_data(DEFAULT_TICKERS, START_DATE, END_DATE, proxy=PROXY_SETTINGS)

    if daily_returns is not None and not daily_returns.empty:
        # Step 2: Optimize Portfolio
        optimized_weights, performance_metrics = optimize_portfolio_for_cvar(daily_returns, beta_cvar=DEFAULT_BETA_CVAR)

        if optimized_weights is not None and performance_metrics is not None:
            # Step 3: Prepare data for saving
            results_to_save = {
                "weights": optimized_weights,
                "performance": performance_metrics,
                "tickers_used": list(daily_returns.columns), # Save the list of tickers actually used
                "last_updated": pd.Timestamp.now(tz='UTC') # Use timezone-aware timestamp
            }

            # Step 4: Save results
            try:
                joblib.dump(results_to_save, MODEL_FILE)
                logging.info(f"Optimization results saved to {MODEL_FILE}")
            except Exception as e:
                logging.error(f"Error saving optimization results to {MODEL_FILE}: {e}", exc_info=True)
        else:
            logging.error("Could not generate optimization results to save.")
    else:
        logging.error("Failed to fetch or process data for default tickers. Cannot proceed with optimization.")

    end_time_script = time.time()
    logging.info(f"Portfolio optimization process finished. Total time: {end_time_script - start_time_script:.2f} seconds.")