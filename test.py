# test.py
import yfinance as yf
import pandas as pd
import datetime
import numpy as np # Needed for isnull checks later potentially

# --- Configuration ---
# Tickers that caused the error in the previous run
TICKERS_TO_TEST = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'ICICIBANK.NS', 'BAJFINANCE.NS']
# Date range from the previous run
START_DATE = '2020-05-08'
END_DATE = '2025-05-08'

print(f"--- Starting test.py ---")
print(f"Tickers: {TICKERS_TO_TEST}")
print(f"Date Range: {START_DATE} to {END_DATE}")

daily_returns = pd.DataFrame() # Initialize empty DataFrame

# --- Fetch Data ---
print(f"\nAttempting to fetch data...")
try:
    data = yf.download(
        TICKERS_TO_TEST,
        start=START_DATE,
        end=END_DATE,
        progress=True, # Keep progress bar for testing
        group_by='ticker',
        timeout=30
        # proxy=None # Add proxy if needed
    )

    # --- !! INSPECTION CODE !! ---
    if data.empty:
        print("\nERROR: No data returned from yfinance. Download might have failed or returned empty.")
    else:
        print("\n--- Downloaded Data Columns ---")
        # Use print options to see the full column structure if it's wide
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
             print(data.columns)

        print("\n--- Downloaded Data Head (First 5 Rows) ---")
        # Use print options to see all columns if wide
        with pd.option_context('display.max_rows', 10, 'display.max_columns', None, 'display.width', 1000):
            print(data.head())

        print("\n--- Downloaded Data Tail (Last 5 Rows) ---")
        # Also check the end of the data
        with pd.option_context('display.max_rows', 10, 'display.max_columns', None, 'display.width', 1000):
            print(data.tail())

        print("\n--- Proceeding to process tickers... ---")
    # --- !! END OF INSPECTION CODE !! ---

    # --- Process Data (Similar to fetch_stock_data logic) ---
    if not data.empty:
        price_data_frames = []
        valid_tickers_downloaded = list(data.columns.get_level_values(0).unique())

        for ticker_symbol in TICKERS_TO_TEST:
            print(f"Processing: {ticker_symbol}...")
            if ticker_symbol not in valid_tickers_downloaded:
                 print(f"  -> Warning: Data for requested {ticker_symbol} not found in download result.")
                 continue
            try:
                # Access the DataFrame for the specific ticker from the MultiIndex columns
                ticker_specific_data = data[ticker_symbol]

                price_series = None
                # Check for 'Adj Close' first
                if 'Adj Close' in ticker_specific_data.columns and not ticker_specific_data['Adj Close'].isnull().all():
                    print(f"  -> Found 'Adj Close' for {ticker_symbol}")
                    price_series = ticker_specific_data['Adj Close'].rename(ticker_symbol)
                # Fallback to 'Close' if 'Adj Close' isn't valid
                elif 'Close' in ticker_specific_data.columns and not ticker_specific_data['Close'].isnull().all():
                     print(f"  -> Found 'Close' for {ticker_symbol} (using as fallback)")
                     price_series = ticker_specific_data['Close'].rename(ticker_symbol)
                else:
                    # If neither is valid, log a warning and skip
                    print(f"  -> Warning: No valid price data ('Adj Close' or 'Close') found for {ticker_symbol}.")
                    continue # Skip this ticker

                # Further validation (optional, but good practice)
                if price_series.count() < 20:
                    print(f"  -> Warning: Ticker {ticker_symbol} has too few valid data points ({price_series.count()}). Excluding.")
                    continue

                price_data_frames.append(price_series)
                print(f"  -> Successfully processed {ticker_symbol}")

            except KeyError as ke:
                # Catch the specific error if column access fails unexpectedly
                print(f"  -> ERROR: KeyError processing {ticker_symbol}: Column '{ke}' not found!")
                # This is where the original error likely happened
                continue # Skip this ticker
            except Exception as e_ticker:
                print(f"  -> Warning: Unexpected error processing data for ticker {ticker_symbol}: {e_ticker}")
                continue

        # --- Combine and Calculate Returns ---
        if price_data_frames:
            print("\nCombining price data...")
            adj_close_combined = pd.concat(price_data_frames, axis=1)
            # Drop rows with *any* NaN to ensure calculations work across all tickers
            final_price_data = adj_close_combined.dropna(axis=0, how='any')
            print(f"Shape after combining and dropping NaNs: {final_price_data.shape}")

            if final_price_data.empty or len(final_price_data) < 2: # Need at least 2 rows for pct_change
                 print("ERROR: Price data is insufficient after cleaning.")
            else:
                print("Calculating daily returns...")
                daily_returns = final_price_data.pct_change().dropna() # Drop first row of NaNs from pct_change

                if daily_returns.empty:
                    print("ERROR: Could not calculate returns (result is empty).")
                else:
                    print("\n--- Calculated Daily Returns Head ---")
                    print(daily_returns.head())
                    print(f"\nSuccessfully calculated daily returns for {list(daily_returns.columns)}")
        else:
            print("\nERROR: No valid price series could be extracted.")

except Exception as e:
    # Catch errors during the download itself or unexpected errors before processing loop
    print(f"\nERROR during yfinance download or initial processing: {e}")

# --- Final Status ---
if not daily_returns.empty:
    print("\n--- test.py finished: Daily returns generated successfully. ---")
else:
    print("\n--- test.py finished: Failed to generate daily returns. Check errors above. ---")