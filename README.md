# 💼 Financial Portfolio Management & Optimization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akshay-birla-03/financial_portfolio_manangement/blob/main/notebooks/Run_in_Colab.ipynb)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](#)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](#)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

An interactive portfolio analytics and **optimization** app: track a basket of assets,
compute risk/return metrics, run mean-variance optimization, and visualise allocation and
performance — served through a **Streamlit** dashboard, backed by pre-trained per-stock
LSTM + ARIMA models.

▶️ **Run it now, no setup:** click the **Open in Colab** badge — it clones, installs and
loads the optimised portfolio (and can launch the full dashboard).

## Features

- **Optimization**: mean-variance optimization via \`scipy.optimize\` (maximise Sharpe ratio)
- **Risk metrics**: expected return, volatility, Sharpe ratio, per-asset weights
- **Live prices** via \`yfinance\`; pre-trained models per stock in \`models/\`
- **Interactive UI**: Streamlit + Plotly charts for allocation and performance

## Quickstart (local)

\`\`\`bash
git clone https://github.com/akshay-birla-03/financial_portfolio_manangement.git
cd financial_portfolio_manangement
pip install -r requirements.txt
streamlit run app.py           # opens the dashboard at http://localhost:8501
\`\`\`

Retrain the per-stock models with \`python train_and_save_models.py\`.

## Project layout

\`\`\`
app.py                       # Streamlit dashboard (main entry)
train_and_save_models.py     # trains + saves per-stock LSTM/ARIMA models
test.py                      # quick checks
models/                      # trained artifacts (HDFC, INFY, RELIANCE, TATAELXSI) + optimized_portfolio.joblib
notebooks/Run_in_Colab.ipynb # one-click Colab runner
requirements.txt
\`\`\`

## Tech

Python · Streamlit · Plotly · SciPy · scikit-learn · TensorFlow/Keras · statsmodels · yfinance · pandas · NumPy

## Disclaimer

For research and educational use only — **not** financial advice.

---
Author: **Akshay Birla** · [GitHub](https://github.com/akshay-birla-03) · MIT License
