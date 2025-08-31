# Microsoft Stock Data Cleaning & Preparation
   
   This project loads, cleans, and engineers features from Microsoft stock price data using Python and pandas.
   
   ## Features
   
   - Loads raw CSV stock data from `dataset/Microsoft_stock_data.csv`
   - Cleans missing and invalid values
   - Removes duplicates
   - Converts columns to appropriate types
   - Engineers time-series and volatility features
   - Outputs a cleaned CSV file
   
   ## Requirements
   
   - Python 3.7+
   - pandas
   
   ## Usage
   
   1. Place `Microsoft_stock_data.csv` in the `dataset` directory (`dataset/Microsoft_stock_data.csv`).
   2. Run:
   
      ```bash
      python main.py