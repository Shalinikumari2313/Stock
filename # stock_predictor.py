# stock_predictor.py

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Ask user for stock symbol
ticker = input("Enter stock symbol (e.g., AAPL, TCS.NS): ").strip()

# 2. Download historical data
print(f"\n Downloading data for {ticker} ...")
data = yf.download(ticker, start='2018-01-01', end='2023-01-01')
# Check if data was fetched
if data.empty:
    print(" Failed to fetch data. Please check the stock symbol.")
    exit()
print(data.head())

# 3. Prepare the data
data = data[['Close']].dropna()
data['Days'] = range(len(data))
X = data[['Days']]
y = data['Close']

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Predict
y_pred = model.predict(X_test)

# 7. Evaluate
print("\n Model Performance:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# 8. Plot results
plt.figure(figsize=(12,6))
plt.scatter(X_test, y_test, color='blue', label='Actual Price')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Price')
plt.xlabel('Days since 2018')
plt.ylabel(f'{ticker} Stock Closing Price')
plt.title(f'{ticker} Stock Price Prediction using Linear Regression')
plt.legend()
plt.grid(True)
plt.show()
