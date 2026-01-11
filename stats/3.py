# 3. Program on Time series: GroupBy Mechanics to display in data vector, multivariate
# time series and forecasting formats


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
# -----------------------------------------------------------
# Create sample time-series data (NO CSV NEEDED)
# -----------------------------------------------------------
date_range = pd.date_range(start='2020-01-01', periods=60, freq='M')
data = pd.DataFrame({
 'Value': np.random.randint(100, 200, size=60)
}, index=date_range)
# -----------------------------------------------------------
# GroupBy Mechanics
# -----------------------------------------------------------
print("GroupBy Mechanics")
print("------------------")
grouped_data = data.groupby([data.index.year, data.index.month]).mean()
print("Grouped Data:")
print(grouped_data)
# -----------------------------------------------------------
# Vector Format (Resampling)
# -----------------------------------------------------------
print("\nVector Format")
print("--------------")
quarterly_data = data.resample('Q').mean()
print("Quarterly Data:")
print(quarterly_data)
# -----------------------------------------------------------
# Multivariate Time Series
# -----------------------------------------------------------
print("\nMultivariate Time Series")
print("------------------------")
multivariate_data = pd.DataFrame({
 'Var1': np.random.normal(0, 1, len(data)),
 'Var2': np.random.normal(0, 1, len(data)),
 'Var3': np.random.normal(0, 1, len(data))
}, index=data.index)
print("Multivariate Data:")
print(multivariate_data)
# -----------------------------------------------------------
# Forecasting
# -----------------------------------------------------------
print("\nForecasting")
print("------------")
# Decomposition (works only on single column)
decomposition = seasonal_decompose(data['Value'], model='additive', period=12)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
print("\nDecomposition:")
print("Trend:\n", trend)
print("\nSeasonal:\n", seasonal)
print("\nResidual:\n", residual)
# -----------------------------------------------------------
# ARIMA Model
# -----------------------------------------------------------
model = ARIMA(data['Value'], order=(1, 1, 1))
model_fit = model.fit()
print("\nARIMA Model Summary:")
print(model_fit.summary())
# Forecast 12 months ahead
forecast = model_fit.forecast(steps=12)
print("\nForecast Values:")
print(forecast)
# -----------------------------------------------------------
# Plotting
# -----------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(data, label='Original')
plt.plot(quarterly_data, label='Quarterly')
plt.plot(trend, label='Trend')
plt.plot(seasonal, label='Seasonal')
plt.plot(residual, label='Residual')
plt.plot(forecast, label='Forecast')
plt.legend()
plt.title("Time Series Analysis and Forecasting")
plt.show()