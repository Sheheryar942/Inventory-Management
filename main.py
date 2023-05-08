# Import libraries
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from scipy.optimize import minimize

# Load data
data = pd.read_csv('inventory_data.csv')

# Preprocess data
...

# Fit ARIMA model
model = ARIMA(data, order=(1,1,1))
model_fit = model.fit()

# Forecast future demand
forecast = model_fit.forecast(steps=30)

# Optimize inventory levels
def objective(x):
    return -np.sum(np.multiply(forecast, x))

constraints = [{'type': 'ineq', 'fun': lambda x: x[0] + x[1] - 50},
               {'type': 'ineq', 'fun': lambda x: x[2] - 20},
               {'type': 'ineq', 'fun': lambda x: x[3] - 30}]
x0 = np.array([10, 20, 10, 20])
result = minimize(objective, x0, method='SLSQP', constraints=constraints)

# Print optimal inventory levels
print('Optimal inventory levels:')
print(result.x)
