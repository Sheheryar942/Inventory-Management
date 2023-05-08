# Import libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create random data for inventory management
n_samples = 1000
n_products = 5

# Generate random sales data
sales_data = np.random.randint(50, 100, size=(n_samples, n_products))

# Generate random inventory data
inventory_data = np.random.randint(10, 30, size=(n_samples, n_products))

# Generate random price data
price_data = np.random.uniform(1, 10, size=n_products)

# Generate random promotion data
promotion_data = np.random.randint(0, 2, size=(n_samples, n_products))

# Generate random demand data
demand_data = np.sum(sales_data, axis=1)

# Combine data into DataFrame
data = pd.DataFrame(sales_data, columns=[f'Product {i+1} Sales' for i in range(n_products)])
data = data.join(pd.DataFrame(inventory_data, columns=[f'Product {i+1} Inventory' for i in range(n_products)]))
data['Price'] = price_data
data = data.join(pd.DataFrame(promotion_data, columns=[f'Product {i+1} Promotion' for i in range(n_products)]))
data['Demand'] = demand_data

# Split data into training and testing sets
X = data.drop('Demand', axis=1)
y = data['Demand']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean squared error:', mse)
print('R^2 score:', r2)
