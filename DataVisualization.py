import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Load data
windsor_data_path = "HousePrices.csv"
windsor_data = pd.read_csv(windsor_data_path)

# Define features and target variable
windsor_features = ["lotsize", 'bedrooms', 'bathrooms', 'stories', 'garage']
X = windsor_data[windsor_features]
y = windsor_data.price

# Split into training and validation sets
trainX, valX, trainy, valy = train_test_split(X, y, random_state=1)

# Train Random Forest model
windsor_model_ranFor = RandomForestRegressor()
windsor_model_ranFor.fit(trainX, trainy)

# Predict with Random Forest model
prediction_ranFor = windsor_model_ranFor.predict(valX)
mae_ranFor = mean_absolute_error(valy, prediction_ranFor)

# Train Linear Regression model
windsor_model_mulReg = LinearRegression()
windsor_model_mulReg.fit(trainX, trainy)

# Predict with Linear Regression model
prediction_mulReg = windsor_model_mulReg.predict(valX)
mae_linReg = mean_absolute_error(valy, prediction_mulReg)

# Calculate errors
error_ranFor = valy - prediction_ranFor
error_linReg = valy - prediction_mulReg

# Plotting
plt.figure(figsize=(14, 6))

# Plot Random Forest errors
plt.subplot(1, 2, 1)
plt.scatter(valy, error_ranFor, color='blue', alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Random Forest Errors')
plt.xlabel('Actual Prices')
plt.ylabel('Prediction Error')

# Plot Linear Regression errors
plt.subplot(1, 2, 2)
plt.scatter(valy, error_linReg, color='green', alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Linear Regression Errors')
plt.xlabel('Actual Prices')
plt.ylabel('Prediction Error')

plt.tight_layout()
plt.show()

# Print MAEs
print('MAE Random Forest:', mae_ranFor)
print('MAE Linear Regression:', mae_linReg)

# Bar chart of MAEs
models = ['Random Forest', 'Linear Regression']
maes = [mae_ranFor, mae_linReg]

plt.figure(figsize=(8, 6))
plt.bar(models, maes, color=['blue', 'green'])
plt.title('MAE Comparison')
plt.ylabel('Mean Absolute Error')
plt.show()
