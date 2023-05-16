import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the price data from the CSV file
df = pd.read_csv('1m.csv')

# Extract the 'y' values as the target variable
y = df['y'].values

# Define the window size for input sequence
window_size = 20

# Create the input sequence and target values
X_train = y[:500].reshape(-1, 1)
y_train = y[500:520]
X_test = y[500:520].reshape(-1, 1)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Plot the training data, predictions, and actual 20 numbers
plt.plot(range(500), y[:500], label='Training Data')
plt.plot(range(500, 520), y_train, 'o', label='Actual Next 20 Numbers')
plt.plot(range(500, 520), y_pred, 'r--', label='Predictions')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Training Data vs Predictions vs Actual Next 20 Numbers')
plt.legend()
plt.show()