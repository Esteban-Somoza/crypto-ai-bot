import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

# Read the CSV file into a DataFrame
df = pd.read_csv('1m.csv')

# Extract the 'y' column from the DataFrame
y = df['y']

# Generate x values based on the number of rows in the DataFrame
x = np.arange(len(y))

# Reshape the x values to a 2D array
x = x.reshape(-1, 1)

# Train a neural network with a single hidden layer
model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000)

# Fit the model to the data
model.fit(x, y)

# Generate y values based on the learned curve
y_fit = model.predict(x)

# Plot the original data and the learned curve
plt.plot(x, y, label='Original Data')
plt.plot(x, y_fit, label='Learned Curve')
plt.xlabel('Index')
plt.ylabel('y')
plt.title('Original Data vs Learned Curve')
plt.legend()
plt.grid(True)
plt.show()