import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv('1m.csv')

# Extract the 'y' column from the DataFrame
y = df['y']

# Generate x values based on the number of rows in the DataFrame
x = np.arange(len(y))

# Perform polynomial regression to fit a curve
degree = 20  # Adjust the degree of the polynomial
coeffs = np.polyfit(x, y, degree)
curve_fit = np.poly1d(coeffs)

# Generate y values based on the fitted curve
y_fit = curve_fit(x)

# Find the indices of local maximum and minimum points
max_indices = np.r_[True, y_fit[1:] > y_fit[:-1]] & np.r_[y_fit[:-1] > y_fit[1:], True]
min_indices = np.r_[True, y_fit[1:] < y_fit[:-1]] & np.r_[y_fit[:-1] < y_fit[1:], True]

# Add points at the maximum and minimum positions
x_max = x[max_indices]
y_max = y_fit[max_indices]
x_min = x[min_indices]
y_min = y_fit[min_indices]

# Plot the original data, the fitted curve, and the additional points
plt.plot(x, y, label='Original Data')
plt.plot(x, y_fit, label='Fitted Curve')
plt.plot(x_max, y_max, 'ro', label='Max Points')
plt.plot(x_min, y_min, 'go', label='Min Points')
plt.xlabel('Index')
plt.ylabel('y')
plt.title('Original Data vs Fitted Curve with Max/Min Points')
plt.legend()
plt.grid(True)
plt.show()

# Import the machine learning library
from sklearn.svm import SVC

# Create the machine learning model
model = SVC()

# Train the model on the dataset of curves
model.fit(x_max, y_max)

# Test the model on a dataset of curves
predictions = model.predict(x_min)
# Print the accuracy of the model
accuracy = np.mean(predictions == y_min)

print('Accuracy:', accuracy)

# Predict in a 10 number window if we are approaching a maximum or minimum
predictions_window = []
for i in range(len(y)):
    if i + 9 < len(y):
        predictions_window.append(model.predict(y[i:i+10]))

# Print the accuracy of the model in a 10 number window
accuracy_window = np.mean(predictions_window == y[9:])
y = y.values.reshape(-1, 1)

print('Accuracy in a 10 number window:', accuracy_window)

# Plot the accuracy chart
plt.plot(range(len(y)), accuracy_window, label='Predictions')
plt.plot(range(len(y)), y[9:], label='Actual')
plt.xlabel('Index')
plt.ylabel('y')
plt.title('Accuracy Chart')
plt.legend()
plt.grid(True)
plt.show()