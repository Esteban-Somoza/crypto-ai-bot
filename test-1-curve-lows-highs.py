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

# Create arrays to store positive and negative intervals
positive_intervals = []
negative_intervals = []

# Find positive and negative intervals
for i in range(len(max_indices) - 1):
    if max_indices[i]:
        start_index = x[i]
        end_index = x[i + 1]
        positive_intervals.append((start_index, end_index))
    elif min_indices[i]:
        start_index = x[i]
        end_index = x[i + 1]
        negative_intervals.append((start_index, end_index))

# Color the areas of positive and negative intervals
for start, end in positive_intervals:
    plt.fill_between(x[start:end], y_fit[start:end], color='green', alpha=0.3)

for start, end in negative_intervals:
    plt.fill_between(x[start:end], y_fit[start:end], color='red', alpha=0.3)

# Plot the original data, the fitted curve, and the additional points
plt.plot(x, y, label='Original Data')
plt.plot(x, y_fit, label='Fitted Curve')
plt.plot(x[max_indices], y_fit[max_indices], 'ro', label='Max Points')
plt.plot(x[min_indices], y_fit[min_indices], 'go', label='Min Points')

plt.xlabel('Index')
plt.ylabel('y')
plt.title('Original Data vs Fitted Curve with Max/Min Points and Trend Areas')
plt.legend()
plt.grid(True)
plt.show()