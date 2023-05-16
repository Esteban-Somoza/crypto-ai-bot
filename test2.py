import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the data from the CSV file
data = pd.read_csv('1m.csv')

# Extract the 'y' column as a list of values
y_values = data['y'].tolist()

# Generate x values (assuming they are sequential integers)
x_values = list(range(len(y_values)))

# Fit the data to a polynomial curve
degree = 15  # Adjust the degree of the polynomial as needed
coefficients = np.polyfit(x_values, y_values, degree)
curve = np.polyval(coefficients, x_values)

# Plotting the data as a line graph
plt.plot(x_values, y_values, color='blue', linewidth=1)

# Plotting the polynomial curve
plt.plot(x_values, curve, color='orange', linewidth=2)

# Determine the overall trend based on the curve
trend = []
for i in range(1, len(y_values)):
    if curve[i] > curve[i-1]:
        trend.append('Uptrend')
    elif curve[i] < curve[i-1]:
        trend.append('Downtrend')
    else:
        trend.append('Neutral')

# Find the maximum and minimum points of the curve
max_points = []
min_points = []
for i in range(1, len(y_values) - 1):
    if trend[i-1] == 'Downtrend' and trend[i] == 'Uptrend':
        max_points.append((x_values[i], y_values[i]))
    elif trend[i-1] == 'Uptrend' and trend[i] == 'Downtrend':
        min_points.append((x_values[i], y_values[i]))

# Coloring the regions based on the trend
current_trend = trend[0]
start_index = 0

for i in range(1, len(trend)):
    if trend[i] != current_trend:
        end_index = i - 1

        if current_trend == 'Uptrend':
            plt.fill_between(x_values[start_index:end_index+1], y_values[start_index:end_index+1],
                             color='green', alpha=0.3)
        elif current_trend == 'Downtrend':
            plt.fill_between(x_values[start_index:end_index+1], y_values[start_index:end_index+1],
                             color='red', alpha=0.3)

        current_trend = trend[i]
        start_index = i

# Fill the last region if it's not neutral
if current_trend == 'Uptrend':
    plt.fill_between(x_values[start_index:], y_values[start_index:],
                     color='green', alpha=0.3)
elif current_trend == 'Downtrend':
    plt.fill_between(x_values[start_index:], y_values[start_index:],
                     color='red', alpha=0.3)

# Plotting the maximum points of the curve
max_x_values, max_y_values = zip(*max_points)
plt.scatter(max_x_values, max_y_values, color='red', marker='o')

# Plotting the minimum points of the curve
min_x_values, min_y_values = zip(*min_points)
plt.scatter(min_x_values, min_y_values, color='green', marker='o')

# Customizing the plot
plt.title('Trend Analysis')
plt.xlabel('Data Point')
plt.ylabel('Y Values')
plt.ylim(27000, max(y_values))  # Set the y-axis limits
plt.grid(True)

# Display the plot
plt.show()