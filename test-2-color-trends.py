import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the CSV file
data = pd.read_csv('1m.csv')

# Extract the 'y' column as a list of values
y_values = data['y'].tolist()

# Generate x values (assuming they are sequential integers)
x_values = list(range(len(y_values)))

# Plotting the data as a line graph
plt.plot(x_values, y_values, color='blue', linewidth=1)

# Determine the trend for each data point
trend = []
for i in range(1, len(y_values)):
    if y_values[i] > y_values[i-1]:
        trend.append('Uptrend')
    elif y_values[i] < y_values[i-1]:
        trend.append('Downtrend')
    else:
        trend.append('Neutral')

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

# Customizing the plot
plt.title('Trend Analysis')
plt.xlabel('Data Point')
plt.ylabel('Y Values')
plt.ylim(27000, max(y_values))  # Set the y-axis limits
plt.grid(True)

# Display the plot
plt.show()