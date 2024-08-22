import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the data
data = pd.read_csv('Fish_data.csv')

# Extract the features and labels
# Extract the first two columns (freshwater and seawater scale diameters)
X = data.iloc[:, :2].values
y = data.iloc[:, 2].values   # Extract the third column (label)

# Initialize the weights and bias and learning rate
w1, w2 = 102, -28
b = 5.0
beta = 0.5

# Store the weights and bias for each iteration
weights = [(w1, w2, b)]

# Train the perceptron
for epoch in range(200):  # 2 epochs
    for i in range(len(X)):
        x1, x2 = X[i]
        # Calculate the predicted value
        z = w1 * x1 + w2 * x2 + b
        y_pred = 1 if z >= 0 else 0

        # Calculate the error
        error = y[i] - y_pred

        # Update the weights and bias
        w1 += beta * error * x1
        w2 += beta * error * x2
        b += beta * error

        weights.append((w1, w2, b))

# Extract the final weights and bias
w1, w2, b = weights[-1]

# Print the final decision boundary
print(
    f"The equation of the final decision boundary is: x2 = -({w1} * x1 + {b}) / {w2}")

# Plot the decision boundaries
x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 400)

plt.figure(figsize=(10, 8))

# Plot the final decision boundary
x2_boundary = -w1 / w2 * x1_range - b / w2
plt.plot(x1_range, x2_boundary, label='Final Boundary', color='green')

# Plot the data points
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue',
            label='Canadian (0)', marker='o', s=100)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red',
            label='Alaskan (1)', marker='x', s=100)

plt.xlabel('$x_1$ (Ring diameter in fresh water)')
plt.ylabel('$x_2$ (Ring diameter in salt water)')
plt.legend()
plt.title('Perceptron Classification Boundaries')
plt.grid(True)
plt.show()
