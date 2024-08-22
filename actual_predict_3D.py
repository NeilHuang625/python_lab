import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, r2_score

# Load your data
df = pd.read_csv('heat_influx_noth_south.csv')

# Prepare the data
X = df[['North', 'South']].values
y = df['HeatFlux'].values

print(X)
# Define the model
model = Sequential()
model.add(Dense(1, input_dim=2, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X, y, epochs=3000, verbose=1)

# Predict the values
y_pred = model.predict(X)

# Calculate MSE and R2 score
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f'MSE: {mse}')
print(f'R2 Score: {r2}')

# Extract weights
weights = model.layers[0].get_weights()
w1 = weights[0][0][0]
w2 = weights[0][1][0]
b = weights[1][0]

# Print the neuron function (linear equation)
print(f'The neuron function is: y = {w1} * x1 + {w2} * x2 + {b}')

# Create a grid for x, y values
y_g = np.linspace(X[:, 0].min(), X[:, 0].max(), num=100)
x_g = np.linspace(X[:, 1].min(), X[:, 1].max(), num=100)
x_grid, y_grid = np.meshgrid(x_g, y_g)
X_grid = np.column_stack((x_grid.ravel(), y_grid.ravel()))

# Predict the values for the grid
y_grid_pred = model.predict(X_grid)
y_grid_pred = y_grid_pred.reshape(x_grid.shape)

# Plot the data and the network function
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, color='b', label='Actual Data')
# ax.plot_surface(x_grid, y_grid, y_grid_pred, color='r',
#                 alpha=0.5, label='Neuron function')

# Plot the predicted data
ax.scatter(X[:, 0], X[:, 1], y_pred.flatten(),
           color='r', label='Predicted Data')

ax.set_xlabel('North')
ax.set_ylabel('South')
ax.set_zlabel('Heat Influx')
ax.legend(loc='upper right')

plt.show()
