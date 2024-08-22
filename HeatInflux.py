import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, r2_score

# Load your data
df = pd.read_csv('heat_influx_noth_south.csv')

# Prepare the data
X = df['North'].values.reshape(-1, 1)
y = df['HeatFlux'].values

# Define the model
model = Sequential()
model.add(Dense(1, input_dim=1, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X, y, epochs=3000, verbose=0)

# Predict the values
y_pred = model.predict(X)

# Calculate MSE and R2 score
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f'MSE: {mse}')
print(f'R2 Score: {r2}')

# Extract weights
weights = model.layers[0].get_weights()
w = weights[0][0][0]
b = weights[1][0]

# Print the neuron function (linear equation)
print(f'The neuron function is: y = {w} * x + {b}')

# Plot the neuron function on data
plt.scatter(X, y, label='Data')
plt.plot(X, w * X + b, 'r', label='Neuron function')
plt.xlabel('North')
plt.ylabel('HeatFlux')
plt.legend()
plt.show()
