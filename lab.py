import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
Y = np.array([0, 0.75, 1.5, 2.25, 3])

# Create the single neuron model using the Keras API
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(2,),
          activation='linear'))  # Change input_shape to 2
model.compile(optimizer='ADAM', metrics=['accuracy'], loss=['mse'])

# Train the perceptron using stochastic gradient descent
history = model.fit(X, Y, epochs=1000, batch_size=25, verbose=1)

# predict output for the set of input values
predicted_y = model.predict(X)
weights = model.layers[0].get_weights()
print(predicted_y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# plot actual output vs predicted output
ax.scatter(X[:, 0], X[:, 1], Y, color='r', label='Actual values')
ax.scatter(X[:, 0], X[:, 1], predicted_y, color='c', label='Predicted values')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.legend(loc='upper right')
plt.show()
