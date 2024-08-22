import numpy as np
import matplotlib.pyplot as plt

values = [(46, -225, 4.5), (110.5, -15, 5.0), (54.5, -212, 4.5), (119, -2, 5)]
labels = ['Boundary after update 0', 'Boundary after update 1',
          'Boundary after update 2', 'Boundary after update 3']
linestyles = ['-.', '--', '-', ':']
colors = ['blue', 'red', 'yellow', 'green']  # Changed the colors

for (w1, w2, a), label, linestyle, color in zip(values, labels, linestyles, colors):
    m = -w1/w2
    b = -a/w2
    x = np.linspace(60, 180, 400)
    y = m*x+b
    plt.plot(x, y, label=label, linestyle=linestyle, color=color)

# Plotting the data points
plt.scatter(112, 394, color='red', label='Canadian (0)')
plt.scatter(129, 420, color='blue', label='Alaskan (1)')

plt.xticks(np.arange(80, 180, 20))
plt.yticks(np.arange(0, 3000, 300))

plt.xlabel('x')
plt.ylabel('y')
plt.title('Perceptron Decision Boundary')
plt.legend()
plt.grid(True)
plt.show()
