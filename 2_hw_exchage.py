import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 这里设函数为y=3x+2
x_data = np.array([1.0, 2.0, 3.0])
y_data = np.array([5.0, 8.0, 11.0])

def forward(x, w, b):
    return x * w + b

def loss(x, y, w, b):
    y_pred = forward(x, w, b)
    return (y_pred - y) ** 2

W = np.arange(0.0, 4.1, 0.1)
B = np.arange(0.0, 4.1, 0.1)
w, b = np.meshgrid(W, B)

loss_values = np.zeros_like(w)

for i in range(len(W)):
    for j in range(len(B)):
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            loss_val = loss(x_val, y_val, W[i], B[j])
            l_sum += loss_val
            loss_values[i, j] = l_sum / 3

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(w, b, loss_values, cmap='viridis')

ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('Loss')

plt.show()
