"""Particle only, forward Euler"""

import numpy as np
import matplotlib.pyplot as plt

M = 1.0
m = 3.0e-6
G = 1.0

X = np.array([0, 0, 0])  # the position vector of the Sun
x = np.array([1, 0, 0])  # the position vector of the Earth
V = np.array([0, 0, 0])  # the velocity vector of the Sun
v = np.array([0, 1, 0])  # the velocity vector of the Earth

dt = 0.01
t_end = 100

x_vec = []  # x of the Earth
y_vec = []  # y of the Earth
X_vec = []  # x of the Sun
Y_vec = []  # y of the Sun

for t in np.linspace(0, t_end, int(t_end/dt)):
    # acc Sun --> Earth
    a = -G * M * (x - X) / np.linalg.norm(x - X) ** 3
    # acc Earth --> Sun
    A = -G * m * (X - x) / np.linalg.norm(X - x) ** 3

    # advance the positions and velocities
    x = x + v * dt
    X = X + V * dt
    v = v + a * dt
    V = V + A * dt

    # store the data for plotting

    x_vec.append(x[0])
    y_vec.append(x[1])
    X_vec.append(X[0])
    Y_vec.append(X[1])

# make the plot
plt.plot(x_vec, y_vec)
plt.plot(X_vec, Y_vec)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()
