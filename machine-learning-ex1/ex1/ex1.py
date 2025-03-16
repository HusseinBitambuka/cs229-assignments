import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas as pd
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import axes3d, Axes3D
from computeCost import *
from gradientDescent import *
from plotData import *
import pandas as pd

# ===================== Part 1: Plotting =====================
print('Plotting Data...')

# using pandas instead of numpy for data loading bcause it's more fun
data = pd.read_csv('/home/husseinbitambuka/Dev/cs229-assignments/machine-learning-ex1/ex1/ex1data1.txt', delimiter = ",", names = ["population","profit"])
print(f"data size is: {data.size}")

print(data.head())
X = data["population"].to_numpy()
y = data["profit"].to_numpy()

m = len(X)



# plotting the data

plot_data(X,y)



input('Program paused. Press ENTER to continue')

# ===================== Part 2: Gradient descent =====================
print('Running Gradient Descent...')

X = np.c_[np.ones(m), X]  # Add a column of ones to X
theta = np.zeros(2)  # initialize fitting parameters

# Some gradient descent settings
iterations = 1500
alpha = 0.01

# Compute and display initial cost
print('Initial cost : ' + str(compute_cost(X, y, theta)) + ' (This value should be about 32.07)')

theta, J_history = gradient_descent(X, y, theta, alpha, iterations)

print('Theta found by gradient descent: ' + str(theta.reshape(2)))

# Plot the linear fit
plt.figure(0)
line1, = plt.plot(X[:, 1], np.dot(X, theta), label='Linear Regression')
plt.legend(handles=[line1])

input('Program paused. Press ENTER to continue')

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot(np.array([1, 3.5]), theta)
print('For population = 35,000, we predict a profit of {:0.3f} (This value should be about 4519.77)'.format(predict1*10000))
predict2 = np.dot(np.array([1, 7]), theta)
print('For population = 70,000, we predict a profit of {:0.3f} (This value should be about 45342.45)'.format(predict2*10000))

input('Program paused. Press ENTER to continue')

# ===================== Part 3: Visualizing J(theta0, theta1) =====================
print('Visualizing J(theta0, theta1) ...')

theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

xs, ys = np.meshgrid(theta0_vals, theta1_vals)
J_vals = np.zeros(xs.shape)

# Fill out J_vals
for i in range(0, theta0_vals.size):
    for j in range(0, theta1_vals.size):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        J_vals[i][j] = compute_cost(X, y, t)

J_vals = np.transpose(J_vals)
# Create the 3D surface plot
fig1 = plt.figure(figsize=(10, 7))
ax = fig1.add_subplot(111, projection='3d')
ax.plot_surface(xs, ys, J_vals, cmap='viridis', edgecolor='none')

ax.set_xlabel(r'$\theta_0$')
ax.set_ylabel(r'$\theta_1$')
ax.set_zlabel(r'$J(\theta_0, \theta_1)$')
ax.set_title('3D Surface Plot of $J(\\theta_0, \\theta_1)$')

# Create the contour plot
plt.figure(figsize=(8, 6))
lvls = np.logspace(-2, 3, 20)
contour = plt.contour(xs, ys, J_vals, levels=lvls, norm=LogNorm())
plt.clabel(contour, inline=True, fontsize=8)

# Plot the current values of theta on the contour plot
plt.plot(theta[0], theta[1], 'r+', markersize=10, label="Theta values")

plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')
plt.title('Contour Plot of $J(\\theta_0, \\theta_1)$')
plt.legend()

plt.show()

input('ex1 Finished. Press ENTER to exit')
