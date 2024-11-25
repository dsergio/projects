import numpy as np
import matplotlib.pyplot as plt

# Generate some synthetic data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add bias term (intercept)
X_b = np.c_[np.ones((X.shape[0], 1)), X]  # add x0 = 1 to each instance

# Gradient Descent function
def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    m = len(y)
    theta = np.random.randn(2, 1)  # random initialization
    for iteration in range(n_iterations):
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta -= learning_rate * gradients
    return theta

# Run gradient descent
theta_best = gradient_descent(X_b, y)

# Print the results
print("Estimated parameters (theta):", theta_best)

# Plotting the results
plt.scatter(X, y)
plt.plot(X, X_b.dot(theta_best), color='red', linewidth=2)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression with Gradient Descent")
plt.show()
