import numpy as np


def activation_function(x):
    """type : step function"""
    return 1 if x >= 0 else 0


def perceptron(x, w, b):
    return activation_function(np.dot(x, w) + b)


# tests
# Example inputs (x1, x2)
x = np.array([1, 0])
# Input vector
w = np.array([0.5, -0.6])  # Weights
b = 0.2
# Bias
# Output
y = perceptron(x, w, b)
print("Output:", y)
