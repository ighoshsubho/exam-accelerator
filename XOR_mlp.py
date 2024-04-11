import numpy as np

# Activation function (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Input dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Expected output (XOR gate)
y = np.array([[0], [1], [1], [0]])

# Seed for random weight initialization
np.random.seed(1)

# Initialize weights and biases
weights1 = np.random.randn(2, 2)  # Weights for the input to hidden layer
bias1 = np.random.randn(1, 2)     # Bias for the hidden layer
weights2 = np.random.randn(2, 1)  # Weights for the hidden to output layer
bias2 = np.random.randn(1, 1)     # Bias for the output layer

# Training parameters
epochs = 10000
learning_rate = 0.1

# Training loop
for epoch in range(epochs):
    # Forward propagation
    layer1 = sigmoid(np.dot(X, weights1.T) + bias1)
    layer2 = sigmoid(np.dot(layer1, weights2.T) + bias2)

    # Compute error
    error = y - layer2

    # Backpropagation
    delta2 = error * sigmoid_derivative(layer2)
    delta1 = np.dot(delta2, weights2) * sigmoid_derivative(layer1)

    # Update weights and biases
    weights2 += learning_rate * np.dot(layer1.T, delta2)
    bias2 += learning_rate * np.sum(delta2, axis=0, keepdims=True)
    weights1 += learning_rate * np.dot(X.T, delta1)
    bias1 += learning_rate * np.sum(delta1, axis=0)

# Test the model
print("Input | Predicted Output")
for i in range(len(X)):
    layer1 = sigmoid(np.dot(X[i], weights1.T) + bias1)
    layer2 = sigmoid(np.dot(layer1, weights2.T) + bias2)
    print(f"{X[i]} | {round(layer2[0][0])}")
