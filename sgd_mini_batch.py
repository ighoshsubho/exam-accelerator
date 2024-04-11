import numpy as np

def stochastic_gradient_descent(X, y, theta, learning_rate, num_iterations):
    """
    Performs stochastic gradient descent to find the optimal theta values.
    
    Parameters:
    X (numpy.ndarray): Input features.
    y (numpy.ndarray): Target variable.
    theta (numpy.ndarray): Initial values of the parameters.
    learning_rate (float): The learning rate for the gradient descent algorithm.
    num_iterations (int): The number of iterations to run the gradient descent algorithm.
    
    Returns:
    numpy.ndarray: The optimal values of the parameters.
    """
    m = len(y)  # Number of examples
    
    for _ in range(num_iterations):
        for i in range(m):
            # Compute the hypothesis and error for a single example
            hypothesis = np.dot(X[i], theta)
            error = hypothesis - y[i]
            
            # Compute the gradient for a single example
            gradient = np.dot(X[i], error)
            
            # Update the parameters
            theta = theta - learning_rate * gradient
    
    return theta

def mini_batch_gradient_descent(X, y, theta, learning_rate, num_iterations, batch_size):
    """
    Performs mini-batch gradient descent to find the optimal theta values.
    
    Parameters:
    X (numpy.ndarray): Input features.
    y (numpy.ndarray): Target variable.
    theta (numpy.ndarray): Initial values of the parameters.
    learning_rate (float): The learning rate for the gradient descent algorithm.
    num_iterations (int): The number of iterations to run the gradient descent algorithm.
    batch_size (int): The size of the batch for mini-batch gradient descent.
    
    Returns:
    numpy.ndarray: The optimal values of the parameters.
    """
    m = len(y)  # Number of examples
    num_batches = m // batch_size
    
    for _ in range(num_iterations):
        for i in range(num_batches):
            # Compute the hypothesis and error for a batch of examples
            start = i * batch_size
            end = (i + 1) * batch_size
            batch_X = X[start:end]
            batch_y = y[start:end]
            hypothesis = np.dot(batch_X, theta)
            error = hypothesis - batch_y
            
            # Compute the gradient for the batch
            gradient = np.dot(batch_X.T, error) / batch_size
            
            # Update the parameters
            theta = theta - learning_rate * gradient
    
    return theta

def batch_gradient_descent(X, y, theta, learning_rate, num_iterations):
    """
    Performs batch gradient descent to find the optimal theta values.
    
    Parameters:
    X (numpy.ndarray): Input features.
    y (numpy.ndarray): Target variable.
    theta (numpy.ndarray): Initial values of the parameters.
    learning_rate (float): The learning rate for the gradient descent algorithm.
    num_iterations (int): The number of iterations to run the gradient descent algorithm.
    
    Returns:
    numpy.ndarray: The optimal values of the parameters.
    """
    m = len(y)  # Number of examples
    
    for _ in range(num_iterations):
        # Compute the hypothesis and error for the entire dataset
        hypothesis = np.dot(X, theta)
        error = hypothesis - y
        
        # Compute the gradient for the entire dataset
        gradient = np.dot(X.T, error) / m
        
        # Update the parameters
        theta = theta - learning_rate * gradient
    
    return theta

# Example usage
X = data.drop('cnt', axis=1).values  # Input features
y = data['cnt'].values  # Target variable

# Normalize the features
X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)

# Stochastic Gradient Descent
theta_sgd = np.zeros(X_normalized.shape[1])
theta_sgd = stochastic_gradient_descent(X_normalized, y, theta_sgd, learning_rate=0.01, num_iterations=1000)
print("Stochastic Gradient Descent:")
print(f"Optimal theta: {theta_sgd}")

# Mini-Batch Gradient Descent
theta_mini_batch = np.zeros(X_normalized.shape[1])
theta_mini_batch = mini_batch_gradient_descent(X_normalized, y, theta_mini_batch, learning_rate=0.01, num_iterations=100, batch_size=32)
print("\nMini-Batch Gradient Descent:")
print(f"Optimal theta: {theta_mini_batch}")

# Batch Gradient Descent
theta_batch = np.zeros(X_normalized.shape[1])
theta_batch = batch_gradient_descent(X_normalized, y, theta_batch, learning_rate=0.01, num_iterations=100)
print("\nBatch Gradient Descent:")
print(f"Optimal theta: {theta_batch}")
