import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function
def cost_function(theta, X, y):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    J = (-1 / m) * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h)))
    return J

# Gradient function
def gradient(theta, X, y):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    grad = (1 / m) * np.dot(X.T, (h - y))
    return grad

# Gradient Descent
def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)
    
    for i in range(num_iters):
        theta = theta - alpha * gradient(theta, X, y)
        J_history[i] = cost_function(theta, X, y)
    
    return theta, J_history

# One-vs-Rest (OVR) Logistic Regression
def ovr_logistic_regression(X_train, y_train, X_test, y_test):
    num_classes = np.unique(y_train).size
    theta_ovr = []
    y_pred_ovr = np.zeros_like(y_test)
    
    for i in range(num_classes):
        # Prepare the binary labels for the current class
        y_train_binary = (y_train == i).astype(int)
        
        # Train the binary logistic regression model
        theta, _ = gradient_descent(X_train, y_train_binary, np.zeros(X_train.shape[1]), 0.01, 1000)
        theta_ovr.append(theta)
        
        # Make predictions on the test set
        y_pred_ovr[:, i] = sigmoid(np.dot(X_test, theta))
    
    # Choose the class with the highest probability
    y_pred_ovr = np.argmax(y_pred_ovr, axis=1)
    return y_pred_ovr

# One-vs-One (OVO) Logistic Regression
def ovo_logistic_regression(X_train, y_train, X_test, y_test):
    num_classes = np.unique(y_train).size
    theta_ovo = []
    y_pred_ovo = np.zeros_like(y_test)
    
    for i in range(num_classes):
        for j in range(i+1, num_classes):
            # Prepare the binary labels for the current class pair
            y_train_binary = (y_train == i).astype(int) - (y_train == j).astype(int)
            
            # Train the binary logistic regression model
            theta, _ = gradient_descent(X_train, y_train_binary, np.zeros(X_train.shape[1]), 0.01, 1000)
            theta_ovo.append(theta)
            
            # Make predictions on the test set
            y_pred_ovo += (sigmoid(np.dot(X_test, theta)) > 0).astype(int)
    
    # Choose the class with the most "votes"
    y_pred_ovo = np.argmax(y_pred_ovo.reshape(y_test.shape[0], -1), axis=1)
    return y_pred_ovo

# Training and Evaluation
y_pred_ovr = ovr_logistic_regression(X_train, y_train, X_test, y_test)
y_pred_ovo = ovo_logistic_regression(X_train, y_train, X_test, y_test)

# Evaluation metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# OVR Evaluation
ovr_cm = confusion_matrix(y_test, y_pred_ovr)
ovr_accuracy = accuracy_score(y_test, y_pred_ovr)
ovr_precision = precision_score(y_test, y_pred_ovr, average='macro')
ovr_recall = recall_score(y_test, y_pred_ovr, average='macro')
ovr_f1 = f1_score(y_test, y_pred_ovr, average='macro')

print("One-vs-Rest (OVR) Logistic Regression:")
print("Confusion Matrix:\n", ovr_cm)
print(f"Accuracy: {ovr_accuracy:.2f}")
print(f"Precision: {ovr_precision:.2f}")
print(f"Recall: {ovr_recall:.2f}")
print(f"F1-Score: {ovr_f1:.2f}")

# OVO Evaluation
ovo_cm = confusion_matrix(y_test, y_pred_ovo)
ovo_accuracy = accuracy_score(y_test, y_pred_ovo)
ovo_precision = precision_score(y_test, y_pred_ovo, average='macro')
ovo_recall = recall_score(y_test, y_pred_ovo, average='macro')
ovo_f1 = f1_score(y_test, y_pred_ovo, average='macro')

print("\nOne-vs-One (OVO) Logistic Regression:")
print("Confusion Matrix:\n", ovo_cm)
print(f"Accuracy: {ovo_accuracy:.2f}")
print(f"Precision: {ovo_precision:.2f}")
print(f"Recall: {ovo_recall:.2f}")
print(f"F1-Score: {ovo_f1:.2f}")
