# Import required packages
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

# One-vs-Rest (OVR) Logistic Regression
from sklearn.linear_model import LogisticRegression
ovr_classifier = LogisticRegression(random_state=0, multi_class='ovr')
ovr_classifier.fit(X_train, y_train)
ovr_y_pred = ovr_classifier.predict(X_test)

# One-vs-One (OVO) Logistic Regression
ovo_classifier = LogisticRegression(random_state=0, multi_class='ovo')
ovo_classifier.fit(X_train, y_train)
ovo_y_pred = ovo_classifier.predict(X_test)

# Evaluation Metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# OVR Evaluation
ovr_cm = confusion_matrix(y_test, ovr_y_pred)
ovr_accuracy = accuracy_score(y_test, ovr_y_pred)
ovr_precision = precision_score(y_test, ovr_y_pred)
ovr_recall = recall_score(y_test, ovr_y_pred)
ovr_f1 = f1_score(y_test, ovr_y_pred)

print("One-vs-Rest (OVR) Logistic Regression:")
print("Confusion Matrix:\n", ovr_cm)
print(f"Accuracy: {ovr_accuracy:.2f}")
print(f"Precision: {ovr_precision:.2f}")
print(f"Recall: {ovr_recall:.2f}")
print(f"F1-Score: {ovr_f1:.2f}")

# OVO Evaluation
ovo_cm = confusion_matrix(y_test, ovo_y_pred)
ovo_accuracy = accuracy_score(y_test, ovo_y_pred)
ovo_precision = precision_score(y_test, ovo_y_pred)
ovo_recall = recall_score(y_test, ovo_y_pred)
ovo_f1 = f1_score(y_test, ovo_y_pred)

print("\nOne-vs-One (OVO) Logistic Regression:")
print("Confusion Matrix:\n", ovo_cm)
print(f"Accuracy: {ovo_accuracy:.2f}")
print(f"Precision: {ovo_precision:.2f}")
print(f"Recall: {ovo_recall:.2f}")
print(f"F1-Score: {ovo_f1:.2f}")
