import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from scipy.linalg import lstsq

# Load the sample dataset
from sklearn.datasets import load_boston
boston = load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['cnt'] = boston.target  # Assuming 'cnt' is the target variable

# 1. EDA & Visualization
# Visualize the hour (hr) column with an appropriate plot
plt.figure(figsize=(10, 6))
sns.countplot(x='hr', data=data)
plt.title('Busy Hours for Bike Sharing')
plt.show()

# Visualize the distribution of count, casual, and registered variables
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
sns.histplot(data['cnt'], kde=True)
plt.title('Distribution of Total Rentals')
plt.subplot(1, 3, 2)
sns.histplot(data['casual'], kde=True)  # Replace 'casual' with your column name
plt.title('Distribution of Casual Rentals')
plt.subplot(1, 3, 3)
sns.histplot(data['registered'], kde=True)  # Replace 'registered' with your column name
plt.title('Distribution of Registered Rentals')
plt.tight_layout()
plt.show()

# Describe the relation of weekday, holiday, and working day
# ... (Add code to analyze the relationship)

# Visualize the month-wise count of casual and registered for 2011 and 2012
# ... (Add code to create stacked bar charts)

# Analyze the correlation between features with a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# 2. Pre-processing and Data Engineering
# Drop unwanted columns
data = data.drop(['unwanted_column1', 'unwanted_column2'], axis=1)  # Replace with your column names

# Identify categorical and continuous variables
categorical_vars = ['season', 'holiday', 'workingday', 'weathersit']  # Replace with your categorical variables
continuous_vars = ['temp', 'atemp', 'humidity', 'windspeed']  # Replace with your continuous variables

# Feature scaling
scaler = MinMaxScaler()
data[continuous_vars] = scaler.fit_transform(data[continuous_vars])

# One-hot encoding
encoder = OneHotEncoder(handle_unknown='ignore')
data_encoded = pd.DataFrame(encoder.fit_transform(data[categorical_vars]).toarray())
data_encoded = data_encoded.join(data.drop(categorical_vars, axis=1))

# Specify features and targets
X = data_encoded.drop('cnt', axis=1)
y = data_encoded['cnt']

# 3. Implement Linear Regression using Normal Equation
# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Implementation using Normal Equation
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))  # Add a column of ones for the bias term
theta = lstsq(X_train, y_train, cond=None, overwrite_a=False, overwrite_b=False, check_finite=True, lapack_driver='gelsd')[0]

# 4. Linear Regression using Scikit-learn
# Fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Calculate the error
mse = np.mean((y_test - y_pred) ** 2)
print(f'Mean Squared Error: {mse:.2f}')

# Calculate R-squared
r_squared = model.score(X_test, y_test)
print(f'R-squared: {r_squared:.2f}')
