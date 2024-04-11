import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('online_retail_data.csv')

# Data Pre-processing function
def preprocess_data(data):
    # Remove redundant data
    data.drop_duplicates(inplace=True)
    
    # Handle cancelled and adjusted orders
    data = data[~data['InvoiceNo'].str.contains('C|A', na=False)]
    
    # Handle null values
    data.dropna(subset=['Quantity', 'UnitPrice'], inplace=True)
    
    # Remove irrelevant transactions
    irrelevant_codes = ['POST', 'PADS', 'M', 'DOT', 'C2', 'BANK CHARGES']
    data = data[~data['StockCode'].isin(irrelevant_codes)]
    
    # Handle outliers
    # ... (Implement outlier detection and handling)
    
    # Create DayOfWeek column
    data['DayOfWeek'] = pd.to_datetime(data['InvoiceDate']).dt.day_name()
    
    return data

# Preprocess the data
data = preprocess_data(data)

# Understanding new insights
# 1. Check for free items
free_items = data[(data['Quantity'] > 0) & (data['UnitPrice'] == 0)].shape[0]
print(f"Number of free items: {free_items}")

# 2. Number of transactions per country
country_counts = data['Country'].value_counts()
country_counts.plot(kind='bar', figsize=(10, 6))
plt.title('Number of Transactions per Country')
plt.xlabel('Country')
plt.ylabel('Number of Transactions')
plt.show()

# 3. Ratio of repeat and single-time purchasers
repeat_customers = data['CustomerID'].value_counts()[data['CustomerID'].value_counts() > 1]
single_customers = data['CustomerID'].value_counts()[data['CustomerID'].value_counts() == 1]
repeat_ratio = repeat_customers.sum() / (repeat_customers.sum() + single_customers.sum())
print(f"Ratio of repeat purchasers: {repeat_ratio:.2f}")
plt.pie([repeat_customers.sum(), single_customers.sum()], labels=['Repeat', 'Single'], autopct='%1.1f%%')
plt.axis('equal')
plt.show()

# 4. Heatmap of UnitPrice per month and day of the week
data['Month'] = pd.to_datetime(data['InvoiceDate']).dt.month_name()
unit_price_pivot = data.pivot_table(values='UnitPrice', index='Month', columns='DayOfWeek', aggfunc='mean')
sns.heatmap(unit_price_pivot, annot=True, cmap='YlGnBu')
plt.title('Average Unit Price per Month and Day of the Week')
plt.show()

# 5. Top 10 customers and items
top_customers = data.groupby('CustomerID')['Quantity'].sum().sort_values(ascending=False).head(10)
top_items = data.groupby('StockCode')['Quantity'].sum().sort_values(ascending=False).head(10)
print("Top 10 Customers by Total Quantity:")
print(top_customers)
print("\nTop 10 Items by Total Quantity:")
print(top_items)

# Feature Engineering and Transformation
data['TotalSpent'] = data['Quantity'] * data['UnitPrice']
data = data.groupby('CustomerID').agg({'TotalSpent': 'sum', 'Quantity': 'sum'}).reset_index()

# Drop unwanted columns
data.drop(columns=['InvoiceNo', 'StockCode', 'Description', 'InvoiceDate', 'Country'], inplace=True)

# Scale the data
scaler = StandardScaler()
data[['TotalSpent', 'Quantity']] = scaler.fit_transform(data[['TotalSpent', 'Quantity']])

# Clustering
# K-Means
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10)
kmeans.fit(data[['TotalSpent', 'Quantity']])
data['KMeans_Cluster'] = kmeans.labels_

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(data[['TotalSpent', 'Quantity']])
data['DBSCAN_Cluster'] = dbscan.labels_

# Visualize clusters
plt.figure(figsize=(10, 6))
plt.scatter(data['TotalSpent'], data['Quantity'], c=data['KMeans_Cluster'], cmap='viridis')
plt.title('K-Means Clustering')
plt.xlabel('TotalSpent')
plt.ylabel('Quantity')
plt.colorbar()
plt.show()

# Train a supervised algorithm
X = data[['TotalSpent', 'Quantity']]
y = data['KMeans_Cluster']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy of Logistic Regression on Clustered Data: {accuracy:.2f}")
