import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# Assuming the dataset is loaded into a Pandas DataFrame named 'df'

# 1. Handle the null values by removing or replacing
df = df.dropna()
df = df.fillna(df.mean())

# 2. Remove unwanted columns
df = df.drop(['EDUC', 'SES'], axis=1)

# 3. Feature scaling
scaler = StandardScaler()
df[['MR delay', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']] = scaler.fit_transform(df[['MR delay', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']])

# 4. Encode categorical features into numeric
le = LabelEncoder()
df['Group'] = le.fit_transform(df['Group'])

# 5. Identify feature and target, and split into train and test
X = df.drop('CDR', axis=1)
y = df['CDR']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Plot the distribution of all the variables using a histogram
df.hist(figsize=(12, 8))
plt.show()

# 7. Visualize the frequency of Age
plt.figure(figsize=(8, 6))
df['Age'].hist()
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Frequency of Age')
plt.show()

# 8. How many people have Alzheimer? Visualize with an appropriate plot
plt.figure(figsize=(8, 6))
df['CDR'].value_counts().plot(kind='bar')
plt.xlabel('CDR')
plt.ylabel('Count')
plt.title('Number of people with Alzheimer\'s')
plt.show()

# 9. Calculate the correlation of features and plot the heatmap
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='YlOrRd')
plt.title('Correlation Heatmap')
plt.show()

# 10. Model training and evaluation using SVM
svm = SVC()
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
