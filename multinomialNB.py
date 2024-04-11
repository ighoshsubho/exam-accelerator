import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk import FreqDist
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Exercise 1: Read the UpdatedResumeDataset.csv dataset
df = pd.read_csv('UpdatedResumeDataset.csv')

# Exercise 2: Display all the categories of resumes and their counts in the dataset
print(df['Category'].value_counts())

# Exercise 3: Create the count plot of different categories
sns.countplot(x='Category', data=df)
plt.show()

# Exercise 4: Create a pie plot depicting the percentage of resume distributions category-wise
category_counts = df['Category'].value_counts()
plt.figure(figsize=(10, 6))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', colors=plt.get_cmap('Set2')(np.linspace(0, 1, len(category_counts))))
plt.axis('equal')
plt.show()

# Exercise 5: Convert all the Resume text to lower case
df['Resume'] = df['Resume'].apply(lambda x: x.lower())

# Exercise 6: Define a function to clean the resume text
import re

def clean_resume_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove RT, cc
    text = re.sub(r'(rt|cc)', '', text)
    
    # Remove hashtags and mentions
    text = re.sub(r'#|\@\w+', '', text)
    
    # Remove punctuations
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# Apply the cleaning function and store the cleaned text in a new column
df['Cleaned_Resume'] = df['Resume'].apply(clean_resume_text)

# Exercise 7: Use nltk package to find the most common words from the cleaned resume column
stop_words = set(stopwords.words('english'))
words = []
for text in df['Cleaned_Resume']:
    words.extend([word for word in text.split() if word not in stop_words])

freq_dist = FreqDist(words)
print(freq_dist.most_common(10))

# Exercise 8: Convert the categorical variable Category to a numerical feature
le = LabelEncoder()
df['Category_Encoded'] = le.fit_transform(df['Category'])

# Exercise 9: Convert the text to feature vectors by applying tfidf vectorizer
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(df['Cleaned_Resume'])

# Exercise 10: Split the data into train and test sets, and apply Naive Bayes Classifier
X_train, X_test, y_train, y_test = train_test_split(X, df['Category_Encoded'], test_size=0.2, random_state=42)

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)
y_pred = nb_classifier.predict(X_test)

# Evaluate the model predictions
from sklearn.metrics import accuracy_score, classification_report
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
