import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.corpus import stopwords
from nltk import FreqDist

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
df['Resume'] = df['Resume'].str.lower()

# Exercise 6: Define a function to clean the resume text
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
categories = df['Category'].unique()
category_to_index = {category: i for i, category in enumerate(categories)}
df['Category_Encoded'] = df['Category'].map(category_to_index)

# Exercise 9: Convert the text to feature vectors
from collections import defaultdict

def get_vocab(texts):
    vocab = defaultdict(int)
    for text in texts:
        for word in text.split():
            vocab[word] += 1
    return vocab

vocab = get_vocab(df['Cleaned_Resume'])
vocab_size = len(vocab)

def tfidf(text):
    word_counts = text.split()
    feature_vector = [0] * vocab_size
    for word in word_counts:
        feature_vector[list(vocab.keys()).index(word)] += 1
    total_words = sum(word_counts)
    feature_vector = [count / total_words * np.log(len(df) / (vocab[word] + 1)) for count, word in zip(feature_vector, vocab)]
    return feature_vector

X = [tfidf(text) for text in df['Cleaned_Resume']]

# Exercise 10: Split the data and apply Naive Bayes Classifier
from collections import Counter

def naive_bayes_classify(X_train, y_train, X_test):
    class_counts = Counter(y_train)
    prior_probs = {category: count / len(y_train) for category, count in class_counts.items()}

    feature_probs = {}
    for category in class_counts:
        category_texts = df.loc[df['Category_Encoded'] == category, 'Cleaned_Resume']
        category_vocab = get_vocab(category_texts)
        feature_probs[category] = {word: (category_vocab[word] + 1) / (sum(category_vocab.values()) + vocab_size) for word in vocab}

    y_pred = []
    for x_test in X_test:
        posteriors = {category: np.log(prior_probs[category]) for category in class_counts}
        for category in class_counts:
            for i, value in enumerate(x_test):
                if value > 0:
                    word = list(vocab.keys())[i]
                    posteriors[category] += np.log(feature_probs[category].get(word, 1e-9))
        y_pred.append(max(posteriors, key=posteriors.get))
    return y_pred

X_train = X[:int(0.8 * len(X))]
y_train = df['Category_Encoded'][:int(0.8 * len(X))]
X_test = X[int(0.8 * len(X)):]
y_test = df['Category_Encoded'][int(0.8 * len(X)):]

y_pred = naive_bayes_classify(X_train, y_train, X_test)

from sklearn.metrics import accuracy_score, classification_report
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
