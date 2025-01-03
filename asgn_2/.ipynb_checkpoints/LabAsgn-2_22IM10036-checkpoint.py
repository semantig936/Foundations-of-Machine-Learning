# Name: Semanti Ghosh
# Roll No - 22IM10036
#Python Library Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import datasets
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
import string
import nltk
from nltk import corpus
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords') 

#Utility function to
# convert text to lower after removing punctuations
# and removing English stopwords.
def clean_util(text):
    punc_rmv = [char for char in text if char not in string.punctuation] # Punctuation remove
    punc_rmv = "".join(punc_rmv)
    # Removing Stopwords (i.e., "a", "an", "the", "and", "but", "in", "on", "of", and "with" etc.) 
    stopword_rmv = [w.strip().lower() for w in punc_rmv.split() if w.strip().lower() not in stopwords.words('english')] 

    return " ".join(stopword_rmv)

#Main program
# Load the dataset without specifying categories
newsgroups_data = fetch_20newsgroups(subset='all') #Loading all documents test & train

# List all available categories
# Get the category names
categories = newsgroups_data.target_names
print(categories)
# Get the target of each document
targets = newsgroups_data.target

print(len(targets))
category_counts = np.bincount(targets)
for category, count in zip(categories, category_counts):
    print(f"{category}: {count} documents")
    
categories = ['alt.atheism', 'comp.graphics', 'sci.space', 'talk.politics.guns', 'rec.sport.hockey']
newsgroups = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)# List all available categories

categories = ['sci.med', 'comp.graphics', 'sci.space', 'sci.electronics', 'rec.sport.hockey']
#newsgroups = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)# List train dataset
#Apply the following - random_state=42 for reproducible results everytime
newsgroups = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)# List full dataset
df = pd.DataFrame({'data': newsgroups.data, 'target': newsgroups.target})
category_counts = df['target'].value_counts()
print("Documents per category:\n", category_counts)
min_docs_per_category = category_counts.min()
balanced_df = df.groupby('target').apply(lambda x: x.sample(min_docs_per_category, random_state=42)).reset_index(drop=True)
print("Balanced dataset:\n", balanced_df['target'].value_counts())

#Applying the cleaning utility function on dataframe
balanced_df['data'] = balanced_df['data'].apply(clean_util)

df.head()

#Train the Multinomial Naiva Bayes Classifier
# Using TfidfVectorizer from sklearn.feature_extraction.text with alpha=0.1

texts = balanced_df['data']
y = balanced_df['target']

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(texts)
# Applying random_state=42 for reproducable results
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #20% test data
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
nb = MultinomialNB(alpha=0.01)
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

# Model Performance

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=vectorizer.get_feature_names_out(), yticklabels=vectorizer.get_feature_names_out())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

