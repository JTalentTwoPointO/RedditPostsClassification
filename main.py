# Step 1: Setup and Library Imports
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Step 2: Load Data
data = pd.read_csv('path_to_your_reddit_data.csv')
print(data.head())

# Step 3: Initial EDA
print(data.info())
print(data['text'].describe())

# Visualization: Distribution of Post Lengths
data['post_length'] = data['text'].apply(len)
plt.hist(data['post_length'], bins=50)
plt.title('Distribution of Post Lengths')
plt.xlabel('Length of Post')
plt.ylabel('Frequency')
plt.show()

# Step 4: Data Preprocessing
stop_words = set(stopwords.words('english'))

def preprocess(text):
    # Remove special characters, links, and stopwords
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = text.lower()
    tokens = [word for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

data['cleaned_text'] = data['text'].apply(preprocess)

# Step 5: Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['cleaned_text'])
y = data['label']  # Ensure you have a label column for Pro-Palestine (0) / Pro-Israel (1)

# Step 6: Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 7: Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))