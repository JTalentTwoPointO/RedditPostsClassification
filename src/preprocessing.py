# src/preprocessing.py

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    """ Clean and preprocess text data """
    # Initialize the stop words and lemmatizer
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Convert text to lowercase
    text = text.lower()

    # Remove URLs, special characters, and digits
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = re.sub(r'\d+', '', text)

    # Tokenize and remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

def prepare_data(df, text_column):
    """ Preprocess the data by cleaning the text """
    df[text_column] = df[text_column].astype(str)  # Ensure all entries are strings
    df['cleaned_text'] = df[text_column].apply(clean_text)
    return df

def vectorize_text(text_series):
    """ Vectorize the cleaned text using TF-IDF """
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(text_series)
    return X, vectorizer

def train_test_split_data(df, target_column):
    """ Split the data into training and testing sets """
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df[target_column], test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test