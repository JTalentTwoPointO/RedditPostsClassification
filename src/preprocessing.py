# src/preprocessing.py

import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure you download necessary resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize the text
    words = nltk.word_tokenize(text)

    # Remove stop words and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    # Reconstruct text from cleaned words
    return ' '.join(words)


def vectorize_text(texts):
    """Vectorizes text using TF-IDF."""
    vectorizer = TfidfVectorizer(max_features=5000)
    return vectorizer.fit_transform(texts)

# Placeholder to demonstrate usage
if __name__ == "__main__":
    # Example usage with placeholder text data
    sample_data = pd.DataFrame({
        'text': ['Sample post pro Palestine', 'Sample post pro Israel']
    })
    sample_data['cleaned_text'] = sample_data['text'].apply(clean_text)
    X = vectorize_text(sample_data['cleaned_text'])
    print(X.shape)