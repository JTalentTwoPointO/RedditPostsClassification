# modeling.py

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import LatentDirichletAllocation

def train_logistic_regression(X_train, y_train):
    """Trains a Logistic Regression model."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluates the model using test data and prints metrics."""
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

def perform_topic_modeling(X_train_tfidf, n_topics=3):
    """Performs topic modeling using Latent Dirichlet Allocation (LDA)."""
    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=10, random_state=42)
    lda.fit(X_train_tfidf)
    return lda

def display_topics(model, vectorizer, n_top_words=10):
    """Displays the top words for each topic in the LDA model."""
    for i, topic in enumerate(model.components_):
        print(f"Top {n_top_words} words for topic #{i}:")
        print([vectorizer.get_feature_names_out()[index] for index in topic.argsort()[-n_top_words:]])