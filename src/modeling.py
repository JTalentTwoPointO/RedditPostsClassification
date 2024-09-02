# src/modeling.py

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def train_logistic_regression(X, y):
    """Trains a Logistic Regression model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and prints the classification report."""
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

# Placeholder to demonstrate usage
if __name__ == "__main__":
    # Example usage with placeholder data
    from preprocessing import vectorize_text

    # Fake data for demonstration
    sample_data = ['Sample pro Palestine', 'Sample pro Israel']
    labels = [0, 1]  # 0 = Pro-Palestine, 1 = Pro-Israel

    X = vectorize_text(sample_data)
    model, X_test, y_test = train_logistic_regression(X, labels)
    evaluate_model(model, X_test, y_test)