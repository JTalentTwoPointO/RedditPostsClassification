import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize
text = "This is a test."
tokens = word_tokenize(text)
print(tokens)
