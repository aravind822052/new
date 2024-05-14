import streamlit as st
from sklearn.naive_bayes import MultinomialNB
import numpy as np

# Streamlit app title
st.title("Simple Naive Bayes Classifier")

# Example training data
training_text = [
    "I love this sandwich.",
    "This is an amazing place!",
    "I feel very good about these beers.",
    "This is my best work.",
    "What an awesome view",
    "I do not like this restaurant",
    "I am tired of this stuff.",
    "I can't deal with this",
    "He is my sworn enemy!",
    "My boss is horrible."
]
training_labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 1 for positive sentiment, 0 for negative sentiment

# Tokenize the training data
vocab = set()
for text in training_text:
    vocab.update(text.lower().split())

# Create a dictionary to map words to indices
word_to_idx = {word: i for i, word in enumerate(vocab)}

# Transform training data into feature vectors
X_train = np.zeros((len(training_text), len(vocab)))
for i, text in enumerate(training_text):
    for word in text.lower().split():
        X_train[i, word_to_idx[word]] += 1

# Initialize and train the Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, training_labels)

# Classify the input text
text_input = st.text_input("Enter text to classify:", "")
if text_input:
    # Tokenize the input text
    input_vector = np.zeros(len(vocab))
    for word in text_input.lower().split():
        if word in word_to_idx:
            input_vector[word_to_idx[word]] += 1
    
    # Make prediction
    prediction = clf.predict([input_vector])
    
    # Display the result
    if prediction[0] == 1:
        st.write("Positive sentiment!")
    else:
        st.write("Negative sentiment!")
