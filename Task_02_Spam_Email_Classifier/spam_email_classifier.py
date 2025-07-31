# Load the dataset
import pandas as pd

df = pd.read_csv('spam_email.csv')
df.head()


# Explore the data
display(df.head())
display(df.isnull().sum())
df.info()


# Preprocess the text data
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['cleaned_text'] = df['text'].apply(clean_text)
display(df.head())

tfidf_vectorizer = TfidfVectorizer(max_features=5000) # Limiting features to 5000 for manageability
X = tfidf_vectorizer.fit_transform(df['cleaned_text'])
print("Shape of the TF-IDF feature matrix:", X.shape)


# Split the dataset
from sklearn.model_selection import train_test_split

y = df['spam']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


# Train a classification model
from sklearn.naive_bayes import MultinomialNB

# Instantiate the Multinomial Naive Bayes model
model = MultinomialNB()

# Train the model using the training data
model.fit(X_train, y_train)


# Evaluate the model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
