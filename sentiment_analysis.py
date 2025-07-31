# Load the data
import pandas as pd
training_df = pd.read_csv('twitter_training.csv')
validation_df = pd.read_csv('twitter_validation.csv')


# Explore the data
display(training_df.head())
display(validation_df.head())
display(training_df.isnull().sum())
display(validation_df.isnull().sum())
display(training_df.info())
display(validation_df.info())


column_names = ['Tweet_ID', 'Entity', 'Sentiment', 'Tweet_Content']
training_df = pd.read_csv('twitter_training.csv', header=None, names=column_names)
validation_df = pd.read_csv('twitter_validation.csv', header=None, names=column_names)

display(training_df.head())
display(validation_df.head())
display(training_df.isnull().sum())
display(validation_df.isnull().sum())
display(training_df.info())
display(validation_df.info())


# Preprocess the text data
import nltk
nltk.download('punkt')
nltk.download('stopwords')

import nltk
nltk.download('punkt_tab')

import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

training_df['cleaned_tweet'] = training_df['Tweet_Content'].apply(preprocess_text)
validation_df['cleaned_tweet'] = validation_df['Tweet_Content'].apply(preprocess_text)

display(training_df[['Tweet_Content', 'cleaned_tweet']].head())
display(validation_df[['Tweet_Content', 'cleaned_tweet']].head())


# Label encoding
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
training_df['sentiment_encoded'] = label_encoder.fit_transform(training_df['Sentiment'])
validation_df['sentiment_encoded'] = label_encoder.transform(validation_df['Sentiment'])

display(training_df[['Sentiment', 'sentiment_encoded']].head())
display(validation_df[['Sentiment', 'sentiment_encoded']].head())


# Split the data
from sklearn.model_selection import train_test_split

X = training_df['cleaned_tweet']
y = training_df['sentiment_encoded']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

print(f"Original training data shape: {training_df.shape}")
print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_val shape: {y_val.shape}")


# Vectorize the text data
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_tfidf = tfidf_vectorizer.transform(X_val)

print(f"Shape of X_train_tfidf: {X_train_tfidf.shape}")
print(f"Shape of X_val_tfidf: {X_val_tfidf.shape}")


# Train a sentiment analysis model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)


# Evaluate the model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_pred = model.predict(X_val_tfidf)

accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, average='weighted')
recall = recall_score(y_val, y_pred, average='weighted')
f1 = f1_score(y_val, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")


# Test the model
validation_predictions = model.predict(X_val_tfidf)
