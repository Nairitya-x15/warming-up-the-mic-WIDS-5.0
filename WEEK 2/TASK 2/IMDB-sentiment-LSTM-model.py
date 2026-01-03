import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

warnings.filterwarnings("ignore", category=FutureWarning)
print("Training lstm sentiment model....")

import pandas as pd
import string
import joblib
import re
import time
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, f1_score
df = pd.read_csv('WEEK 1/IMDB-Dataset.csv')

def remove_html_tags(text):
    pattern= re.compile('<.*?>')
    return pattern.sub(r'', text)
df["review"]= df['review'].apply(remove_html_tags)

def remove_punctuation_translate(input_string):
    translator= str.maketrans('','',string.punctuation)
    cleaned_string= input_string.translate(translator)
    return cleaned_string
df["review"]= df['review'].apply(remove_punctuation_translate)

df['review']= df['review'].str.lower()

df["sentiment"]= df["sentiment"].map({
    "positive": 1,
    "negative": 0
})

from sklearn.model_selection import train_test_split
X = df["review"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
      random_state=567,
      stratify=y
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 20000

tokenizer = Tokenizer(
    num_words=vocab_size,
    oov_token="<OOV>"
)
tokenizer.fit_on_texts(X_train)


X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)


max_len = 200

X_train_pad = pad_sequences(
    X_train_seq,
    maxlen=max_len,
    padding='post',
    truncating='post'
)

X_test_pad = pad_sequences(
    X_test_seq,
    maxlen=max_len,
    padding='post',
    truncating='post'
)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

start_time = time.time()
model = Sequential([
   Embedding(input_dim=vocab_size, output_dim=128),
    LSTM(128, return_sequences=False),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(
    X_train_pad,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

lstm_train_time = time.time() - start_time



y_train_pred_lstm = (model.predict(X_train_pad) > 0.5).astype(int).ravel()
y_test_pred_lstm = (model.predict(X_test_pad) > 0.5).astype(int).ravel()


model.save("WEEK 2/imdb_lstm_model.keras")
joblib.dump(tokenizer, "WEEK 2/tokenizer.joblib")

lstm_results = {
    "Train Accuracy": accuracy_score(y_train, y_train_pred_lstm),
    "Test Accuracy": accuracy_score(y_test, y_test_pred_lstm),
    "Train F1-score": f1_score(y_train, y_train_pred_lstm),
    "Test F1-score": f1_score(y_test, y_test_pred_lstm),
    "Training Time (s)": lstm_train_time,
}

lstm_results["Overfitting Gap (Acc)"] = (
    lstm_results["Train Accuracy"] - lstm_results["Test Accuracy"]
)


print("Model has been trained successfully ")
print(lstm_results)

joblib.dump(lstm_results, "WEEK 2/MINI-PROJECT/lstm-results.pkl")
