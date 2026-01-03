import pandas as pd
import re
import string
import joblib
import time
from sklearn.metrics import accuracy_score, f1_score
print("training tf-idf sentiment classifier...")

df=pd.read_csv('WEEK 1/IMDB-Dataset.csv')

df["review"]= df["review"].str.lower();

def remove_html_tags(text):
    pattern= re.compile('<.*?>')
    return pattern.sub(r'', text)
df["review"]= df['review'].apply(remove_html_tags)

def remove_punctuation_translate(input_string):
    translator= str.maketrans('','',string.punctuation)
    cleaned_string= input_string.translate(translator)
    return cleaned_string
df["review"]= df['review'].apply(remove_punctuation_translate)

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

from sklearn.feature_extraction.text import TfidfVectorizer
start_time = time.time()
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1,2),
    min_df=5,
    max_df=0.9
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

tfidf_train_time = time.time() - start_time
y_test_pred = model.predict(X_test_tfidf)

y_train_pred = model.predict(X_train_tfidf)

joblib.dump(model, "WEEK 1/sentiment_model.pkl")
joblib.dump(vectorizer, "WEEK 1/tfidf_vectorizer.pkl")

print("Sentiment Model has been saved successfully")


tfidf_results = {
    "Train Accuracy": accuracy_score(y_train, y_train_pred),
    "Test Accuracy": accuracy_score(y_test, y_test_pred),
    "Train F1-score": f1_score(y_train, y_train_pred),
    "Test F1-score": f1_score(y_test, y_test_pred),
    "Training Time (s)": tfidf_train_time,
}

tfidf_results["Overfitting Gap (Acc)"] = (
    tfidf_results["Train Accuracy"] - tfidf_results["Test Accuracy"]
)
print(tfidf_results)

joblib.dump(tfidf_results, "WEEK 2/MINI-PROJECT/tfidfresult.pkl")