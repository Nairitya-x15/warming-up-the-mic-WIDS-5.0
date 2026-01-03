import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

warnings.filterwarnings("ignore", category=FutureWarning)
import re
import string
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences



user_review = input("Enter the Text for which you want to test sentiment: ")


model = load_model("WEEK 2/imdb_lstm_model.keras")
tokenizer = joblib.load("WEEK 2/tokenizer.joblib")

def remove_html_tags(text):
    pattern= re.compile('<.*?>')
    return pattern.sub(r'', text)


def remove_punctuation_translate(input_string):
    translator= str.maketrans('','',string.punctuation)
    cleaned_string= input_string.translate(translator)
    return cleaned_string


def predict_sentiment(text):
    text = remove_html_tags(text)
    text = remove_punctuation_translate(text)
    text = text.lower()
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=200, padding='post',truncating='post')
    pred = model.predict(pad)[0][0]
    return "Positive" if pred > 0.5 else "Negative"

review = predict_sentiment(user_review)

print("Predicted Sentiment: ",review)