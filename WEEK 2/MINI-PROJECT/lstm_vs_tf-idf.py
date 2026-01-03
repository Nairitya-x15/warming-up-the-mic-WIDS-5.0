import joblib
import pandas as pd

tfidf_results = joblib.load('WEEK 2/MINI-PROJECT/tfidfresult.pkl')
lstm_results = joblib.load('WEEK 2/MINI-PROJECT/lstm-results.pkl')

comparison_df = pd.DataFrame(
    [tfidf_results, lstm_results],
    index=["TF-IDF + Logistic Regression", "LSTM"]
)

print("FINAL COMPARISON TABLE:")
print(comparison_df)