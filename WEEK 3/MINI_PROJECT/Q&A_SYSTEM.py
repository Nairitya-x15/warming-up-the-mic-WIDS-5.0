import os
import logging
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
cont = input("ENTER YOUR CONTEXT : ")
ques = input("ENTER YOUR QUESTION : ")

from transformers import pipeline

qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

result = qa({
    "question": ques,
    "context":  cont
})

print("ANSWER : ",result["answer"])
