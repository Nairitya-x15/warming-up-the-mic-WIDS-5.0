import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

warnings.filterwarnings("ignore", category=FutureWarning)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

sms= input("ENTER MESSAGE TO BE CLASSIFIED AS HAM OR SPAM : ")
print("Loading.....")

model_dir = "WEEK 3\TASK 1\spam_classifier"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

model.eval()
import torch.nn.functional as F

inputs = tokenizer(
    sms,
    return_tensors="pt",
    truncation=True,
    padding=True
)

with torch.no_grad():
    outputs = model(**inputs)

probs = F.softmax(outputs.logits, dim=1)[0]

if(probs[0]>=probs[1]):
    print("THIS MESSAGE IS A: HAM")
else:
    print("THIS MESSAGE IS A: SPAM")

print("HAM PROBABILITY:", probs[0].item())
print("SPAM PROBABILITY:", probs[1].item())

