from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

with open("WEEK 3/TASK 2/data.txt", "w", encoding="utf-8") as f:
    for item in dataset["train"]:
        if item["text"].strip():
            f.write(item["text"] + "\n")


import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="WEEK 3/TASK 2/data.txt",
    model_prefix="WEEK 3/TASK 2/custom_sp",
    vocab_size=20000,
    model_type="unigram",
    character_coverage=1.0,
    unk_id=0,
    bos_id=1,
    eos_id=2,
    pad_id=3
)

print("Sentence Piece Tokenizer is trained and saved successfully")


sp = spm.SentencePieceProcessor()
sp.load("WEEK 3/TASK 2/custom_sp.model")

text = input("ENTER ANY SENTENCE TO TEST THE TOKENIZER : ")

tokens = sp.encode(text, out_type=str)
ids = sp.encode(text, out_type=int)

print("Original text: ")
print(text)

print("Tokens: ")
print(tokens)

print("Token IDs: ")
print(ids)

decoded_text = sp.decode(ids)

print("Decoded text: ")
print(decoded_text)
