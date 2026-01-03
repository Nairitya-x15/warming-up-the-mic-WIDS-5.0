# WIDS 5.0 -- WEEK 3 ASSIGNMENT
### TASK 1: Fine-tune a BERT model for text classification
-> I Have used a sms_span dataset from huggingface dataset library for this model.
<br>
-> firstly split the dataset into train and test dataset.
<br>
-> and then load the tokenizer and tokenize the dataset
<br>
-> then load the bert-base-uncased model.
<br>
-> freeze all bert layers except last two and unfreeze pooler layer.
<br>
-> and then load and define evaluation metrics.
<br>
-> set training configuration and train the model.
<br>

### TASK 2: Train a custom SentencePiece tokenizer
-> loads the WikiText-2-raw-v1 dataset using the Hugging Face datasets library.
<br>
-> import sentencepiece library
<br>
-> specifying parameters such as vocabulary size, unigram model type etc.
<br>
-> model is automatically saved as custom_sp.model
<br>
-> then test the model by running it and giving user input and it will return tokens and tokens id.

### TASK 3: Use Hugging Face transformers for inference
-> loads the prebuilt spam classifier model in task1
<br>
-> take input from user
<br>
-> bring model to evaluation mode
<br>
-> tokenize the input 
<br>
-> Model inference is performed inside a no_grad block to prevent gradient computation, making prediction faster and more memory-efficient.
<br>
-> convert logits to probabilities using softmax
<br>
-> compare probabilities to classify as spam of ham.

### MINI PROJECT :--
#### Question and Answer system
-> The system is pretty simple
<br>
-> use huggingface transformer and pipeline to load the prebuilt model: distilbert-base-cased-distilled-squad
<br>
-> create a object of this model
<br>
-> pass user question and context into it and will give the desired output.


# END OF WEEK 3



