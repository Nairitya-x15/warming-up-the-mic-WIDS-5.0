# WIDS 5.0 -- WEEK 2 ASSIGNMENT
### TASK 1: Train Word2Vec/GloVe on a small text corpus
-> I Have used a small essay on pollution to implement Word2vec embedding technique.
<br>
-> firstly done some simple preprocessing using gensim library
<br>
-> and then use the Word2Vec class of gensim library to do word embedding.
<br>
-> also used PCA to plot a 3d scatter graph of word embedding.
<br>
### TASK 2: Build an LSTM sentiment classifier
->This task has been done in 2 parts.
<br>

#### 1) SENTIMENT MODEL TRAINING :--
-> Firstly import important packages and library and remove future warnings
<br>
-> Read the IMDB-Dataset.csv
<br>
-> Clean dataset by removing html tags and punctuation marks.
<br>
-> Split the dataset into 4:1 ratio in training set and testing dataset respectively.
<br>
-> using tensorflow library to perform tokenization and asigning each token a word index.
<br>
-> only the most frequent 20,000 words are given integer numbers rest are given "oov" token.
<br>
-> converting texts to sequences using word indexes and then doing padding so that each sequence is of same length.
<br>
-> compiling the model and then training it.
<br>
-> and then saving the model.

#### 1) SENTIMENT PREDICTION :--
-> Load the sentiment model using tensorflow load model.
<br>
->Take input text from user
<br>
-> Perform basic text cleaning on input text like html tag removal, remove punctuation mark.
<br>
->And use sentiment model to predict sentiment of input text

### TASK 3: MINI PROJECT
#### TF-IDF vs LSTM MODEL :--
-> For this purpose i had already created and saved a .pkl file for lstm results and tf-idf results while training them.
<br>
->use joblib library to load these results files
<br>
-> since i already calculated accuracy,f1 score, training time, overfitting behaviour so i just need to print these results using those pkl files.


# END OF WEEK 2



