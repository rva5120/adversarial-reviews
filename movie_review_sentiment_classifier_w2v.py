# --- Movie Reviews Classifier by Sentiment ---
# In this script, we use the pre-trained word embeddings to encode movie reviews.
# Then, we train the model on what positive and negative reviews are. And we test
# the model on test reviews.
#	*CNN Model with 1 Embedding, Convo1D, MaxPooling1D, Flatten and Dense w/sigmoid
#	*Reviews are integer encoded before they are sent to the Embedding layer. The
#	integer maps to the index of a specific vector in the embedding layer.
#	*Each document is a review

from build_vocabulary import build_vocab, load_doc
from prepare_w2v_embeddings import generate_w2v_embeddings

import string
from os import listdir

from numpy import asarray
from numpy import array
from numpy import zeros

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


# clean_doc	turn doc/review into a one item array ['this is a positive review :)']
def clean_doc(doc, vocab):
	# split review into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	tokens = [w.translate(None, string.punctuation) for w in tokens]
	# filter out tokens not in vocab
	tokens = [w for w in tokens if w in vocab]
	tokens = ' '.join(tokens)
	# return clean review
	return tokens

# process_doc	load all docs in the directory to clean and generate testing data
def process_docs(directory, vocab, is_train):
	documents = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if is_train and filename.startswith('cv9'):
			continue
		if not is_train and not filename.startswith('cv9'):
			continue
		# create the full path to the file
		path = directory + '/' + filename
		# load the doc
		doc = load_doc(path)
		# clean doc
		tokens = clean_doc(doc, vocab)
		# add to list
		documents.append(tokens)
	return documents

# load_embedding	load embedding as a dictionary
def load_embedding(filename):
	# load embedding into memory, skip first line
	file = open(filename, 'r')
	lines = file.readlines()[1:]
	file.close()
	# create a map of words to vectors
	embedding = dict()
	for line in lines:
		parts = line.split()
		# key is string word, value is numpy array for vectors
		embedding[parts[0]] = asarray(parts[1:], dtype='float32')
	return embedding

# get_weight_matrix	create a weight matrix for the Embedding later from a loaded embedding
def get_weight_matrix(embedding, vocab):
	# total vocabulary size plus 0 for unknow words
	vocab_size = len(vocab) + 1
	# define weight matrix dimensions with all 0
	weight_matrix = zeros((vocab_size, 100))
	# step vocab, store vectors using the Tokenizer's integer mapping
	for word, i in vocab.items():
		weight_matrix[i] = embedding.get(word)
	return weight_matrix

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

# generate word2vec embeddings
generate_w2v_embeddings()

# -- Training Data --
# load all training reviews
positive_docs = process_docs('txt_sentoken/pos', vocab, True)
negative_docs = process_docs('txt_sentoken/neg', vocab, True)
train_docs = negative_docs + positive_docs

# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
print(train_docs)
tokenizer.fit_on_texts(train_docs)

# sequence encode
encoded_docs = tokenizer.texts_to_sequences(train_docs)
# pad sequences
max_length = max([len(s.split()) for s in train_docs])
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define training labels
ytrain = array([0 for _ in range(900)] + [1 for _ in range(900)])
# --- --- ---

# -- Testing Data --
# load all testing reviews
positive_docs = process_docs('txt_sentoken/pos', vocab, False)
negative_docs = process_docs('txt_sentoken/neg', vocab, False)
test_docs = negative_docs + positive_docs

# fit the tokenizer on the documents
tokenizer.fit_on_texts(test_docs)

# sequence encode
encoded_docs = tokenizer.texts_to_sequences(test_docs)
# pad sequences
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define testing labels
ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])
# --- --- ---

# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1

# load embedding from file
raw_embedding = load_embedding('embedding_word2vec.txt')
# get vectors in the right order
embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index)
# create the embedding layer
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_vectors], input_length=max_length, trainable=True)

# define model
model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
print(model.summary())

# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit network
model.fit(Xtrain, ytrain, epochs=10, verbose=2)

# evaluate
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc*100))
