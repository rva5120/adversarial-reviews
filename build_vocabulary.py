# --- Prepare Data ---

# This script does the following:
#	- Split tokens by white space delimiters
#	- Remove all punctuations from words
#	- Remove all words that known stop words
#	- Remove all words that have a length <= 1 character


from os import listdir
from nltk.corpus import stopwords
from collections import Counter
import string

# load_doc	loads doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# process_docs	walk through all files in the folder
def process_docs(directory, vocab):
	for filename in listdir(directory):
		# skip files that do not have the right extension
		if filename.endswith(".txt"):
			# create the full path of the file to open
			path = directory + '/' + filename
			# add doc to vocab
			add_doc_to_vocab(path, vocab)
			print('Loaded vocabulary of doc %s' % filename)

# clean_doc	turn doc into clean tokens
def clean_doc(doc):
	# split into tokens using white space delimiters
	tokens = doc.split()
	# remove punctuation from each token
	tokens = [w.translate(None, string.punctuation) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stopwords
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	return tokens

# add_doc_to_vocab	process document and add its words to the vocabulary
def add_doc_to_vocab(filename, vocab):
	# load doc
	doc = load_doc(filename)
	# clean doc
	tokens = clean_doc(doc)
	# update counts
	vocab.update(tokens)

# save_list	save tokens to file (one per line)
def save_list(lines, filename):
	file = open(filename, 'w')
	for line in lines:
		data = '\n'.join(line)
		file.write(data)
	file.close()

# build_vocab	build vocabulary of all reviews
def build_vocab():
	# define vocab
	vocab = Counter()
	# add all docs to vocab
	process_docs('txt_sentoken/neg', vocab)
	process_docs('txt_sentoken/pos', vocab)
	# print the size of the vocab
	print(len(vocab))
	# print the top words in the vocab
	print(vocab.most_common(50))
	# keep tokens with > 5 occurrences
	min_occ = 2
	tokens = [k for k,c in vocab.items() if c >= min_occ]
	print(len(tokens))
	# save tokens to a vocabulary file
	save_list(tokens, 'vocab.txt')

