from build_vocabulary import build_vocab, load_doc, clean_doc, save_list
from os import listdir
from gensim.models import Word2Vec
import string

# doc_to_clean_line	load doc, clean and return lines
def doc_to_clean_line(doc, vocab):
	clean_lines = list()
	lines = doc.splitlines()
	for line in lines:
		# split tokens by white space
		tokens = line.split()
		# remove punctuation from each token
		tokens = [w.translate(None, string.punctuation) for w in tokens]
		# filter by vocab
		tokens = [w for w in tokens if w in vocab]
		clean_lines.append(tokens)
	return clean_lines

# process_docs	load all docs in a directory
def process_docs(directory, vocab, is_train):
	lines = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# training files do not start with cv9
		# and testing files start with cv9
		if is_train and filename.startswith('cv9'):
			continue
		if not is_train and not filename.startwith('cv9'):
			continue
		# create full path of the file to open
		path = directory + '/' + filename
		# load and clean the doc
		doc = load_doc(path)
		doc_lines = doc_to_clean_line(doc, vocab)
		# add to list
		lines += doc_lines
	return lines

''' Run this to generate word embeddings using the word2vec algorithm '''
def generate_w2v_embeddings():
	# build vocabulary
	#build_vocab() --> called by "main"

	# load vocabulary
	vocab_filename = 'vocab.txt'
	vocab = load_doc(vocab_filename)
	vocab = vocab.split()
	vocab = set(vocab)

	# prepare negative reviews
	negative_lines = process_docs('txt_sentoken/neg', vocab, True)
	save_list(negative_lines, 'negative.txt')

	# prepare positive reviews
	positive_lines = process_docs('txt_sentoken/pos', vocab, True)
	save_list(positive_lines, 'positive.txt')

	# set training reviews
	sentences = negative_lines + positive_lines
	print('Total training sentences: %d' % len(sentences))

	# train Word2Vec model
	model = Word2Vec(sentences, size=100, window=5, workers=8, min_count=1)

	# summarize vocabulary size in model
	words = list(model.wv.vocab)
	print('Vocabulary size: %d' % len(words))

	# save model in ASCII (word2vec) format
	filename = 'embedding_word2vec.txt'
	model.wv.save_word2vec_format(filename, binary=False)
