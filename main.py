import logging
import pickle
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize, wordpunct_tokenize, RegexpTokenizer
from collections import OrderedDict

#Read SNLI training file
trainingSet = pd.read_csv('snli_1.0/snli_1.0_train.txt', delimiter='\t')
#Read SNLI test file
testSet = pd.read_csv('snli_1.0/snli_1.0_test.txt', delimiter='\t')
#Read SNLI validation (val) file
valSet = pd.read_csv('snli_1.0/snli_1.0_dev.txt', delimiter='\t')
#Load word2vec NLP feature learning algorithm
word2vec = Word2Vec.load_word2vec_format('snli_1.0/GoogleNews-vectors-negative300.bin', binary=True)

len(word2vec.vocab)

re_tokenize = RegexpTokenizer('\w+')
#keep track of words in vocab
inv_words = OrderedDict()
#keep track of words in training set but NOT in vocab
oov_words_in_train = set()
#This function takes a sentence, checks if each word is in the word2vec vocabulary
#returns the number of words in the training set that are NOT in vocab
def check_sentence(s):
	count = 0
	for r in s:
		if type(r) != str:
			print r
			count += 1
			continue
		words = word_tokenize(r)
		for w in words:
			if w in inv_words or w in oov_words_in_train:
				continue
			if w not in word2vec:
				count += 1
				oov_words_in_train.add(w)
			else:
				inv_words[w] = word2vec.vocab[w].index
	return count
#convert each sentence into list of indices
trainingSet[['sentence1', 'sentence2']].apply(check_sentence)

#keep track of words in test set but NOT in training and NOT in vocab
oov_words_not_train = set()


#This function takes a sentence, checks if each word is in the word2vec vocabulary
#returns the number of words in the validation and test sets that are NOT in vocab
def check_sentence_for_val_testSets(s):
	count = 0
	for r in s:
		if type(r) != str:
			print r
			count += 1
			continue
		words = word_tokenize(r)
		for w in words:
			if w in inv_words or w in oov_words_in_train or w in oov_words_not_train:
				continue
			if w not in word2vec:
				count += 1
				oov_words_not_train.add(w)
			else:
				inv_words[w] = word2vec.vocab[w].index
	return count


#concatenate the validation and test sets
val_testSets = pd.concat([valSet, testSet], ignore_index = True)
#convert each sentence to list of indices
#count number of words in val and test sets that are NOT in vocab or training
val_testSets[['sentence1', 'sentence2']].apply(check_sentence_for_val_testSets)

#store ALL words in dict
dictionary = OrderedDict()
#initialize dict index
index = 0
#store words in data set AND vocab in dict
for k in inv_words:
	dictionary[k] = index
	index += 1
#store words in test and val sets but NOT vocab or training in dict
for k in oov_words_not_train:
	dictionary[k] = index
	index += 1
#store words in training but NOT vocab in dict
for k in oov_words_in_train:
	dictionary[k] = index
	index += 1
#dump dict into respective file
dictionary_fileName = 'dictionary.pkl'
with open(dictionary_fileName, 'wb') as f:
	pickle.dump(dictionary, f)
#A list of indices for the words in the data set and vocab
inv_indices = list(inv_words.values())
#Convert each word index into numeric representation with 300 features
inv_W = word2vec.syn0[inv_indices]

#assign random numeric reps for words NOT in vocab
rsg = np.random.RandomState(919)
#Out-of-vocabulary words encountered at inference
#time on the validation and test corpus are set to fixed random vectors
#-http://arxiv.org/pdf/1509.06664v4.pdf
oov_not_train_W = (rsg.rand(len(oov_words_not_train), word2vec.vector_size) - 0.5) / 10.0

unchanged_W = np.concatenate([inv_W, oov_not_train_W])

#"Out-ofvocabulary words in the training set are randomly initialized by sampling values 
#uniformly from (0.05, 0.05) and optimized during training" 
oov_in_train_W = (rsg.rand(len(oov_words_in_train), word2vec.vector_size) - 0.5) / 10.0
#test
#np.all([np.all(word2vec.syn0[i2] == unchanged_W[i1]) for i1, i2 in enumerate(inv_indices)])

unchanged_W_fileName = 'unchanged_W.pkl'
with open(unchanged_W_fileName, 'wb') as f:
    pickle.dump(unchanged_W, f)

oov_in_train_W_fileName = 'oov_in_train_W.pkl'
with open(oov_in_train_W_fileName, 'wb') as f:
    pickle.dump(oov_in_train_W, f)


#This function takes a pair of premise and hypothesis sentences
#returns a list where each item consists of two lists and a label
#each list consists of the indexes of each word in either the premise or hypothesis
#and the label corresponds to the output classification 
def to_ids(r):
    premise_words = word_tokenize(r.sentence1)
    hypo_words = word_tokenize(r.sentence2)
    premise_ids = []
    for w in premise_words:
        premise_ids.append(dictionary[w])
    hypo_ids = []
    for w in hypo_words:
        hypo_ids.append(dictionary[w])
    r.loc['sentence1'] = premise_ids
    r.loc['sentence2'] = hypo_ids
    return r


trainingSet = trainingSet.fillna('')
#convert val set to index lists
converted_train = trainingSet.apply(to_ids, axis=1)

valSet = valSet.fillna('')
#convert val set to index lists
converted_val = valSet.apply(to_ids, axis=1)


testSet = testSet.fillna('')
#convert test set to index lists
converted_test = testSet.apply(to_ids, axis=1)

#add column labels
saved_columns = ['sentence1', 'sentence2', 'gold_label']
converted_train = converted_train[saved_columns]
converted_val = converted_val[saved_columns]
converted_test = converted_test[saved_columns]

#dump processed data sets into respective files
converted_train_fileName = 'converted_train.pkl'
with open(converted_train_fileName, 'wb') as f:
    pickle.dump(converted_train, f)


converted_val_fileName = 'converted_val.pkl'
with open(converted_val_fileName, 'wb') as f:
    pickle.dump(converted_val, f)

converted_test_fileName = 'converted_test.pkl'
with open(converted_test_fileName, 'wb') as f:
    pickle.dump(converted_test, f)





