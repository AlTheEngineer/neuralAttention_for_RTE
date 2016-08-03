import logging
import pickle
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize, wordpunct_tokenize, RegexpTokenizer
from collections import OrderedDict

trainingSet = pd.read_csv('snli_1.0_train.txt', delimiter='\t')

testSet = pd.read_csv('snli_1.0_test.txt', delimiter='\t')

valSet = pd.read_csv('snli_1.0_dev.txt', delimiter='\t')

word2vec = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

len(word2vec.vocab)

re_tokenize = RegexpTokenizer('\w+')

inv_words, oov_words_in_train = OrderedDict(), set()

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

trainingSet[['sentence1', 'sentence2']].apply(check_sentence)

oov_words_not_train = set()

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



val_testSets = pd.concat([valSet, testSet], ignore_index = True)

val_testSets[['sentence1', 'sentence2']].apply(check_sentence_for_val_testSets)

index = 0
dictionary = OrderedDict()

for k in inv_words:
	dictionary[k] = index
	index += 1

for k in oov_words_not_train:
	dictionary[k] = index
	index += 1

for k in oov_words_in_train:
	dictionary[k] = index
	index += 1

dictionary_fileName = 'dictionary.pkl'
with open(dictionary_fileName, 'wb') as f:
	pickle.dump(dictionary, f)

inv_indices = list(inv_words.values())
inv_W = word2vec.syn0[inv_indices]

rsg = np.random.RandomState(919)
oov_not_train_W = (rsg.rand(len(oov_words_not_train), word2vec.vector_size) - 0.5) / 10.0

unchanged_W = np.concatenate([inv_W, oov_not_train_W])

oov_in_train_W = (rsg.rand(len(oov_words_in_train), word2vec.vector_size) - 0.5) / 10.0

np.all([np.all(word2vec.syn0[i2] == unchanged_W[i1]) for i1, i2 in enumerate(inv_indices)])

unchanged_W_fileName = 'unchanged_W.pkl'
with open(unchanged_W_fileName, 'wb') as f:
    pickle.dump(unchanged_W, f)

oov_in_train_W_fileName = 'oov_in_train_W.pkl'
with open(oov_in_train_W_fileName, 'wb') as f:
    pickle.dump(oov_in_train_W, f)

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
converted_train = trainingSet.apply(to_ids, axis=1)

valSet = valSet.fillna('')
converted_val = valSet.apply(to_ids, axis=1)

testSet = testSet.fillna('')
converted_test = testSet.apply(to_ids, axis=1)

saved_columns = ['sentence1', 'sentence2', 'gold_label']

converted_train = converted_train[saved_columns]
converted_val = converted_val[saved_columns]
converted_test = converted_test[saved_columns]

converted_train_fileName = 'converted_train.pkl'
with open(converted_train_fileName, 'wb') as f:
    pickle.dump(converted_train, f)


converted_val_fileName = 'converted_val.pkl'
with open(converted_val_fileName, 'wb') as f:
    pickle.dump(converted_val, f)

converted_test_fileName = 'converted_test.pkl'
with open(converted_test_fileName, 'wb') as f:
    pickle.dump(converted_test, f)





