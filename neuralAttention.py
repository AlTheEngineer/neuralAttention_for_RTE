# coding: utf-8

# In[1]:

from __future__ import print_function

import pickle
import sys

import numpy
import numpy as np
import pandas as pd

import theano
import theano.tensor as T
import lasagne
import time

from layers import EmbeddedLayer, Encoder, DenseLayer, Decoder



# prepare data for training
def prepare(data):
    seqs_premise = []
    seqs_hypothesis = []
    for cc in data['sentence1']:
        seqs_premise.append(cc)
    for cc in data['sentence2']:
        seqs_hypothesis.append(cc)
    seqs_p = seqs_premise
    seqs_h = seqs_hypothesis

    lengths_p = [len(s) for s in seqs_p]
    lengths_h = [len(s) for s in seqs_h]

    n_samples = len(seqs_p)
    maxlen_p = numpy.max(lengths_p) + 1
    maxlen_h = numpy.max(lengths_h) + 1

    premise = numpy.zeros((n_samples, maxlen_p))
    hypothesis = numpy.zeros((n_samples, maxlen_h))
    premise_masks = numpy.zeros((n_samples, maxlen_p))
    hypothesis_masks = numpy.zeros((n_samples, maxlen_h))
    for idx, [s_t, s_h] in enumerate(zip(seqs_p, seqs_h)):
        assert lengths_h[idx] == len(s_h)
        premise[idx, :lengths_p[idx]] = s_t
        premise_masks[idx, :lengths_p[idx]] = 1
        hypothesis[idx, :lengths_h[idx]] = s_h
        hypothesis_masks[idx, :lengths_h[idx]] = 1
    labels = []
    for gl in data['gold_label']:
        if gl == 'entailment':
            value = 2
        elif gl == 'contradiction':
            value = 1
        elif gl == 'neutral':
            value = 0
        else:
            raise ValueError('unknown gold_label {0}'.format(gl))
        labels.append(value)

    labels = np.array(labels)
    return (premise.astype('int32'),
            premise_masks.astype('int32'),
            hypothesis.astype('int32'),
            hypothesis_masks.astype('int32'),
            labels.astype('int32'))


print('Loading data ...')
train_data, val_data, test_data = (None, None, None)
with open('./snli/converted_train.pkl', 'rb') as f:
    print('Loading training set ...')
    train_data = pickle.load(f)
    print(len(train_data))
    filtered_s2 = train_data.sentence2.apply(lambda s2: len(s2) != 0)
    train_data = train_data[filtered_s2]
    print(len(train_data))
    train_data = train_data[train_data.gold_label != '-']
    train_data = train_data.reset_index()
    print("Size of training set: "+str(len(train_data)))

with open('./snli/converted_dev.pkl', 'rb') as f:
    print('Loading validation set ...')
    val_data = pickle.load(f)
    print(len(val_data))
    filtered_s2 = val_data.sentence2.apply(lambda s2: len(s2) != 0)
    val_data = val_data[filtered_s2]
    print("Size of validation set: "+str(len(val_data)))
    #remove samples with missing data
    val_data = val_data[val_data.gold_label != '-']
    val_data = val_data.reset_index()
    print(len(val_data))
with open('./snli/converted_test.pkl', 'rb') as f:
    print('Loading test set ...')
    test_data = pickle.load(f)
    print(len(test_data))
    filtered_s2 = test_data.sentence2.apply(lambda s2: len(s2) != 0)
    test_data = test_data[filtered_s2]
    print(len(test_data))
    test_data = test_data[test_data.gold_label != '-']
    test_data = test_data.reset_index()
    print(len(test_data))

premise_max = 82 + 1
hypothesis_max = 62 + 1

def main(num_epochs=10, k=100, 
	 batch_size=128,
         display_freq=100,
         save_freq=1000,
         load_previous=False,
         attention=True,
         word_by_word=True, 
	 p=0, mode='word_by_word'):
    print('num_epochs: {}'.format(num_epochs))
    print('k: {}'.format(k))
    print('batch_size: {}'.format(batch_size))
    print('display_frequency: {}'.format(display_freq))
    print('save_frequency: {}'.format(save_freq))
    print('load previous: {}'.format(load_previous))
    print('attention: {}'.format(attention))
    print('word_by_word: {}'.format(word_by_word))
    save_filename = './snli/{}_model.npz'.format(mode)
    print("Building network ...")
    premise_var = T.imatrix('premise_var')
    premise_mask = T.imatrix('premise_mask')
    hypo_var = T.imatrix('hypo_var')
    hypo_mask = T.imatrix('hypo_mask')
    unchanged_W = pickle.load(open('./snli/unchanged_W.pkl', 'rb'))
    unchanged_W = unchanged_W.astype('float32')
    unchanged_W_shape = unchanged_W.shape
    oov_in_train_W = pickle.load(open('./snli/oov_in_train_W.pkl', 'rb'))
    oov_in_train_W = oov_in_train_W.astype('float32')
    oov_in_train_W_shape = oov_in_train_W.shape
    print('unchanged_W.shape: {0}'.format(unchanged_W_shape))
    print('oov_in_train_W.shape: {0}'.format(oov_in_train_W_shape))
    # hyperparameters
    learning_rate = 0.001
    l2_weight = 0.
#Input layers
    l_premise = lasagne.layers.InputLayer(shape=(None, premise_max), input_var=premise_var)
    l_premise_mask = lasagne.layers.InputLayer(shape=(None, premise_max), input_var=premise_mask)
    l_hypo = lasagne.layers.InputLayer(shape=(None, hypothesis_max), input_var=hypo_var)
    l_hypo_mask = lasagne.layers.InputLayer(shape=(None, hypothesis_max), input_var=hypo_mask)
#Embedded layers
    premise_embedding = EmbeddedLayer(l_premise, unchanged_W, unchanged_W_shape,
                                        oov_in_train_W, oov_in_train_W_shape,
                                        p=p)
#weights shared with premise_embedding
    hypo_embedding = EmbeddedLayer(l_hypo, unchanged_W=premise_embedding.unchanged_W,
                                     unchanged_W_shape=unchanged_W_shape,
                                     oov_in_train_W=premise_embedding.oov_in_train_W,
                                     oov_in_train_W_shape=oov_in_train_W_shape,
                                     p=p,
                                     dropout_mask=premise_embedding.dropout_mask)
#Dense layers
    l_premise_linear = DenseLayer(premise_embedding, k,
                                   nonlinearity=lasagne.nonlinearities.linear)
    l_hypo_linear = DenseLayer(hypo_embedding, k,
                                W=l_premise_linear.W, b=l_premise_linear.b,
                                nonlinearity=lasagne.nonlinearities.linear)

    encoder = Encoder(l_premise_linear, k, peepholes=False, mask_input=l_premise_mask)
#initialized with encoder final hidden state
    decoder = Decoder(l_hypo_linear, k, cell_init=encoder, peepholes=False, mask_input=l_hypo_mask,
                                encoder_mask_input=l_premise_mask,
                                attention=attention,
                                word_by_word=word_by_word)
    if p > 0.:
        print('apply dropout rate {} to decoder'.format(p))
        decoder = lasagne.layers.DropoutLayer(decoder, p)
    l_softmax = lasagne.layers.DenseLayer(decoder, 
	    				  num_units=3,
            nonlinearity=lasagne.nonlinearities.softmax)
    target_var = T.ivector('target_var')

#lasagne.layers.get_output produces a variable for the output of the net
    prediction = lasagne.layers.get_output(l_softmax, deterministic=False)
#The network output will have shape (n_batch, 3);
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    cost = loss.mean()
    if l2_weight > 0.:
       #apply l2 regularization
        print('apply l2 penalty to all layers, weight: {}'.format(l2_weight))
        regularized_layers = {encoder: l2_weight,
                              decoder: l2_weight}
        l2_penalty = lasagne.regularization.regularize_network_params(l_softmax,
                                                                      lasagne.regularization.l2) * l2_weight
        cost += l2_penalty
#Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_softmax, trainable=True)
#Compute adam updates for training
    print("Computing updates ...")
    updates = lasagne.updates.adam(cost, all_params, 
					learning_rate=learning_rate)

    test_prediction = lasagne.layers.get_output(l_softmax, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # lasagne.objectives.categorical_accuracy()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train_fn = theano.function([premise_var, premise_mask, hypo_var, hypo_mask, target_var],
                               cost, updates=updates)
    val_fn = theano.function([premise_var, premise_mask, hypo_var, hypo_mask, target_var],
                             [test_loss, test_acc])
    print("Training ...")

    print('train_data.shape: {0}'.format(train_data.shape))
    print('val_data.shape: {0}'.format(val_data.shape))
    print('test_data.shape: {0}'.format(test_data.shape))
    try:
        # Finally, launch the training loop.
        print("Training started...")
        # iterate over epochs:
        for epoch in range(num_epochs):
            # In each epoch, do a full pass over the training data:
            shuffled_train_data = train_data.reindex(np.random.permutation(train_data.index))
            train_err = 0
            train_acc = 0
            train_batches = 0
            start_time = time.time()
            display_at = time.time()
            save_at = time.time()
            for start_i in range(0, len(shuffled_train_data), batch_size):
                batched_data = shuffled_train_data[start_i:start_i + batch_size]
                ps, p_masks, hs, h_masks, labels = prepare(batched_data)
                train_err += train_fn(ps, p_masks, hs, h_masks, labels)
                err, acc = val_fn(ps, p_masks, hs, h_masks, labels)
                train_acc += acc
                train_batches += 1
                # display
                if train_batches % display_freq == 0:
                    print("Seen {:d} samples, time used: {:.3f}s".format(
                        start_i + batch_size, time.time() - display_at))
                    print("  current training loss:\t\t{:.6f}".format(train_err / train_batches))
                    print("  current training accuracy:\t\t{:.6f}".format(train_acc / train_batches))
                # do tmp save model
                if train_batches % save_freq == 0:
                    print('saving to ..., time used {:.3f}s'.format(time.time() - save_at))
                    np.savez(save_filename,
                             *lasagne.layers.get_all_param_values(l_softmax))
                    save_at = time.time()

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for start_i in range(0, len(val_data), batch_size):
                batched_data = val_data[start_i:start_i + batch_size]
                ps, p_masks, hs, h_masks, labels = prepare(batched_data)
                err, acc = val_fn(ps, p_masks, hs, h_masks, labels)
                val_err += err
                val_acc += acc
                val_batches += 1

            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                    epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  training accuracy:\t\t{:.2f} %".format(
                    train_acc / train_batches * 100))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(
                    val_acc / val_batches * 100))

            # After training, we compute and print the test error:
            test_err = 0
            test_acc = 0
            test_batches = 0
            for start_i in range(0, len(test_data), batch_size):
                batched_data = test_data[start_i:start_i + batch_size]
                ps, p_masks, hs, h_masks, labels = prepare(batched_data)
                err, acc = val_fn(ps, p_masks, hs, h_masks, labels)
                test_err += err
                test_acc += acc
                test_batches += 1
            # print("Final results:")
            print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
            print("  test accuracy:\t\t{:.2f} %".format(
                    test_acc / test_batches * 100))
            filename = './snli/{}_model_epoch{}.npz'.format(mode, epoch + 1)
            print('saving to {}'.format(filename))
            np.savez(filename,
                     *lasagne.layers.get_all_param_values(l_softmax))

        # Optionally, you could now dump the network weights to a file like this:
        # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
        #
        # And load them again later on like this:
        # with np.load('model.npz') as f:
        #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        # lasagne.layers.set_all_param_values(network, param_values)
    except KeyboardInterrupt:
        print('exit ...')


if __name__ == '__main__':
    attention = True
    word_by_word = True
    mode = 'word_by_word'
    if len(sys.argv) == 2:
        mode = sys.argv[1]
        if mode == 'condition':
            attention = False
            word_by_word = False
        elif mode == 'attention':
            word_by_word = False
        elif mode == 'word_by_word':
            attention = True
            word_by_word = True
        else:
            print('doesn\'t recognize mode {}'.format(mode))
            print('only supports [condition|attention|word_by_word]')
            sys.exit(1)

    main(num_epochs=20, batch_size=30,
         display_freq=1000,
         load_previous=False,
         attention=attention,
         word_by_word=word_by_word,
         mode=mode,
	 p=0,
	 )

