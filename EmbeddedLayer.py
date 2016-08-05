import numpy as np

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import lasagne
from lasagne.layers import Gate
from lasagne import nonlinearities
from lasagne import init
from lasagne.utils import unroll_scan
from lasagne.layers import MergeLayer, Layer, InputLayer, DenseLayer



class EmbeddedLayer(lasagne.layers.Layer):

    def __init__(self, incoming, unchanged_W, unchanged_W_shape,
                 oov_in_train_W, oov_in_train_W_shape,
                 p=0.5, rescale=True, dropout_mask=None,
                 **kwargs):
        #initialize parent layer
        super(EmbeddedLayer, self).__init__(incoming, **kwargs)
        #output layer size
        self.output_size = unchanged_W_shape[1]
        #parametrize layers
        self.unchanged_W = self.add_param(unchanged_W, unchanged_W_shape,
                                          name="unchanged_W",
                                          trainable=False,
                                          regularizable=False)
        self.oov_in_train_W = self.add_param(oov_in_train_W,
                                             oov_in_train_W_shape, 
                                             name='oov_in_train_W')
        #concatenate layers
        self.W = T.concatenate([self.unchanged_W, self.oov_in_train_W])
        #drop out rate
        self.p = p
        self.rescale = rescale
        if dropout_mask is None:
            dropout_mask = RandomStreams(np.random.randint(1, 2147462579)).binomial(self.W.shape,
                                                                               p=1 - self.p,
                                                                               dtype=self.W.dtype)
        self.dropout_mask = dropout_mask

    def get_output_shape_for(self, input_shape):
        return input_shape + (self.output_size, )

    def get_output_for(self, input, deterministic=False, **kwargs):
        W = self.W
        if not deterministic and self.p != 0:
            print('apply dropout mask id {} to embedding matrix ...'.format(id(self.dropout_mask)))
            print('dropout rate is {}'.format(self.p))
            print('input var is {}'.format(input))
            one = T.constant(1)
            retain_prob = one - self.p
            if self.rescale:
                W /= retain_prob
            W = W * self.dropout_mask
        return W[input]


