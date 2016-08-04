# -*- coding: utf-8 -*-

# Created by junfeng on 5/12/16.

# logging config
import logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
'''
__all__ = [
    "EmbedLayer",
    "CustomDense"
    "LSTMEncoder",
    "LSTMDecoder",
]
'''
_rng = np.random

#The following class is largely adopted from Lasagne's lasagne.layers.layer class with
#some minor modifications.

# This class concerns the embedded layer 
class EmbedLayer(lasagne.layers.Layer):
# incoming = input layer
    def __init__(self, incoming, unchanged_W, unchanged_W_shape,
                 oov_in_train_W, oov_in_train_W_shape,
                 p=0.5, rescale=True, dropout_mask=None,
                 **kwargs):
        #refers to parent class without explicit naming
        super(EmbedLayer, self).__init__(incoming, **kwargs)
        self.output_size = unchanged_W_shape[1] # feature dimn (300)
        #layer params initialized in constructor (lasagne.layers.layer.add_param())
        self.unchanged_W = self.add_param(unchanged_W, unchanged_W_shape,
                                          name="unchanged_W",
                                          trainable=False,
                                          regularizable=False)
        self.oov_in_train_W = self.add_param(oov_in_train_W,
                                             oov_in_train_W_shape, name='oov_in_train_W')
        self.W = T.concatenate([self.unchanged_W, self.oov_in_train_W])
        self.p = p #drop out rate
        self.rescale = rescale
        if dropout_mask is None:
            dropout_mask = RandomStreams(_rng.randint(1, 2147462579)).binomial(self.W.shape,
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


