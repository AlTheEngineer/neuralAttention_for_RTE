
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
from Encoder import Encoder

class Decoder(lasagne.layers.LSTMLayer):

    def __init__(self, incoming, num_units, ingate=Gate(), forgetgate=Gate(),
                 cell=Gate(W_cell=None, nonlinearity=nonlinearities.tanh), outgate=Gate(),
                 nonlinearity=nonlinearities.tanh, cell_init=init.Constant(0.), 
                 hid_init=init.Constant(0.),
                 backwards=False, learn_init=False, peepholes=True, 
                 gradient_steps=-1, grad_clipping=0,
                 precompute_input=True, mask_input=None,
                 encoder_mask_input=None, attention=False, word_by_word=False, **kwargs):
        super(Decoder, self).__init__(incoming, num_units, ingate, 
                                                forgetgate, cell, outgate, nonlinearity,
                                                cell_init, hid_init, backwards, learn_init, 
                                                peepholes, gradient_steps,
                                                grad_clipping, False, precompute_input, 
                                                mask_input, True,
                                                **kwargs)
        self.attention = attention
        self.word_by_word = word_by_word
        # encoder mask
        self.encoder_mask_incoming_index = -1
        if encoder_mask_input is not None:
            self.input_layers.append(encoder_mask_input)
            self.input_shapes.append(encoder_mask_input.output_shape)
            self.encoder_mask_incoming_index = len(self.input_layers)-1
        # check encoder
        if not isinstance(self.cell_init, Encoder) \
                or self.num_units != self.cell_init.num_units:
            raise ValueError('cell_init must be Encoder'
                             ' and num_units should equal')
        self.r_init = None
        self.r_init = self.add_param(init.Constant(0.),
                                     (1, num_units), name="r_init",
                                     trainable=False, regularizable=False)
        if self.word_by_word:
            # rewrites
            self.attention = True
        if self.attention:
            if not isinstance(encoder_mask_input, lasagne.layers.Layer):
                raise ValueError('Attention mechnism needs encoder mask layer')
            # initializes attention weights
            self.W_y_attend = self.add_param(init.Normal(0.1), (num_units, num_units), 'W_y_attend')
            self.W_h_attend = self.add_param(init.Normal(0.1), (num_units, num_units), 'W_h_attend')
            # doesn't need transpose
            self.w_attend = self.add_param(init.Normal(0.1), (num_units, 1), 'w_attend')
            self.W_p_attend = self.add_param(init.Normal(0.1), (num_units, num_units), 'W_p_attend')
            self.W_x_attend = self.add_param(init.Normal(0.1), (num_units, num_units), 'W_x_attend')
            if self.word_by_word:
                self.W_r_attend = self.add_param(init.Normal(0.1), (num_units, num_units), 'W_r_attend')
                self.W_t_attend = self.add_param(init.Normal(0.1), (num_units, num_units), 'W_t_attend')

    def get_output_shape_for(self, input_shapes):
        return super(Decoder, self).get_output_shape_for(input_shapes)

    def get_output_for(self, inputs, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable

        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``. When the hidden state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with. When the cell state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with. When both the cell state and the hidden state are
            being pre-filled `inputs[-2]` is the hidden state, while
            `inputs[-1]` is the cell state.

        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        cell_init = None
        encoder_hs = None
        encoder_mask = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]
        if self.encoder_mask_incoming_index > 0:
            # (n_batch, n_time_steps)
            encoder_mask = inputs[self.encoder_mask_incoming_index]
            encoder_mask = encoder_mask.astype('float32')
        cell_init = inputs[self.cell_init_incoming_index]
        if self.attention:
            # (n_batch, n_time_steps, n_features)
            encoder_hs = cell_init[0]
            # encoder_mask is # (n_batch, n_time_steps, 1)
            encoder_hs = encoder_hs * encoder_mask.dimshuffle(0, 1, 'x')
        cell_init = cell_init[1]

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape

        # Stack input weight matrices into a (num_inputs, 4*num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_ingate, self.W_in_to_forgetgate,
             self.W_in_to_cell, self.W_in_to_outgate], axis=1)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_ingate, self.W_hid_to_forgetgate,
             self.W_hid_to_cell, self.W_hid_to_outgate], axis=1)

        # Stack biases into a (4*num_units) vector
        b_stacked = T.concatenate(
            [self.b_ingate, self.b_forgetgate,
             self.b_cell, self.b_outgate], axis=0)

        if self.precompute_input:
            # Because the input is given for all time steps, we can
            # precompute_input the inputs dot weight matrices before scanning.
            # W_in_stacked is (n_features, 4*num_units). input is then
            # (n_time_steps, n_batch, 4*num_units).
            input = T.dot(input, W_in_stacked) + b_stacked

        # At each call to scan, input_n will be (n_time_steps, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        # Create single recurrent computation step function
        # input_n is the n'th vector of the input
        def step(input_n, cell_previous, hid_previous, previous_r, *args):
            if not self.precompute_input:
                input_n = T.dot(input_n, W_in_stacked) + b_stacked

            # Calculate gates pre-activations and slice
            gates = input_n + T.dot(hid_previous, W_hid_stacked)

            # Clip gradients
            if self.grad_clipping:
                gates = theano.gradient.grad_clip(
                    gates, -self.grad_clipping, self.grad_clipping)

            # Extract the pre-activation gate values
            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.peepholes:
                # Compute peephole connections
                ingate += cell_previous*self.W_cell_to_ingate
                forgetgate += cell_previous*self.W_cell_to_forgetgate

            # Apply nonlinearities
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)

            # Compute new cell value
            cell = forgetgate*cell_previous + ingate*cell_input

            if self.peepholes:
                outgate += cell*self.W_cell_to_outgate
            outgate = self.nonlinearity_outgate(outgate)

            # Compute new hidden unit activation
            hid = outgate*self.nonlinearity(cell)
            r = previous_r
            if self.attention and self.word_by_word:
                mh = T.dot(hid, self.W_h_attend) + T.dot(previous_r, self.W_r_attend)
                # mh is (n_batch, 1, n_features)
                mh = mh.dimshuffle(0, 'x', 1)
                M = T.dot(encoder_hs, self.W_y_attend) + mh
                # (n_batch, n_time_steps, n_features)
                M = nonlinearities.tanh(M)
                # alpha is (n_batch, n_time_steps, 1)
                alpha = T.dot(M, self.w_attend)
                # now is (n_batch, n_time_steps)
                alpha = T.flatten(alpha, 2)
                # 0 after softmax is not 0, fuck, my mistake.
                # when i > encoder_seq_len, fill alpha_i to -np.inf
                # alpha = T.switch(encoder_mask, alpha, -np.inf)
                alpha = T.nnet.softmax(alpha)
                # apply encoder_mask to alpha
                # encoder_mask is (n_batch, n_time_steps)
                # when i > encoder_seq_len, alpha_i should be 0.
                # actually not need mask, but in case of error
                # alpha = alpha * encoder_mask
                alpha = alpha.dimshuffle(0, 1, 'x')
                weighted_encoder = T.sum(encoder_hs * alpha, axis=1)
                r = weighted_encoder + nonlinearities.tanh(T.dot(previous_r, self.W_t_attend))

            return [cell, hid, r]

        def step_masked(input_n, mask_n, cell_previous, hid_previous, previous_r, *args):
            cell, hid, r = step(input_n, cell_previous, hid_previous, previous_r, *args)

            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)
            r = T.switch(mask_n, r, previous_r)
            return [cell, hid, r]

        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = input
            step_fun = step

        ones = T.ones((num_batch, 1))
        if not isinstance(self.hid_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(ones, self.hid_init)

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked]
        # The "peephole" weight matrices are only used when self.peepholes=True
        if self.peepholes:
            non_seqs += [self.W_cell_to_ingate,
                         self.W_cell_to_forgetgate,
                         self.W_cell_to_outgate]

        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]

        r_init = T.dot(ones, self.r_init)
        if self.attention and self.word_by_word:
            non_seqs += [self.W_y_attend,
                         self.W_h_attend,
                         self.W_r_attend,
                         self.w_attend,
                         self.W_t_attend,
                         encoder_hs,
                         # encoder_mask
                         ]
        # Scan op iterates over first dimension of input and repeatedly
        # applies the step function
        cell_out, hid_out, r_out = theano.scan(
            fn=step_fun,
            sequences=sequences,
            outputs_info=[cell_init, hid_init, r_init],
            go_backwards=self.backwards,
            truncate_gradient=self.gradient_steps,
            non_sequences=non_seqs,
            strict=True)[0]
        # (n_batch, n_features)
        hid_N = hid_out[-1]
        out = hid_N
        if self.attention:
            if self.word_by_word:
                r_N = r_out[-1]
            else:
                mh = T.dot(hid_N, self.W_h_attend)
                mh = mh.dimshuffle(0, 'x', 1)
                M = T.dot(encoder_hs, self.W_y_attend) + mh
                # (n_batch, n_time_steps, n_features)
                M = nonlinearities.tanh(M)
                alpha = T.dot(M, self.w_attend)
                # (n_batch, n_time_steps)
                alpha = T.flatten(alpha, 2)
                # when i > encoder_seq_len, fill alpha_i to -np.inf
                # alpha = T.switch(encoder_mask, alpha, -np.inf)
                alpha = T.nnet.softmax(alpha)
                # apply encoder_mask to alpha
                # encoder_mask is (n_batch, n_time_steps)
                # when i > encoder_seq_len, alpha_i should be 0.
                # actually not need mask, but in case of error
                # alpha = alpha * encoder_mask
                alpha = alpha.dimshuffle(0, 1, 'x')
                # (n_batch, n_features)
                r_N = T.sum(encoder_hs * alpha, axis=1)
            out = nonlinearities.tanh(T.dot(r_N, self.W_p_attend) + T.dot(hid_N, self.W_x_attend))
        return out



