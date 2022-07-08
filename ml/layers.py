from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .inits import zeros

flags = tf.app.flags
FLAGS = flags.FLAGS

# DISCLAIMER:
# Boilerplate parts of this code file were originally forked from
# https://github.com/tkipf/gcn
# which itself was very inspired by the keras package

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).
    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name','name_weights', 'logging', 'model_size', 'jump_type', 'jumping_knowledge',
                          'hidden_dim_1','hidden_dim_2', 'geto_dims','geto_vec_dim','use_geto',
                          'subcomplex_weight'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        name_w = kwargs.get('name_weights')
        if name_w:
            name = name+name_w
        self.name = name
        self.vars = {}
        # self.sub_vars = {}
        self.grad_free_vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])

        # for var in self.sub_vars:
        #     tf.summary.histogram(self.name + '/sub_vars/' + var, self.sub_vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, dropout=0., 
                 act=tf.nn.relu, placeholders=None, bias=True, featureless=False, 
                 sparse_inputs=False, use_geto=False,**kwargs):
        super(Dense, self).__init__(**kwargs)

        self.dropout = dropout

        self.act = act
        self.featureless = featureless
        self.bias = bias
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.use_geto = use_geto

        # helper variable for sparse dropout
        self.sparse_inputs = sparse_inputs
        if sparse_inputs:
            self.num_features_nonzero = placeholders['num_features_nonzero']

        # if not use_geto:
        with tf.variable_scope(self.name + '_vars'):
            self.vars[self.name+'_weights'] = tf.get_variable(self.name+'_weights', shape=(input_dim, output_dim),
                                         dtype=tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer(),
                                         regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay))

            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()
        # else:
        #     with tf.variable_scope(self.name + '_vars'):
        #         self.geto_vars[self.name + '_weights'] = tf.get_variable(self.name + '_weights',
        #                                                             shape=(input_dim, output_dim),
        #                                                             dtype=tf.float32,
        #                                                             initializer=tf.contrib.layers.xavier_initializer(),
        #                                                             regularizer=tf.contrib.layers.l2_regularizer(
        #                                                                 FLAGS.weight_decay))
        #         if self.bias:
        #             self.geto_vars['bias'] = zeros([output_dim], name='bias')
        #
        #     #if self.logging:
        #     #    self._log_vars()

    def _call(self, inputs, name='call'):
        x = inputs

        x = tf.nn.dropout(x, keep_prob=1-self.dropout, name=name+'dp')

        # transform
        # if not self.use_geto:
        output = tf.matmul(x, self.vars[self.name+'_weights'], name=name+'mm')
        # bias
        if self.bias:
            output += self.vars['bias']
        # else:
        #     output = tf.matmul(x, self.geto_vars[self.name + '_weights'], name=name + 'mm')
        #     # bias
        #     if self.bias:
        #         output += self.geto_vars['bias']



        return self.act(output)
