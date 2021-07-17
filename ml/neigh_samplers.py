from __future__ import division
from __future__ import print_function

from .layers import Layer

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS


"""
Classes that are used to sample node neighborhoods
"""

class UniformNeighborSampler(Layer):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """
    def __init__(self, adj_info, geto_adj_info=None, name='neighbor_sampler', **kwargs):
        super(UniformNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info
        self.geto_adj_info = geto_adj_info
        self.name = name

    def _call(self, inputs):
        ids, num_samples = inputs
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids, name = self.name)

        adj_lists_T = tf.transpose(adj_lists)
        indices = tf.range(start=0, limit=tf.shape(adj_lists_T)[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)
        shuffled_adj_lists_T = tf.gather(adj_lists_T, shuffled_indices)
        adj_lists = tf.transpose(shuffled_adj_lists_T)

        #tf.random_shuffle(tf.transpose(adj_lists)), name = self.name)
        adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples], name=self.name)
        if self.geto_adj_info is not None:
            geto_adj_lists = tf.nn.embedding_lookup(self.geto_adj_info, ids, name=self.name+'geto')

            geto_adj_lists_T = tf.transpose(geto_adj_lists)
            shuffled_geto_adj_lists_T = tf.gather(geto_adj_lists_T, shuffled_indices)
            geto_adj_lists = tf.transpose(shuffled_geto_adj_lists_T)

            geto_adj_lists = tf.slice(geto_adj_lists, [0, 0], [-1, num_samples], name=self.name+'geto')
            return adj_lists , geto_adj_lists
        return adj_lists, self.geto_adj_info


#
# sampling distribution weighted by geometric-topological attributes
# geo-topo informed sampling as apposed to i.i.d.
class GeToInformedNeighborSampler(Layer):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """
    def __init__(self, adj_info, geto_adj_info=None, adj_probs=None, resampling_rate=0,
                 name='geto_informed', **kwargs):
        super(GeToInformedNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info
        self.geto_adj_info = geto_adj_info
        self.name = name
        self.previous_sample_ids = None
        print("    * : prob weight size: ", adj_probs.shape)
        self.adj_probs = tf.Variable(tf.constant(adj_probs
                                                    , dtype=tf.float32), trainable=False)
        self.resampling_rate = resampling_rate

    def _call(self, inputs):
        ids, num_samples = inputs

        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids, name = self.name)

        adj_lists_T = tf.transpose(adj_lists)##
        total_num_adj_samples = tf.shape(adj_lists_T)[0] ##
        total_num_adj_samples = total_num_adj_samples + self.resampling_rate * total_num_adj_samples

        geto_adj_lists = tf.nn.embedding_lookup(self.geto_adj_info, ids, name=self.name+'geto')

        geto_adj_lists_T = tf.transpose(geto_adj_lists)##

        hidden_geto_prob = tf.nn.embedding_lookup(self.adj_probs, geto_adj_lists)

        weighted_sample_indices = tf.random.categorical(tf.math.log([tf.reshape(hidden_geto_prob, [-1])]),
                                                        total_num_adj_samples)

        weighted_adj_lists_T = tf.gather(adj_lists_T, weighted_sample_indices[0]) ##
        weighted_adj_lists = tf.transpose(weighted_adj_lists_T) ##
        weighted_adj_lists = tf.slice(weighted_adj_lists, [0, 0], [-1, num_samples], name=self.name)


        weighted_geto_adj_lists_T = tf.gather(geto_adj_lists_T, weighted_sample_indices[0]) ##
        weighted_geto_adj_lists = tf.transpose(weighted_geto_adj_lists_T)
        weighted_geto_adj_lists = tf.slice(weighted_geto_adj_lists, [0, 0], [-1, num_samples], name=self.name+'geto')

        return weighted_adj_lists , weighted_geto_adj_lists