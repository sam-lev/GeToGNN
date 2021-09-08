from __future__ import division
from __future__ import print_function
import numpy as np
import sys

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
        ids, num_samples, _, _, _, _, _, _, _ = inputs
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
    def __init__(self, adj_info, layer_info=None, geto_adj_info=None, adj_probs=None,
                 geto_dims=None, resampling_rate=0, batch_size=None,
                 name='geto_informed', **kwargs):
        super(GeToInformedNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info
        self.geto_adj_info = geto_adj_info
        self.layer_info = layer_info
        self.name = name
        self.previous_sample_ids = None
        print("    * : prob weight size: ", adj_probs.shape)
        #self.adj_probs = tf.placeholder(tf.constant(adj_probs
        #                                            , dtype=tf.float32), trainable=False)
        #self.tensor = tf.placeholder(dtype=tf.float32, shape=(self.adj_probs.shape[1]))
        #self.hidden_geto_samples = tf.placeholder(dtype=tf.float32, shape=batch_size)
        self.batch_size = batch_size
        self.resampling_rate = resampling_rate

    def _call(self, inputs):
        ids, num_samples, geto_ids, geto_elms, geto_dims, support_size, hop, layer_info, batch_size  = inputs

        num_sample_list = [layer_i.num_samples for layer_i in layer_info]
        # layers = range(len(num_samples))



        print("    * : GETIDS ARE NONE", geto_ids )
        self.geto_dims = geto_dims#geto_elms[0].shape[1]

        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids, name = self.name)
        adj_lists_T = tf.transpose(adj_lists)##
        indices = tf.range(start=0, limit=tf.shape(adj_lists_T)[0], dtype=tf.int32)
        #adj_lists_T = tf.gather(adj_lists_T, indices)
        #adj_lists = tf.transpose(adj_lists_T)
        total_num_adj_samples = tf.shape(adj_lists_T)[0] ##
        total_num_adj_samples = total_num_adj_samples + self.resampling_rate * total_num_adj_samples

        geto_adj_lists = tf.nn.embedding_lookup(self.geto_adj_info, geto_ids, name=self.name+'geto')
        geto_adj_lists_T = tf.transpose(geto_adj_lists)
        geto_adj_lists_T_nbr = tf.gather(geto_adj_lists_T, indices)
        geto_adj_lists_nbr = tf.transpose(geto_adj_lists_T_nbr)

        geto_adj_lists_all = tf.slice(geto_adj_lists_nbr, [0, 0], [-1, num_samples], name=self.name + 'geto')


        geto_sample_all = []
        geto_sample_all.append(tf.reshape(geto_adj_lists_all, [support_size * batch_size, ]))
        hidden_geto_all = [tf.nn.embedding_lookup(geto_elms,
                                                  node_samples) for node_samples in geto_sample_all]


        hidden_geto_prob = []#self.reduce_pca(hidden_geto_samples)

        neigh_geto_dims = [batch_size * support_size,#support_size,
                           #num_sample_list[len(num_sample_list) - hop - 1],
                           geto_dims]
        check = 0
        for neigh_geto in hidden_geto_all:
            getodims = tf.shape(neigh_geto)
            tf.print("    *: getodims",getodims, sys.stdout)
            tf.print("    *: neighbor: ",neigh_geto,sys.stdout)

            geto_batch_size = getodims[0]
            num_nbr_geto = getodims[1]
            # [nodes * sampled neighbors] x [hidden_dim]

            # neighbor geto mlp
            #geto_dims_in = based on what layer and specified output
            #hidden_geto_reshaped = tf.reshape(neigh_geto,
            #                       (geto_batch_size * num_nbr_geto, self.geto_dims))
            hidden_geto_reshaped = tf.reshape(neigh_geto,
                                       neigh_geto_dims)
            tensor_pca = tf.map_fn(self.reduce_pca, hidden_geto_reshaped)

            if len(hidden_geto_prob) == 0:
                hidden_geto_prob.append(tensor_pca)
                continue

            # compute distance between seld and neighbor pca comps
            dist_2_components = np.sqrt((hidden_geto_prob[0][:,0] - tensor_pca[:,0])**2 +
                                        (hidden_geto_prob[0][:, 1] - tensor_pca[:, 1]) ** 2)
            #if check < 10:
            #    print('    * : ', dist_2_components)
            #    check+=1

            weight_difference = 1.0 - dist_2_components

            hidden_geto_prob.append(weight_difference)



        weighted_sample_indices = tf.random.categorical(tf.math.log([tf.reshape(hidden_geto_prob, [-1])]),
                                                        total_num_adj_samples)

        weighted_adj_lists_T = tf.gather(adj_lists_T, weighted_sample_indices[0]) ##
        weighted_adj_lists = tf.transpose(weighted_adj_lists_T) ##
        weighted_adj_lists = tf.slice(weighted_adj_lists, [0, 0], [-1, num_samples], name=self.name+'geto')

        #geto_adj_lists_T = tf.transpose(geto_adj_lists)
        weighted_geto_adj_lists_T = tf.gather(geto_adj_lists_T, weighted_sample_indices[0]) ##
        weighted_geto_adj_lists = tf.transpose(weighted_geto_adj_lists_T)
        weighted_geto_adj_lists = tf.slice(weighted_geto_adj_lists, [0, 0], [-1, num_samples], name=self.name+'geto')

        return weighted_adj_lists , weighted_geto_adj_lists

    def _map_slice(self, input):
        sigma, geto_dims, n_dimensions = input
        sigma = tf.slice(sigma, [0, 0], [geto_dims, n_dimensions])
        return sigma



    def reduce_pca(self, X):
        keep_info=0
        x_shape = tf.shape(X)
        X_reshaped = tf.reshape(X, (2 , x_shape[0] / 2))# (8 , x_shape[0] / 8))
        singular_values, u, v = tf.linalg.svd(X_reshaped)
        sigma = tf.linalg.tensor_diag(singular_values)

        tf.print(singular_values, output_stream=sys.stdout)
        tf.print(x_shape, output_stream=sys.stdout)

        singular_values_sum = tf.reduce_sum(singular_values, axis=0) #tf.map_fn(sum,singular_values)
        normalized_singular_values = singular_values / singular_values_sum
        # Create the aggregated ladder of kept information per dimension
        ladder = np.cumsum(normalized_singular_values)
        # Get the first index which is above the given information threshold
        #index = next(idx for idx, value in enumerate(ladder) if value >= keep_info) + 1
        n_dimensions = 2#index
        # with self.graph.as_default():
        # Cut out the relevant part from sigma
        slice_input = (sigma, self.geto_dims, n_dimensions)
        # sigma = tf.map_fn(self._map_slice, slice_input)
        slice_dim = tf.shape(X_reshaped)[0]
        sigma = tf.slice(sigma, [0, 0], [slice_dim, n_dimensions])
        # PCA
        pca = tf.matmul(u, sigma)

        return pca
