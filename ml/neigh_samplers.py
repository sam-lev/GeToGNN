from __future__ import division
from __future__ import print_function
import numpy as np
import sys
from .layers import Layer
import tensorflow as tf

from ml.utils import pout

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
        pout([" WEIGHTED MESSAGE PASSING "])
        self.summary_writer = tf.compat.v1.summary.FileWriter('./log-dir/debug/')
        super(GeToInformedNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info
        self.geto_adj_info = geto_adj_info
        self.layer_info = layer_info
        self.name = name
        self.previous_sample_ids = None
        pout(["    * : prob weight size: ", adj_probs.shape])

        #self.adj_probs = tf.placeholder(tf.constant(adj_probs
        #                                            , dtype=tf.float32), trainable=False)
        #self.tensor = tf.placeholder(dtype=tf.float32, shape=(self.adj_probs.shape[1]))
        #self.hidden_geto_samples = tf.placeholder(dtype=tf.float32, shape=batch_size)
        self.batch_size = batch_size
        self.resampling_rate = resampling_rate

        self.check = 0

    def _call(self, inputs):
        ids, num_samples, geto_ids, geto_elms, geto_dims, support_size, hop, layer_info, batch_size  = inputs

        num_sample_list = [layer_i.num_samples for layer_i in layer_info]
        # layers = range(len(num_samples))



        #b print("    * : GETIDS ARE NONE", geto_ids )
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


        geto_sampled_all = []
        geto_sampled_all.append(tf.reshape(geto_adj_lists_all, [support_size * batch_size, ]))
        hidden_geto_all = [tf.nn.embedding_lookup(geto_elms,
                                                  node_samples) for node_samples in geto_sampled_all]#sample_all]

        target_nodes = tf.nn.embedding_lookup(geto_elms, ids, name='self_weighted_agg'+self.name)
        target_nodes = tf.expand_dims(target_nodes, axis=1)
        hop_target = [target_nodes]

        hidden_geto_prob = []#self.reduce_pca(hidden_geto_samples)
        # batch_size, num_neighbors, self.hidden_dim
        #neigh_geto_dims = [batch_size*, num_samples,
        #                   #num_sample_ list[len(num_sample_list) - hop - 1],
        #                   geto_dims]

        # self_shape = tf.shape(target_nodes)
        # target_shape = [batch_size, self_shape[0],geto_dims]
        #f.reshape(target_nodes, target_shape)#

        pout(["added support", support_size,"num samples", num_samples])
        hop = 0
        for hidden_neighbor in hidden_geto_all:
            neigh_shape = tf.shape(hidden_neighbor)
            neigh_geto_dims = [batch_size*support_size , geto_dims]

            target_node = target_nodes[0]#hop_target[hop]
            hop+=1


            # tensor_pca = tf.map_fn(self.reduce_pca, hidden_geto_reshaped)
            # dist_2_components = np.sqrt((hidden_geto_prob[0][:,0] - tensor_pca[:,0])**2 +
            #                             (hidden_geto_prob[0][:, 1] - tensor_pca[:, 1]) ** 2)
            hidden_neighbor_reshaped = tf.reshape(hidden_neighbor, neigh_geto_dims)
            # target_shape = [batch_size, num_samples, geto_dims]
            # target_nodes = tf.reshape(target_nodes, target_shape)#
            # target_nodes = tf.concat([target_nodes,target_nodes],axis=1)
            #hop_target.append(target_nodes)#hidden_neighbor_reshaped)

            # geto_reshaped = tf.reshape(hidden_geto_all,#neigh_geto,
            #                             neigh_geto_dims)
            #tf.reduce_mean(
            dist_2_components =  tf.sqrt(tf.reduce_sum(tf.square(
                tf.subtract(target_node, hidden_neighbor_reshaped, name='subtract1')
                    # tf.reduce_sum(
                    #     tf.concat([self.target_node_geto , tf.multiply(geto_reshaped,-1.0)],axis=2),
                    #     axis=2, keepdims=True)
                ),
                    axis=1)#,keepdims=True)
                ,name='dist_embedd')#, axis=1)

            # dist_2_components = tf.map_fn(fn=lambda n: tf.reduce_mean(
            #     tf.sqrt(tf.reduce_sum(tf.square(self.target_node_geto - n), 1))),
            #                               elems=geto_reshaped, parallel_iterations=100)



            #dist_2_components = self.euclideanMeanDistance(self.target_node_geto, geto_reshaped)
            # f
            # self.check < 10:
            # neigh_shape = tf.shape(geto_reshaped)
            # dist_summary = tf.summary.tensor_summary('dist comp', dist_2_components)
            # self_shape = tf.shape(self.target_node_geto)
            # neigh_summary = tf.summary.tensor_summary('shape neigh', neigh_shape)
            # self_summary = tf.summary.tensor_summary('shape self', self_shape)
            pout(["dist comp"])#, dist_2_components.eval()])
            tf.print(dist_2_components,output_stream=sys.stdout)  # , [dist_2_components])#, message=" distance comp: ")
            # # pout(['neigh sahape',neigh_shape.numpy()])
            # tf.print(neigh_shape)  # ,[neigh_shape])#,message=" neighbor shape: ")
            # # pout(['self shaope',self_shape.numpy()])
            # tf.print(self_shape)  # ,[self_shape])#,message=" self shape: ")
            # self.summary_writer.add_summary(dist_summary)
            # self.summary_writer.add_summary(neigh_summary)  # ,self.check)
            self.check += 1


            #weight_difference = 1.0 - dist_2_components

            hidden_geto_prob.append(dist_2_components)#weight_difference)


        # [batch_size, num_samples]
        # which should be [batch_size, num_neighbors]
        weighted_sample_indices = tf.random.categorical(tf.math.log([tf.reshape(hidden_geto_prob, [-1])]),
                                                        total_num_adj_samples)
        # w_summary = tf.summary.tensor_summary('weighted indices', weighted_sample_indices)


        # self.summary_writer.add_summary(w_summary)#, self.check)
        # weighted_sample_indices_T = tf.transpose(weighted_sample_indices)

        #pout(["weighted_indices", weighted_sample_indices.numpy().shape])
        #pout(["weighted indices transpose", weighted_sample_indices_T])
        # if self.check < 10:
        #     #pout(['weighted indices', weighted_sample_indices.numpy()])
        #     tf.print(weighted_sample_indices)#, [weighted_sample_indices])#, message=" weighted probs indices :")

        # !!!!!!!!!!!!!!!!!!!!!!!!!! ???????????????
        # weighted_adj_lists_T = tf.gather(adj_lists_T, weighted_sample_indices[0]) ##

        weighted_adj_lists_T = tf.gather(adj_lists_T, weighted_sample_indices[0])#_T)#[0])  ##
        weighted_adj_lists = tf.transpose(weighted_adj_lists_T) ##
        weighted_adj_lists = tf.slice(weighted_adj_lists, [0, 0], [-1, num_samples], name=self.name+'geto')

        # !!!!!!!!!!!!!!!!!!!!!! ??????????????????????????
        weighted_geto_adj_lists_T = tf.gather(geto_adj_lists_T, weighted_sample_indices[0])#_T)#[0]) ##
        weighted_geto_adj_lists = tf.transpose(weighted_geto_adj_lists_T)
        weighted_geto_adj_lists = tf.slice(weighted_geto_adj_lists, [0, 0], [-1, num_samples], name=self.name+'geto')

        return weighted_adj_lists , weighted_geto_adj_lists

    def _map_slice(self, input):
        sigma, geto_dims, n_dimensions = input
        sigma = tf.slice(sigma, [0, 0], [geto_dims, n_dimensions])
        return sigma

    def euclideanMeanDistance(self, y):
        x = self.target_node_geto
        dist = tf.sqrt(tf.reduce_sum(tf.square(x - y), 1))
        return tf.reduce_mean(dist)

    def euclideanDistance(self, x, y):
        dist = tf.sqrt(tf.reduce_sum(tf.square(x - y), 1))
        return dist

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
