from collections import namedtuple

import tensorflow as tf
import math
import numpy as np

from .layers import *
from .metrics import *
from ml.utils import pout,

from .prediction import BipartiteEdgePredLayer
from .aggregators import MeanAggregator, MaxPoolingAggregator, \
    MeanPoolingAggregator, SeqAggregator, GCNAggregator, TwoMaxLayerPoolingAggregator,\
    GeToMeanPoolAggregator

flags = tf.app.flags
FLAGS = flags.FLAGS

# DISCLAIMER:
# Boilerplate parts of this code file were originally forked from
# https://github.com/tkipf/gcn
# which itself was very inspired by the keras package

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'model_size', 'jumping_knowledge', 'concat',
                          'jump_type', 'hiddin_dim_1','hiddin_dim_2','geto_loss'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.sub_loss = 0
        self.weight_loss = 0
        self.geto_loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None
        self.geto_opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)
        self.geto_opt_op = self.optimizer.minimize(self.geto_loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    """ A standard multi-layer perceptron """
    def __init__(self, placeholders, dims, categorical=True, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.dims = dims
        self.input_dim = dims[0]
        self.output_dim = dims[-1]
        self.placeholders = placeholders
        self.categorical = categorical

        self.inputs = placeholders['features']
        self.labels = placeholders['labels']

        #lr = tf.get_variable('learning_rate', initializer=FLAGS.learning_rate, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        if self.categorical:
            self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                    self.placeholders['labels_mask'])
        # L2
        else:
            diff = self.labels - self.outputs
            self.loss += tf.reduce_sum(tf.sqrt(tf.reduce_sum(diff * diff, axis=1)))

    def _accuracy(self):
        if self.categorical:
            self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                    self.placeholders['labels_mask'])

    def _add_layer(self, in_dim, out_dim):
        l = Dense(input_dim=in_dim,
                         output_dim=out_dim,
                         act=tf.nn.relu,
                         dropout=self.placeholders['dropout'],
                         sparse_inputs=False,
                         logging=self.logging)
        return l

    def _add_transition(self, in_dim, out_dim):
        l = Dense(input_dim=in_dim,
                         output_dim=out_dim,
                         act=lambda x: x,
                         dropout=self.placeholders['dropout'],
                         logging=self.logging)
        #l = tf.nn.pool('pool', l)
        return l

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=self.dims[1],
                                 act=tf.nn.relu,
                                 dropout=self.placeholders['dropout'],
                                 sparse_inputs=False,
                                 logging=self.logging))
        
        for i in range(self.depth - 2):
            l = self._add_layer(self.dims[1], self.dims[1])
            self.layers.append(l)
        #l = add_transition(self.dims[1], self.dims[1]) #perform pooling ect
        

        self.layers.append(Dense(input_dim=self.dims[1],
                                 output_dim=self.output_dim,
                                 act=lambda x: x,
                                 dropout=self.placeholders['dropout'],
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)

class GeneralizedModel(Model):
    """
    Base class for models that aren't constructed from traditional, sequential layers.
    Subclasses must set self.outputs in _build method

    (Removes the layers idiom from build method of the Model class)
    """

    def __init__(self, **kwargs):
        super(GeneralizedModel, self).__init__(**kwargs)
        

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)
        self.geto_opt_op = self.optimizer.minimize(self.geto_loss)

# SAGEInfo is a namedtuple that specifies the parameters 
# of the recursive GraphSAGE layers
SAGEInfo = namedtuple("SAGEInfo",
    ['layer_name', # name of the layer (to get feature embedding etc.)
     'neigh_sampler', # callable neigh_sampler constructor
     'num_samples',
     'output_dim' # the output (i.e., hidden) dimension
    ])

class SampleAndAggregate(GeneralizedModel):
    """
    Base implementation of unsupervised GraphSAGE
    """

    def __init__(self, placeholders, features, adj, degrees,
                 layer_infos, geto_adj_info = None, depth = 1,
                 geto_elements=None, geto_weights = None,
                 concat=True, jumping_knowledge=False, jump_type=None,
                 multilevel_concat = False,
                 aggregator_type="mean",
            model_size="small", identity_dim=0, geto_loss=False,
                 hidden_dim_1_agg=None, hidden_dim_2_agg=None,
                 hidden_dim_1=None, hidden_dim_2=None,
            **kwargs):
        '''
        Args:
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features. 
                        NOTE: Pass a None object to train in featureless mode (identity features for nodes)!
            - adj: Numpy array with adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees. 
            - layer_infos: List of SAGEInfo namedtuples that describe the parameters of all 
                   the recursive layers. See SAGEInfo definition above.
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
            - identity_dim: Set to positive int to use identity features (slow and cannot generalize, but better accuracy)
        '''
        super(SampleAndAggregate, self).__init__(**kwargs)
        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "seq":
            self.aggregator_cls = SeqAggregator
        elif aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
            self.aggregator_cls.jumping_knowledge =  jumping_knowledge
        elif aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "gcn":
            self.aggregator_cls = GCNAggregator
        elif aggregator_type == "getomaxpool":
            self.aggregator_cls = GeToMeanPoolAggregator
            self.aggregator_cls.jumping_knowledge = jumping_knowledge
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)

        self.aggregator_type = aggregator_type

        self.hidden_geto_agg = "hidden" in aggregator_type or "edge" in aggregator_type

        if hidden_dim_1_agg is not None:
            #print(">>> hidden dim 1 ", hidden_dim_1_agg,)
            self.aggregator_cls.hidden_dim_1 = hidden_dim_1_agg
        if hidden_dim_2_agg is not None:
            self.aggregator_cls.hidden_dim_2 = hidden_dim_2_agg

        # get info from placeholders...
        self.inputs1 = placeholders["batch1"]
        self.inputs2 = placeholders["batch2"]

        self.subcomplex_weights = placeholders['subcomplex_weight']
        self.sub_batch0 = placeholders['sub_batch0']
        # self.sub_batch1 = placeholders['sub_batch1']
        self.model_size = model_size
        self.adj_info = adj

        # For geometric / topologically informed aggregation
        if geto_elements is not None and geto_adj_info is not None:
            self.geto_elements = tf.Variable(tf.constant(geto_elements
                                                         ,dtype=tf.float32), trainable=False)
            self.geto_adj_info = geto_adj_info
            self.hidden_geto_dict = {}
        else:
            self.geto_elements = geto_elements
            self.geto_adj_info = geto_adj_info
            self.hidden_geto_dict = None
        self.getoinputs1 = placeholders["getobatch1"] if self.geto_elements is not None else None
        self.getoinputs2 = placeholders["getobatch2"] if self.geto_elements is not None else None
        self.sub_getoinputs1 = placeholders["sub_getobatch"] if self.geto_elements is not None else None
        self.geto_loss = geto_loss

        #for weighted loss or aggregation
        if geto_weights is not None:
            self.geto_weights = tf.Variable(tf.constant(geto_weights
                                                        ,dtype=tf.float32), trainable=False)
        if identity_dim > 0:
           self.embeds = tf.get_variable("node_embeddings", [adj.get_shape().as_list()[0], identity_dim])
        else:
           self.embeds = None
        if features is None: 
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            self.features = self.embeds
        else:
            self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
            if not self.embeds is None:
                self.features = tf.concat([self.embeds, self.features], axis=1)
        self.degrees = degrees
        self.concat = concat
        self.multilevel_concat = multilevel_concat
        self.jumping_knowledge = jumping_knowledge
        self.hidden_dim_1 = hidden_dim_1_agg
        self.hidden_dim_2 = hidden_dim_2_agg
        self.jump_type = jump_type
        self.depth = depth
        
        self.dims = [(0 if features is None else features.shape[1]) + identity_dim]

        self.dims_geto_elms = [(0 if geto_elements is None else geto_elements.shape[1]) + identity_dim]

        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.dims_geto_elms.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])



        self.batch_size = placeholders["batch_size"]
        self.sub_batch0_size = placeholders["sub_batch0_size"]
        # self.sub_batch1_size = placeholders["sub_batch1_size"]
        self.scale_graph = 0

        self.placeholders = placeholders
        self.layer_infos = layer_infos

        self.sampler_type = layer_infos[0].neigh_sampler.name

        #lr = tf.get_variable('learning_rate', initializer=FLAGS.learning_rate, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()


    def sample(self, inputs=None, layer_infos=None,
               sup_batch_size=None, geto_inputs=None,
               geto_elms=None, geto_dims=None,
               sub_samples=False,

               shuffled_idx = None,

               sub_ids = [0], sub_inputs=None):
        """ Sample neighbors to be the supportive fields for multi-layer convolutions.

        Args:
            inputs: batch inputs
            sup_batch_size: the number of inputs (different for batch inputs and negative samples).
        """
        
        if sup_batch_size is None:
            sup_batch_size = self.batch_size

        sub_batch0_size = self.sub_batch0_size
        # sub_batch1_size = self.sub_batch1_size

        def sup_only():
            return inputs

        def levelset():
            return sub_inputs

        sub_sample_dict = {}
        for sub_idx in sub_ids:
            sb_name = 'sub_batch'+str(sub_idx)
            sb_sz_name = sb_name+'_size'
            sb_lb_name = sb_name+'_labels'
            sub_sample_dict[sb_name] = [sub_inputs[sb_name]]
            # self.subbatch_dict[sb_sz_name] = placeholders[sb_sz_name]
            # self.subbatch_dict[sb_lb_name] = placeholders[sb_lb_name]



        # sub_samples = sub_sample_dict['sub_batch0']

        # 0th index is target node, nbrs appended from sampler
        # must concat to preserve label ordering
        if inputs is not None:
            samples = [inputs]#[tf.concat([inputs,sub_inputs],axis=0)]




        def mixed_level_sets():
            sub_out =  [tf.concat([inputs,sub_inputs],axis=0)]
            return sub_out

        def sub_only():
            return sub_inputs

        geto_samples = None if self.geto_adj_info is None else [geto_inputs]
        # size of convolution support at each layer per node
        sub_support_size = 1
        sup_support_size = 1
        support_sizes = [sub_support_size]
        sup_support_sizes = [sup_support_size]
        for k in range(len(layer_infos)):
            t = len(layer_infos) - k - 1
            sub_support_size *= layer_infos[t].num_samples
            sup_support_size = sup_support_size * layer_infos[t].num_samples

            sampler = layer_infos[t].neigh_sampler

            full_bs_w_support = sup_support_size * sup_batch_size

            sample_input = (samples[k], layer_infos[t].num_samples,
                            None, None , None, None, None, full_bs_w_support, k,
                            sub_ids, sub_sample_dict, shuffled_idx)#, sup_samples[k])

            # if self.geto_adj_info is None:

            node, _, suBnodes, shuffled_idx = sampler(sample_input)


            # suBnodes = tf.reshape(suBnodes, [full_bs_w_support, ])

            # samp = tf.reshape(node, [full_bs_w_support, ])
            samples.append(node)
            # sub_samples.append(suBnodes)

            sup_support_sizes.append(sup_support_size)
            support_sizes.append(sub_support_size)
            # else:
            #     # would need to adjust geto return if using geto #!
            #     node, node_geto, suBnodes, shuffled_idx  = sampler(sample_input)
            #     geto_samples.append(tf.reshape(node_geto, [sub_support_size * sup_batch_size, ]))
            #     samples.append(tf.reshape(node, [sub_support_size * sup_batch_size, ]))
            #     sub_samples.append(tf.reshape(suBnodes, [sub_support_size * sub_batch0_size, ]))
            #     support_sizes.append(sub_support_size)
        return samples, support_sizes, geto_samples, sub_sample_dict, sup_support_sizes, geto_samples

    def aggregate(self, samples, input_features, dims, num_samples, support_sizes,
                  batch_size=None, aggregators=None, name=None, concat=False, model_size="small",
                  jumping_knowledge=False, jump_type=None, hidden_dim_1=None, hidden_dim_2=None,
                  hidden_geto=False, geto_dims=None, geto_elms=None, getosamples=None, geto_loss=False,
                  sub_samples_dict=None, geto_subsamples=None,
                  sub_batch0_size=None, geto_sub_batch0_size=None,
                  sub_support_size=None, sub_ids = [0]
                  ):
        """ At each layer, aggregate hidden representations of neighbors to compute the hidden representations 
            at next layer.
        Args:
            samples: a list of samples of variable hops away for convolving at each layer of the
                network. Length is the number of layers + 1. Each is a vector of node indices.
            input_features: the input features for each sample of various hops away.
            dims: a list of dimensions of the hidden representations from the input layer to the
                final layer. Length is the number of layers + 1.
            num_samples: list of number of samples for each layer.
            support_sizes: the number of nodes to gather information from for each layer.
            batch_size: the number of inputs (different for batch inputs and negative samples).
        Returns:
            The hidden representation at the final layer for all nodes in batch
        """
        use_geto = False
        if batch_size is None:
            batch_size = self.batch_size
        batch_size = self.placeholders['batch_size']
        sub_batch0_size = self.placeholders['sub_batch0_size']
        # sub_batch2_size = self.placeholders['sub_batch2_size']

        self.super_present, self.sub_present = (batch_size != 0 and batch_size is not None), (sub_batch0_size != 0 and sub_batch0_size is not None)


        if geto_dims is None:
            geto_dims = [0] * len(num_samples)
        # length: number of layers + 1

        hidden = [tf.nn.embedding_lookup(input_features, node_samples) for node_samples in samples]
        hidden_geto_elm = None if geto_elms is None else [tf.nn.embedding_lookup(geto_elms, node_samples) for node_samples in getosamples]


        embedd_weight = self.placeholders['subcomplex_weight']
        hidden_sub = [tf.nn.embedding_lookup(input_features, node_samples) for node_samples in sub_samples_dict]

        hidden_sub_dict = {}
        for sub_idx in sub_ids:
            sb_name = 'sub_batch'+str(sub_idx)
            sb_sz_name = sb_name+'_size'
            sb_lb_name = sb_name+'_labels'
            hidden_sub_dict[sb_name] = [tf.nn.embedding_lookup(input_features, node_samples) for node_samples in sub_samples_dict[sb_name]]
            # self.subbatch_dict[sb_sz_name] = placeholders[sb_sz_name]
            # self.subbatch_dict[sb_lb_name] = placeholders[sb_lb_name]


        new_agg = aggregators is None

        name = self.aggregator_type

        if new_agg:
            aggregators = []
        for layer in range(len(num_samples)):
            if new_agg:
                dim_mult = 2 if concat and (layer != 0) else 1
                # dim_mult = 2 if concat and (layer != 0) else 1
                dim_mult_geto = 2 if concat and (layer != 0) and self.hidden_geto_agg else 1
                geto_dims_in = layer if self.hidden_geto_agg else 0

                if self.multilevel_concat:
                    dim_mult = dim_mult * 2 if (layer != 0) else dim_mult
                    dim_mult_geto = dim_mult_geto * 2 if (layer != 0) and self.hidden_geto_agg else dim_mult_geto

                input_dim = dim_mult * dims[layer]

                if self.concat and self.multilevel_concat:
                    input_dim_sub = input_dim//2 if (layer != 0) else input_dim
                else:
                    input_dim_sub = input_dim


                batch_diff = 1
                # aggregator at current layer
                if layer == len(num_samples) - 1:
                    aggregator = self.aggregator_cls(input_dim=input_dim,
                                                     input_dim_sub=input_dim_sub,
                                                     output_dim=dims[layer + 1],
                                                     sub_ids = sub_ids,
                                                     act=lambda x : x,
                                                     dropout=self.placeholders['dropout'],
                                                     subcomplex_weight = embedd_weight,
                                                     jumping_knowledge=jumping_knowledge,
                                                     jump_type=jump_type,
                                                     multilevel_concat = self.multilevel_concat,
                                                     hidden_dim_1 = hidden_dim_1, hidden_dim_2 = hidden_dim_2,
                                                     name=name+str(layer), concat=concat, model_size=model_size,
                                                     geto_loss=geto_loss,
                                                     geto_dims_in= dim_mult_geto * geto_dims[geto_dims_in],
                                                     geto_dims_out = geto_dims[layer+1],
                                                     bs = batch_size,
                                                     sub_bs = sub_batch0_size,
                                                     diff_bs = batch_diff)
                else:
                    aggregator = self.aggregator_cls(input_dim=input_dim,
                                                     input_dim_sub=input_dim_sub,
                                                     output_dim=dims[layer + 1],
                                                     sub_ids=sub_ids,
                                                     dropout=self.placeholders['dropout'], geto_loss=geto_loss,
                                                     subcomplex_weight=embedd_weight,
                                                     jumping_knowledge=jumping_knowledge,
                                                     jump_type=jump_type,
                                                     multilevel_concat = self.multilevel_concat,
                                                     hidden_dim_1 = hidden_dim_1, hidden_dim_2 = hidden_dim_2,
                                                     name=name+str(layer), concat=concat, model_size=model_size,
                                                     geto_dims_in= dim_mult_geto * geto_dims[geto_dims_in],
                                                     geto_dims_out = geto_dims[layer+1],
                                                     bs=batch_size,
                                                     sub_bs=sub_batch0_size,
                                                     diff_bs = batch_diff
                                                     )
                aggregators.append(aggregator)
            else:
                aggregator = aggregators[layer]
            # hidden representation at current layer for all support nodes that are various hops away
            next_hidden = []
            next_hidden_sub = []
            next_geto_hidden = []
            next_hidden_sub_dict = {}
            for sub_idx in sub_ids:
                sb_name = 'sub_batch' + str(sub_idx)
                sb_sz_name = sb_name + '_size'
                sb_lb_name = sb_name + '_labels'
                next_hidden_sub_dict[sb_name] = []
                # as layer increases, the number of support nodes needed decreases
            for hop in range(len(num_samples) - layer):
                dim_mult = 2 if concat and (layer != 0) else 1
                dim_mult_geto = 2 if concat and (layer != 0) and self.hidden_geto_agg else 1
                geto_dims_in = layer if self.hidden_geto_agg else 0

                if self.multilevel_concat:
                    dim_mult = dim_mult * 2 if (layer != 0) else dim_mult
                    dim_mult_geto = dim_mult_geto * 2 if (layer != 0) and self.hidden_geto_agg else dim_mult_geto

                if self.concat and self.multilevel_concat:
                    dim_sub_neigh = dim_mult//2 * dims[layer] if (layer != 0) else dim_mult * dims[layer]
                else:
                    dim_sub_neigh = dim_mult * dims[layer]

                def super_neighs():
                    self.scale_graph += 1
                    # pout(['super pesent agg'])
                    include_sub_size = sub_batch0_size #if self.scale_graph == 1 or (layer == 0) else 0
                    init_cat_embeds = 1 if self.scale_graph == 1 or (layer == 0) else 0
                    neigh_dims = [batch_size * support_sizes[hop],
                                  num_samples[len(num_samples) - hop - 1],
                                  dim_mult * dims[layer]]
                    hidden_next_hop = hidden[hop + 1] #if self.scale_graph == 1 or (layer == 0) else hidden[hop + 1]
                    hidden_target   =  hidden[hop]
                    neigh_hop_reshaped = tf.reshape(hidden_next_hop,
                                                                   neigh_dims)
                    neigh_feat_reshaped = (hidden_target, neigh_hop_reshaped)
                    #self.super_present = True
                    return neigh_feat_reshaped
                def super_absent():
                    # pout(['super absent agg'])
                    include_sub_size = sub_batch0_size #if self.scale_graph == 1 or (layer == 0) else 0
                    dims_agg_target = tf.shape(hidden[hop])
                    neigh_feat_reshaped = (tf.ones(dims_agg_target),
                                           tf.ones((batch_size * support_sizes[hop],
                                                    num_samples[len(num_samples) - hop - 1],
                                                    dim_mult * dims[layer])))  # (None, None)#
                    return neigh_feat_reshaped


                #if sublevel_present:

                def sub_neighs():
                    # pout(['sub pesent agg'])
                    sub_neigh_dims = [batch_size * sub_support_size[hop],
                                      num_samples[len(num_samples) - hop - 1],
                                      dim_sub_neigh]
                    sub_neigh_feat_reshaped = (hidden_sub[hop], tf.reshape(hidden_sub[hop + 1],
                                                                           sub_neigh_dims))
                    #self.sub_present = True
                    return sub_neigh_feat_reshaped
                def sub_absent():
                    # pout(['sub absent agg'])
                    sub_neigh_feat_reshaped = (tf.ones(tf.shape(hidden_sub[hop])),
                                               tf.ones((batch_size * sub_support_size[hop],#sub_batch0_size * sub_support_size[hop],
                                               num_samples[len(num_samples) - hop - 1],
                                               dim_sub_neigh)))
                    return sub_neigh_feat_reshaped




                # neigh_geto_dims = [batch_size * support_sizes[hop] ,
                #                    num_samples[len(num_samples) - hop - 1],
                #                     dim_mult_geto * geto_dims[geto_dims_in]]




                neigh_feat_reshaped = super_neighs()
                # tf.cond(tf.not_equal(batch_size, 0),  # tf.constant(0)),
                #                                  super_neighs,
                #                                  super_absent)




                sub_neigh_feat_reshaped = sub_neighs()
                # tf.cond(tf.not_equal(sub_batch0_size, 0),
                #                                   sub_neighs,
                #                                   sub_absent)


                # agg_feat_reshaped = neigh_feat_reshaped

                if geto_elms is None:
                    #if sub_complex_present and super_level_present:
                    node_and_neighbors = (# agg_feat_reshaped[0],agg_feat_reshaped[1],
                                          neigh_feat_reshaped[0], neigh_feat_reshaped[1],
                                          sub_neigh_feat_reshaped[0], sub_neigh_feat_reshaped[1])
                else:
                    if self.hidden_geto_agg:
                        node_and_neighbors = 1
                    else:
                        node_and_neighbors = (# agg_feat_reshaped[0],agg_feat_reshaped[1],
                                              neigh_feat_reshaped[0], neigh_feat_reshaped[1],
                                              sub_neigh_feat_reshaped[0], sub_neigh_feat_reshaped[1])
                #) #tf.reshape(hidden_geto_elm[hop + 1], neigh_geto_dims)
                if geto_elms is None or not self.hidden_geto_agg:
                    h, sub_h = aggregator(node_and_neighbors)
                    next_hidden.append(h)
                    next_hidden_sub.append(sub_h)
                    # next_hidden_agg.append(agg_h)
                else:
                    h, geto_h = aggregator(node_and_neighbors)
                    next_hidden.append(h)
                    next_geto_hidden.append(geto_h)
                    # next_hidden_agg.append(agg_h)
            hidden = next_hidden
            hidden_sub = next_hidden_sub
            # hidden_agg = next_hidden_agg
            hidden_geto_elm = next_geto_hidden if geto_elms is not None and self.hidden_geto_agg else hidden_geto_elm

        if self.jumping_knowledge and not new_agg:
            rev_hidden = hidden#_agg#next_hidden
            rev_hidden.reverse()
            #print(">>> rev hidden ", rev_hidden)
            h_jump = rev_hidden[0]
            # for idx, l_vec in enumerate(rev_hidden[1:]):
            #     # if even jump every other so as to get to layer 1 and 2 hops
            #     # if odd do every third so as to get farthest and last two
            #     skip = idx+1+(len(hidden)%2)
            #     if idx +1 < len(hidden)-1:# and skip%2 != 0:
            #         #h_jump = tf.concat([h_jump, hidden[idx + 1]], axis=1)
            #         if  'maxpool' in self.jump_type:
            #             h_next = tf.reduce_max(hidden[idx + 1], axis=1)
            #         elif 'meanpool' in self.jump_type:
            #             h_next = tf.reduce_mean(hidden[idx + 1], axis=1)
            #         else:
            #             h_next = hidden[idx + 1]
            #         from_h_next = h_next#tf.matmul(h_next, self.vars['neigh_weights'])
            #         from_h = h_jump#tf.matmul(h_jump, self.vars["self_weights"])
            #
            #         if 'cat' in self.jump_type:
            #             h_jump = tf.concat([from_h, from_h_next], axis=1)
            #         else:
            #             h_jump = tf.add_n([from_h, from_h_next])
            from_h_target = hidden[0]
            from_h_last   = hidden[-1]

            if 'cat' in self.jump_type:
                h_jump = tf.concat([from_h_target, from_h_last], axis=1)
            else:
                h_jump = tf.add_n([from_h_target, from_h_last])


            hidden[0] = h_jump

        def _return_hidden_super():
            return hidden[0]
        def _return_hidden_sub():
            return hidden_sub[0]
        def _return_empty():
            return tf.zeros((0))

        ret_hidden_super = _return_hidden_super()
        # tf.cond(tf.not_equal(batch_size, 0),
        #                               _return_hidden_super,
        #                               _return_hidden_super)#_return_empty)

        ret_hidden_sub = _return_hidden_sub()
        # tf.cond(tf.not_equal(sub_batch0_size, 0),
        #                                   _return_hidden_sub,
        #                                   _return_hidden_sub)#_return_empty)

        if geto_elms is not None:
            if self.sampler_type == 'geto_informed':
                self.hidden_geto_dict = {geto_id:hidden_rep for geto_id, hidden_rep in zip(getosamples,hidden_geto_elm)}
            return hidden[0], aggregators, hidden_geto_elm[0]
        else:
            return ret_hidden_super, ret_hidden_sub, aggregators, hidden_geto_elm

    def _build(self):
        labels = tf.reshape(
                tf.cast(self.placeholders['batch2'], dtype=tf.int64),
                [self.batch_size, 1])
        self.neg_samples, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels,
            num_true=1,
            num_sampled=FLAGS.neg_sample_size,
            unique=False,
            range_max=len(self.degrees),
            distortion=0.75,
            unigrams=self.degrees.tolist()))

        dim_mult = 2 if self.concat else 1
        dim_mult_geto =  2 if self.concat else 1

        if self.multilevel_concat:
            dim_mult = dim_mult * 2
            dim_mult_geto = dim_mult_geto * 2
        #print("    * : GETO IS NONE", self.geto_elements is None)
        # perform "convolution"
        samples1, support_sizes1, getosamples1 = self.sample(self.inputs1,
                                                             self.layer_infos,
                                                             geto_inputs=self.getoinputs1,
                                                             geto_elms=self.geto_elements, #list(self.hidden_geto_dict.values())
                                                             geto_dims=dim_mult * self.dims_geto_elms[-1])
        samples2, support_sizes2, getosamples2 = self.sample(self.inputs2,
                                                             self.layer_infos,
                                                             geto_inputs=self.getoinputs2,
                                                             geto_elms=self.geto_elements,#list(self.hidden_geto_dict.values())
                                                             geto_dims=dim_mult * self.dims_geto_elms[-1])
        sub_samples1, sub_support_sizes1, sub_getosamples1 = self.sample(self.sub_batch1,
                                                                         self.layer_infos,
                                                                         geto_inputs=self.sub_getoinputs1,
                                                                         geto_elms=self.geto_elements,
                                                                         sup_batch_size=self.sub_batch0_size,
                                                                         geto_dims=dim_mult * self.dims_geto_elms[-1])

        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]

        self.outputs1, self.aggregators, self.getooutputs1 = self.aggregate(samples1, [self.features],
                                                                            self.dims, num_samples,
                                                                            support_sizes1,
                                                                            sub_samples=sub_samples1,
                                                                            sub_support_size=sub_support_sizes1,
                                                                            sub_batch0_size=self.sub_batch0_size,
                                                                            geto_subsamples=sub_getosamples1,
                                                                            geto_dims=self.dims_geto_elms,
                                                                            concat=self.concat,
                                                                            model_size=self.model_size,
                                                                            jumping_knowledge=self.jumping_knowledge,
                                                                            jump_type=self.jump_type,
                                                                            hidden_dim_1=self.hidden_dim_1,
                                                                            hidden_dim_2=self.hidden_dim_2,
                                                                            geto_elms=self.geto_elements,
                                                                            getosamples=getosamples1)

        # self.sub_outputs1, self.sub_aggregators, self.sub_getooutputs1 = self.aggregate(sub_samples1, [self.features],
        #                                                                     self.dims, num_samples,
        #                                                                     sub_support_sizes1,
        #                                                                     geto_dims=self.dims_geto_elms,
        #                                                                     concat=self.concat,
        #                                                                     model_size=self.model_size,
        #                                                                     jumping_knowledge=self.jumping_knowledge,
        #                                                                     jump_type=self.jump_type,
        #                                                                     hidden_dim_1=self.hidden_dim_1,
        #                                                                     hidden_dim_2=self.hidden_dim_2,
        #                                                                     geto_elms=self.geto_elements,
        #                                                                     getosamples=sub_getosamples1)

        self.outputs2, _, self.getooutputs2 = self.aggregate(samples2, [self.features], self.dims, num_samples,
                                                             support_sizes2, aggregators=self.aggregators,
                                                             concat=self.concat,
                                                             model_size=self.model_size, geto_dims=self.dims_geto_elms,
                                                             jumping_knowledge=self.jumping_knowledge,
                                                             jump_type=self.jump_type,
                                                             hidden_dim_1=self.hidden_dim_1,
                                                             hidden_dim_2=self.hidden_dim_2,
                                                             geto_elms=self.geto_elements, getosamples=getosamples2)

        neg_samples, neg_support_sizes, neg_getosamples = self.sample(self.neg_samples, self.layer_infos,
                                                                      FLAGS.neg_sample_size)
        self.neg_outputs, _, self.neg_geto_outputs = self.aggregate(neg_samples, [self.features], self.dims,
                                                                    num_samples,
                                                                    neg_support_sizes, batch_size=FLAGS.neg_sample_size,
                                                                    aggregators=self.aggregators,
                                                                    concat=self.concat, model_size=self.model_size,
                                                                    geto_dims=self.dims_geto_elms,
                                                                    jumping_knowledge=self.jumping_knowledge,
                                                                    jump_type=self.jump_type,
                                                                    hidden_dim_1=self.hidden_dim_1,
                                                                    hidden_dim_2=self.hidden_dim_2,
                                                                    geto_elms=self.geto_elements,
                                                                    getosamples=neg_getosamples)


        self.link_pred_layer = BipartiteEdgePredLayer(dim_mult * self.dims[-1],
                                                      dim_mult * self.dims[-1], self.placeholders, act=tf.nn.sigmoid,
                                                      bilinear_weights=False,
                                                      name='edge_predict')

        self.outputs1 = tf.nn.l2_normalize(self.outputs1, 1)
        # self.sub_outputs1 = tf.nn.l2_normalize(self.sub_outputs1, 1)
        self.outputs2 = tf.nn.l2_normalize(self.outputs2, 1)
        self.neg_outputs = tf.nn.l2_normalize(self.neg_outputs, 1)

        if self.geto_loss and self.hidden_geto_agg:
            self.geto_link_pred_layer = BipartiteEdgePredLayer(dim_mult_geto * self.dims_geto_elms[-1],
                                                          dim_mult_geto * self.dims_geto_elms[-1], self.placeholders,
                                                          act=tf.nn.sigmoid,
                                                          bilinear_weights=False,
                                                          name='edge_predict',
                                                               use_geto=False)

            self.getooutputs1 = tf.nn.l2_normalize(self.getooutputs1, 1)
            self.getooutputs2 = tf.nn.l2_normalize(self.getooutputs2, 1)
            self.neg_geto_outputs = tf.nn.l2_normalize(self.neg_geto_outputs, 1)

    def build(self):
        self._build()

        # TF graph management
        self._loss()
        self._accuracy()
        self.loss = self.loss / tf.cast(self.batch_size, tf.float32)
        self.sub_loss = self.sub_loss / tf.cast(self.batch_size, tf.float32)
        self.weight_loss = self.weight_loss / tf.cast(self.batch_size, tf.float32)
        #self.loss = tf.divide(self.loss, tf.cast(self.batch_size, tf.float32), name="loss" )
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) 
                for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)

        if False:#self.geto_elements is not None and self.hidden_geto_agg:
            self.geto_loss = self.geto_loss / tf.cast(self.batch_size, tf.float32)
            # self.loss = tf.divide(self.loss, tf.cast(self.batch_size, tf.float32), name="loss" )
            geto_grads_and_vars = self.optimizer.compute_gradients(self.geto_loss)
            clipped_geto_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                      for grad, var in geto_grads_and_vars]
            self.geto_grad, _ = clipped_geto_grads_and_vars[0]
            self.geto_opt_op = self.optimizer.apply_gradients(clipped_geto_grads_and_vars)

    def _loss(self):
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        self.loss += self.link_pred_layer.loss(self.outputs1, self.outputs2, self.neg_outputs)

        if False:#self.geto_elements is not None and self.hidden_geto_agg:
            for aggregator in self.aggregators:
                for var in aggregator.vars.values():
                    self.geto_loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

            self.geto_loss += self.link_pred_layer.loss(self.getooutputs1,
                                                        self.getooutputs2, self.neg_geto_outputs)
            tf.summary.scalar('geto_loss', self.geto_loss)

        tf.summary.scalar('loss', self.loss)



    def _accuracy(self):
        # shape: [batch_size]
        aff = self.link_pred_layer.affinity(self.outputs1, self.outputs2)
        # shape : [batch_size x num_neg_samples]
        self.neg_aff = self.link_pred_layer.neg_cost(self.outputs1, self.neg_outputs)
        self.neg_aff = tf.reshape(self.neg_aff, [self.batch_size, FLAGS.neg_sample_size])
        _aff = tf.expand_dims(aff, axis=1)
        self.aff_all = tf.concat(axis=1, values=[self.neg_aff, _aff])
        size = tf.shape(self.aff_all)[1]
        _, indices_of_ranks = tf.nn.top_k(self.aff_all, k=size)
        _, self.ranks = tf.nn.top_k(-indices_of_ranks, k=size)
        self.mrr = tf.reduce_mean(tf.div(1.0, tf.cast(self.ranks[:, -1] + 1, tf.float32)))
        tf.summary.scalar('mrr', self.mrr)


class Node2VecModel(GeneralizedModel):
    def __init__(self, placeholders, dict_size, degrees, name=None,
                 nodevec_dim=50, lr=0.001, **kwargs):
        """ Simple version of Node2Vec/DeepWalk algorithm.

        Args:
            dict_size: the total number of nodes.
            degrees: numpy array of node degrees, ordered as in the data's id_map
            nodevec_dim: dimension of the vector representation of node.
            lr: learning rate of optimizer.
        """

        super(Node2VecModel, self).__init__(**kwargs)

        self.placeholders = placeholders
        self.degrees = degrees
        self.inputs1 = placeholders["batch1"]
        self.inputs2 = placeholders["batch2"]

        self.batch_size = placeholders['batch_size']
        self.hidden_dim = nodevec_dim

        # following the tensorflow word2vec tutorial
        self.target_embeds = tf.Variable(
                tf.random_uniform([dict_size, nodevec_dim], -1, 1),
                name="target_embeds")
        self.context_embeds = tf.Variable(
                tf.truncated_normal([dict_size, nodevec_dim],
                stddev=1.0 / math.sqrt(nodevec_dim)),
                name="context_embeds")
        self.context_bias = tf.Variable(
                tf.zeros([dict_size]),
                name="context_bias")

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)

        self.build()

    def _build(self):
        labels = tf.reshape(
                tf.cast(self.placeholders['batch2'], dtype=tf.int64),
                [self.batch_size, 1])
        self.neg_samples, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels,
            num_true=1,
            num_sampled=FLAGS.neg_sample_size,
            unique=True,
            range_max=len(self.degrees),
            distortion=0.75,
            unigrams=self.degrees.tolist()))

        self.outputs1 = tf.nn.embedding_lookup(self.target_embeds, self.inputs1)
        self.outputs2 = tf.nn.embedding_lookup(self.context_embeds, self.inputs2)
        self.outputs2_bias = tf.nn.embedding_lookup(self.context_bias, self.inputs2)
        self.neg_outputs = tf.nn.embedding_lookup(self.context_embeds, self.neg_samples)
        self.neg_outputs_bias = tf.nn.embedding_lookup(self.context_bias, self.neg_samples)

        self.link_pred_layer = BipartiteEdgePredLayer(self.hidden_dim, self.hidden_dim,
                self.placeholders, bilinear_weights=False)

    def build(self):
        self._build()
        # TF graph management
        self._loss()
        self._minimize()
        self._accuracy()

    def _minimize(self):
        self.opt_op = self.optimizer.minimize(self.loss)

    def _loss(self):
        aff = tf.reduce_sum(tf.multiply(self.outputs1, self.outputs2), 1) + self.outputs2_bias
        neg_aff = tf.matmul(self.outputs1, tf.transpose(self.neg_outputs)) + self.neg_outputs_bias
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(aff), logits=aff)
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(neg_aff), logits=neg_aff)
        loss = tf.reduce_sum(true_xent) + tf.reduce_sum(negative_xent)
        self.loss = loss / tf.cast(self.batch_size, tf.float32)
        tf.summary.scalar('loss', self.loss)
        
    def _accuracy(self):
        # shape: [batch_size]
        aff = self.link_pred_layer.affinity(self.outputs1, self.outputs2)
       # shape : [batch_size x num_neg_samples]
        self.neg_aff = self.link_pred_layer.neg_cost(self.outputs1, self.neg_outputs)
        self.neg_aff = tf.reshape(self.neg_aff, [self.batch_size, FLAGS.neg_sample_size])
        _aff = tf.expand_dims(aff, axis=1)
        self.aff_all = tf.concat(axis=1, values=[self.neg_aff, _aff])
        size = tf.shape(self.aff_all)[1]
        _, indices_of_ranks = tf.nn.top_k(self.aff_all, k=size)
        _, self.ranks = tf.nn.top_k(-indices_of_ranks, k=size)
        self.mrr = tf.reduce_mean(tf.div(1.0, tf.cast(self.ranks[:, -1] + 1, tf.float32)))
        tf.summary.scalar('mrr', self.mrr)
