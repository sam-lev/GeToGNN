import tensorflow as tf
import numpy as np
from .layers import Layer, Dense
from .inits import glorot, zeros, normxav_lrhype
from ml.ops import euclideanDistance
from ml.utils import pout

class MeanAggregator(Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,geto_loss=False,
            dropout=0., bias=False, act=tf.nn.relu, 
            name=None, concat=False, **kwargs):
        super(MeanAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([neigh_input_dim, output_dim],
                                                        name='neigh_weights')
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                                        name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs

        neigh_vecs = tf.nn.dropout(neigh_vecs,keep_prob=1-self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, keep_prob=1-self.dropout)
        neigh_means = tf.reduce_mean(neigh_vecs, axis=1)
       
        # [nodes] x [out_dim]
        from_neighs = tf.matmul(neigh_means, self.vars['neigh_weights'])

        from_self = tf.matmul(self_vecs, self.vars["self_weights"])
         
        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)

class GCNAggregator(Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    Same matmul parameters are used self vector and neighbor vectors.
    """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,geto_loss=False,
            dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        super(GCNAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['weights'] = glorot([neigh_input_dim, output_dim],
                                                        name='neigh_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs

        neigh_vecs = tf.nn.dropout(neigh_vecs, keep_prob=1-self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, keep_prob=1-self.dropout)
        means = tf.reduce_mean(tf.concat([neigh_vecs, 
            tf.expand_dims(self_vecs, axis=1)], axis=1), axis=1)

        # [nodes] x [out_dim]
        output = tf.matmul(means, self.vars['weights'])

        # bias
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)

class HiddenGeToMaxPoolAggregator(Layer):
    """ Aggregates via max-pooling over MLP functions.
    """

    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
                 dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False,
                 hidden_dim_1=None, hidden_dim_2=None, jumping_knowledge=False,
                 geto_dims_in=None, geto_dims_out=None, geto_vec_dim=None,geto_loss=False,
                 **kwargs):
        super(HiddenGeToMaxPoolAggregator, self).__init__(**kwargs)

        print("    * : Aggregator is HiddenGeToMaxpool")

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat
        self.dim_mult = 2 if concat else 1
        self.jumping_knowledge = jumping_knowledge

        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2

        self.pool_then_combine = False

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if self.hidden_dim_1 is not None:
            hidden_dim = self.hidden_dim = self.hidden_dim_1
        else:
            if model_size == "small":
                hidden_dim = self.hidden_dim = 201 * 3  # 256
            elif model_size == "big":
                hidden_dim = self.hidden_dim = 201 * 4  # 1024

        self.mlp_layers = []
        self.mlp_layers.append(Dense(input_dim=neigh_input_dim,
                                     output_dim=hidden_dim,
                                     act=tf.nn.relu,
                                     dropout=dropout,
                                     sparse_inputs=False,
                                     name='feat_'+name,
                                     logging=self.logging,
                                     use_geto=False))
        self.geto_mlp_layers = []
        self.geto_mlp_layers.append(Dense(input_dim=geto_dims_in,
                                     output_dim=hidden_dim,
                                     act=tf.nn.relu,
                                     dropout=dropout,
                                     sparse_inputs=False,
                                     name='geto_'+name,
                                     logging=self.logging,
                                     use_geto=geto_loss))

        with tf.variable_scope(self.name + name + '_vars'):
            combine_out = output_dim if self.pool_then_combine else hidden_dim
            self.vars['neigh_feat_weights'] = glorot([hidden_dim,  combine_out],
                                                name='neigh_feat_weights')

            self.vars['self_feat_weights'] = glorot([input_dim, output_dim],
                                               name='self_feat_weights')
            if not geto_loss:
                self.vars['self_geto_weights'] = glorot([geto_dims_in, output_dim],
                                                   name='self_geto_weights')
            else:
                self.geto_vars['self_geto_weights'] = glorot([geto_dims_in, output_dim],
                                                        name='self_geto_weights')
            self.vars['self_geto_feat_weights'] = glorot([output_dim, output_dim],
                                                          name='self_geto_feat_weights')

            combine_out = output_dim if self.pool_then_combine else hidden_dim
            self.vars['neigh_geto_weights'] = glorot([hidden_dim, combine_out],
                                               name='neigh_geto_weights')
            if not geto_loss:
                self.vars['neigh_geto_only'] = glorot([hidden_dim, output_dim],
                                                         name='neigh_geto_only')
            else:
                self.vars['neigh_geto_only'] = glorot([hidden_dim, output_dim],
                                                      name='neigh_geto_only')
            ##self.vars['neigh_feat_only'] = glorot([hidden_dim, output_dim],
            ##                                      name='neigh_feat_only')
            combine_in = output_dim if self.pool_then_combine else hidden_dim
            self.vars['neigh_geto_feat_weights'] = glorot([combine_in,output_dim],
                                               name='neigh_geto_feat_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim
        self.geto_dims_in=geto_dims_in
        self.geto_loss = geto_loss


    def _call(self, inputs):
        self_vecs, neigh_vecs, self_geto_elms, neigh_geto_elms = inputs
        neigh_h = neigh_vecs
        neigh_geto = neigh_geto_elms

        dims = tf.shape(neigh_h)
        getodims = tf.shape(neigh_geto)
        selfshape = tf.shape(self_geto_elms)
        batch_size = dims[0]
        num_neighbors = dims[1]

        geto_batch_size = getodims[0]
        num_nbr_geto = getodims[1]
        # [nodes * sampled neighbors] x [hidden_dim]



        # neighbor geto mlp
        geto_reshaped = tf.reshape(neigh_geto,
                                   (geto_batch_size * num_nbr_geto, self.geto_dims_in))
        for l in self.geto_mlp_layers:
            geto_reshaped = l(geto_reshaped)




        if self.pool_then_combine:
            neigh_geto_only = tf.reshape(geto_reshaped,
                                         (batch_size, num_neighbors, self.hidden_dim))
            from_neigh_geto_pooled = tf.reduce_max(neigh_geto_only, axis=1)
            neigh_geto = from_neigh_geto_pooled
        else:
            neigh_geto = geto_reshaped


        if not self.geto_loss:
            from_neigh_geto = tf.matmul(neigh_geto, self.vars['neigh_geto_weights'])
        else:
            from_neigh_geto = tf.matmul(neigh_geto, self.geto_vars['neigh_geto_weights'])

        if not self.pool_then_combine:
            from_neigh_geto = tf.reshape(from_neigh_geto,
                                         (batch_size, num_neighbors, self.hidden_dim))

        # neighbor fetures mlp
        h_reshaped = tf.reshape(neigh_h,
                                (batch_size * num_neighbors, self.neigh_input_dim))
        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)


        if self.pool_then_combine:
            neigh_h = tf.reshape(h_reshaped,
                                 (batch_size, num_neighbors, self.hidden_dim))
            neigh_h = tf.reduce_max(neigh_h, axis=1)
        else:
            neigh_h = h_reshaped

        from_neigh_feat = tf.matmul(neigh_h, self.vars['neigh_feat_weights'])

        if not self.pool_then_combine:
            from_neigh_feat = tf.reshape(from_neigh_feat,
                                 (batch_size, num_neighbors, self.hidden_dim))

        # element wise multiplication of nbr's geto an featurer embeddings
        ##neigh_geto_feat_reshaped = tf.multiply(from_neigh_geto, from_neigh_feat)
        ##neigh_geto_feat_reshaped = tf.concat([neigh_geto, neigh_h], axis=1)
        neigh_geto_feat = tf.multiply(from_neigh_geto, from_neigh_feat)#, transpose_a=True)

        if not self.pool_then_combine:
            neigh_geto_feat = tf.reduce_max(neigh_geto_feat, axis=1)


        from_neigh_geto_feat = tf.matmul(neigh_geto_feat,
                                                 self.vars['neigh_geto_feat_weights'])

        ##neigh_feat_pooled = tf.reduce_mean(neigh_h, axis=1)
        ##from_neigh_feat_only = tf.matmul(neigh_feat_pooled, self.vars['neigh_feat_only'])
        if not self.geto_loss:
            from_self_geto = tf.matmul(self_geto_elms, self.vars['self_geto_weights'])
        else:
            from_self_geto = tf.matmul(self_geto_elms, self.geto_vars['self_geto_weights'])

        from_self_feat = tf.matmul(self_vecs, self.vars["self_feat_weights"])

        from_self_geto_feat = tf.multiply(from_self_geto , from_self_feat)

        ##from_self_geto_feat = tf.matmul(self_geto_feat, self.vars['self_geto_feat_weights'])

        neigh_geto_only = tf.reshape(geto_reshaped,
                                     (batch_size, num_neighbors, self.hidden_dim))
        neigh_geto_pooled = tf.reduce_max(neigh_geto_only, axis=1)
        from_neigh_geto_pooled = tf.matmul(neigh_geto_pooled, self.vars['neigh_geto_only'])


        if not self.concat:
            node_output = tf.add_n([from_self_geto_feat , from_neigh_geto_feat])
            geto_output = tf.add_n([from_self_geto, from_neigh_geto_pooled])
        else:
            node_output = tf.concat([from_self_geto_feat , from_neigh_geto_feat], axis=1)
            geto_output = tf.concat([from_self_geto, from_neigh_geto_pooled], axis=1)
            #[from_self, from_nbr_geto_augmented_feature], axis=1)

        # bias
        if self.bias:
            node_output += self.vars['bias']
            geto_output += self.vars['bias']

        return self.act(node_output) , self.act(geto_output)

class HiddenGeToMeanPoolAggregator(Layer):
    """ Aggregates via max-pooling over MLP functions.
    """

    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
                 dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False,
                 hidden_dim_1=None, hidden_dim_2=None, jumping_knowledge=False,
                 geto_dims_in=None, geto_dims_out=None, geto_vec_dim=None,geto_loss=False,
                 **kwargs):
        super(HiddenGeToMeanPoolAggregator, self).__init__(**kwargs)

        print("    * : Aggregator is HiddenGeToMeanpool")

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat
        self.dim_mult = 2 if concat else 1
        self.jumping_knowledge = jumping_knowledge

        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2

        self.pool_then_combine = False

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if self.hidden_dim_1 is not None:
            hidden_dim = self.hidden_dim = self.hidden_dim_1
        else:
            if model_size == "small":
                hidden_dim = self.hidden_dim = 201 * 3  # 256
            elif model_size == "big":
                hidden_dim = self.hidden_dim = 201 * 4  # 1024

        self.mlp_layers = []
        self.mlp_layers.append(Dense(input_dim=neigh_input_dim,
                                     output_dim=hidden_dim,
                                     act=tf.nn.relu,
                                     dropout=dropout,
                                     sparse_inputs=False,
                                     name='feat_'+name,
                                     logging=self.logging,
                                     use_geto=False))
        self.geto_mlp_layers = []
        self.geto_mlp_layers.append(Dense(input_dim=geto_dims_in,
                                     output_dim=hidden_dim,
                                     act=tf.nn.relu,
                                     dropout=dropout,
                                     sparse_inputs=False,
                                     name='geto_'+name,
                                     logging=self.logging,
                                     use_geto=geto_loss))

        with tf.variable_scope(self.name + name + '_vars'):
            combine_out = output_dim if self.pool_then_combine else hidden_dim
            self.vars['neigh_feat_weights'] = glorot([hidden_dim,  combine_out],
                                                name='neigh_feat_weights')

            self.vars['self_feat_weights'] = glorot([input_dim, output_dim],
                                               name='self_feat_weights')
            if not geto_loss:
                self.vars['self_geto_weights'] = glorot([geto_dims_in, output_dim],
                                                   name='self_geto_weights')
            else:
                self.geto_vars['self_geto_weights'] = glorot([geto_dims_in, output_dim],
                                                        name='self_geto_weights')
            self.vars['self_geto_feat_weights'] = glorot([output_dim, output_dim],
                                                          name='self_geto_feat_weights')

            combine_out = output_dim if self.pool_then_combine else hidden_dim
            self.vars['neigh_geto_weights'] = glorot([hidden_dim, combine_out],
                                               name='neigh_geto_weights')
            if not geto_loss:
                self.vars['neigh_geto_only'] = glorot([hidden_dim, output_dim],
                                                         name='neigh_geto_only')
            else:
                self.vars['neigh_geto_only'] = glorot([hidden_dim, output_dim],
                                                      name='neigh_geto_only')
            ##self.vars['neigh_feat_only'] = glorot([hidden_dim, output_dim],
            ##                                      name='neigh_feat_only')
            combine_in = output_dim if self.pool_then_combine else hidden_dim
            self.vars['neigh_geto_feat_weights'] = glorot([combine_in,output_dim],
                                               name='neigh_geto_feat_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim
        self.geto_dims_in=geto_dims_in
        self.geto_loss = geto_loss


    def _call(self, inputs):
        self_vecs, neigh_vecs, self_geto_elms, neigh_geto_elms = inputs
        neigh_h = neigh_vecs
        neigh_geto = neigh_geto_elms

        dims = tf.shape(neigh_h)
        getodims = tf.shape(neigh_geto)
        selfshape = tf.shape(self_geto_elms)
        batch_size = dims[0]
        num_neighbors = dims[1]

        geto_batch_size = getodims[0]
        num_nbr_geto = getodims[1]
        # [nodes * sampled neighbors] x [hidden_dim]



        # neighbor geto mlp
        geto_reshaped = tf.reshape(neigh_geto,
                                   (geto_batch_size * num_nbr_geto, self.geto_dims_in))
        for l in self.geto_mlp_layers:
            geto_reshaped = l(geto_reshaped)




        if self.pool_then_combine:
            neigh_geto_only = tf.reshape(geto_reshaped,
                                         (batch_size, num_neighbors, self.hidden_dim))
            from_neigh_geto_pooled = tf.reduce_mean(neigh_geto_only, axis=1)
            neigh_geto = from_neigh_geto_pooled
        else:
            neigh_geto = geto_reshaped


        if not self.geto_loss:
            from_neigh_geto = tf.matmul(neigh_geto, self.vars['neigh_geto_weights'])
        else:
            from_neigh_geto = tf.matmul(neigh_geto, self.geto_vars['neigh_geto_weights'])

        if not self.pool_then_combine:
            from_neigh_geto = tf.reshape(from_neigh_geto,
                                         (batch_size, num_neighbors, self.hidden_dim))

        # neighbor fetures mlp
        h_reshaped = tf.reshape(neigh_h,
                                (batch_size * num_neighbors, self.neigh_input_dim))
        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)


        if self.pool_then_combine:
            neigh_h = tf.reshape(h_reshaped,
                                 (batch_size, num_neighbors, self.hidden_dim))
            neigh_h = tf.reduce_mean(neigh_h, axis=1)
        else:
            neigh_h = h_reshaped

        from_neigh_feat = tf.matmul(neigh_h, self.vars['neigh_feat_weights'])

        if not self.pool_then_combine:
            from_neigh_feat = tf.reshape(from_neigh_feat,
                                 (batch_size, num_neighbors, self.hidden_dim))

        # element wise multiplication of nbr's geto an featurer embeddings
        ##neigh_geto_feat_reshaped = tf.multiply(from_neigh_geto, from_neigh_feat)
        ##neigh_geto_feat_reshaped = tf.concat([neigh_geto, neigh_h], axis=1)
        neigh_geto_feat = tf.multiply(from_neigh_geto, from_neigh_feat)#, transpose_a=True)

        if not self.pool_then_combine:
            neigh_geto_feat = tf.reduce_mean(neigh_geto_feat, axis=1)


        from_neigh_geto_feat = tf.matmul(neigh_geto_feat,
                                                 self.vars['neigh_geto_feat_weights'])

        ##neigh_feat_pooled = tf.reduce_mean(neigh_h, axis=1)
        ##from_neigh_feat_only = tf.matmul(neigh_feat_pooled, self.vars['neigh_feat_only'])
        if not self.geto_loss:
            from_self_geto = tf.matmul(self_geto_elms, self.vars['self_geto_weights'])
        else:
            from_self_geto = tf.matmul(self_geto_elms, self.geto_vars['self_geto_weights'])

        from_self_feat = tf.matmul(self_vecs, self.vars["self_feat_weights"])

        from_self_geto_feat = tf.multiply(from_self_geto , from_self_feat)

        ##from_self_geto_feat = tf.matmul(self_geto_feat, self.vars['self_geto_feat_weights'])

        neigh_geto_only = tf.reshape(geto_reshaped,
                                     (batch_size, num_neighbors, self.hidden_dim))
        neigh_geto_pooled = tf.reduce_mean(neigh_geto_only, axis=1)
        from_neigh_geto_pooled = tf.matmul(neigh_geto_pooled, self.vars['neigh_geto_only'])


        if not self.concat:
            node_output = tf.add_n([from_self_geto_feat , from_neigh_geto_feat])
            geto_output = tf.add_n([from_self_geto, from_neigh_geto_pooled])
        else:
            node_output = tf.concat([from_self_geto_feat , from_neigh_geto_feat], axis=1)
            geto_output = tf.concat([from_self_geto, from_neigh_geto_pooled], axis=1)
            #[from_self, from_nbr_geto_augmented_feature], axis=1)

        # bias
        if self.bias:
            node_output += self.vars['bias']
            geto_output += self.vars['bias']

        return self.act(node_output) , self.act(geto_output)

class GeToEdgeAggregator(Layer):
    """ Aggregates via max-pooling over MLP functions.
    """

    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
                 dropout=0., bias=False, act=tf.nn.elu, pool=tf.reduce_max, pool_geto = tf.reduce_max,
                 name=None, concat=False,
                 hidden_dim_1=None, hidden_dim_2=None, jumping_knowledge=False,
                 geto_dims_in=None, geto_dims_out=None, geto_vec_dim=None,geto_loss=False,
                 **kwargs):
        super(GeToEdgeAggregator, self).__init__(**kwargs)

        print("    * : Aggregator is GetoEdge Aggregator")

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.pool = pool
        self.pool_geto = pool_geto
        self.concat = concat
        self.dim_mult = 2 if concat else 1
        self.jumping_knowledge = jumping_knowledge

        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if self.hidden_dim_1 is not None:
            hidden_dim = self.hidden_dim = self.hidden_dim_1
        else:
            if model_size == "small":
                hidden_dim = self.hidden_dim = 201 * 3  # 256
            elif model_size == "big":
                hidden_dim = self.hidden_dim = 201 * 4  # 1024

        self.mlp_layers = []
        self.mlp_layers.append(Dense(input_dim=neigh_input_dim,
                                     output_dim=hidden_dim,
                                     act=tf.nn.elu,#relu,
                                     dropout=dropout,
                                     sparse_inputs=False,
                                     name='feat_'+name,
                                     logging=self.logging,
                                     use_geto=False))
        self.geto_mlp_layers = []
        self.geto_mlp_layers.append(Dense(input_dim=geto_dims_in,
                                     output_dim=hidden_dim,
                                     act=tf.nn.elu,#softmax,#elu,
                                     dropout=dropout,
                                     sparse_inputs=False,
                                     name='geto_'+name,
                                     logging=self.logging,
                                     use_geto=geto_loss))

        self.geto_concat_mlp_layers = []
        self.geto_concat_mlp_layers.append(Dense(input_dim=1,#geto_dims_in,
                                          output_dim=hidden_dim,
                                          act=tf.nn.elu,#softmax,
                                          dropout=dropout,
                                          sparse_inputs=False,
                                          name='concat_feats_' + name,
                                          logging=self.logging,
                                          use_geto=geto_loss))

        with tf.variable_scope(self.name + name + '_vars'):

            # self feature weights
            self.vars['self_feat_weights'] = glorot([input_dim,  output_dim],
                                               name='self_feat_weights')
            # self geto weights
            if not geto_loss:
                self.vars['self_geto_weights'] = glorot([geto_dims_in, output_dim],
                                                   name='self_geto_weights')
            else:
                self.geto_vars['self_geto_weights'] = glorot([geto_dims_in, output_dim],
                                                        name='self_geto_weights')

            # combined self/neighbor geto embedding
            self.vars['combined_geto'] = glorot([hidden_dim,  output_dim],
                                                          name='combined_neigh_weights')

            # combined neighbor geto weigthed embedding
            # 2 * out due to concat
            self.vars['weighted_edge'] = glorot([output_dim,output_dim],
                                                name='weighted_edge')

            # neighbor feature weights
            self.vars['neigh_feat_weights'] = glorot([hidden_dim,  output_dim],
                                                     name='neigh_feat_weights')
            self.vars['neigh_geto_weights'] = glorot([hidden_dim, output_dim],
                                               name='neigh_geto_weights')

            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')


        self.edge_embedding_weighted = True

        if self.edge_embedding_weighted:
            pout(["using weighted embedding tensor"])
        else:
            pout(["Using scalar weighted edge aggregation"])


        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim
        self.geto_dims_in=geto_dims_in
        self.geto_loss = False#geto_loss

    def _call(self, inputs):
        self_vecs, neigh_vecs, self_geto_elms, neigh_geto_elms = inputs
        neigh_h = neigh_vecs
        neigh_geto = neigh_geto_elms

        dims = tf.shape(neigh_h)
        getodims = tf.shape(neigh_geto)
        selfshape = tf.shape(self_geto_elms)
        batch_size = dims[0]
        num_neighbors = dims[1]

        geto_batch_size = getodims[0]
        num_nbr_geto = getodims[1]
        # [nodes * sampled neighbors] x [hidden_dim]

        # [batch size * support, num samples, getodims]
        '''geto_embed_concat = tf.concat([neigh_geto,
                                       tf.expand_dims(self_geto_elms, axis=1)], axis=1)#tf.reduce_mean(, axis=1)
        geto_embed_concat_reshaped = tf.reshape(geto_embed_concat,
                                                (batch_size*(num_neighbors + 1) ,
                                                 self.geto_dims_in))'''

        # neighbor geto mlp
        geto_reshaped = tf.reshape(neigh_geto,
                                   (geto_batch_size * num_nbr_geto, self.geto_dims_in))
        for l in self.geto_mlp_layers:
            geto_reshaped = l(geto_reshaped)
        neigh_geto_embeddings  = tf.reshape(geto_reshaped,
                                       (batch_size * num_neighbors, self.hidden_dim))




        # out dim [batch*num_nbr , output_dim]
        if not self.geto_loss:
            from_neigh_geto = tf.matmul(neigh_geto_embeddings , self.vars['neigh_geto_weights'])
        else:
            from_neigh_geto = tf.matmul(neigh_geto_embeddings , self.geto_vars['neigh_geto_weights'])
        
        neigh_geto_embeddings= tf.reshape(from_neigh_geto,(batch_size * num_neighbors,
                                                           self.output_dim))
        neigh_geto_embeddings_hidden = tf.reshape(from_neigh_geto, (batch_size,
                                                             num_neighbors,
                                                             self.output_dim))
        from_neigh_geto_pool = self.pool_geto(neigh_geto_embeddings_hidden, axis=1)


        # Self Geto Embedding
        if not self.geto_loss:
            from_self_geto = tf.matmul(self_geto_elms, self.vars['self_geto_weights'])
        else:
            from_self_geto = tf.matmul(self_geto_elms, self.geto_vars['self_geto_weights'])



        # neighbor fetures mlp
        h_reshaped = tf.reshape(neigh_h,
                                (batch_size * num_neighbors, self.neigh_input_dim))
        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)
        neigh_h = tf.reshape(h_reshaped,
                             (batch_size * num_neighbors, self.hidden_dim))
        from_neigh_feat = tf.matmul(neigh_h, self.vars['neigh_feat_weights'])
        from_neigh_feat_reshaped = tf.reshape(from_neigh_feat,
                                              (batch_size, num_neighbors, self.output_dim))
        from_neigh_feat_pool = self.pool(from_neigh_feat_reshaped,axis=1)

        # Self Feats Embedding
        from_self_feat = tf.matmul(self_vecs, self.vars["self_feat_weights"])




        if self.edge_embedding_weighted:
            from_feat_embedding_weighted = tf.multiply(neigh_geto_embeddings_hidden,
                                                       from_neigh_feat_reshaped)
            neigh_embedding_weighted = tf.reshape(from_feat_embedding_weighted,
                                              (batch_size, num_neighbors, self.output_dim))




        if self.edge_embedding_weighted:

            neigh_feat_embedding_reshaped = tf.reshape(from_neigh_feat,
                                                      (batch_size * num_neighbors,
                                                       # num_neighbors,
                                                       self.output_dim))

            # [ batch_size, num_nbr, getodimm_in ]
            self_replicated = tf.tile(tf.expand_dims(from_self_geto, axis=1),
                                      [1, num_neighbors, 1])
            self_replicated = tf.reshape(self_replicated,
                                         (batch_size * num_neighbors,
                                          self.output_dim))#self.geto_dims_in))
            # self.output_dim))
            def embedding_euclideanDistance( y ):
                # dist = tf.sqrt(tf.reduce_sum(tf.square(x - y), 1))
                dist = tf.sqrt(tf.reduce_sum(tf.square( #tf.subtract(self_replicated, y, name='subtract1')
                    self_replicated - y), axis=1, keepdims=True), name='dist_embedd')
                return dist
            # [batch_size*num_neighbors,1]
            distance_embeddings = embedding_euclideanDistance(neigh_geto_embeddings)#from_geto_embedding_reshaped)
            #tf.map_fn(_euclideanDistance,from_geto_embedding_reshaped)

            for l in self.geto_concat_mlp_layers:
                distance_embeddings = l(distance_embeddings)
            distance_embeddings_h = tf.reshape(distance_embeddings,
                         (batch_size*num_neighbors, self.hidden_dim))
            edge_weights = tf.matmul(distance_embeddings_h, self.vars['combined_geto'], name='edge_weights')

            nbr_weighted_edge_embedding = tf.reshape(edge_weights, (batch_size * num_neighbors,
                                                       self.output_dim))
            # invert distances to inversely weight by distance
            one_tensor = tf.ones(tf.shape(nbr_weighted_edge_embedding))
            from_nbr_edge_weights = tf.divide(one_tensor, nbr_weighted_edge_embedding)
            from_nbr_edge_weights = tf.reshape(from_nbr_edge_weights,
                                               (batch_size, num_neighbors, self.output_dim))
            from_feat_embedding_weighted = tf.multiply(from_nbr_edge_weights,
                                                       neigh_embedding_weighted)#neigh_feat_embedding_reshaped)
            neigh_embedding_weighted = tf.reshape(from_feat_embedding_weighted,
                                                      (batch_size,
                                                       num_neighbors,
                                                       self.output_dim))


        from_embedding_weighted_pooled  = self.pool(neigh_embedding_weighted,
                                                             axis=1)

        # self_replicated = tf.tile(tf.expand_dims(from_self_geto, axis=1),
        #                           [1, num_neighbors, 1])
        # self_replicated = tf.reshape(self_replicated,
        #                              (batch_size * num_neighbors,
        #                               self.output_dim))




        if not self.concat:
            node_output = tf.add_n([from_self_feat,
                                    from_embedding_weighted_pooled], axis=1)
            geto_output = tf.add_n([from_self_geto, from_embedding_weighted_pooled])
        else:
            node_output = tf.concat([from_self_feat,
                                    from_embedding_weighted_pooled],axis=1)
            # node_output = tf.concat([from_self_feat,from_nbr_weighted_embedding],axis=1)#tf.concat([from_self_feat , edge_weighted_nbrs],axis=1)#from_neigh_feat], axis=1)
            # geto_output = self.pool(tf.reshape(tf.concat([self_replicated,neigh_geto_embeddings], axis=1),
            #                          (batch_size , num_neighbors,
            #                           2*self.output_dim)), axis=1)
            geto_output = tf.concat([from_self_geto, from_embedding_weighted_pooled], axis=1)
            #[from_self, from_nbr_geto_augmented_feature], axis=1)
        # if not self.concat:
        #     node_output = tf.add_n([from_self_feat , from_neigh_embedding])
        # else:
        #     node_output = tf.concat([from_self_feat , from_neigh_embedding], axis=1)
        # bias
        if self.bias:
            node_output += self.vars['bias']
            geto_output += self.vars['bias']

        return self.act(node_output) , self.act(geto_output)

class GeToMeanPoolAggregator(Layer):
    """ Aggregates via max-pooling over MLP functions.
    """

    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
                 dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False,
                 hidden_dim_1=None, hidden_dim_2=None, jumping_knowledge=False,geto_loss=False,
                 geto_dims=None, geto_dims_in= None,geto_dims_out = None, **kwargs):
        super(GeToMeanPoolAggregator, self).__init__(**kwargs)

        print("    * : Aggregator is GeToMeanPool")

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat
        self.jumping_knowledge = jumping_knowledge

        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if self.hidden_dim_1 is not None:
            hidden_dim = self.hidden_dim = self.hidden_dim_1
        else:
            if model_size == "small":
                hidden_dim = self.hidden_dim = 201 * 3  # 256
            elif model_size == "big":
                hidden_dim = self.hidden_dim = 201 * 3  # 1024

        self.mlp_layers = []
        self.mlp_layers.append(Dense(input_dim=neigh_input_dim,
                                     output_dim=hidden_dim,
                                     act=tf.nn.relu,
                                     dropout=dropout,
                                     sparse_inputs=False,
                                     logging=self.logging))

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([hidden_dim, output_dim],
                                                name='neigh_weights')

            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                               name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim
        #print("neighbor input dim")
        #print(neigh_input_dim)
        self.geto_dims=geto_dims
        self.geto_dims_in = geto_dims_in

    def _call(self, inputs):
        self_vecs, neigh_vecs, self_geto_elms, neigh_geto_elms = inputs
        neigh_h = neigh_vecs
        neigh_geto = neigh_geto_elms

        dims = tf.shape(neigh_h)
        getodims = tf.shape(neigh_geto)
        geto_bs = getodims[0]
        num_geto_neigh = getodims[1]
        batch_size = dims[0]
        num_neighbors = dims[1]
        # [nodes * sampled neighbors] x [hidden_dim]




        #geto_reshaped = tf.reshape(neigh_geto,
        #                           (batch_size * num_neighbors, self.geto_dims))
        h_reshaped = tf.reshape(neigh_h, (batch_size , num_neighbors, self.neigh_input_dim))

        neigh_geto_reshaped = tf.reshape(neigh_geto, (geto_bs , num_geto_neigh, self.geto_dims_in))


        neigh_geto_weighted = tf.multiply(h_reshaped, neigh_geto_reshaped)

        h_reshaped = tf.reshape(neigh_geto_weighted,
                                (batch_size * num_neighbors, self.neigh_input_dim))

        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)

        h_reshaped = tf.reshape(h_reshaped,
                                (batch_size, num_neighbors, self.hidden_dim))
        neigh_pooled= tf.reduce_mean(h_reshaped, axis=1)


        from_neighs = tf.matmul(neigh_pooled, self.vars['neigh_weights'])

        #neigh_weighted_h = tf.reshape(neigh_weighted, (batch_size, num_neighbors, self.hidden_dim))


        self_geto_weighted = tf.multiply(self_vecs, self_geto_elms)
        #self_geto_weighted = tf.concat([self_vecs, tf.expand_dims(self_geto_pooled,axis=1)], axis=1)
        from_self_geto_weighted = tf.matmul(self_geto_weighted, self.vars["self_weights"])

        if not self.concat:
            output = tf.add_n([from_self_geto_weighted, from_neighs])
        else:
            output = tf.concat([from_self_geto_weighted, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

class GeToMaxPoolAggregator(Layer):
    """ Aggregates via max-pooling over MLP functions.
    """

    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
                 dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False,
                 hidden_dim_1=None, hidden_dim_2=None, jumping_knowledge=False,geto_loss=False,
                 geto_dims=None, geto_dims_in= None,geto_dims_out = None, **kwargs):
        super(GeToMaxPoolAggregator, self).__init__(**kwargs)

        print("    * : Aggregator is GeToMeanPool")

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat
        self.jumping_knowledge = jumping_knowledge

        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if self.hidden_dim_1 is not None:
            hidden_dim = self.hidden_dim = self.hidden_dim_1
        else:
            if model_size == "small":
                hidden_dim = self.hidden_dim = 201 * 3  # 256
            elif model_size == "big":
                hidden_dim = self.hidden_dim = 201 * 3  # 1024

        self.mlp_layers = []
        self.mlp_layers.append(Dense(input_dim=neigh_input_dim,
                                     output_dim=hidden_dim,
                                     act=tf.nn.relu,
                                     dropout=dropout,
                                     sparse_inputs=False,
                                     logging=self.logging))

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([hidden_dim, output_dim],
                                                name='neigh_weights')

            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                               name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim
        print("neighbor input dim")
        print(neigh_input_dim)
        self.geto_dims=geto_dims
        self.geto_dims_in = geto_dims_in

    def _call(self, inputs):
        self_vecs, neigh_vecs, self_geto_elms, neigh_geto_elms = inputs
        neigh_h = neigh_vecs
        neigh_geto = neigh_geto_elms

        dims = tf.shape(neigh_h)
        getodims = tf.shape(neigh_geto)
        geto_bs = getodims[0]
        num_geto_neigh = getodims[1]
        batch_size = dims[0]
        num_neighbors = dims[1]
        # [nodes * sampled neighbors] x [hidden_dim]

        h_reshaped = tf.reshape(neigh_h, (batch_size, num_neighbors, self.neigh_input_dim))

        neigh_geto_reshaped = tf.reshape(neigh_geto, (geto_bs, num_geto_neigh, self.geto_dims_in))

        neigh_geto_weighted = tf.multiply(h_reshaped, neigh_geto_reshaped)

        h_reshaped = tf.reshape(neigh_geto_weighted,
                                (batch_size * num_neighbors, self.neigh_input_dim))

        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)

        h_reshaped = tf.reshape(h_reshaped,
                                (batch_size, num_neighbors, self.hidden_dim))
        neigh_pooled = tf.reduce_max(h_reshaped, axis=1)

        from_neighs = tf.matmul(neigh_pooled, self.vars['neigh_weights'])

        # neigh_weighted_h = tf.reshape(neigh_weighted, (batch_size, num_neighbors, self.hidden_dim))

        self_geto_weighted = tf.multiply(self_vecs, self_geto_elms)
        # self_geto_weighted = tf.concat([self_vecs, tf.expand_dims(self_geto_pooled,axis=1)], axis=1)
        from_self_geto_weighted = tf.matmul(self_geto_weighted, self.vars["self_weights"])

        if not self.concat:
            output = tf.add_n([from_self_geto_weighted, from_neighs])
        else:
            output = tf.concat([from_self_geto_weighted, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class MaxPoolingAggregator(Layer):
    """ Aggregates via max-pooling over MLP functions.
    """

    def __init__(self, input_dim, input_dim_sub, output_dim, model_size="small", neigh_input_dim=None,
                 sub_neigh_input_dim=None,
                 sub_ids=None,
                 dropout=0., bias=False,
                 act=tf.nn.relu,
                 aggregator=tf.add_n,
                 pool=tf.reduce_mean,
                 multilevel_concat = False,
                 name=None, concat=False, hidden_dim_1=None,
                 hidden_dim_2=None, geto_loss=False,
                 jumping_knowledge=False, geto_dims_in=None, geto_dims_out=None,
                 subcomplex_weight=1.0, bs = 0, sub_bs = 0, diff_bs =1,
                 **kwargs):
        super(MaxPoolingAggregator, self).__init__(**kwargs)

        # print("    * : Aggregator is MaxPool")

        self.dropout = dropout
        self.bias = bias
        self.act = act




        self.concat = concat
        self.multilevel_concat = tf.constant(multilevel_concat,tf.bool)

        self.dim_disrepency = tf.logical_and(tf.constant(multilevel_concat,tf.bool),
                                             tf.constant(concat,tf.bool))

        self.jumping_knowledge = jumping_knowledge

        self.subcomplex_weight = subcomplex_weight

        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2

        self.sub_bs = sub_bs
        self.bs     = bs

        self.agg_sub_level_graphs = True
        self.pool                  = pool

        # if multiply change absent sub filler to one
        self.agg = aggregator

        if neigh_input_dim is None:
            neigh_input_dim = input_dim
        if sub_neigh_input_dim is None:
            sub_neigh_input_dim = input_dim_sub

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if self.hidden_dim_1 is not None:
            hidden_dim = self.hidden_dim = self.hidden_dim_1
        else:
            if model_size == "small":
                hidden_dim = self.hidden_dim = 201 * 3  # 256
            elif model_size == "big":
                hidden_dim = self.hidden_dim = 201 * 3  # 1024



        # self.sub_mlp_l1 = Dense(input_dim=sub_neigh_input_dim,
        #                              output_dim=hidden_dim,
        #                              act=tf.nn.relu,
        #                              dropout=dropout,
        #                              sparse_inputs=False,
        #                              logging=self.logging)
        self.mlp_layers = []
        self.mlp_layers.append(Dense(input_dim=neigh_input_dim,
                                     output_dim=hidden_dim,
                                     act=tf.nn.relu,
                                     dropout=dropout,
                                     sparse_inputs=False,
                                     logging=self.logging))
        self.mlp_layers.append(Dense(input_dim=hidden_dim,
                                     output_dim=hidden_dim,
                                     act=tf.nn.relu,
                                     dropout=dropout,
                                     sparse_inputs=False,
                                     logging=self.logging))

        # self.mlp_layers.append(Dense(input_dim=hidden_dim,
        #                              output_dim=hidden_dim,
        #                              act=tf.nn.relu,
        #                              dropout=dropout,
        #                              sparse_inputs=False,
        #                              logging=self.logging))
        # self.mlp_layers.append(Dense(input_dim=hidden_dim,
        #                              output_dim=hidden_dim,
        #                              act=tf.nn.relu,
        #                              dropout=dropout,
        #                              sparse_inputs=False,
        #                              logging=self.logging))



        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([hidden_dim, output_dim],
                                                name='neigh_weights')

            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                               name='self_weights')

            # self.vars['sub_neigh_weights'] = normxav_lrhype([hidden_dim, output_dim], maxval = 1.,
            #                                         name='sub_neigh_weights')
            #
            # self.vars['self_sub_weights'] = normxav_lrhype([input_dim_sub,output_dim],maxval = 1.,
            #                                        name='self_sub_weights')

            # self.vars['sub_neigh_weight_factor'] = normxav_lrhype([1, output_dim],
            #                                                       maxval=100, minval=10,
            #                                                       name='agg_neigh_weights_factor')
            #
            # self.vars['self_sub_weight_factor'] = normxav_lrhype([1, output_dim],
            #                                                      maxval=100, minval=10,
            #                                                      name='agg_self_weights_factor')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim
        self.sub_neigh_input_dim = sub_neigh_input_dim
        # pout(["hidden 1 ", self.hidden_dim_1,'hidden 2', self.hidden_dim_2])


    def _call(self, inputs):

        self_vecs, neigh_vecs, sub_self_vecs, sub_neigh_vecs = inputs

        self.super_present, self.sub_present = self.bs != 0, self.sub_bs != 0

        neigh_h = neigh_vecs

        batch_size = self.bs

        dims = tf.shape(neigh_h)
        batch_size = dims[0]
        num_neighbors = dims[1]

        sub_batch0_size = self.sub_bs
        sub_dims = tf.shape(sub_neigh_vecs)
        sub_batch0_size = sub_dims[0]
        num_sub_neighbors = sub_dims[1]

        def sublevel_embeddings():
            h_sub_reshaped = tf.reshape(sub_neigh_vecs,
                                        (sub_batch0_size * num_sub_neighbors, self.neigh_input_dim))

            # def sub_mlp():
            #     h_sub_reshaped_sub = self.sub_mlp_l1(h_sub_reshaped)
            #     for l in self.mlp_layers[1:]:
            #         h_sub_reshaped_sub = l(h_sub_reshaped_sub)
            #     return h_sub_reshaped_sub
            def mlp():
                h_sub_reshaped_sub = h_sub_reshaped
                for l in self.mlp_layers:
                    h_sub_reshaped_sub = l(h_sub_reshaped_sub)
                return h_sub_reshaped_sub

            h_sub_reshaped = mlp()
            # tf.cond(self.dim_disrepency,
            #                          sub_mlp,
            #                          mlp)
            sub_neigh_h = tf.reshape(h_sub_reshaped,
                                     (sub_batch0_size, num_sub_neighbors, self.hidden_dim))

            sub_neigh_h = self.pool(sub_neigh_h, axis=1)

            from_sub_neighs = tf.matmul(sub_neigh_h, self.vars['neigh_weights'])

            from_self_sub = tf.matmul(sub_self_vecs, self.vars["self_weights"])

            return from_sub_neighs, from_self_sub, sub_neigh_h

        # def sub_absent():
        #     empty_padding = (tf.zeros((sub_batch0_size, self.output_dim)),
        #                      tf.zeros((sub_batch0_size,self.output_dim)),
        #                      tf.zeros((sub_batch0_size,self.hidden_dim)))
        #     return empty_padding


        # from_sub_neighs, from_sub_self = sublevel_embeddings()
        from_sub_neighs, from_sub_self, sub_neigh_h = sublevel_embeddings()
        # tf.cond(tf.not_equal(self.sub_bs, 0),
        #                                          sublevel_embeddings,
        #                                          sub_absent)



        def concat_sub():
            return tf.concat([from_sub_self, from_sub_neighs],axis=1)
        def add_sub():
            return tf.add_n([from_sub_self, from_sub_neighs])

        sublevel_embedding_h = tf.cond(tf.constant(self.concat,tf.bool),
                                     concat_sub,
                                     add_sub)



        def levelset_weighted_superlevel_embeddings():
            h_reshaped = tf.reshape(neigh_vecs, (batch_size * num_neighbors, self.neigh_input_dim))
            for l in self.mlp_layers:
                h_reshaped = l(h_reshaped)

            neigh_h = tf.reshape(h_reshaped,
                                 (batch_size, num_neighbors, self.hidden_dim))

            neigh_h = self.pool(neigh_h, axis=1)

            from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])

            from_self = tf.matmul(self_vecs, self.vars["self_weights"])

            return from_neighs, from_self

        # def isolated_level_set():
        #     def sup():
        #         h_reshaped = tf.reshape(neigh_vecs,
        #                                 (batch_size * num_neighbors, self.neigh_input_dim))
        #         for l in self.mlp_layers:
        #             h_reshaped = l(h_reshaped)
        #
        #         neigh_h = tf.reshape(h_reshaped,
        #                              (batch_size, num_neighbors, self.hidden_dim))
        #
        #         neigh_h = self.pool(neigh_h, axis=1)
        #
        #         from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        #
        #         from_self = tf.matmul(self_vecs, self.vars["self_weights"])
        #
        #         return from_neighs, from_self
        #     def sub():
        #         return from_sub_neighs, from_sub_self
        #
        #     from_sup_neighs, from_sup_self = tf.cond(tf.not_equal(self.bs, 0),
        #                                              sup,
        #                                              sub)
        #     return from_sup_neighs, from_sup_self

        from_neighs, from_self =levelset_weighted_superlevel_embeddings()
        # tf.cond(tf.logical_and(tf.not_equal(self.bs, 0),
        #                                                 tf.not_equal(self.sub_bs, 0)),
        #                                  levelset_weighted_superlevel_embeddings,
        #                                  isolated_level_set)




        def concat_sup():
            return tf.concat([from_self, from_neighs],axis=1)
        def add_sup():
            return tf.add_n([from_self, from_neighs])

        superlevel_embedding_h = tf.cond(tf.constant(self.concat, tf.bool),
                                     concat_sup,
                                     add_sup)

        # def training_with_sublevel_graph():
        #     return sublevel_embedding_h
        # def testing_superlevel_graph():
        #     return superlevel_embedding_h
        # sublevel_embedding_h  = tf.cond(tf.equal(self.sub_bs, 0),
        #                                          training_with_sublevel_graph,
        #                                          testing_superlevel_graph)
        def mult_agg():
            output = tf.multiply(sublevel_embedding_h, superlevel_embedding_h)

            sub_output = sublevel_embedding_h
            return output, sub_output
        def add_agg():
            output = tf.add_n([sublevel_embedding_h, superlevel_embedding_h])

            sub_output = sublevel_embedding_h
            return output, sub_output
        def no_agg():
            # if self.super_present:

            output = superlevel_embedding_h#tf.concat([sublevel_embedding_h, superlevel_embedding_h], axis=1)
            sub_output = sublevel_embedding_h
            return output, sub_output

        output, sub_output = tf.cond(tf.not_equal(self.subcomplex_weight,-1),
                                     add_agg,
                                     no_agg)

        # output, sub_output = add_agg()
        # bias
        if self.bias:
            #if self.super_present:
            output += self.vars['bias']
            #if self.sub_present:
            sub_output += self.vars['bias']
            # agg_output += self.vars['bias']

        return self.act(output), self.act(sub_output)

class GraphSageMaxPoolingAggregator(Layer):
    """ Aggregates via max-pooling over MLP functions.
    """
    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
                 dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, hidden_dim_1 = None,
                 hidden_dim_2 = None, geto_loss=False,
                                                     multilevel_concat = False,
                 jumping_knowledge = False, geto_dims_in= None, geto_dims_out = None,
                 subcomplex_weight = 1.0, **kwargs):
        super(MaxPoolingAggregator, self).__init__(**kwargs)

        #print("    * : Aggregator is MaxPool")

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat
        self.multilevel_concat = multilevel_concat
        self.jumping_knowledge = jumping_knowledge

        self.subcomplex_weight = subcomplex_weight


        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if self.hidden_dim_1 is not None:
            hidden_dim = self.hidden_dim = self.hidden_dim_1
        else:
            if model_size == "small":
                hidden_dim = self.hidden_dim = 201*3#256
            elif model_size == "big":
                hidden_dim = self.hidden_dim = 201*3#1024


        self.mlp_layers = []
        self.mlp_layers.append(Dense(input_dim=neigh_input_dim,
                                 output_dim=hidden_dim,
                                 act=tf.nn.relu,
                                 dropout=dropout,
                                 sparse_inputs=False,
                                 logging=self.logging))

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([hidden_dim, output_dim],
                                                        name='neigh_weights')
           
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                                        name='self_weights')

            self.vars['sub_neigh_weights'] = glorot([hidden_dim, output_dim],
                                                name='sub_neigh_weights')

            self.vars['self_sub_weights'] = glorot([input_dim, output_dim],
                                               name='self_sub_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim
        #pout(["hidden 1 ", self.hidden_dim_1,'hidden 2', self.hidden_dim_2])

    def _call(self, inputs):
        self_vecs, neigh_vecs, sub_self_vecs, sub_neigh_vecs = inputs
        neigh_h = neigh_vecs

        dims = tf.shape(neigh_h)
        batch_size = dims[0]
        num_neighbors = dims[1]

        sub_dims = tf.shape(sub_neigh_vecs)
        sub_batch0_size = sub_dims[0]
        num_sub_neighbors = sub_dims[1]

        # [nodes * sampled neighbors] x [hidden_dim]
        h_reshaped = tf.reshape(neigh_h, (batch_size * num_neighbors, self.neigh_input_dim))
        h_sub_reshaped = tf.reshape(sub_neigh_vecs, (sub_batch0_size * num_sub_neighbors, self.neigh_input_dim))

        for l in self.mlp_layers:
            h_sub_reshaped = l(h_sub_reshaped)

        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)


        neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim))
        neigh_h = tf.reduce_max(neigh_h, axis=1)

        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])

        from_self = tf.matmul(self_vecs, self.vars["self_weights"])


        sub_neigh_h = tf.reshape(h_sub_reshaped, (sub_batch0_size, num_sub_neighbors, self.hidden_dim))
        sub_neigh_h = tf.reduce_max(sub_neigh_h, axis=1)
        sub_neigh_h = tf.multiply(self.subcomplex_weight, sub_neigh_h)

        from_sub_neighs = tf.matmul(sub_neigh_h, self.vars['sub_neigh_weights'])

        from_self_sub = tf.matmul(self_vecs, self.vars["self_sub_weights"])
        from_self_sub = tf.multiply(self.subcomplex_weight, from_self_sub)

        from_neighs = tf.add_n([from_sub_neighs, from_neighs])
        from_self = tf.add_n([from_self_sub, from_self])
        
        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
            sub_output = tf.add_n([from_self_sub, from_sub_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)
            sub_output = tf.concat([from_self_sub, from_sub_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']
            sub_output += self.vars['bias']
       
        return self.act(output), self.act(sub_output)

class MeanPoolingAggregator(Layer):
    """ Aggregates via mean-pooling over MLP functions.
    """
    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
                 hidden_dim_1=None, hidden_dim_2=None, geto_loss=False,
            dropout=0., jumping_knowledge=False, bias=False, act=tf.nn.relu,
                 name=None, concat=False, geto_dims_in= None,geto_dims_out = None, **kwargs):
        super(MeanPoolingAggregator, self).__init__(**kwargs)

        #print("    * : Aggregator is MeanPool")

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2

        self.jumping_knowledge = jumping_knowledge

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if self.hidden_dim_1 is not None:
            hidden_dim = self.hidden_dim = self.hidden_dim_1
        else:
            if model_size == "small":
                hidden_dim = self.hidden_dim = 201*3#256
            elif model_size == "big":
                hidden_dim = self.hidden_dim = 201*3#1024

        self.mlp_layers = []
        self.mlp_layers.append(Dense(input_dim=neigh_input_dim,
                                 output_dim=hidden_dim,
                                 act=tf.nn.relu,
                                 dropout=dropout,
                                 sparse_inputs=False,
                                 logging=self.logging))

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([hidden_dim, output_dim],
                                                        name='neigh_weights')
           
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                                        name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim
        pout(["hidden 1 ", self.hidden_dim_1,'hidden 2', self.hidden_dim_2])

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs
        neigh_h = neigh_vecs

        dims = tf.shape(neigh_h)
        batch_size = dims[0]
        num_neighbors = dims[1]
        # [nodes * sampled neighbors] x [hidden_dim]
        h_reshaped = tf.reshape(neigh_h, (batch_size * num_neighbors, self.neigh_input_dim))

        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)
        neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim))

        neigh_h = tf.reduce_mean(neigh_h, axis=1)
        
        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])
        
        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)


class TwoMaxLayerPoolingAggregator(Layer):
    """ Aggregates via pooling over two MLP functions.
    """
    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False,geto_loss=False,
                 geto_dims_in= None,geto_dims_out = None, **kwargs):
        super(TwoMaxLayerPoolingAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if self.hidden_dim_1 is not None:
            hidden_dim_1 = self.hidden_dim_1
        else:
            if model_size == "small":
                hidden_dim_1 = self.hidden_dim_1 = 512#256
            elif model_size == "big":
                hidden_dim_1 = self.hidden_dim_1 = 1024

        if self.hidden_dim_2 is not None:
            hidden_dim_2 = self.hidden_dim_2
        else:
            if model_size == "small":
                hidden_dim_2 = self.hidden_dim_2 = 256#256
            elif model_size == "big":
                hidden_dim_2 = self.hidden_dim_2 = 512#1024



        self.mlp_layers = []
        self.mlp_layers.append(Dense(input_dim=neigh_input_dim,
                                 output_dim=hidden_dim_1,
                                 act=tf.nn.relu,
                                 dropout=dropout,
                                 sparse_inputs=False,
                                 logging=self.logging))
        self.mlp_layers.append(Dense(input_dim=hidden_dim_1,
                                 output_dim=hidden_dim_2,
                                 act=tf.nn.relu,
                                 dropout=dropout,
                                 sparse_inputs=False,
                                 logging=self.logging))


        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([hidden_dim_2, output_dim],
                                                        name='neigh_weights')
           
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                                        name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs
        neigh_h = neigh_vecs

        dims = tf.shape(neigh_h)
        batch_size = dims[0]
        num_neighbors = dims[1]
        # [nodes * sampled neighbors] x [hidden_dim]
        h_reshaped = tf.reshape(neigh_h, (batch_size * num_neighbors, self.neigh_input_dim))

        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)
        neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim_2))
        neigh_h = tf.reduce_max(neigh_h, axis=1)
        
        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])
        
        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)

class SeqAggregator(Layer):
    """ Aggregates via a standard LSTM.
    """
    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,geto_loss=False,
            dropout=0., bias=False, act=tf.nn.relu, name=None,  concat=False, **kwargs):
        super(SeqAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if model_size == "small":
            hidden_dim = self.hidden_dim = 128
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 256

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([hidden_dim, output_dim],
                                                        name='neigh_weights')
           
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                                        name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim
        self.cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim)

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs

        dims = tf.shape(neigh_vecs)
        batch_size = dims[0]
        initial_state = self.cell.zero_state(batch_size, tf.float32)
        used = tf.sign(tf.reduce_max(tf.abs(neigh_vecs), axis=2))
        length = tf.reduce_sum(used, axis=1)
        length = tf.maximum(length, tf.constant(1.))
        length = tf.cast(length, tf.int32)

        with tf.variable_scope(self.name) as scope:
            try:
                rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                        self.cell, neigh_vecs,
                        initial_state=initial_state, dtype=tf.float32, time_major=False,
                        sequence_length=length)
            except ValueError:
                scope.reuse_variables()
                rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                        self.cell, neigh_vecs,
                        initial_state=initial_state, dtype=tf.float32, time_major=False,
                        sequence_length=length)
        batch_size = tf.shape(rnn_outputs)[0]
        max_len = tf.shape(rnn_outputs)[1]
        out_size = int(rnn_outputs.get_shape()[2])
        index = tf.range(0, batch_size) * max_len + (length - 1)
        flat = tf.reshape(rnn_outputs, [-1, out_size])
        neigh_h = tf.gather(flat, index)

        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])
         
        #output = tf.add_n([from_self, from_neighs])

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)

