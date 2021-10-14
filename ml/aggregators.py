import tensorflow as tf

from .layers import Layer, Dense
from .inits import glorot, zeros

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

        neigh_vecs = tf.nn.dropout(neigh_vecs, 1-self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1-self.dropout)
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

        neigh_vecs = tf.nn.dropout(neigh_vecs, 1-self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1-self.dropout)
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
                 dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False,
                 hidden_dim_1=None, hidden_dim_2=None, jumping_knowledge=False,
                 geto_dims_in=None, geto_dims_out=None, geto_vec_dim=None,geto_loss=False,
                 **kwargs):
        super(GeToEdgeAggregator, self).__init__(**kwargs)

        print("    * : Aggregator is GetoEdge Aggregator")

        self.dropout = dropout
        self.bias = bias
        self.act = act
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
            self.vars['neigh_feat_weights'] = glorot([hidden_dim,  output_dim],
                                                name='neigh_feat_weights')

            self.vars['self_feat_weights'] = glorot([input_dim, output_dim],
                                               name='self_feat_weights')
            if not geto_loss:
                self.vars['self_geto_weights'] = glorot([geto_dims_in, output_dim],
                                                   name='self_geto_weights')
            else:
                self.geto_vars['self_geto_weights'] = glorot([geto_dims_in, output_dim],
                                                        name='self_geto_weights')
            ##self.vars['self_geto_feat_weights'] = glorot([output_dim, output_dim],
            ##                                              name='self_geto_feat_weights')
            self.vars['neigh_geto_weights'] = glorot([hidden_dim, output_dim],
                                               name='neigh_geto_weights')
            if not geto_loss:
                self.vars['neigh_geto_only'] = glorot([hidden_dim, output_dim],
                                                         name='neigh_geto_only')
            else:
                self.vars['neigh_geto_only'] = glorot([hidden_dim, output_dim],
                                                      name='neigh_geto_only')
            ##self.vars['neigh_feat_only'] = glorot([hidden_dim, output_dim],
            ##                                      name='neigh_feat_only')
            ##self.vars['neigh_geto_feat_weights'] = glorot([output_dim,output_dim],
            ##                                   name='neigh_geto_feat_weights')

            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim
        print("    * neighbor input dim")
        print(neigh_input_dim)
        self.geto_dims_in=geto_dims_in
        self.geto_loss = geto_loss
        print("    * geto dim")
        print(geto_dims_out)

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
        neigh_geto = tf.reshape(geto_reshaped,
                            (batch_size, num_neighbors, self.hidden_dim))#self.hidden_dim))
        neigh_geto = tf.reduce_mean(neigh_geto, axis=1)
        if not self.geto_loss:
            from_neigh_geto = tf.matmul(neigh_geto, self.vars['neigh_geto_weights'])
        else:
            from_neigh_geto = tf.matmul(neigh_geto, self.geto_vars['neigh_geto_weights'])


        # neighbor fetures mlp
        h_reshaped = tf.reshape(neigh_h,
                                (batch_size * num_neighbors, self.neigh_input_dim))
        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)
        neigh_h = tf.reshape(h_reshaped,
                             (batch_size, num_neighbors, self.hidden_dim))
        neigh_h = tf.reduce_mean(neigh_h, axis=1)
        from_neigh_feat = tf.matmul(neigh_h, self.vars['neigh_feat_weights'])

        # element wise multiplication of nbr's geto an featurer embeddings
        ##neigh_geto_feat_reshaped = tf.multiply(from_neigh_geto, from_neigh_feat)
        ##neigh_geto_feat_reshaped = tf.concat([neigh_geto, neigh_h], axis=1)
        ''' * neigh_geto_feat = tf.multiply(from_neigh_geto, from_neigh_feat)#, transpose_a=True)
        '''

        '''neigh_geto_feat = tf.reshape(neigh_geto_feat,
                             (batch_size, num_neighbors, self.hidden_dim))'''

        '''neigh_geto_feat_pooled = tf.reduce_mean(neigh_geto_feat, axis=1)'''

        '''*from_neigh_geto_feat = tf.matmul(neigh_geto_feat,
                                                 self.vars['neigh_geto_feat_weights'])
        '''
        #from_neigh_geto_feat = tf.multiply(neigh_geto_feat_pooled,
        #                                                  self.vars['neigh_geto_feat_weights'])
        #neigh_geto_pooled = neigh_geto_feat_pooled#tf.reduce_mean(neigh_geto, axis=1)
        if not self.geto_loss:
            from_neigh_geto_only = from_neigh_geto# tf.matmul(neigh_geto_pooled, self.vars['neigh_geto_only'])
        else:
            from_neigh_geto_only = from_neigh_geto #tf.matmul(neigh_geto_pooled, self.geto_vars['neigh_geto_only'])

        ##neigh_feat_pooled = tf.reduce_mean(neigh_h, axis=1)
        ##from_neigh_feat_only = tf.matmul(neigh_feat_pooled, self.vars['neigh_feat_only'])
        if not self.geto_loss:
            from_self_geto = tf.matmul(self_geto_elms, self.vars['self_geto_weights'])
        else:
            from_self_geto = tf.matmul(self_geto_elms, self.geto_vars['self_geto_weights'])
        from_self_feat = tf.matmul(self_vecs, self.vars["self_feat_weights"])

        ''' *self_geto_feat = tf.multiply(from_self_geto , from_self_feat)
        ##self_geto_feat = tf.concat([from_self_geto , from_self_feat], axis=1)
        from_self_geto_feat = tf.matmul(self_geto_feat, self.vars['self_geto_feat_weights'])
        '''
        ##from_self = tf.matmul(self_vecs, self.vars["self_weights"])
        ##from_self_nbr_geto_weighted = tf.multiply(from_neigh_geto, from_self)

        """geto_feat_means_h = tf.reduce_mean(tf.concat([from_neigh_geto,
                                                from_neighs],axis=1), axis=1)
        geto_feat_means_h = tf.reduce_mean(tf.concat([tf.expand_dims(from_neigh_geto, axis=1),
                                                      tf.expand_dims(from_neighs, axis=1)], axis=1), axis=1)"""


        #from_nbr_geto_augmented_feature = tf.matmul( from_geto_feat_combined_h,
        #                                         self.vars['geto_feature_weights'])

        if not self.concat:
            node_output = tf.add_n([from_self_feat , from_neigh_feat])
            geto_output = tf.add_n([from_self_geto, from_neigh_geto_only])
        else:
            node_output = tf.concat([from_self_feat , from_neigh_feat], axis=1)
            geto_output = tf.concat([from_self_geto, from_neigh_geto_only], axis=1)
            #[from_self, from_nbr_geto_augmented_feature], axis=1)

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
    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, hidden_dim_1 = None,
                 hidden_dim_2 = None, geto_loss=False,
            jumping_knowledge = False, geto_dims_in= None,geto_dims_out = None, **kwargs):
        super(MaxPoolingAggregator, self).__init__(**kwargs)

        print("    * : Aggregator is MaxPool")

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

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs
        neigh_h = neigh_vecs

        dims = tf.shape(neigh_h)
        batch_size = dims[0]
        num_neighbors = dims[1]
        # [nodes * sampled neighbors] x [hidden_dim]
        h_reshaped = tf.reshape(neigh_h, (batch_size * num_neighbors, self.neigh_input_dim))

        ## jumping knowledge applied here, concat each layer, not just neighbor
        #jumping_knowledge = []
        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)
        #    if self.jumping_knowledge:
        #        jumping_knowledge.append(h_reshaped)

        #if self.jumping_knowledge:
        #    h_jump = jumping_knowledge[-1]
        #    for idx, l_vec in enumerate(jumping_knowledge[::-1]):
        #        h_jump = tf.concat([h_jump, jumping_knowledge[idx+1]], axis=1)
        #    h_reshaped = h_jump

        neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim))
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

class MeanPoolingAggregator(Layer):
    """ Aggregates via mean-pooling over MLP functions.
    """
    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
                 hidden_dim_1=None, hidden_dim_2=None, geto_loss=False,
            dropout=0., jumping_knowledge=False, bias=False, act=tf.nn.relu,
                 name=None, concat=False, geto_dims_in= None,geto_dims_out = None, **kwargs):
        super(MeanPoolingAggregator, self).__init__(**kwargs)

        print("    * : Aggregator is MeanPool")

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

