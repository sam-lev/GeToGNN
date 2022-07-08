import tensorflow as tf
import numpy as np

from .models import SampleAndAggregate, GeneralizedModel
from .layers import Dense
from .aggregators import MeanAggregator, MaxPoolingAggregator, MeanPoolingAggregator\
    , SeqAggregator, GCNAggregator, TwoMaxLayerPoolingAggregator, \
    GeToMeanPoolAggregator, GeToMaxPoolAggregator,HiddenGeToMeanPoolAggregator,HiddenGeToMaxPoolAggregator,\
    GeToEdgeAggregator
from ml.utils import pout, get_subgraph_attr, append_subgraph_attr, map_fn_subgraph_attr

flags = tf.app.flags
FLAGS = flags.FLAGS

class SupervisedGraphsage(SampleAndAggregate):
    """Implementation of supervised GraphSAGE."""

    def __init__(self, num_classes,
            placeholders, features, adj, degrees,
            layer_infos, concat=True, aggregator_type="mean",
                 geto_elements = None, geto_adj_info = None, geto_weights=None,
                 jumping_knowledge=False,hidden_dim_1_agg = None, hidden_dim_2_agg = None,
                 jump_type='pool',
                 multilevel_concat=False,

                 name='', positive_class_weight = 1.0, geto_loss=False
                 ,model_size="small", sigmoid_loss=False, identity_dim=0,
                 lr_decay_step=None,
                 sub_adj= None,
                 subgraph_dict = None,
                **kwargs):
        '''
        Args:
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features.
            - adj: Numpy array with adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees. 
            - layer_infos: List of SAGEInfo namedtuples that describe the parameters of all 
                   the recursive layers. See SAGEInfo definition above.
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
            - sigmoid_loss: Set to true if nodes can belong to multiple classes
        '''

        GeneralizedModel.__init__(self, **kwargs)

        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "seq":
            self.aggregator_cls = SeqAggregator
        elif aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator#
        elif aggregator_type == "gcn":
            self.aggregator_cls = GCNAggregator
        elif aggregator_type == "geto-meanpool":
            self.aggregator_cls = GeToMeanPoolAggregator
            self.aggregator_cls.jumping_knowledge = jumping_knowledge
        elif aggregator_type == "geto-maxpool":
            self.aggregator_cls = GeToMaxPoolAggregator
            self.aggregator_cls.jumping_knowledge = jumping_knowledge
        elif aggregator_type == "hidden-geto_meanpool":
            self.aggregator_cls = HiddenGeToMeanPoolAggregator
        elif aggregator_type == "hidden-geto_maxpool":
            self.aggregator_cls = HiddenGeToMaxPoolAggregator
        elif aggregator_type == 'geto-edge':
            self.aggregator_cls = GeToEdgeAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)

        self.aggregator_type = aggregator_type

        self.hidden_geto_agg = "hidden" in aggregator_type or "edge" in aggregator_type

        if hidden_dim_1_agg is not None:
            #print(">>> hidden dim 1 ", hidden_dim_1_agg,)
            self.aggregator_cls.hidden_dim_1 = hidden_dim_1_agg
        if hidden_dim_2_agg is not None:
            self.aggregator_cls.hidden_dim_2 = hidden_dim_2_agg

        # For geometric / topologically informed aggregation
        if geto_elements is not None and geto_adj_info is not None:
            self.geto_elements = tf.Variable(tf.constant(geto_elements
                                                         , dtype=tf.float32), trainable=False)
            self.geto_adj_info = geto_adj_info
            self.hidden_geto_dict = {}
        else:
            self.geto_elements = geto_elements
            self.geto_adj_info = geto_adj_info
            self.hidden_geto_dict = None

        self.geto_loss = geto_loss

        # for weighted loss or aggregation
        if geto_weights is not None:
            self.geto_weights = tf.Variable(tf.constant(geto_weights
                                                        , dtype=tf.float32), trainable=False)

        # get info from placeholders...
        self.inputs1 = placeholders["batch"]

        self.sub_ids = np.arange(int(kwargs['total_sublevel_sets']))
        self.subbatch_dict = {}
        for sub_idx in self.sub_ids:
            sb_name = 'sub_batch'+str(sub_idx)

            pout(('sub_name', sb_name))

            sb_sz_name = sb_name+'_size'
            sb_lb_name = sb_name+'_labels'
            sb_loss_name = sb_name + '_loss'
            self.subbatch_dict[sb_name] = placeholders[sb_name]
            self.subbatch_dict[sb_sz_name] = placeholders[sb_sz_name]
            self.subbatch_dict[sb_lb_name] = placeholders[sb_lb_name]
            self.subbatch_dict[sb_loss_name] = 0

        self.sup_on_sub_dict = {}
        for sub_idx in self.sub_ids:
            sb_name = 'sub_batch'+str(sub_idx)

            sb_sz_name = sb_name+'_size'
            sb_lb_name = sb_name+'_labels'
            sb_loss_name = sb_name + '_loss'
            self.sup_on_sub_dict[sb_name] = placeholders[sb_name]
            self.sup_on_sub_dict[sb_sz_name] = placeholders[sb_sz_name]
            self.sup_on_sub_dict[sb_lb_name] = placeholders[sb_lb_name]
            self.sup_on_sub_dict[sb_loss_name] = 0

        self.sub_inputs0 = placeholders['sub_batch0']
        # self.sub_inputs2 = placeholders['sub_batch2']
        self.getoinputs1 = placeholders["getobatch"] if self.geto_elements is not None else None
        self.sub_getoinputs1 = placeholders["sub_getobatch"] if self.geto_elements is not None else None
        self.subcomplex_weight = placeholders['subcomplex_weight']
        self.model_size = model_size
        self.adj_info = adj

        # check = 100
        # if check < 200:
        #     pout(["features at graph idx ", check])
        #     pout(["feat", self.features[check]])
        #     check += 100

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

        self.subgraph_dict = subgraph_dict
        if subgraph_dict is not None:
            for subgraph_name, subgraph in subgraph_dict.items():
                if identity_dim > 0:
                   sub_embeds = tf.get_variable("node_embeddings", [sub_adj.get_shape().as_list()[0], identity_dim])
                else:
                   sub_embeds = None
                if subgraph.features is None:
                    if identity_dim == 0:
                        raise Exception("Must have a positive value for identity feature dimension if no input features given.")
                    sub_features = sub_embeds
                else:
                    sub_features = tf.Variable(tf.constant(subgraph.features, dtype=tf.float32), trainable=False)
                    if not self.embeds is None:
                        sub_features = tf.concat([sub_embeds, sub_features], axis=1)
                subgraph.features = sub_features
                self.subgraph_dict[subgraph_name] = subgraph

        self.degrees = degrees
        self.concat = concat
        self.hidden_geto = self.hidden_geto_agg
        self.jumping_knowledge = jumping_knowledge
        self.jump_type = jump_type
        self.hidden_dim_1 = hidden_dim_1_agg
        self.hidden_dim_2 = hidden_dim_2_agg
        self.num_classes = num_classes
        self.sigmoid_loss = sigmoid_loss
        self.dims = [(0 if features is None else features.shape[1]) + identity_dim]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.dims_geto_elms = [(0 if geto_elements is None else geto_elements.shape[1]) + identity_dim]
        self.dims_geto_elms.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.batch_size = placeholders["batch_size"]
        self.sub_batch0_size = placeholders["sub_batch0_size"]
        # self.sub_batch2_size = placeholders['sub_batch2_size']
        # self.sub_batch0_id = placeholders['sub_batch0_id']

        self.superlevel_adj_id = tf.constant(kwargs['total_sublevel_sets'])
        self.multilevel_concat = multilevel_concat
        self.total_sublevel_sets = kwargs['total_sublevel_sets']

        self.scale_graph = 0
        self.placeholders = placeholders
        self.layer_infos = layer_infos
        self.positive_class_weight = positive_class_weight

        if geto_elements is not None and geto_adj_info is not None:
            self.geto_elements = tf.Variable(tf.constant(geto_elements
                                                         , dtype=tf.float32), trainable=False)
            self.hidden_geto_elements = tf.Variable(tf.constant(geto_elements
                                                         , dtype=tf.float32), trainable=False)
            self.geto_adj_info = geto_adj_info
        else:
            self.geto_elements = geto_elements
            self.geto_adj_info = geto_adj_info

        self.sampler_type = layer_infos[0].neigh_sampler.name


        #
        # optimizers
        #
        self.global_step = tf.Variable(0, trainable=False)
        epsilon = kwargs['epsilon']

        sub_lr = FLAGS.learning_rate

        if lr_decay_step is not None:
            lr_start = FLAGS.learning_rate
            step_rate = lr_decay_step
            learning_rate = tf.compat.v1.train.exponential_decay(lr_start,
                                                                 self.global_step,
                                                                 step_rate, 0.95, staircase=True)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                    epsilon=epsilon)
            self.sub_optimizer = tf.train.AdamOptimizer(learning_rate=sub_lr,
                                                    epsilon=epsilon)
        else:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate,
                                                    epsilon=epsilon)
            self.sub_optimizer = tf.train.AdamOptimizer(learning_rate=sub_lr,
                                                        epsilon=epsilon)


        self.build()#name=name)

        #print("aggregator hidden dim:",self.aggregator_cls.hidden_dim_1)
        #print("aggregator hidden dim:", self.aggregator_cls.hidden_dim_2)

    def build(self):

        # sub_complex_present = self.sub_batch0_size != 0
        # super_level_present = self.batch_size != 0
        dim_mult = 2 if self.concat else 1
        dim_mult = dim_mult * 2 if self.multilevel_concat else dim_mult
        dim_mult_sub = dim_mult //2 if self.multilevel_concat else dim_mult

        pout(["concat", self.concat])
        # dim_mult_geto = dim_mult_geto * 2 if (layer != 0) and self.hidden_geto_agg else dim_mult_geto
        # if self.total_sublevel_sets != 0:
        #     dim_mult *= (self.total_sublevel_sets + 1)
        self.superlevel_present = False
        self.sublevel_present   = False


        samples1, support_sizes1, getosamples1, sub_samples_dict, sub_support_sizes1, sub_getosamples1 = self.sample(
            self.inputs1,
            self.layer_infos,
            geto_inputs=self.sub_getoinputs1,
            geto_elms=self.geto_elements,
            # list(self.hidden_geto_dict.values())
            geto_dims=dim_mult *
                      self.dims_geto_elms[
                          -1],
            sub_ids=self.sub_ids,
            sub_inputs=self.subbatch_dict)



        #sub_samples_dict = sub_sample_dict['sub_batch0']



        
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
        self.outputs1, self.sub_outputs1_dict, self.aggregators, self.getooutputs1 = self.aggregate(samples1,
                                                                                               [self.features],
                                                                                               self.dims,
                                                                                               num_samples,
                                                                                               support_sizes1,
                                                                                                sub_ids=self.sub_ids,
                                                                                               sub_samples_dict=sub_samples_dict,
                                                                                               sub_support_size=sub_support_sizes1,
                                                                                               sub_batch0_size=self.sub_batch0_size,
                                                                                                    subgraph_dict=self.subgraph_dict,
                                                                                               geto_subsamples=sub_getosamples1,
                                                                                               concat=self.concat,
                                                                                               hidden_geto=self.hidden_geto,
                                                                                               model_size=self.model_size,
                                                                                               jumping_knowledge=self.jumping_knowledge,
                                                                                               hidden_dim_1=self.hidden_dim_1,
                                                                                               hidden_dim_2=self.hidden_dim_2,
                                                                                               geto_dims=self.dims_geto_elms,
                                                                                               geto_elms=self.geto_elements,
                                                                                               getosamples=getosamples1,
                                                                                               geto_loss=self.geto_loss)

        self.node_pred = Dense(dim_mult * self.dims[-1], self.num_classes,
                               dropout=self.placeholders['dropout'],
                               act=lambda x: x,
                               use_geto=False)  # ,name=name)


        # self.sub_node_pred = Dense(dim_mult_sub * self.dims[-1], self.num_classes,
        #                        dropout=self.placeholders['dropout'],
        #                        act=lambda x: x,
        #                        use_geto=False,
        #                        name_weights='sub')


        def normed_super_outputs():
            return tf.nn.l2_normalize(self.outputs1, 1)

        self.outputs1 = normed_super_outputs()

        self.multi_sub_opt_op = []
        self.collected_sub_gradients = []
        self.collected_sup_on_sub_gradients = []
        # self.collected_losses = []#
        self.collected_sub_preds = []
        self.collected_sup_on_sub_preds = []
        all_sub_grad_and_var = {}

        self.collected_sup_on_sub_preds = []
        # for sub_name, sub_out in self.sub_outputs1_dict.items():
        for sb_idx in self.sub_ids:
            sub_name = 'sub_batch' + str(sb_idx)
            pout(('in build sub name', sub_name))
            sb_sz_name = sub_name + '_size'
            sb_lb_name = sub_name + '_labels'
            sb_loss_name = sub_name + '_loss'
            sb_loss = self.subbatch_dict[sb_loss_name]
            sub_out_i = self.sub_outputs1_dict[sub_name]


            sub_outputs_i = tf.nn.l2_normalize(sub_out_i, 1)

            self.see_grad_count = 0

            # TF graph management
            # now use loss from sublevel embedding



            sub_labels = self.placeholders[sub_name+'_labels']
            '''
            sub_node_preds = self.sub_node_pred(sub_outputs_i)  # !!!!!!!!!!!!!!     !!!!!!!!!!!
            sub_loss = self._subloss(sb_loss, sub_node_preds, sub_labels, sub_name)
            self.subbatch_dict[sb_loss_name] = sub_loss
            '''



            sup_on_sub_node_preds = self.node_pred(sub_outputs_i)
            self.sup_on_sub_dict[sub_name + '_preds'] = sup_on_sub_node_preds
            sp_sb_loss = self.sup_on_sub_dict[sb_loss_name]
            self.sup_on_sub_dict[sb_lb_name] = sub_labels

            loss, sup_on_sub_loss = self._loss(sup_on_sub_preds=self.sup_on_sub_dict[sub_name + '_preds'],
                                               sub_preds=None,
                                               sup_on_sub_loss=sp_sb_loss,
                                               sub_loss=None,
                                               sub_labels=sub_labels,
                                               sub_name=sub_name)

            self.loss += sup_on_sub_loss

            self.sup_on_sub_dict[sb_loss_name] += sup_on_sub_loss
            self.collected_sup_on_sub_losses[sb_idx] = self.sup_on_sub_dict[sb_loss_name]
            sup_on_sub_grads_and_vars = self.optimizer.compute_gradients(self.sup_on_sub_dict[sb_loss_name])
            sup_on_sub_clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                                 for grad, var in sup_on_sub_grads_and_vars]
            sp_sb_grad, _ = sup_on_sub_clipped_grads_and_vars[0]
            self.collected_sup_on_sub_gradients.append(sup_on_sub_clipped_grads_and_vars)

            sup_on_sub_preds = self.predict( self.sup_on_sub_dict[sub_name + '_preds'] )
            self.collected_sup_on_sub_preds.append(sup_on_sub_preds)



            '''
            grads_and_vars_sub = self.sub_optimizer.compute_gradients(sub_loss)
            clipped_grads_and_vars_sub = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                          for grad, var in grads_and_vars_sub]
            all_sub_grad_and_var[sub_name] = clipped_grads_and_vars_sub
            self.sub_grad, _ = clipped_grads_and_vars_sub[0]
            # self.multi_opt_op.append(clipped_grads_and_vars_sub)
            self.collected_sub_gradients.append(clipped_grads_and_vars_sub)
            self.collected_sub_losses[sb_idx] = sub_loss
            
            sub_preds = self.sub_predict(sub_node_preds)
            self.collected_sub_preds.append(sub_preds)
            '''



        # independent subgraph classifier opt
        '''
        self.multi_sub_opt_op = [self.sub_optimizer.apply_gradients(grad_and_var) for grad_and_var in self.collected_sub_gradients]
        '''
        # opt on sup classifier over subgraph
        # separate optimizer??
        self.multi_sup_on_sub_opt = [self.sub_optimizer.apply_gradients(grads_and_vars) for grads_and_vars in
                                     self.collected_sup_on_sub_gradients]



        # loss and decent for super level
        def pred_sup_embedding_loss():
            return self.node_pred(self.outputs1), self.placeholders['labels'], tf.constant(False, tf.bool)



        self.node_preds, self.labels, cont = pred_sup_embedding_loss()

        self._loss(sup_on_sub_preds = None, sub_preds=None, sup_on_sub_loss = None,
                   sub_loss = None, sub_labels = None, sub_name = None)


        # def combine_losses():
        #     for sb_idx in self.sub_ids:
        #         sub_name = 'sub_batch' + str(sb_idx)
        #         sb_loss_name = sub_name + '_loss'
        #         sub_loss = self.subbatch_dict[sb_loss_name]
        #         sup_on_sup_loss = self.sup_on_sub_dict[sb_loss_name]
        #         self.loss = tf.add(self.loss, sup_on_sup_loss)#sub_loss)
        #     return self.loss
        # def no_agg():
        #     return self.loss
        #
        # self.loss = tf.cond(tf.not_equal(self.subcomplex_weight,-2),
        #                                  combine_losses,
        #                                  no_agg)






        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]


        self.heirarchical_grads_vars = clipped_grads_and_vars


        self.opt_op = self.optimizer.apply_gradients(self.heirarchical_grads_vars, name="opt_op")

        self.preds = self.predict()

        self.increment_global_step = tf.assign(self.global_step, self.global_step + 1)





    def _loss(self, sup_on_sub_preds = None, sub_preds = None,
              sup_on_sub_loss = None,sub_loss = None,
              sub_labels = None, sub_name = None ):
        # Weight decay loss
        if sub_preds is None and sup_on_sub_preds is None:
            for ag_id, aggregator in enumerate(self.aggregators):
                var_id = 0
                for var_key, var in aggregator.vars.items():#.values()):
                    if 'sub' not in var_key:
                        self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
                        tf.summary.histogram('aggregator_{0}_weight_{1}'.format(ag_id,var_id),var)
                    var_id += 1
        else:
            for ag_id, aggregator in enumerate(self.aggregators):
                var_id = 0
                for var_key, var in aggregator.vars.items():#.values()):   ###!!!! !!!!!!!!
                    if 'sub' in var_key:
                        sup_on_sub_loss  += FLAGS.weight_decay * tf.nn.l2_loss(var)
                        tf.summary.histogram('aggregator_{0}_weight_{1}'.format(ag_id,var_id),var)
                    var_id += 1
        if sub_preds is None and sup_on_sub_preds is None:
            for var in self.node_pred.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        else:
            for var in self.node_pred.vars.values():
                sup_on_sub_loss += FLAGS.weight_decay * tf.nn.l2_loss(var)     # ! !!!!!!!!!!  !!! !!  !!



        # classification loss
        # self.positive_class_weight = 1.0
        if self.sigmoid_loss:# and self.positive_class_weight == 1.0:
            pout(("USING SIGMOID LOSS"))
            if sub_preds is None and sup_on_sub_preds is None:
                self.loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.node_preds,
                        labels=self.labels))#self.placeholders['labels']))
            elif sup_on_sub_preds is not None:
                sb_loss_name = sub_name+'_loss'
                sp_sb_loss = sup_on_sub_loss
                sp_sb_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=sup_on_sub_preds,
                        labels=sub_labels))
                # self.sup_on_sub_dict[sb_loss_name] = sp_sb_loss

                # self.loss += sp_sb_loss
            elif sub_preds is not None:
                dif_pred_embeds = sub_preds #tf.subtract(sup_on_sub_preds, sub_preds)
                embed_logit_dif_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits= dif_pred_embeds,
                                                                                              labels=sub_labels))
                self.loss += embed_logit_dif_loss

        elif self.positive_class_weight == 1.0:

            if sub_preds is None and sup_on_sub_preds is None:
                self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        logits=self.node_preds,
                        labels=self.labels))#self.placeholders['labels']))
            elif sup_on_sub_preds is not None:
                sb_loss_name = sub_name+'_loss'
                sp_sb_loss = sup_on_sub_loss
                sp_sb_loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        logits=sup_on_sub_preds,
                        labels=sub_labels))
                # self.sup_on_sub_dict[sb_loss_name] = sp_sb_loss
                # self.loss += sp_sb_loss
            elif sub_preds is not None:
                dif_pred_embeds = sub_preds#tf.subtract(sup_on_sub_preds, sub_preds)
                embed_logit_dif_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= dif_pred_embeds,
                                                                                              labels=sub_labels))
                self.loss += embed_logit_dif_loss





        tf.summary.scalar('loss', self.loss)
        if sup_on_sub_preds is not None:
            tf.summary.scalar('sup_on_sub_loss', self.sup_on_sub_dict[sub_name+'_loss'])

        return self.loss, sup_on_sub_loss

    def _subloss(self, sub_loss, sub_node_preds, sub_labels, sub_name):

        include_global_loss = False##
        # Weight decay loss
        for ag_id, aggregator in enumerate(self.aggregators):
            var_id = 0
            for var_key, var in aggregator.vars.items():
                if True:#'sub' in var_key:
                    sub_loss +=   FLAGS.weight_decay * tf.nn.l2_loss(var)
                    tf.summary.histogram('aggregator_{0}_weight_{1}_subloss'.format(ag_id, var_id), var)
                var_id += 1

        for var in self.sub_node_pred.vars.values():                 # !!!!!!!!!!!!!  !!!!!!!!!!!!!
            sub_loss +=   FLAGS.weight_decay * tf.nn.l2_loss(var)

        # levelset loss
        # classification loss
        def levelset_pred_loss(sub_loss):
            level_set_factor = 0.5
            if self.sigmoid_loss:# and self.positive_class_weight == 1.0:
                sub_loss +=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=sub_node_preds,
                    labels=sub_labels))
            elif self.positive_class_weight == 1.0:
                sub_loss +=  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=sub_node_preds,
                    labels=sub_labels))
            else:
                sub_loss +=  tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
                    logits=sub_node_preds,
                    labels=sub_labels,
                    pos_weight=self.positive_class_weight))
            return sub_loss

        sub_loss += levelset_pred_loss(sub_loss)
        #self.sub_loss += sub_loss
        #self.sub_loss_dict[sub_name] = sub_loss
        # tf.cond(tf.not_equal(self.sub_batch0_size, 0),  # tf.shape(self.sub_outputs1)[0], 0),
        #                                 levelset_pred_loss,
        #                                 levelset_pred_loss)
        #self.loss += self.sub_loss
        tf.summary.scalar('sub_loss_'+sub_name, sub_loss)
        return sub_loss

    def multi_optimization(self):
        for level, grad_var in enumerate(self.collected_sub_gradients):
            self.optimizer.apply_gradients(grad_var, name="sub_opt_op"+str(level))

    def predict(self, preds = None):
        node_preds = self.node_preds if preds is None else preds
        if self.sigmoid_loss:
            return tf.nn.sigmoid(node_preds, name="preds")
        else:
            return tf.nn.softmax(node_preds, name="preds")

    def sub_predict(self, sub_node_preds):
        if self.sigmoid_loss:
            return tf.nn.sigmoid(sub_node_preds, name="geto_preds")
        else:
            return tf.nn.softmax(sub_node_preds, name="geto_preds")
