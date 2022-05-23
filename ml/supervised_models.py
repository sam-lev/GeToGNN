import tensorflow as tf
import numpy as np

from .models import SampleAndAggregate, GeneralizedModel
from .layers import Dense
from .aggregators import MeanAggregator, MaxPoolingAggregator, MeanPoolingAggregator\
    , SeqAggregator, GCNAggregator, TwoMaxLayerPoolingAggregator, \
    GeToMeanPoolAggregator, GeToMaxPoolAggregator,HiddenGeToMeanPoolAggregator,HiddenGeToMaxPoolAggregator,\
    GeToEdgeAggregator
from ml.utils import pout

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
                 total_sublevel_sets = 0,
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

        self.sub_ids = np.arange(total_sublevel_sets)
        self.subbatch_dict = {}
        for sub_idx in self.sub_ids:
            sb_name = 'sub_batch'+str(sub_idx)
            sb_sz_name = sb_name+'_size'
            sb_lb_name = sb_name+'_labels'
            self.subbatch_dict[sb_name] = placeholders[sb_name]
            self.subbatch_dict[sb_sz_name] = placeholders[sb_sz_name]
            self.subbatch_dict[sb_lb_name] = placeholders[sb_lb_name]
        self.sub_inputs0 = placeholders['sub_batch0']
        # self.sub_inputs2 = placeholders['sub_batch2']
        self.getoinputs1 = placeholders["getobatch"] if self.geto_elements is not None else None
        self.sub_getoinputs1 = placeholders["sub_getobatch"] if self.geto_elements is not None else None
        self.subcomplex_weights = placeholders['subcomplex_weight']
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
        self.superlevel_adj_id = tf.constant(total_sublevel_sets)
        self.multilevel_concat = multilevel_concat
        self.total_sublevel_sets = total_sublevel_sets
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

        # self.sub_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        #pout(["    * supgeto: learning rate",FLAGS.learning_rate])

        self.build()#name=name)

        #print("aggregator hidden dim:",self.aggregator_cls.hidden_dim_1)
        #print("aggregator hidden dim:", self.aggregator_cls.hidden_dim_2)

    def build(self):

        # sub_complex_present = self.sub_batch0_size != 0
        # super_level_present = self.batch_size != 0
        dim_mult = 2 if self.concat else 1
        dim_mult = dim_mult * 2 if self.multilevel_concat else dim_mult

        pout(["concat", self.concat])
        # dim_mult_geto = dim_mult_geto * 2 if (layer != 0) and self.hidden_geto_agg else dim_mult_geto
        # if self.total_sublevel_sets != 0:
        #     dim_mult *= (self.total_sublevel_sets + 1)
        self.superlevel_present = False
        self.sublevel_present   = False


        samples1, support_sizes1, getosamples1, sub_sample_dict, sub_support_sizes1, sub_getosamples1 = self.sample(
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



        sub_samples_dict = sub_sample_dict['sub_batch0']



        
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
        self.outputs1, self.sub_outputs1, self.aggregators, self.getooutputs1 = self.aggregate(samples1,
                                                                                               [self.features],
                                                                                               self.dims,
                                                                                               num_samples,
                                                                                               support_sizes1,
                                                                                               sub_samples_dict=sub_samples_dict,
                                                                                               sub_support_size=sub_support_sizes1,
                                                                                               sub_batch0_size=self.sub_batch0_size,
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

        def normed_super_outputs():
            return tf.nn.l2_normalize(self.outputs1, 1)
        def normed_sub_outputs():
            sub_outputs1 = tf.nn.l2_normalize(self.sub_outputs1, 1)
            #outputs1 = tf.concat([self.outputs1, self.sub_outputs1], axis=0)

            return sub_outputs1#, outputs1#, labels

        def _absent():
            return tf.zeros(tf.shape(self.outputs1))#, self.outputs1#, self.placeholders['labels']

        self.outputs1 = normed_super_outputs()
        # tf.cond(tf.not_equal(tf.shape(self.outputs1)[0], 0),
        #                         _normed_super_outputs,
        #                         _absent)
        self.sub_outputs1 = normed_sub_outputs()
        # tf.cond(tf.not_equal(self.sub_batch0_size, 0),  #tf.shape(self.sub_outputs1)[0], 0),
        #                             _normed_sub_outputs,
        #                             _absent)

        # self.outputs1     = tf.add_n([self.outputs1, self.sub_outputs1])
        # self.outputs1 = tf.multiply(self.outputs1, self.sub_outputs1)
        # self.outputs1 = tf.concat([self.outputs1, self.sub_outputs1], axis=1)

        self.see_grad_count = 0


        #dim_mult += int(self.hidden_geto) if self.concat else 0

        self.node_pred = Dense(dim_mult*self.dims[-1], self.num_classes,
                dropout=self.placeholders['dropout'],
                act=lambda x : x,
                               use_geto=False)#,name=name)


        # TF graph management
        # now use loss from sublevel embedding
        def pred_sub_embedding_loss():
            node_preds = self.node_pred(self.sub_outputs1)

            labels = self.placeholders['sub_batch0_labels']


            return node_preds, labels, tf.constant(True,tf.bool)
        def pred_sup_embedding_loss():
            return self.node_pred(self.outputs1), self.placeholders['labels'], tf.constant(False,tf.bool)


        #def gradient_decent_for_sub_embeddings():
        self.sub_node_preds, self.sub_labels, cont = pred_sub_embedding_loss()

        self._subloss()
        grads_and_vars_sub = self.optimizer.compute_gradients(self.sub_loss)
        clipped_grads_and_vars_sub = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                  for grad, var in grads_and_vars_sub]

        sub_grad, _ = clipped_grads_and_vars_sub[0]
        # def apply_sub_gradients():
        self.sub_opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars_sub, name="sub_opt_op")
        # return sub_opt_op
        # def no_update():
        #     x = tf.Variable([1, 2, 3], dtype=tf.float32)
        #     grad = tf.constant([0.1, 0.2, 0.3])
        #     sub_opt_op = self.optimizer.apply_gradients(zip([grad], [x]), name="sub_opt_op")
        #     return sub_opt_op
        # self.sub_opt_op = tf.cond(tf.not_equal(self.subcomplex_weights, -1),
        #                           apply_sub_gradients,
        #                           no_update)

        self.sub_preds = self.sub_predict()




        self.node_preds, self.labels, cont = pred_sup_embedding_loss()
        # #tf.cond(cont,
        #                                              pred_sup_embedding_loss,
        #                                              dummy_out)
        self.agg_losses = True
        self._loss()
        self.loss += self.sub_loss
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                  for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars, name="opt_op")
        self.preds = self.predict()


        # self.node_preds = self.node_pred(self.outputs1)
        # self.labels = self.placeholders['labels']
        #
        #
        #
        # if self.geto_loss and self.hidden_geto_agg:
        #     self.getoouputs1 = tf.nn.l2_normalize(self.getooutputs1, 1)
        #     self.geto_node_pred = Dense(dim_mult * self.dims_geto_elms[-1], self.num_classes,
        #                            dropout=self.placeholders['dropout'],
        #                            act=lambda x: x,
        #                                 use_geto=False)
        #     self.geto_node_preds = self.geto_node_pred(self.getooutputs1)
        #
        #
        # self._loss()
        # grads_and_vars = self.optimizer.compute_gradients(self.loss)
        # #see_values = [[('grad', grad), ('var', var)] for grad, var in grads_and_vars]
        #
        # if self.geto_loss and self.hidden_geto_agg:
        #     geto_grads_and_vars = self.optimizer.compute_gradients(self.geto_loss)
        #     see_values_geto = [[('grad', grad), ('var', var)] for grad, var in grads_and_vars]
        #     geto_clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
        #                                    for grad, var in geto_grads_and_vars]
        #     self.geto_grad, _ = geto_clipped_grads_and_vars[0]
        #     self.geto_opt_op = self.optimizer.apply_gradients(geto_clipped_grads_and_vars,
        #                                                       name="geto_opt_op")
        #
        # clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
        #         for grad, var in grads_and_vars]
        #
        #
        # self.grad, _ = clipped_grads_and_vars[0]
        # self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars, name="opt_op")
        #
        #
        #
        #
        # self.preds = self.predict()





    def _loss(self):
        # Weight decay loss
        for ag_id, aggregator in enumerate(self.aggregators):
            for var_id, var in enumerate(aggregator.vars.values()):
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
                tf.summary.histogram('aggregator_{0}_weight_{1}'.format(ag_id,var_id),var)
        if False:#self.geto_elements is not None and self.hidden_geto_agg:
            # Weight decay loss
            for aggregator in self.aggregators:
                if len(aggregator.vars.values()) != 0:
                    for var in aggregator.vars.values():
                        self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        for var in self.node_pred.vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
       
        # classification loss
        if self.sigmoid_loss and self.positive_class_weight == 1.0:
            self.loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.node_preds,
                    labels=self.labels))#self.placeholders['labels']))
        elif self.positive_class_weight == 1.0:
            self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.node_preds,
                    labels=self.labels))#self.placeholders['labels']))
        else:
            #print("--------------- using weighted loss")
            self.loss += tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
                logits=self.node_preds,
                labels=self.labels,#self.placeholders['labels'],
                pos_weight = self.positive_class_weight))


        tf.summary.scalar('loss', self.loss)
        return self.loss

    def _subloss(self):
        include_global_loss = False##
        # Weight decay loss

        for ag_id, aggregator in enumerate(self.aggregators):
            for var_id, var in enumerate(aggregator.vars.values()):

                self.weight_loss +=   FLAGS.weight_decay * tf.nn.l2_loss(var)
                tf.summary.histogram('aggregator_{0}_weight_{1}_subloss'.format(ag_id, var_id), var)


        for var in self.node_pred.vars.values():
            # if include_global_loss:
            #     self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
            self.weight_loss +=   FLAGS.weight_decay * tf.nn.l2_loss(var)

        self.sub_loss += self.weight_loss

        # # levelset loss
        # # classification loss
        def levelset_pred_loss():
            level_set_factor = 0.5
            if self.sigmoid_loss and self.positive_class_weight == 1.0:
                self.sub_loss +=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.sub_node_preds,
                    labels=self.sub_labels))
                if include_global_loss:
                    self.loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.sub_node_preds,
                        labels=self.sub_labels))
            elif self.positive_class_weight == 1.0:
                self.sub_loss +=  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.sub_node_preds,
                    labels=self.sub_labels))
                if include_global_loss:
                    self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        logits=self.sub_node_preds,
                        labels=self.sub_labels))
            else:
                self.sub_loss +=  tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
                    logits=self.sub_node_preds,
                    labels=self.sub_labels,
                    pos_weight=self.positive_class_weight))
                if include_global_loss:
                    self.loss += tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
                        logits=self.sub_node_preds,
                        labels=self.sub_labels,
                        pos_weight=self.positive_class_weight))
            return self.sub_loss

        self.sub_loss = tf.cond(tf.not_equal(self.sub_batch0_size, 0),  # tf.shape(self.sub_outputs1)[0], 0),
                                        levelset_pred_loss,
                                        levelset_pred_loss)
        #self.loss += self.sub_loss
        tf.summary.scalar('sub_loss', self.sub_loss)
        return self.sub_loss, self.weight_loss



    def predict(self):
        if self.sigmoid_loss:
            return tf.nn.sigmoid(self.node_preds, name="preds")
        else:
            return tf.nn.softmax(self.node_preds, name="preds")

    def sub_predict(self):
        if self.sigmoid_loss:
            return tf.nn.sigmoid(self.sub_node_preds, name="geto_preds")
        else:
            return tf.nn.softmax(self.sub_node_preds, name="geto_preds")
