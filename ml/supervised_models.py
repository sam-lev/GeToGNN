import tensorflow as tf

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
                 name='', positive_class_weight = 1.0, geto_loss=False
                 ,model_size="small", sigmoid_loss=False, identity_dim=0,
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
        self.getoinputs1 = placeholders["getobatch"] if self.geto_elements is not None else None
        self.model_size = model_size
        self.adj_info = adj
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

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        #pout(["    * supgeto: learning rate",FLAGS.learning_rate])

        self.build()#name=name)

        #print("aggregator hidden dim:",self.aggregator_cls.hidden_dim_1)
        #print("aggregator hidden dim:", self.aggregator_cls.hidden_dim_2)

    def build(self):



        dim_mult = 2 if self.concat else 1
        samples1, support_sizes1, getosamples1 = self.sample(self.inputs1,
                                                             self.layer_infos,
                                                             geto_inputs=self.getoinputs1,
                                                             geto_elms=self.geto_elements,#list(self.hidden_geto_dict.values())
                                                             geto_dims=dim_mult * self.dims_geto_elms[-1])



        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
        self.outputs1, self.aggregators, self.getooutputs1 = self.aggregate(samples1, [self.features],
                                                                            self.dims,
                                                         num_samples,support_sizes1, concat=self.concat,
                                                         hidden_geto = self.hidden_geto,
                                                         model_size=self.model_size,
                                                         jumping_knowledge = self.jumping_knowledge,
                                                         hidden_dim_1 = self.hidden_dim_1,
                                                         hidden_dim_2 = self.hidden_dim_2,
                                                         geto_dims=self.dims_geto_elms,
                                                         geto_elms=self.geto_elements,
                                                         getosamples=getosamples1,
                                                                            geto_loss=self.geto_loss)

        self.outputs1 = tf.nn.l2_normalize(self.outputs1, 1)

        self.see_grad_count = 0

        dim_mult = 2 if self.concat else 1
        #dim_mult += int(self.hidden_geto) if self.concat else 0

        self.node_pred = Dense(dim_mult*self.dims[-1], self.num_classes,
                dropout=self.placeholders['dropout'],
                act=lambda x : x,
                               use_geto=False)#,name=name)

        # TF graph management
        self.node_preds = self.node_pred(self.outputs1)


        if self.geto_loss and self.hidden_geto_agg:
            self.getoouputs1 = tf.nn.l2_normalize(self.getooutputs1, 1)
            self.geto_node_pred = Dense(dim_mult * self.dims_geto_elms[-1], self.num_classes,
                                   dropout=self.placeholders['dropout'],
                                   act=lambda x: x,
                                        use_geto=False)
            self.geto_node_preds = self.geto_node_pred(self.getooutputs1)


        self._loss()
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        see_values = [[('grad', grad), ('var', var)] for grad, var in grads_and_vars]

        if self.geto_loss and self.hidden_geto_agg:
            geto_grads_and_vars = self.optimizer.compute_gradients(self.geto_loss)
            see_values_geto = [[('grad', grad), ('var', var)] for grad, var in grads_and_vars]
            geto_clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                           for grad, var in geto_grads_and_vars]
            self.geto_grad, _ = geto_clipped_grads_and_vars[0]
            self.geto_opt_op = self.optimizer.apply_gradients(geto_clipped_grads_and_vars,
                                                              name="geto_opt_op")

        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) 
                for grad, var in grads_and_vars]


        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars, name="opt_op")

        self.preds = self.predict()
        if self.geto_loss and self.hidden_geto_agg:
            self.geto_preds = self.geto_predict()



    def _loss(self):
        # Weight decay loss
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
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
                    labels=self.placeholders['labels']))
        elif self.positive_class_weight == 1.0:
            self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.node_preds,
                    labels=self.placeholders['labels']))
        else:
            #print("--------------- using weighted loss")
            self.loss += tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
                logits=self.node_preds,
                labels=self.placeholders['labels'],
                pos_weight = self.positive_class_weight))

        if self.geto_loss and self.hidden_geto_agg:
            # Weight decay loss
            for aggregator in self.aggregators:
                for var in aggregator.geto_vars.values():
                    self.geto_loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
            # Weight decay loss
            for var in self.geto_node_pred.geto_vars.values():
                self.geto_loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

            # classification loss
            if self.sigmoid_loss and self.positive_class_weight == 1.0:
                self.geto_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.geto_node_preds,
                    labels=self.placeholders['labels']))
            elif self.positive_class_weight == 1.0:
                self.geto_loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.geto_node_preds,
                    labels=self.placeholders['labels']))
            else:
                #print("--------------- using weighted loss")
                self.geto_loss += tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
                    logits=self.geto_node_preds,
                    labels=self.placeholders['labels'],
                    pos_weight=self.positive_class_weight))

            tf.summary.scalar('geto_loss', self.geto_loss)
        tf.summary.scalar('loss', self.loss)

    def predict(self):
        if self.sigmoid_loss:
            return tf.nn.sigmoid(self.node_preds, name="preds")
        else:
            return tf.nn.softmax(self.node_preds, name="preds")

    def geto_predict(self):
        if self.sigmoid_loss:
            return tf.nn.sigmoid(self.geto_node_preds, name="geto_preds")
        else:
            return tf.nn.softmax(self.geto_node_preds, name="geto_preds")
