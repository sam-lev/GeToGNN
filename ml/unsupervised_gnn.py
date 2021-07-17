#!/home/sci/samlev/bin/bin/python3
# SBATCH --time=21-00:00:00 # walltime, abbreviated by -t
# SBATCH --mem=60G
# SBATCH -o slurm-%j.out-%N # name of the stdout, using the job number (%j) and the first node (%N)
# SBATCH -e slurm-%j.err-%N # name of the stderr, using the job number (%j) and the first node (%N)
# SBATCH --gres=gpu:1

from __future__ import division
from __future__ import print_function

import sys
import os
import time
import tensorflow as tf
import numpy as np
import sklearn
from sklearn import metrics

import time

from .models import SampleAndAggregate, SAGEInfo, Node2VecModel
from .minibatch import EdgeMinibatchIterator
from .neigh_samplers import UniformNeighborSampler
from .utils import load_data

from localsetup import LocalSetup
from .utils import random_walk_embedding
from .utils import format_data

# supervised trainer imports
from .supervised_models import SupervisedGraphsage
from .models import SAGEInfo
from .minibatch import NodeMinibatchIterator
from .neigh_samplers import UniformNeighborSampler
from .utils import load_data
from .LinearRegression import LinearRegression

#
# ii. Could update loss given msc behaviours with relation to 'negative' and
#       'positive' arcs-loss dependendt on aggregation of previous to current
#       both negative and positive samples.
# ii. Could add layers to allow convergence faster. Currently model is a
#       two layer multiperceptron
# i.  Could add adaptive hyperparamters e.g. loss drop given some epoch
#       number or exponential/other decay
# iv. Persistence affording subsets of graphs could be to a benefit in
#       explanding training set size or allowing training over set with
#       persistence dependence e.g. birth/death of msc added as hyperparmeter
#       during training. Start simple with high and have training set change by
#       lowering persitence iteratively.
# v.  Construct an aggregator that propogates along arcs depending on
#       persistence. Persistence Weighted Aggregator which is more likely to
#       move along high more persistent arcs. Weighted random walk based on
#       persistence.
# i.  Increase/diversify validation set
#
# Geometric attributes:
#   geomretric no overlap each pixel part of arc is unique arc.
#   instead of each saddle with 4 incident, instead heres arc heres two nodes
#   at end.
#   extend across lines, cosine similarity of normals
#   laplacian adds dark spot on each line so tangential maxima connect in order to connect to minumum and not loose info for bounded region w/ in max
#   send classification image
#   train on full image classify full image
#   Look into neighborhood properties based on geomretry
#
# -change weight initialization (currently xavier which is good w/ l2 regularization so would need to cater weight to feature attributes)
# - normalize features around zero instead of mean
# -add layers(?)
# -add identity element for features
# -try to overfit small set
# -play with negative bc loss meant to diverge pos from neg samples in embedding
#  - number neg samples plays large role cross entropy log of class seems to be log of number negative used.

class gnn:
    def __init__(self, aggregator='graphsage_mean', env=None, msc_collection=None):
        self.aggregator = aggregator
        self.graph = None
        self.features = None
        self.id_map = None
        self.node_classes = None
        self.msc_collection = msc_collection
        self.slurm = env if env is None else env == 'slurm'
        self.params_set = False

    def set_parameters(self, G=None, feats=None, id_map=None, walks=None, class_map=None
                       , train_prefix='', load_walks=False, number_negative_samples=None,
                       nx_idx_to_getoelm_idx = None,
                       geto_weights = None,
                       geto_elements=None
                       , number_positive_samples=None, embedding_file_out=''
                       , learning_rate=None, depth=2, epochs=200, batch_size=512
                       , positive_arcs=[], negative_arcs=[]
                       , max_degree=64 * 3, degree_l1=25, degree_l2=10, degree_l3=0
                       , weight_decay=0.001, polarity=6, use_embedding=None
                       , random_context=True, total_features=1
                       , gpu=0, val_model='cvt', model_size='small', sigmoid=False, env='multivax'
                       , dim_1=256
                       , dim_2=256,
                       base_log_dir='./',
                       jumping_knowledge=True, concat=True,
                       jump_type='pool',
                       hidden_dim_1=None, hidden_dim_2=None):

        if not self.params_set:
            ## variables not actually used but implemented for later development
            self.train_prefix = train_prefix
            self.G = G
            self.feats = feats
            self.id_map = id_map
            self.walks = walks
            self.class_map = class_map
            self.positive_arcs = positive_arcs
            self.negative_arcs = negative_arcs
            self.val_model = val_model

            self.LocalSetup = LocalSetup()
            self.load_walks = load_walks

            self.use_embedding = use_embedding

            slurm = self.slurm if self.slurm is not None else env == 'slurm'
            if slurm != 'slurm':
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

            # Set random seed
            seed = 123
            np.random.seed(seed)
            tf.set_random_seed(seed)

            # Settings
            self.flags = tf.app.flags
            flags = self.flags
            self.FLAGS = self.flags.FLAGS

            self.number_positive_samples = 0
            self.number_negative_samples = 4
            if number_negative_samples:
                self.number_negative_samples = number_negative_samples

            self.model_name = embedding_file_out

            if not learning_rate:
                learning_rate = 0.001
            self.learning_rate = learning_rate

            self.epochs = epochs

            self.depth = depth

            # define graph embedding dimensionality
            # dimension 2x used value with concat
            dim = int(474. / 10.)

            concat = False  # mean aggregator only one to perform concat
            self.dim_feature_space = int((dim + 1) / 2) if concat else dim

            # end vomit#####################################
            self.nx_idx_to_getoelm_idx = nx_idx_to_getoelm_idx
            self.geto_weights = geto_weights
            self.geto_elements = geto_elements
            self.jump_type = jump_type
            self.concat = concat

            tf.app.flags.DEFINE_boolean('log_device_placement', False,
                                        "Whether to log device placement.")
            # core params..
            self.flags.DEFINE_string('model', self.aggregator,
                                     'model names. See README for possible values.')  # mean aggregator does not perform concat
            self.flags.DEFINE_float('learning_rate', self.learning_rate, 'initial learning rate.')
            self.flags.DEFINE_integer('drop_1', 2, 'epoch to reduce learning rate first time  by a tenth')
            self.flags.DEFINE_integer('drop_2', 175, 'epoch to reduce learning rate for the second time by a tenth')
            self.flags.DEFINE_string("model_size", model_size, "Can be big or small; model specific def'ns")
            self.flags.DEFINE_string('train_prefix', train_prefix,
                                     'name of the object file that stores the training data. must be specified.')

            self.flags.DEFINE_string('model_name', self.model_name, 'name of the embedded graph model file is created.')

            self.flags.DEFINE_integer('depth', self.depth,
                                      'epoch to reduce learning rate for the second time by a tenth')  # I added this, journal advocates depth of 2 but loss seems to improve with more

            # left to default values in main experiments
            self.flags.DEFINE_integer('epochs', self.epochs, 'number of epochs to train.')
            self.flags.DEFINE_float('dropout', 0.0, 'dropout rate (1 - keep probability).')
            self.flags.DEFINE_float('weight_decay', weight_decay, 'weight for l2 loss on embedding matrix.')
            self.flags.DEFINE_integer('max_degree', max_degree, 'maximum node degree.')  # 64*3
            self.flags.DEFINE_integer('samples_1', degree_l1,
                                      'number of samples in layer 1')  # neighborhood sample size-currently set to whats used in paper, list of samples of variable hops away for convolving at each layer Length #layers +1
            self.flags.DEFINE_integer('samples_hidden', degree_l1, 'number of samples in hidden layers')
            self.flags.DEFINE_integer('samples_2', degree_l2,
                                      'number of users samples in layer 2')  # neighborhoos sample size
            flags.DEFINE_integer('samples_3', degree_l3,
                                 'number of users samples in layer 3. (Only for mean model)')  # 0
            flags.DEFINE_integer('dim_1', dim_1, 'Size of output dim (final is 2x this, if using concat)')
            flags.DEFINE_integer('dim_2', dim_2, 'Size of output dim (final is 2x this, if using concat)')
            flags.DEFINE_integer('hidden_dim_1_agg', hidden_dim_1,
                                 'hidden dimension of aggregator')
            flags.DEFINE_integer('hidden_dim_2_agg', hidden_dim_2,
                                 'hidden dimension of aggregator')
            self.flags.DEFINE_boolean('random_context', random_context, 'Whether to use random context or direct edges')
            self.flags.DEFINE_integer('neg_sample_size', polarity,
                                      'number of negative samples')  # paper hard set to twenty rather than actual negative. defines the 'weight' on which neighboring negative nodes have on the loss function allowing a spread in the embedding space of positive and negative samples.
            self.flags.DEFINE_integer('batch_size', batch_size, 'minibatch size.')
            self.flags.DEFINE_integer('n2v_test_epochs', 1, 'Number of new SGD epochs for n2v.')  # node to vector
            self.flags.DEFINE_integer('identity_dim', 0,
                                      'Set to positive value to use identity embedding features of that dimension. Default 0.')
            self.flags.DEFINE_boolean('jumping_knowledge', jumping_knowledge,
                                      'whether to use jumping knowledge approach for graph embedding')
            # logging, saving, validation settings etc.
            # logging, saving, validation settings etc.
            self.flags.DEFINE_boolean('save_embeddings', True,
                                      'whether to save embeddings for all nodes after training')
            self.flags.DEFINE_string('base_log_dir', base_log_dir, 'base directory for logging and saving embeddings')
            self.flags.DEFINE_integer('validate_iter', 1000, "how often to run a validation minibatch.")
            self.flags.DEFINE_integer('validate_batch_size', 5, "how many nodes per validation sample.")
            self.flags.DEFINE_integer('gpu', gpu, "which gpu to use.")
            self.flags.DEFINE_string('env', 'multivax', 'environment to manage data paths and gpu use')
            self.flags.DEFINE_integer('print_every', 650, "How often to print training info.")
            self.flags.DEFINE_integer('max_total_steps', 10 ** 10, "Maximum total number of iterations")

            if slurm != 'slurm':
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.FLAGS.gpu)

            self.GPU_MEM_FRACTION = 0.95

            self.params_set = True

    def train(self, G=None, feats=None, id_map=None, walks=None, class_map=None,
              train_prefix='', load_walks=False, number_negative_samples=None,
              nx_idx_to_getoelm_idx=None,
              geto_weights = None,
              geto_elements = None
              , number_positive_samples=None, embedding_file_out=''
              , learning_rate=None, depth=3, epochs=200, batch_size=512
              , positive_arcs=[], negative_arcs=[]
              , weight_decay=0.001, polarity=6, use_embedding=None
              , max_degree=64 * 3, degree_l1=25, degree_l2=10, degree_l3=0
              , gpu=0, env='mutivax', sigmoid=False, val_model='cvt', model_size="small"
              , dim_1=256
              , dim_2=256,
              base_log_dir='./',
              concat = True,
              hidden_dim_1=None,
              hidden_dim_2=None,
              jumping_knowledge=False,
              jump_type='pool',
              total_features=1):

        if load_walks is not None or walks is not None:
            random_context = True
        else:
            random_context = False

        self.set_parameters(G=G, feats=feats, id_map=id_map, walks=walks, class_map=class_map
                            , train_prefix=train_prefix, load_walks=load_walks,
                            number_negative_samples=number_negative_samples,
                            nx_idx_to_getoelm_idx=nx_idx_to_getoelm_idx,
                            geto_weights=geto_weights,
                            geto_elements=geto_elements
                            , number_positive_samples=number_positive_samples, embedding_file_out=embedding_file_out
                            , learning_rate=learning_rate, depth=depth, epochs=epochs, batch_size=batch_size
                            , positive_arcs=positive_arcs, negative_arcs=negative_arcs
                            , max_degree=max_degree, degree_l1=degree_l1, degree_l2=degree_l2, degree_l3=degree_l3
                            , weight_decay=weight_decay, polarity=polarity, use_embedding=use_embedding
                            , random_context=random_context, total_features=total_features
                            , gpu=gpu, val_model=val_model, sigmoid=sigmoid, env=env, model_size=model_size
                            , dim_1=dim_1
                            , dim_2=dim_2,
                            base_log_dir=base_log_dir,
                            concat=concat,
                            jumping_knowledge=jumping_knowledge,
                            jump_type=jump_type,
                            hidden_dim_1=hidden_dim_1, hidden_dim_2=hidden_dim_2)

        train_data = self.get_data()
        # begin training
        print('Begin GNN training')
        print('')
        # if self.msc_collection is None:
        self._train(train_data[:-2])
        # else:
        #    self.batch_train(self.msc_collection)

    def log_dir(self):
        log_dir = os.path.join(self.FLAGS.base_log_dir , 'embedding') # + self.FLAGS.train_prefix.split("/")[-2]
        #log_dir += "embedding"#/{model:s}_{model_size:s}_{lr:0.6f}/".format(
        #    model=self.FLAGS.model,
        #    model_size=self.FLAGS.model_size,
        #    lr=self.FLAGS.learning_rate)
        #if not os.path.exists(log_dir):
        #    os.makedirs(log_dir)
        return log_dir

    # Define model evaluation function
    def evaluate(self, sess, model, minibatch_iter, size=None):
        t_test = time.time()
        feed_dict_val = minibatch_iter.val_feed_dict(size)
        outs_val = sess.run([model.loss, model.ranks, model.mrr],
                            feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)

    def incremental_evaluate(self, sess, model, minibatch_iter, size):
        t_test = time.time()
        finished = False
        val_losses = []
        val_mrrs = []
        iter_num = 0
        while not finished:
            feed_dict_val, finished, _ = minibatch_iter.incremental_val_feed_dict(size, iter_num)
            iter_num += 1
            outs_val = sess.run([model.loss, model.ranks, model.mrr],
                                feed_dict=feed_dict_val)
            val_losses.append(outs_val[0])
            val_mrrs.append(outs_val[2])
        return np.mean(val_losses), np.mean(val_mrrs), (time.time() - t_test)

    def save_val_embeddings(self, sess, model, minibatch_iter, size, out_dir, mod=""):
        val_embeddings = []
        finished = False
        seen = set([])
        nodes = []
        iter_num = 0
        name = "val"
        while not finished:
            feed_dict_val, finished, edges = minibatch_iter.incremental_embed_feed_dict(size, iter_num)
            iter_num += 1
            outs_val = sess.run([model.loss, model.mrr, model.outputs1],
                                feed_dict=feed_dict_val)
            # ONLY SAVE FOR embeds1 because of planetoid
            for i, edge in enumerate(edges):
                if not edge[0] in seen:
                    val_embeddings.append(outs_val[-1][i, :])
                    nodes.append(edge[0])
                    seen.add(edge[0])
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        val_embeddings = np.vstack(val_embeddings)
        np.save(os.path.join(out_dir, name + ".npy"), val_embeddings)
        with open(os.path.join(out_dir, name + ".txt"), "w") as fp:
            fp.write("\n".join(map(str, nodes)))

    def get_graph(self):
        return self.G

    def construct_placeholders(self):
        # Define placeholders
        placeholders = {
            'batch1': tf.placeholder(tf.int32, shape=(None), name='batch1'),
            'batch2': tf.placeholder(tf.int32, shape=(None), name='batch2'),
            # negative samples for all nodes in the batch
            'neg_samples': tf.placeholder(tf.int32, shape=(None,),
                                          name='neg_sample_size'),
            'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
            'batch_size': tf.placeholder(tf.int32, name='batch_size'),
        }
        return placeholders

    def _train(self, train_data, test_data=None):
        G = train_data[0]
        features = train_data[1]
        id_map = train_data[2]

        if not features is None:
            # pad with dummy zero vector
            features = np.vstack([features, np.zeros((features.shape[1],))])

        context_pairs = train_data[3] if self.FLAGS.random_context else None
        placeholders = self.construct_placeholders()
        minibatch = EdgeMinibatchIterator(G,
                                          id_map,
                                          placeholders,
                                          nx_idx_to_getoelm_idx=self.nx_idx_to_getoelm_idx,
                                          batch_size=self.FLAGS.batch_size,
                                          max_degree=self.FLAGS.max_degree,
                                          num_neg_samples=self.FLAGS.neg_sample_size,
                                          context_pairs=context_pairs)
        adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
        adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

        geto_adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.geto_adj.shape)
        geto_adj_info = tf.Variable(geto_adj_info_ph, trainable=False, name="geto_adj_info")

        if self.FLAGS.model == 'graphsage_mean':
            # Create model
            # for more layers add layers to MLP in models.py as well as
            # add SAGEInfo nodes for more layers [layer name, neighbor sampler,
            #               number neighbors sampled, out dim]
            sampler = UniformNeighborSampler(adj_info, geto_adj_info=geto_adj_info)
            layer_infos = [SAGEInfo("node_gsmean", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1)]  # ,
            # SAGEInfo("node_maxpool", sampler, FLAGS.samples_2, FLAGS.dim_2)]
            for i in range(3, self.depth + 1):
                # layer_infos.append(SAGEInfo("node_maxpool", sampler, FLAGS.samples_1, FLAGS.dim_1))
                layer_infos.append(SAGEInfo("node_gsmean", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2))
            layer_infos.append(SAGEInfo("node_gsmean", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2))

            model = SampleAndAggregate(placeholders,
                                       features,
                                       adj_info,
                                       minibatch.deg,

                                       concat=self.concat,
                                       jumping_knowledge=self.FLAGS.jumping_knowledge,
                                       jump_type=self.jump_type,
                                       hidden_dim_1_agg=self.FLAGS.hidden_dim_1_agg,
                                       hidden_dim_2_agg=self.FLAGS.hidden_dim_2_agg,
                                       layer_infos=layer_infos,
                                       depth=self.depth,
                                       model_size=self.FLAGS.model_size,
                                       identity_dim=self.FLAGS.identity_dim,
                                       logging=True)
            """callbacks=[
                ModelSaver(), # Record state graph at intervals during epochs
                InferenceRunner(dataset_train,
                                [ScalarStats('cost'), ClassificationError()]), #Compare to validation set
                ScheduledHyperParamSetter('learning_rate',
                                          [(1, 0.1), (args.drop_1, 0.01), (args.drop_2, 0.001)]) # denote current hyperparameters
            ],"""
        elif self.FLAGS.model == 'gcn':
            # Create model
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, 2 * self.FLAGS.dim_1)]
            for l in range(self.depth - 2):
                layer = SAGEInfo("node", sampler, self.FLAGS.samples_1, 2 * self.FLAGS.dim_hidden)
                layer_infos.append(layer)
            layer_infos.append(SAGEInfo("node", sampler, self.FLAGS.samples_2, 2 * self.FLAGS.dim_2))
            # layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, 2*self.FLAGS.dim_1),
            #                    SAGEInfo("node", sampler, self.FLAGS.samples_2, 2*self.FLAGS.dim_2)]

            model = SampleAndAggregate(placeholders,
                                       features,
                                       adj_info,
                                       minibatch.deg,
                                       layer_infos=layer_infos,
                                       aggregator_type="gcn",
                                       model_size=self.FLAGS.model_size,
                                       identity_dim=self.FLAGS.identity_dim,
                                       concat=False,
                                       logging=True)

        elif self.FLAGS.model == 'graphsage_seq':
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1),
                           SAGEInfo("node", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2)]

            model = SampleAndAggregate(placeholders,
                                       features,
                                       adj_info,
                                       minibatch.deg,
                                       layer_infos=layer_infos,
                                       identity_dim=self.FLAGS.identity_dim,
                                       aggregator_type="seq",
                                       model_size=self.FLAGS.model_size,
                                       logging=True)

        elif self.FLAGS.model == 'graphsage_maxpool':
            sampler = UniformNeighborSampler(adj_info, geto_adj_info=geto_adj_info)
            layer_infos = [SAGEInfo("node_maxpool", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1)]  # ,
            # SAGEInfo("node_maxpool", sampler, FLAGS.samples_2, FLAGS.dim_2)]
            for i in range(3, self.depth + 1):
                # layer_infos.append(SAGEInfo("node_maxpool", sampler, FLAGS.samples_1, FLAGS.dim_1))
                layer_infos.append(SAGEInfo("node_maxpool", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2))
            layer_infos.append(SAGEInfo("node_maxpool", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2))

            model = SampleAndAggregate(placeholders,
                                       features,
                                       adj_info,
                                       minibatch.deg,
                                       geto_weights=self.geto_weights,
                                       geto_elements=self.geto_elements,
                                       geto_adj_info=geto_adj_info,
                                       concat=self.concat,
                                       jumping_knowledge=self.FLAGS.jumping_knowledge,
                                       jump_type=self.jump_type,
                                       hidden_dim_1_agg=self.FLAGS.hidden_dim_1_agg,
                                       hidden_dim_2_agg=self.FLAGS.hidden_dim_2_agg,
                                       layer_infos=layer_infos,
                                       aggregator_type="maxpool",
                                       model_size=self.FLAGS.model_size,
                                       identity_dim=self.FLAGS.identity_dim,
                                       logging=True)
            """callbacks = [
                ScheduledHyperParamSetter('learning_rate',
                                          [(1, 0.001), (self.FLAGS.drop_1, 0.0001), (self.FLAGS.drop_2, 0.00001)])
                ]"""
        elif self.FLAGS.model == 'getomaxpool':
            sampler = UniformNeighborSampler(adj_info, geto_adj_info=geto_adj_info)
            layer_infos = [SAGEInfo("node_getomaxpool", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1)]  # ,
            # SAGEInfo("node_maxpool", sampler, FLAGS.samples_2, FLAGS.dim_2)]
            for i in range(3, self.depth + 1):
                # layer_infos.append(SAGEInfo("node_maxpool", sampler, FLAGS.samples_1, FLAGS.dim_1))
                layer_infos.append(SAGEInfo("node_getomaxpool", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2))
            layer_infos.append(SAGEInfo("node_getomaxpool", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2))

            model = SampleAndAggregate(placeholders,
                                       features,
                                       adj_info,
                                       minibatch.deg,
                                       geto_elements=self.geto_elements,
                                       geto_adj_info=geto_adj_info,
                                       concat=self.concat,
                                       jumping_knowledge=self.FLAGS.jumping_knowledge,
                                       jump_type=self.jump_type,
                                       hidden_dim_1_agg=self.FLAGS.hidden_dim_1_agg,
                                       hidden_dim_2_agg=self.FLAGS.hidden_dim_2_agg,
                                       layer_infos=layer_infos,
                                       aggregator_type="getomaxpool",
                                       model_size=self.FLAGS.model_size,
                                       identity_dim=self.FLAGS.identity_dim,
                                       logging=True)
        elif self.FLAGS.model == 'graphsage_meanpool':
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node_meanpool", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1)]  # ,
            # SAGEInfo("node_maxpool", sampler, FLAGS.samples_2, FLAGS.dim_2)]
            for i in range(3, self.depth + 1):
                # layer_infos.append(SAGEInfo("node_maxpool", sampler, FLAGS.samples_1, FLAGS.dim_1))
                layer_infos.append(SAGEInfo("node_meanpool", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2))
            layer_infos.append(SAGEInfo("node_meanpool", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2))
            model = SampleAndAggregate(placeholders,
                                       features,
                                       adj_info,
                                       minibatch.deg,
                                       concat=self.concat,
                                       jumping_knowledge=self.FLAGS.jumping_knowledge,
                                       jump_type=self.jump_type,
                                       hidden_dim_1_agg=self.FLAGS.hidden_dim_1_agg,
                                       hidden_dim_2_agg=self.FLAGS.hidden_dim_2_agg,
                                       layer_infos=layer_infos,
                                       aggregator_type="meanpool",
                                       model_size=self.FLAGS.model_size,
                                       identity_dim=self.FLAGS.identity_dim,
                                       logging=True)

        elif self.FLAGS.model == 'n2v':
            model = Node2VecModel(placeholders, features.shape[0],
                                  minibatch.deg,
                                  # 2x because graphsage uses concat
                                  nodevec_dim=2 * self.FLAGS.dim_1,
                                  lr=self.FLAGS.learning_rate)
        else:
            raise Exception('Error: model name unrecognized.')

        config = tf.ConfigProto(log_device_placement=self.FLAGS.log_device_placement)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = self.GPU_MEM_FRACTION
        config.allow_soft_placement = True

        # Initialize session
        sess = tf.Session(config=config)
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.log_dir(), sess.graph)

        # Init variables
        sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph: minibatch.adj, geto_adj_info_ph:minibatch.geto_adj})

        # Train model

        train_shadow_mrr = None
        shadow_mrr = None

        total_steps = 0
        avg_time = 0.0
        epoch_val_costs = []

        train_adj_info = tf.assign(adj_info, minibatch.adj)
        val_adj_info = tf.assign(adj_info, minibatch.test_adj)

        if self.generate_embedding:
            self.FLAGS.epochs = 1

        for epoch in range(self.FLAGS.epochs):
            minibatch.shuffle()

            iter = 0
            print('...')
            print('Epoch: %04d' % (epoch + 1))
            print('...')

            epoch_val_costs.append(0)
            while not minibatch.end():
                # Construct feed dictionary
                feed_dict = minibatch.next_minibatch_feed_dict()
                feed_dict.update({placeholders['dropout']: self.FLAGS.dropout})

                t = time.time()
                # Training step
                outs = sess.run([merged, model.opt_op, model.loss, model.ranks, model.aff_all,
                                 model.mrr, model.outputs1], feed_dict=feed_dict)
                train_cost = outs[2]
                train_mrr = outs[5]
                if train_shadow_mrr is None:
                    train_shadow_mrr = train_mrr  #
                else:
                    train_shadow_mrr -= (1 - 0.99) * (train_shadow_mrr - train_mrr)

                if iter % self.FLAGS.validate_iter == 0:
                    # Validation
                    sess.run(val_adj_info.op)
                    val_cost, ranks, val_mrr, duration = self.evaluate(sess, model, minibatch,
                                                                       size=self.FLAGS.validate_batch_size)
                    sess.run(train_adj_info.op)
                    epoch_val_costs[-1] += val_cost
                if shadow_mrr is None:
                    shadow_mrr = val_mrr
                else:
                    shadow_mrr -= (1 - 0.99) * (shadow_mrr - val_mrr)

                if total_steps % self.FLAGS.print_every == 0:
                    summary_writer.add_summary(outs[0], total_steps)

                # Print results
                avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)

                if total_steps % self.FLAGS.print_every == 0:
                    print("Iter:", '%04d' % iter,
                          "train_loss=", "{:.5f}".format(train_cost),
                          "train_mrr=", "{:.5f}".format(train_mrr),
                          "train_mrr_ema=", "{:.5f}".format(train_shadow_mrr),  # exponential moving average
                          "val_loss=", "{:.5f}".format(val_cost),
                          "val_mrr=", "{:.5f}".format(val_mrr),
                          "val_mrr_ema=", "{:.5f}".format(shadow_mrr),  # exponential moving average
                          "time=", "{:.5f}".format(avg_time))

                iter += 1
                total_steps += 1

                if total_steps > self.FLAGS.max_total_steps:
                    break

            if total_steps > self.FLAGS.max_total_steps:
                break

        print("Optimization Finished.")
        # Modify for embedding of new graph
        #   adj_info for unseen needed ( node connectivity to rest of new graph)
        #   modify minibatch to accomodate new adj_info of unseen
        #   also feature matrix for unseen graph
        if self.FLAGS.save_embeddings:
            sess.run(val_adj_info.op)

            self.save_val_embeddings(sess, model, minibatch, self.FLAGS.validate_batch_size, self.log_dir())

            if self.FLAGS.model == "n2v":
                # stopping the gradient for the already trained nodes
                train_ids = tf.constant(
                    [[id_map[n]] for n in G.nodes_iter() if not G.node[n]['val'] and not G.node[n]['test']],
                    dtype=tf.int32)
                test_ids = tf.constant([[id_map[n]] for n in G.nodes_iter() if G.node[n]['val'] or G.node[n]['test']],
                                       dtype=tf.int32)
                update_nodes = tf.nn.embedding_lookup(model.context_embeds, tf.squeeze(test_ids))
                no_update_nodes = tf.nn.embedding_lookup(model.context_embeds, tf.squeeze(train_ids))
                update_nodes = tf.scatter_nd(test_ids, update_nodes, tf.shape(model.context_embeds))
                no_update_nodes = tf.stop_gradient(
                    tf.scatter_nd(train_ids, no_update_nodes, tf.shape(model.context_embeds)))
                model.context_embeds = update_nodes + no_update_nodes
                sess.run(model.context_embeds)

                # run random walks
                from .utils import run_random_walks
                nodes = [n for n in G.nodes_iter() if G.node[n]["val"] or G.node[n]["test"]]
                start_time = time.time()
                pairs = run_random_walks(G, nodes, num_walks=50)
                walk_time = time.time() - start_time

                test_minibatch = EdgeMinibatchIterator(G,
                                                       id_map,
                                                       placeholders, batch_size=self.FLAGS.batch_size,
                                                       max_degree=self.FLAGS.max_degree,
                                                       num_neg_samples=self.FLAGS.neg_sample_size,
                                                       context_pairs=pairs,
                                                       n2v_retrain=True,
                                                       fixed_n2v=True)

                start_time = time.time()
                print("Doing test training for n2v.")
                test_steps = 0
                for epoch in range(self.FLAGS.n2v_test_epochs):
                    test_minibatch.shuffle()
                    while not test_minibatch.end():
                        feed_dict = test_minibatch.next_minibatch_feed_dict()
                        feed_dict.update({placeholders['dropout']: self.FLAGS.dropout})
                        outs = sess.run([model.opt_op, model.loss, model.ranks, model.aff_all,
                                         model.mrr, model.outputs1], feed_dict=feed_dict)
                        if test_steps % self.FLAGS.print_every == 0:
                            print("Iter:", '%04d' % test_steps,
                                  "train_loss=", "{:.5f}".format(outs[1]),
                                  "train_mrr=", "{:.5f}".format(outs[-2]))
                        test_steps += 1
                train_time = time.time() - start_time
                self.save_val_embeddings(sess, model, minibatch, self.FLAGS.validate_batch_size, self.log_dir(),
                                         mod="-test")
                print("Total time: ", train_time + walk_time)
                print("Walk time: ", walk_time)
                print("Train time: ", train_time)

    def get_data(self, train_data=None):
        print("Loading training data..")
        # train_data = load_data(FLAGS.train_prefix)

        ### load MSC data
        print('loading msc graph data')

        self.generate_embedding = self.use_embedding is not None
        if self.use_embedding is not None:
            self.G = self.use_embedding[0]  # graph to embed
            self.feats = self.use_embedding[1]  # features of graph to embed
            self.id_map = self.use_embedding[2]  # learned embedding id map
            self.walks = self.use_embedding[3] if self.load_walks is None else []
            self.class_map = []
        # Collect training data
        # train from saved file, assumes pre-labeled train/test nodes
        if self.msc_collection is None:
            if self.train_prefix and self.G is None:
                print("loading graph data for gnn training")
                self.train_prefix = self.train_prefix

                train_data = load_data(self.train_prefix, load_walks=self.load_walks, scheme_required=True,
                                       train_or_test='train')

                self.number_negative_samples = train_data[len(train_data) - 2]
                self.number_positive_samples = train_data[len(train_data) - 1]
                number_samples = self.number_negative_samples + self.number_positive_samples
                proportion_negative = int(number_samples / float(self.number_negative_samples))

            # train from passed graph, assumed pre-labeled(/processed)
            # graph with test/train nodes
            elif self.G is not None and self.feats is not None and self.id_map is not None and self.class_map is not None and not self.train_prefix:
                train_prefix = 'nNeg-' + str(self.number_negative_samples) + 'nPos-' + str(self.number_positive_samples)
                print("using pre-processed graph data for gnn training")
                self.number_negative_samples = self.number_negative_samples
                number_samples = self.number_negative_samples + self.number_positive_samples
                proportion_negative = int(number_samples / float(self.number_negative_samples))
                train_data = (self.G, self.feats, self.id_map, self.walks, self.class_map, [], [])

            # train from cvt sampled graph and respective in/out arcs as train
            elif self.positive_arcs and self.negative_arcs:
                train_data = load_data(self.positive_arcs, self.negative_arcs, load_walks=self.load_walks,
                                       scheme_required=True,
                                       train_or_test='train')
                self.number_negative_samples = len(self.negative_arcs)
                number_samples = len(self.positive_arcs) + len(self.negative_arcs)
                proportion_negative = int(number_samples / float(self.number_negative_samples))
            # keep labeled (test/train) graph for later use in testing
            self.graph = train_data[0]
            self.features = train_data[1]
            self.id_map = train_data[2]
            self.node_classes = train_data[4]

            if self.load_walks:
                walks = []
                if isinstance(self.graph.nodes()[0], int):
                    conversion = lambda n: int(n)
                else:
                    conversion = lambda n: n
                with open(self.load_walks + "-walks.txt") as fp:
                    for line in fp:
                        walks.append(map(conversion, line.split()))

            train_data = (
                train_data[0], train_data[1], train_data[2], walks, train_data[4], train_data[5], train_data[6])
        return train_data

    def batch_train(self, msc_collection, test_data=None):
        mscbatch = []

        training_msc_dataset = msc_collection[0]
        persistence_values = msc_collection[1]
        blur_sigmas = msc_collection[2]
        for image, msc, mask, segmentation in training_msc_dataset:
            # = training_msc_dataset[0]
            pers = 0  # temp
            blur = 0
            msc = msc[(persistence_values[pers], blur_sigmas[blur])]
            training_sample = self.format_msc_feature_graph(image, msc, mask, segmentation, persistence_values[pers],
                                                            blur_sigmas[blur])
            G = training_sample[0]
            features = training_sample[1]
            id_map = training_sample[2]

            if not features is None:
                # pad with dummy zero vector
                features = np.vstack([features, np.zeros((features.shape[1],))])

            context_pairs = training_sample[3] if self.FLAGS.random_context else None
            placeholders = self.construct_placeholders()
            minibatch = EdgeMinibatchIterator(G,
                                              id_map,
                                              placeholders, batch_size=self.FLAGS.batch_size,
                                              max_degree=self.FLAGS.max_degree,
                                              num_neg_samples=self.FLAGS.neg_sample_size,
                                              context_pairs=context_pairs)
            adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
            adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

        if self.FLAGS.model == 'graphsage_mean':
            # Create model
            # for more layers add layers to MLP in models.py as well as
            # add SAGEInfo nodes for more layers [layer name, neighbor sampler,
            #               number neighbors sampled, out dim]
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1),
                           SAGEInfo("node", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2)]

            model = SampleAndAggregate(placeholders,
                                       features,
                                       adj_info,
                                       minibatch.deg,
                                       layer_infos=layer_infos,
                                       depth=self.depth,
                                       model_size=self.FLAGS.model_size,
                                       identity_dim=self.FLAGS.identity_dim,
                                       logging=True)
            """callbacks=[
                ModelSaver(), # Record state graph at intervals during epochs
                InferenceRunner(dataset_train,
                                [ScalarStats('cost'), ClassificationError()]), #Compare to validation set
                ScheduledHyperParamSetter('learning_rate',
                                          [(1, 0.1), (args.drop_1, 0.01), (args.drop_2, 0.001)]) # denote current hyperparameters
            ],"""
        elif self.FLAGS.model == 'gcn':
            # Create model
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, 2 * self.FLAGS.dim_1)]
            for l in range(self.depth - 2):
                layer = SAGEInfo("node", sampler, self.FLAGS.samples_hidden, 2 * self.FLAGS.dim_hidden)
                layer_infos.append(layer)
            layer_infos.append(SAGEInfo("node", sampler, self.FLAGS.samples_2, 2 * self.FLAGS.dim_2))
            layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, 2 * self.FLAGS.dim_1),
                           SAGEInfo("node", sampler, self.FLAGS.samples_2, 2 * self.FLAGS.dim_2)]

            model = SampleAndAggregate(placeholders,
                                       features,
                                       adj_info,
                                       minibatch.deg,
                                       layer_infos=layer_infos,
                                       aggregator_type="gcn",
                                       model_size=self.FLAGS.model_size,
                                       identity_dim=self.FLAGS.identity_dim,
                                       concat=False,
                                       logging=True)

        elif self.FLAGS.model == 'graphsage_seq':
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1),
                           SAGEInfo("node", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2)]

            model = SampleAndAggregate(placeholders,
                                       features,
                                       adj_info,
                                       minibatch.deg,
                                       layer_infos=layer_infos,
                                       identity_dim=self.FLAGS.identity_dim,
                                       aggregator_type="seq",
                                       model_size=self.FLAGS.model_size,
                                       logging=True)

        elif self.FLAGS.model == 'graphsage_maxpool':
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1)]
            for l in range(self.depth - 2):
                layer = SAGEInfo("node", sampler, self.FLAGS.samples_hidden, 2 * self.FLAGS.dim_hidden)
                layer_infos.append(layer)
            layer_infos.append(SAGEInfo("node", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2))
            # layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1),
            #                    SAGEInfo("node", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2)]
            model = SampleAndAggregate(placeholders,
                                       features,
                                       adj_info,
                                       minibatch.deg,
                                       layer_infos=layer_infos,
                                       aggregator_type="maxpool",
                                       model_size=self.FLAGS.model_size,
                                       identity_dim=self.FLAGS.identity_dim,
                                       logging=True)
            """callbacks = [
                ScheduledHyperParamSetter('learning_rate',
                                          [(1, 0.001), (self.FLAGS.drop_1, 0.0001), (self.FLAGS.drop_2, 0.00001)])
                ]"""
        elif self.FLAGS.model == 'graphsage_meanpool':
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1)]
            for l in range(self.depth - 2):
                layer = SAGEInfo("node", sampler, self.FLAGS.samples_hidden, self.FLAGS.dim_hidden)
                layer_infos.append(layer)
            layer_infos.append(SAGEInfo("node", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2))
            # layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1),
            #                    SAGEInfo("node", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2)]

            model = SampleAndAggregate(placeholders,
                                       features,
                                       adj_info,
                                       minibatch.deg,
                                       layer_infos=layer_infos,
                                       aggregator_type="meanpool",
                                       model_size=self.FLAGS.model_size,
                                       identity_dim=self.FLAGS.identity_dim,
                                       logging=True)

        elif self.FLAGS.model == 'n2v':
            model = Node2VecModel(placeholders, features.shape[0],
                                  minibatch.deg,
                                  # 2x because graphsage uses concat
                                  nodevec_dim=2 * self.FLAGS.dim_1,
                                  lr=self.FLAGS.learning_rate)
        else:
            raise Exception('Error: model name unrecognized.')

        config = tf.ConfigProto(log_device_placement=self.FLAGS.log_device_placement)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = self.GPU_MEM_FRACTION
        config.allow_soft_placement = True

        # Initialize session
        sess = tf.Session(config=config)
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.log_dir(), sess.graph)

        # Init variables
        sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph: minibatch.adj})

        # Train model

        train_shadow_mrr = None
        shadow_mrr = None

        total_steps = 0
        avg_time = 0.0
        epoch_val_costs = []

        train_adj_info = tf.assign(adj_info, minibatch.adj)
        val_adj_info = tf.assign(adj_info, minibatch.test_adj)

        if self.generate_embedding:
            self.FLAGS.epochs = 1

        for epoch in range(self.FLAGS.epochs):
            minibatch.shuffle()

            iter = 0
            print('...')
            print('Epoch: %04d' % (epoch + 1))
            print('...')

            epoch_val_costs.append(0)
            while not minibatch.end():
                # Construct feed dictionary
                feed_dict = minibatch.next_minibatch_feed_dict()
                feed_dict.update({placeholders['dropout']: self.FLAGS.dropout})

                t = time.time()
                # Training step
                outs = sess.run([merged, model.opt_op, model.loss, model.ranks, model.aff_all,
                                 model.mrr, model.outputs1], feed_dict=feed_dict)
                train_cost = outs[2]
                train_mrr = outs[5]
                if train_shadow_mrr is None:
                    train_shadow_mrr = train_mrr  #
                else:
                    train_shadow_mrr -= (1 - 0.99) * (train_shadow_mrr - train_mrr)

                if iter % self.FLAGS.validate_iter == 0:
                    # Validation
                    sess.run(val_adj_info.op)
                    val_cost, ranks, val_mrr, duration = self.evaluate(sess, model, minibatch,
                                                                       size=self.FLAGS.validate_batch_size)
                    sess.run(train_adj_info.op)
                    epoch_val_costs[-1] += val_cost
                if shadow_mrr is None:
                    shadow_mrr = val_mrr
                else:
                    shadow_mrr -= (1 - 0.99) * (shadow_mrr - val_mrr)

                if total_steps % self.FLAGS.print_every == 0:
                    summary_writer.add_summary(outs[0], total_steps)

                # Print results
                avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)

                if total_steps % self.FLAGS.print_every == 0:
                    print("Iter:", '%04d' % iter,
                          "train_loss=", "{:.5f}".format(train_cost),
                          "train_mrr=", "{:.5f}".format(train_mrr),
                          "train_mrr_ema=", "{:.5f}".format(train_shadow_mrr),  # exponential moving average
                          "val_loss=", "{:.5f}".format(val_cost),
                          "val_mrr=", "{:.5f}".format(val_mrr),
                          "val_mrr_ema=", "{:.5f}".format(shadow_mrr),  # exponential moving average
                          "time=", "{:.5f}".format(avg_time))

                iter += 1
                total_steps += 1

                if total_steps > self.FLAGS.max_total_steps:
                    break

            if total_steps > self.FLAGS.max_total_steps:
                break

        print("Optimization Finished.")
        if self.FLAGS.save_embeddings:
            sess.run(val_adj_info.op)

            self.save_val_embeddings(sess, model, minibatch, self.FLAGS.validate_batch_size, self.log_dir())

            if self.FLAGS.model == "n2v":
                # stopping the gradient for the already trained nodes
                train_ids = tf.constant(
                    [[id_map[n]] for n in G.nodes_iter() if not G.node[n]['val'] and not G.node[n]['test']],
                    dtype=tf.int32)
                test_ids = tf.constant([[id_map[n]] for n in G.nodes_iter() if G.node[n]['val'] or G.node[n]['test']],
                                       dtype=tf.int32)
                update_nodes = tf.nn.embedding_lookup(model.context_embeds, tf.squeeze(test_ids))
                no_update_nodes = tf.nn.embedding_lookup(model.context_embeds, tf.squeeze(train_ids))
                update_nodes = tf.scatter_nd(test_ids, update_nodes, tf.shape(model.context_embeds))
                no_update_nodes = tf.stop_gradient(
                    tf.scatter_nd(train_ids, no_update_nodes, tf.shape(model.context_embeds)))
                model.context_embeds = update_nodes + no_update_nodes
                sess.run(model.context_embeds)

                # run random walks
                from .utils import run_random_walks
                nodes = [n for n in G.nodes_iter() if G.node[n]["val"] or G.node[n]["test"]]
                start_time = time.time()
                pairs = run_random_walks(G, nodes, num_walks=50)
                walk_time = time.time() - start_time

                test_minibatch = EdgeMinibatchIterator(G,
                                                       id_map,
                                                       placeholders, batch_size=self.FLAGS.batch_size,
                                                       max_degree=self.FLAGS.max_degree,
                                                       num_neg_samples=self.FLAGS.neg_sample_size,
                                                       context_pairs=pairs,
                                                       n2v_retrain=True,
                                                       fixed_n2v=True)

                start_time = time.time()
                print("Doing test training for n2v.")
                test_steps = 0
                for epoch in range(self.FLAGS.n2v_test_epochs):
                    test_minibatch.shuffle()
                    while not test_minibatch.end():
                        feed_dict = test_minibatch.next_minibatch_feed_dict()
                        feed_dict.update({placeholders['dropout']: self.FLAGS.dropout})
                        outs = sess.run([model.opt_op, model.loss, model.ranks, model.aff_all,
                                         model.mrr, model.outputs1], feed_dict=feed_dict)
                        if test_steps % self.FLAGS.print_every == 0:
                            print("Iter:", '%04d' % test_steps,
                                  "train_loss=", "{:.5f}".format(outs[1]),
                                  "train_mrr=", "{:.5f}".format(outs[-2]))
                        test_steps += 1
                train_time = time.time() - start_time
                self.save_val_embeddings(sess, model, minibatch, self.FLAGS.validate_batch_size, self.log_dir(),
                                         mod="-test")
                print("Total time: ", train_time + walk_time)
                print("Walk time: ", walk_time)
                print("Train time: ", train_time)

    """Perform classification using learned graph representation from GNN"""
    def classify(self, MSCGNN_infer = None, test_prefix = None,  trained_prefix=None
                 , embedding_prefix=None, embedding_path_name=None, aggregator=None
                 , learning_rate = None, MSCGNN = None, supervised=False, size='small'):
        cwd = './'
        #embedding_path =  os.path.join(cwd,'log-dir',embedding_prefix+'-unsup-json_graphs','graphsage_mean_small_'+'0.100000')
        if embedding_path_name is None and learning_rate is not None:
            embedding_p = embedding_prefix+'-unsup-json_graphs'+'/'+aggregator+'_'+size if not supervised else embedding_prefix+'/'+aggregator+'_'+'small'
            embedding_p += ("_{lr:0.6f}").format(lr=learning_rate)
        else:
            embedding_p = embedding_path_name
        if test_prefix is not None and trained_prefix is not None and not self.G:
            trained_p = os.path.join(cwd,'data','json_graphs',trained_prefix)
            test_p =  os.path.join(cwd,'data','json_graphs',test_prefix)
            trained_prfx = trained_prefix
            test_prfx = test_prefix
            mscgnn_infer = LinearRegression(test_path = test_p, MSCGNN_infer = MSCGNN_infer
                                , test_prefix = test_prfx, trained_path = trained_p
                                 , trained_prefix = trained_prfx, MSCGNN = self
                                , embedding_path = os.path.join(cwd, 'log-dir',embedding_p)).run()

        elif self.G:
             G,feats,id_map, walks\
                 ,class_map, number_negative_samples, number_positive_samples = format_data(dual=self.G,
                                                                                            features=self.features,
                                                                                            node_id=self.node_id,
                                                                                            id_map=self.node_id,
                                                                                            node_classes=self.node_classes,
                                                                                            train_or_test = '',
                                                                                            scheme_required = True, load_walks=False)

             mscgnn_infer = LinearRegression(G=G,
                                  MSCGNN_infer=MSCGNN_infer,
                                features = feats,
                                labels=class_map,
                                num_neg = 10,#len(self.negative_arcs),
                                id_map = id_map,
                                MSCGNN = self,
                                embedding_path = os.path.join(cwd, 'log-dir',embedding_p)).run()
             self.gnn.G = self.G


        if mscgnn_infer is not None:
            return mscgnn_infer
    #if __name__ == '__main__':
    #    tf.app.run()





