#!/home/sci/samlev/bin/bin/python3                                          
#SBATCH --time=21-00:00:00 # walltime, abbreviated by -t                       
#SBATCH --mem=60G                                                             
#SBATCH -o slurm-%j.out-%N # name of the stdout, using the job number (%j) and the first node (%N)                                                            
#SBATCH -e slurm-%j.err-%N # name of the stderr, using the job number (%j) and the first node (%N)                                                            
#SBATCH --gres=gpu:1

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

#supervised trainer imports
from .supervised_models import SupervisedGraphsage
from .models import SAGEInfo
from .minibatch import NodeMinibatchIterator
from .neigh_samplers import UniformNeighborSampler, GeToInformedNeighborSampler
from .utils import load_data
from ml.utils import pout








########
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
################
class gnn:
    # begin vomit
    def __init__(self, aggregator='graphsage_mean', env=None
                 , msc_collection=None, persistence_values=None, model_path=None):
        self.aggregator = aggregator
        self.graph = None
        self.features = None
        self.id_map = None
        self.node_classes = None
        self.msc_collection = msc_collection
        self.persistence_values=persistence_values
        self.slurm = env if env is None else env == 'slurm'
        self.infer = False
        self.model = None
        self.sess = None
        self.model_path = model_path
        self.FLAGS = None
        self.params_set = False
        self.placeholders = None

    def set_parameters(self, G=None, feats=None, id_map=None, walks=None, class_map=None
              , train_prefix='', load_walks=False, number_negative_samples=None
              , number_positive_samples=None, embedding_file_out=''
              , learning_rate=None, depth=2, epochs=200, batch_size=512
              , positive_arcs=[], negative_arcs=[]
                       ,dim_1=128, dim_2=128
              , max_degree=64*3, degree_l1=25, degree_l2=10,degree_l3=0
              , weight_decay=0.001, polarity=6, use_embedding=None
                       , jumping_knowledge=True, concat=True,
                       positive_class_weight = 1.0,
                       jump_type = 'pool',
                       nx_idx_to_getoelm_idx=None,
                       geto_weights=None,
                       geto_elements=None,
                       geto_loss=False,
                       hidden_dim_1=None, hidden_dim_2 = None, random_context=True
              , gpu=0, val_model='cvt', model_size="small", sigmoid=False, env='multivax'):

        if True:#not self.params_set:
            ## variables not actually used but implemented for later development
            self.train_prefix = train_prefix
            self.G = G
            self.feats=feats
            self.id_map=id_map
            self.walks=walks
            self.class_map=class_map
            self.positive_arcs = positive_arcs
            self.negative_arcs=negative_arcs
            self.val_model = val_model

            self.LocalSetup = LocalSetup()
            self.load_walks = load_walks

            self.use_embedding=use_embedding

            slurm = self.slurm if self.slurm is not None else env == 'slurm'
            if slurm != 'slurm':
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

            # Set random seed
            seed = 123
            np.random.seed(seed)
            tf.set_random_seed(seed)

            # Settings
            self.flags = tf.app.flags
            #hack

            self.FLAGS = self.flags.FLAGS

            flags = self.flags

            self.number_positive_samples=0
            self.number_negative_samples = 4
            if number_negative_samples:
                self.number_negative_samples = number_negative_samples

            self.model_name = embedding_file_out

            if not learning_rate:
                learning_rate = 0.001
            self.learning_rate = .003#learning_rate

            self.epochs = 10#epochs

            self.depth = depth

            # define graph embedding dimensionality
            # dimension 2x used value with concat
            dim = int(474. / 10.)

            self.nx_idx_to_getoelm_idx = nx_idx_to_getoelm_idx
            self.geto_weights = geto_weights
            self.geto_elements = geto_elements
            self.use_geto = self.nx_idx_to_getoelm_idx is not None

            self.concat = concat  # mean aggregator only one to perform concat
            self.dim_feature_space = int((dim + 1) / 2) if concat else dim
            self.jump_type = jump_type
            self.geto_loss = geto_loss
            print('.... Jump Type is: ', jump_type)
            #end vomit#####################################


            # Set random seed
            seed = 123
            np.random.seed(seed)
            tf.set_random_seed(seed)

            def del_all_flags(FLAGS):
                flags_dict = FLAGS._flags()
                keys_list = [keys for keys in flags_dict ]
                for keys in keys_list:
                    FLAGS.__delattr__(keys)

            del_all_flags(tf.flags.FLAGS)

            del_all_flags(self.flags.FLAGS)

            tf.app.flags.DEFINE_boolean('log_device_placement', False,
                                        """Whether to log device placement.""")
            # core params..
            flags.DEFINE_string('model', self.aggregator, 'model names. See README for possible values.')
            flags.DEFINE_float('learning_rate', self.learning_rate, 'initial learning rate.')
            flags.DEFINE_string("model_size", model_size, "Can be big or small; model specific def'ns")
            flags.DEFINE_string('train_prefix', '', 'prefix identifying training data. must be specified.')

            # left to default values in main experiments
            flags.DEFINE_integer('epochs', self.epochs, 'number of epochs to train.')
            flags.DEFINE_float('dropout', 0.0, 'dropout rate (1 - keep probability).')
            flags.DEFINE_float('weight_decay', weight_decay, 'weight for l2 loss on embedding matrix.')
            flags.DEFINE_integer('max_degree', max_degree, 'maximum node degree.')#128
            flags.DEFINE_integer('samples_1', degree_l1, 'number of samples in layer 1')#25, number samples per node
            flags.DEFINE_integer('samples_2', degree_l2, 'number of samples in layer 2')#10
            flags.DEFINE_integer('samples_3', degree_l3, 'number of users samples in layer 3. (Only for mean model)')#0
            flags.DEFINE_integer('dim_1', dim_1, 'Size of output dim (final is 2x this, if using concat)')
            flags.DEFINE_integer('dim_2', dim_2, 'Size of output dim (final is 2x this, if using concat)')
            flags.DEFINE_integer('hidden_dim_1_agg', hidden_dim_1,
                                 'hidden dimension of aggregator')
            flags.DEFINE_integer('hidden_dim_2_agg', hidden_dim_2,
                                 'hidden dimension of aggregator')
            flags.DEFINE_boolean('random_context', random_context, 'Whether to use random context or direct edges')#true
            flags.DEFINE_integer('batch_size', batch_size, 'minibatch size.')
            flags.DEFINE_boolean('sigmoid', sigmoid, 'whether to use sigmoid loss')
            flags.DEFINE_float('positive_class_weight', positive_class_weight, 'positive class weight for imbalanced class distribution')
            flags.DEFINE_integer('identity_dim', 0,
                                 'Set to positive value to use identity embedding features of that dimension. Default 0.')

            self.flags.DEFINE_boolean('jumping_knowledge', jumping_knowledge, 'whether to use jumping knowledge approach for graph embedding')
            # logging, saving, validation settings etc.
            self.flags.DEFINE_boolean('save_embeddings', True, 'whether to save embeddings for all nodes after training')
            self.flags.DEFINE_string('base_log_dir', './log-dir', 'base directory for logging and saving embeddings')
            self.flags.DEFINE_integer('validate_iter', 100, "how often to run a validation minibatch.")
            self.flags.DEFINE_integer('validate_batch_size', batch_size//4, "how many nodes per validation sample.")
            flags.DEFINE_integer('gpu', gpu, "which gpu to use.")
            self.flags.DEFINE_string('env', 'multivax', 'environment to manage data paths and gpu use')
            self.flags.DEFINE_integer('print_every', 300, "How often to print training info.")
            self.flags.DEFINE_integer('max_total_steps', 10**10, "Maximum total number of iterations")
            self.flags.DEFINE_string('model_name', self.model_name, 'name of the embedded graph model file is created.')
            self.flags.DEFINE_integer('depth', self.depth,
                                      'epoch to reduce learning rate for the second time by a tenth')  # I added this, journal advocates depth of 2 but loss seems to improve with more

            if slurm != 'slurm':
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.FLAGS.gpu)

            self.GPU_MEM_FRACTION = 0.99


            flags.DEFINE_integer('read_param', 2, 'hack to use argparse elsewhere.')
            self.flags.DEFINE_integer('sample_idx', 2, 'hack to use argparse elsewhere.')

            #self.params_set = True

    def train(self, G=None, feats=None, id_map=None, walks=None, class_map=None
              , train_prefix='', load_walks=False, number_negative_samples=None
              , number_positive_samples=None, embedding_file_out=''
              , learning_rate=None, depth=3, epochs=200, batch_size=512
              , positive_arcs=[], negative_arcs=[]
              , max_degree=64*3, degree_l1=25, degree_l2=10,degree_l3=0
              , dim_1 = 256, dim_2 = 256,
              concat = True, random_context=True,
              nx_idx_to_getoelm_idx=None,
              geto_weights=None,
              geto_elements=None,
              jumping_knowledge = False,
              jump_type = 'pool',
              geto_loss = False,
              hidden_dim_1=None, hidden_dim_2=None,
              positive_class_weight=1.0
              , weight_decay=0.001, polarity=6, use_embedding=None
              , gpu=0, val_model='cvt', sigmoid=False,model_size="small", env='multivax'):

        self.set_parameters(G=G, feats=feats, id_map=id_map, walks=walks, class_map=class_map
              , train_prefix=train_prefix, load_walks=load_walks, number_negative_samples=number_negative_samples
              , number_positive_samples=number_positive_samples, embedding_file_out=embedding_file_out
              , learning_rate=learning_rate, depth=depth, epochs=epochs, batch_size=batch_size
              , positive_arcs=positive_arcs, negative_arcs=negative_arcs
              , max_degree=max_degree, degree_l1=degree_l1, degree_l2=degree_l2,degree_l3=degree_l3
                            ,dim_1=dim_1,dim_2=dim_2,
                            positive_class_weight=positive_class_weight
                            , jumping_knowledge=jumping_knowledge, concat=concat,
                            jump_type = jump_type, random_context=random_context,
                            nx_idx_to_getoelm_idx=nx_idx_to_getoelm_idx,
                            geto_weights=geto_weights,
                            geto_elements=geto_elements,
                            geto_loss=geto_loss,
                            hidden_dim_1=hidden_dim_1, hidden_dim_2=hidden_dim_2
              , weight_decay=weight_decay, polarity=polarity, use_embedding=use_embedding
              , gpu=gpu, val_model=val_model, model_size=model_size, sigmoid=sigmoid, env=env)


        # format or retriev data
        train_data = self.get_data()
        ## use msc graph data
        # begin training
        print('Begin GNN training')
        print('')

        # resulting trained model
        self.model = None
        #if self.msc_collection is None and not self.infer:
        print("%%%%%BEGINING TRAINGING%%%%")
        self._train(train_data[:-2])

        # need to union graphs for batch training, either here or prior
        # i.e. can't iterate over disjoint due to poor implementation
        #elif not self.infer:
        #    print("%%%%%BEGINNING BATCH TRAIN%%%%")
        #    self.batch_train(self.msc_collection)


    def calc_f1(self,  y_true, y_pred):
        if not self.FLAGS.sigmoid:
            y_true = np.argmax(y_true, axis=1)
            y_pred = np.argmax(y_pred, axis=1)
        else:
            y_pred[y_pred > 0.5] = 1
            y_pred[y_pred <= 0.5] = 0
        return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")

    def pred_values(self, pred):
        if not self.FLAGS.sigmoid:
            pred = np.argmax(pred, axis=1)
        else:
            pred[pred > 0.5] = 1
            pred[pred <= 0.5] = 0
        return pred

    # Define model evaluation function
    def evaluate(self, sess, model, minibatch_iter, size=None):
        val_preds = []
        t_test = time.time()
        feed_dict_val, labels = minibatch_iter.node_val_feed_dict(size)
        node_outs_val = sess.run([model.preds, model.loss],
                                 feed_dict=feed_dict_val)

        # add inference labels
        minibatch_iter.update_batch_prediction(node_outs_val[0])

        val_preds = self.pred_values(node_outs_val[0])
        #minibatch_iter.update_batch_prediction(val_preds)

        mic, mac = self.calc_f1(labels, node_outs_val[0])
        return node_outs_val[1], mic, mac, (time.time() - t_test)

    def log_dir(self):
        log_dir = self.FLAGS.base_log_dir + "/sup-" + self.FLAGS.model_name
        #log_dir = self.FLAGS.base_log_dir + "/sup-" + self.FLAGS.train_prefix.split("/")[-2]
        log_dir += "/{model:s}_{model_size:s}_{lr:0.4f}/".format(
            model=self.FLAGS.model,
            model_size=self.FLAGS.model_size,
            lr=self.FLAGS.learning_rate)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def incremental_evaluate(self, sess, model, minibatch_iter, size, test=False, inference=False,
                             infer_feed_dict=None, infer_labels = None):
        t_test = time.time()
        val_losses = []
        val_preds = []
        labels = []
        iter_num = 0
        finished = False

        while not finished:
            if infer_feed_dict is None:
                feed_dict_val, batch_labels, finished, _ = minibatch_iter.incremental_node_val_feed_dict(size,
                                                                                                         iter_num,
                                                                  test=test, inference=inference)
                if inference:
                    finished = True
            else:
                #feed_dict_val, batch_inf, batch_labels = infer_feed_dict#minibatch_iter.inference_feed_dict()
                feed_dict_val = infer_feed_dict
                batch_labels = infer_labels
                finished = True
            node_outs_val = sess.run([model.preds, model.loss],
                                     feed_dict=feed_dict_val)
            val_preds.append(node_outs_val[0])

            # add inference labels
            minibatch_iter.update_batch_prediction(node_outs_val[0])

            preds = self.pred_values(node_outs_val[0])

            #minibatch_iter.update_batch_prediction(val_preds)

            labels.append(batch_labels)
            val_losses.append(node_outs_val[1])
            iter_num += 1
        val_preds = np.vstack(val_preds)
        labels = np.vstack(labels)
        f1_scores = self.calc_f1(labels, val_preds)
        return np.mean(val_losses), f1_scores[0], f1_scores[1], (time.time() - t_test)

    def _make_label_vec(self, node, class_map, num_classes):
        label = class_map[node]
        if isinstance(label, list):
            label_vec = np.array(label, dtype=np.float32)
        else:
            label_vec = np.zeros((num_classes), dtype=np.float32)
            class_ind = class_map[node]
            label_vec[class_ind] = 1
        return label_vec



    def construct_placeholders(self, num_classes, batch=None, labels = None, inf_batch_shape=None,
                               class_map=None, id_map = None, name_prefix='', infer=False):
        # Define placeholders
        inf_batch_shape=None


        bs = tf.placeholder(tf.int32, shape=(), name="batch_size")
        ls = tf.placeholder(tf.float32, shape=(inf_batch_shape, num_classes), name="labels")
        #ls2 = tf.placeholder(tf.float32, shape=(inf_batch_shape, num_classes), name=name_prefix + 'labels2')
        b = tf.placeholder(tf.int32, shape=(inf_batch_shape), name="batch")
        geto_batch = tf.placeholder(tf.int32, shape=(inf_batch_shape), name="getobatch")
        dpo = tf.placeholder_with_default(0., shape=(),  name="dropout")
        #b1 = tf.placeholder(tf.int32, shape=(inf_batch_shape), name=name_prefix + 'batch1')
        if not infer:
            placeholders = {
                'labels': ls,
                'batch': b,
                'getobatch': geto_batch,
                'dropout': dpo,
                'batch_size': bs, #tf.placeholder(tf.int32,inf_batch_shape, name=name_prefix+'batch_size'),
            }
        else:
            placeholders = {
                'batch' : b,
                'labels' : ls,
                'getobatch': geto_batch,
                #'batch1' : b1,                         #tf.placeholder(tf.int32, shape=(None), name='batch1'),
                #'batch2' : ls2,                        #tf.placeholder(tf.int32, shape=(None), name='batch2'),
                # negative samples for all nodes in the batch
                #'neg_samples': tf.placeholder(tf.int32, shape=(None,),
                #    name='neg_sample_size'),
                'dropout': dpo,
                'batch_size' : bs,                    #tf.placeholder(tf.int32, name='batch_size'),
            }
        return placeholders

    def fetch_placeholders(self, batch, labels):


        #bs = tf.get_default_graph().get_tensor_by_name("batch_size")
        #ls = tf.get_default_graph().get_tensor_by_name("labels")
        #b = tf.get_default_graph().get_tensor_by_name("batch")
        #dpo = tf.get_default_graph().get_tensor_by_name("dropout")
        placeholders = {
            'batch': batch,#b,
            'labels': labels, #ls,
            # 'batch1' : b1,                         #tf.placeholder(tf.int32, shape=(None), name='batch1'),
            # 'batch2' : ls2,                        #tf.placeholder(tf.int32, shape=(None), name='batch2'),
            # negative samples for all nodes in the batch
            # 'neg_samples': tf.placeholder(tf.int32, shape=(None,),
            #    name='neg_sample_size'),
            'dropout': 0.,#dpo,
            'batch_size':  len(batch), #bs,  # tf.placeholder(tf.int32, name='batch_size'),
        }
        return placeholders

    def _train(self, train_data, test_data=None):




        G = train_data[0]
        features = train_data[1]
        id_map = train_data[2]
        class_map = train_data[4]
        if isinstance(list(class_map.values())[0], list):
            num_classes = len(list(class_map.values())[0])
        else:
            num_classes = len(set(class_map.values()))

        if not features is None:
            # pad with dummy zero vector
            features = np.vstack([features, np.zeros((features.shape[1],))])

        FLAGS = self.FLAGS

        pout(["    * learning rate", self.FLAGS.learning_rate,
              "epochs",FLAGS.epochs,
              "weight decay",FLAGS.weight_decay])
        context_pairs = train_data[3] if FLAGS.random_context else None
        placeholders = self.construct_placeholders(num_classes)
        self.placeholders = placeholders

        minibatch = NodeMinibatchIterator(G,
                                          id_map,
                                          placeholders,
                                          class_map,
                                          num_classes,
                                          nx_idx_to_getoelm_idx=self.nx_idx_to_getoelm_idx,
                                          batch_size=FLAGS.batch_size,
                                          max_degree=FLAGS.max_degree,
                                          context_pairs=context_pairs)
        self.adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
        adj_info_ph = self.adj_info_ph
        adj_info = tf.Variable(initial_value=minibatch.adj, shape=minibatch.adj.shape
                               , trainable=False, name="adj_info", dtype=tf.int32)

        geto_adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.geto_adj.shape) if self.use_geto else None
        geto_adj_info = tf.Variable(geto_adj_info_ph, trainable=False, name="geto_adj_info") if self.use_geto  else None

        print(" ... Using aggregator: ", FLAGS.model)

        if FLAGS.model == 'graphsage_mean':
            # Create model
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node_gsmean", sampler, FLAGS.samples_1, FLAGS.dim_1)]  # ,
            # SAGEInfo("node_maxpool", sampler, FLAGS.samples_2, FLAGS.dim_2)]
            for i in range(3, self.depth + 1):
                # layer_infos.append(SAGEInfo("node_maxpool", sampler, FLAGS.samples_1, FLAGS.dim_1))
                layer_infos.append(SAGEInfo("node_gsmean", sampler, FLAGS.samples_2, FLAGS.dim_2))
            layer_infos.append(SAGEInfo("node_gsmean", sampler, FLAGS.samples_2, FLAGS.dim_2))


            model = SupervisedGraphsage(num_classes, placeholders,
                                        features,
                                        adj_info,
                                        minibatch.deg,
                                        layer_infos,
                                        concat=self.concat,
                                        geto_loss=self.geto_loss,
                                        jumping_knowledge=FLAGS.jumping_knowledge,
                                        jump_type=self.jump_type,
                                        hidden_dim_1_agg=FLAGS.hidden_dim_1_agg,
                                        hidden_dim_2_agg=FLAGS.hidden_dim_2_agg,
                                        model_size=FLAGS.model_size,
                                        sigmoid_loss=FLAGS.sigmoid,
                                        identity_dim=FLAGS.identity_dim,
                                        logging=True)
        elif FLAGS.model == 'gcn':
            # Create model
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node_gcn", sampler, FLAGS.samples_1, 2 * FLAGS.dim_1),
                           SAGEInfo("node_gcn", sampler, FLAGS.samples_2, 2 * FLAGS.dim_2)]

            model = SupervisedGraphsage(num_classes, placeholders,
                                        features,
                                        adj_info,
                                        minibatch.deg,
                                        geto_loss=self.geto_loss,
                                        jumping_knowledge=FLAGS.jumping_knowledge,
                                        jump_type=self.jump_type,
                                        hidden_dim_agg=FLAGS.hidden_dim_1_agg,
                                        layer_infos=layer_infos,
                                        aggregator_type="gcn",
                                        model_size=FLAGS.model_size,
                                        concat=False,
                                        sigmoid_loss=FLAGS.sigmoid,
                                        identity_dim=FLAGS.identity_dim,
                                        logging=True)

        elif FLAGS.model == 'graphsage_seq':
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node_gsseq", sampler, FLAGS.samples_1, FLAGS.dim_1),
                           SAGEInfo("node_gsseq", sampler, FLAGS.samples_2, FLAGS.dim_2)]

            model = SupervisedGraphsage(num_classes, placeholders,
                                        features,
                                        adj_info,
                                        minibatch.deg,
                                        geto_loss=self.geto_loss,
                                        layer_infos=layer_infos,
                                        aggregator_type="seq",
                                        model_size=FLAGS.model_size,
                                        sigmoid_loss=FLAGS.sigmoid,
                                        identity_dim=FLAGS.identity_dim,
                                        logging=True)

        elif FLAGS.model == 'graphsage_maxpool':
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node_maxpool", sampler, FLAGS.samples_1, FLAGS.dim_1)]#,
                           #SAGEInfo("node_maxpool", sampler, FLAGS.samples_2, FLAGS.dim_2)]
            for i in range(3, self.depth+1):
                #layer_infos.append(SAGEInfo("node_maxpool", sampler, FLAGS.samples_1, FLAGS.dim_1))
                layer_infos.append(SAGEInfo("node_maxpool", sampler, FLAGS.samples_2, FLAGS.dim_2))
            layer_infos.append(SAGEInfo("node_maxpool", sampler, FLAGS.samples_2, FLAGS.dim_2))

            print(">>> Using positive class weight: ", FLAGS.positive_class_weight)

            model = SupervisedGraphsage(num_classes, placeholders,
                                        features,
                                        adj_info,
                                        minibatch.deg,
                                        concat=self.concat,
                                        geto_loss=self.geto_loss,
                                        jumping_knowledge=FLAGS.jumping_knowledge,
                                        jump_type = self.jump_type,
                                        hidden_dim_1_agg=FLAGS.hidden_dim_1_agg,
                                        hidden_dim_2_agg=FLAGS.hidden_dim_2_agg,
                                        positive_class_weight=FLAGS.positive_class_weight,
                                        layer_infos=layer_infos,
                                        aggregator_type="maxpool",
                                        model_size=FLAGS.model_size,
                                        sigmoid_loss=FLAGS.sigmoid,
                                        identity_dim=FLAGS.identity_dim,
                                        logging=True)

        elif FLAGS.model == 'graphsage_meanpool':
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node_meanpool", sampler, FLAGS.samples_1, FLAGS.dim_1)]  # ,
            # SAGEInfo("node_maxpool", sampler, FLAGS.samples_2, FLAGS.dim_2)]
            for i in range(3, self.depth + 1):
                # layer_infos.append(SAGEInfo("node_maxpool", sampler, FLAGS.samples_1, FLAGS.dim_1))
                layer_infos.append(SAGEInfo("node_meanpool", sampler, FLAGS.samples_2, FLAGS.dim_2))
            layer_infos.append(SAGEInfo("node_meanpool", sampler, FLAGS.samples_2, FLAGS.dim_2))

            model = SupervisedGraphsage(num_classes, placeholders,
                                        features,
                                        adj_info,
                                        minibatch.deg,
                                        geto_loss=self.geto_loss,
                                        jumping_knowledge=FLAGS.jumping_knowledge,
                                        jump_type=self.jump_type,
                                        hidden_dim_1_agg=FLAGS.hidden_dim_1_agg,
                                        hidden_dim_2_agg=FLAGS.hidden_dim_2_agg,
                                        layer_infos=layer_infos,
                                        aggregator_type="meanpool",
                                        model_size=FLAGS.model_size,
                                        sigmoid_loss=FLAGS.sigmoid,
                                        identity_dim=FLAGS.identity_dim,
                                        logging=True)
        elif self.FLAGS.model == 'geto-meanpool':
            sampler = UniformNeighborSampler(adj_info, geto_adj_info=geto_adj_info)
            layer_infos = [SAGEInfo("node_getomeanpool", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1)]  # ,
            # SAGEInfo("node_maxpool", sampler, FLAGS.samples_2, FLAGS.dim_2)]
            for i in range(3, self.depth + 1):
                # layer_infos.append(SAGEInfo("node_maxpool", sampler, FLAGS.samples_1, FLAGS.dim_1))
                layer_infos.append(SAGEInfo("node_getomeanpool", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2))
            layer_infos.append(SAGEInfo("node_getomeanpool", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2))

            model = SupervisedGraphsage(num_classes,placeholders,
                                       features,
                                       adj_info,
                                       minibatch.deg,
                                       geto_elements=self.geto_elements,
                                       geto_adj_info=geto_adj_info,
                                       concat=self.concat,
                                        geto_loss=self.geto_loss,
                                       jumping_knowledge=self.FLAGS.jumping_knowledge,
                                       jump_type=self.jump_type,
                                       hidden_dim_1_agg=self.FLAGS.hidden_dim_1_agg,
                                       hidden_dim_2_agg=self.FLAGS.hidden_dim_2_agg,
                                       positive_class_weight=FLAGS.positive_class_weight,
                                       layer_infos=layer_infos,
                                       aggregator_type="geto-meanpool",
                                       model_size=self.FLAGS.model_size,
                                       identity_dim=self.FLAGS.identity_dim,
                                       logging=True)
        elif self.FLAGS.model == 'geto-maxpool':
            sampler = UniformNeighborSampler(adj_info, geto_adj_info=geto_adj_info)
            layer_infos = [SAGEInfo("node_getomeanpool", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1)]  # ,
            # SAGEInfo("node_maxpool", sampler, FLAGS.samples_2, FLAGS.dim_2)]
            for i in range(3, self.depth + 1):
                # layer_infos.append(SAGEInfo("node_maxpool", sampler, FLAGS.samples_1, FLAGS.dim_1))
                layer_infos.append(SAGEInfo("node_getomeanpool", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2))
            layer_infos.append(SAGEInfo("node_getomeanpool", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2))

            model = SupervisedGraphsage(num_classes,placeholders,
                                       features,
                                       adj_info,
                                       minibatch.deg,
                                       geto_elements=self.geto_elements,
                                       geto_adj_info=geto_adj_info,
                                       concat=self.concat,
                                        geto_loss=self.geto_loss,
                                       jumping_knowledge=self.FLAGS.jumping_knowledge,
                                       jump_type=self.jump_type,
                                       hidden_dim_1_agg=self.FLAGS.hidden_dim_1_agg,
                                       hidden_dim_2_agg=self.FLAGS.hidden_dim_2_agg,
                                       positive_class_weight=FLAGS.positive_class_weight,
                                       layer_infos=layer_infos,
                                       aggregator_type="geto-maxpool",
                                       model_size=self.FLAGS.model_size,
                                       identity_dim=self.FLAGS.identity_dim,
                                       logging=True)
        elif self.FLAGS.model == 'geto-sampling_maxpool':
            sampler = GeToInformedNeighborSampler(adj_info,
                                                  geto_adj_info=geto_adj_info,
                                                  adj_probs=self.geto_elements,
                                                  resampling_rate=0)
            layer_infos = [SAGEInfo("node_getomeanpool", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1)]  # ,
            # SAGEInfo("node_maxpool", sampler, FLAGS.samples_2, FLAGS.dim_2)]
            for i in range(3, self.depth + 1):
                # layer_infos.append(SAGEInfo("node_maxpool", sampler, FLAGS.samples_1, FLAGS.dim_1))
                layer_infos.append(SAGEInfo("node_getomeanpool", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2))
            layer_infos.append(SAGEInfo("node_getomeanpool", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2))

            model = SupervisedGraphsage(num_classes, placeholders,
                                        features,
                                        adj_info,
                                        minibatch.deg,
                                        concat=self.concat,
                                        jumping_knowledge=FLAGS.jumping_knowledge,
                                        jump_type=self.jump_type,
                                        geto_loss=self.geto_loss,
                                        hidden_dim_1_agg=FLAGS.hidden_dim_1_agg,
                                        hidden_dim_2_agg=FLAGS.hidden_dim_2_agg,
                                        positive_class_weight=FLAGS.positive_class_weight,
                                        layer_infos=layer_infos,
                                        aggregator_type="maxpool",
                                        model_size=FLAGS.model_size,
                                        sigmoid_loss=FLAGS.sigmoid,
                                        identity_dim=FLAGS.identity_dim,
                                        logging=True)
        elif self.FLAGS.model == 'hidden-geto_meanpool':
            sampler = UniformNeighborSampler(adj_info, geto_adj_info=geto_adj_info)
            layer_infos = [SAGEInfo("node_hidden-geto_maxpool", sampler,
                                    self.FLAGS.samples_1, self.FLAGS.dim_1)]  # ,
            # SAGEInfo("node_maxpool", sampler, FLAGS.samples_2, FLAGS.dim_2)]
            for i in range(3, self.depth + 1):
                # layer_infos.append(SAGEInfo("node_maxpool", sampler, FLAGS.samples_1, FLAGS.dim_1))
                layer_infos.append(SAGEInfo("node_hidden-geto_maxpool", sampler,
                                            self.FLAGS.samples_2, self.FLAGS.dim_2))
            layer_infos.append(SAGEInfo("node_hidden-geto_maxpool", sampler,
                                        self.FLAGS.samples_2, self.FLAGS.dim_1))

            model = SupervisedGraphsage(num_classes, placeholders,
                                        features,
                                        adj_info,
                                        minibatch.deg,
                                        geto_elements=self.geto_elements,
                                        geto_adj_info=geto_adj_info,
                                        concat=self.concat,
                                        geto_loss=self.geto_loss,
                                        jumping_knowledge=FLAGS.jumping_knowledge,
                                        jump_type=self.jump_type,
                                        hidden_dim_1_agg=FLAGS.hidden_dim_1_agg,
                                        hidden_dim_2_agg=FLAGS.hidden_dim_2_agg,
                                        positive_class_weight=FLAGS.positive_class_weight,
                                        layer_infos=layer_infos,
                                        aggregator_type="hidden-geto_meanpool",
                                        model_size=FLAGS.model_size,
                                        sigmoid_loss=FLAGS.sigmoid,
                                        identity_dim=FLAGS.identity_dim,
                                        logging=True)
        elif self.FLAGS.model == 'geto-edge':
            sampler = GeToInformedNeighborSampler(adj_info,
                                                  batch_size=FLAGS.batch_size,
                                                  geto_adj_info=geto_adj_info,
                                                  adj_probs=self.geto_elements,
                                                  resampling_rate=0)
            layer_infos = [SAGEInfo("node_hidden-geto_maxpool", sampler,
                                    self.FLAGS.samples_1, self.FLAGS.dim_1)]  # ,
            # SAGEInfo("node_maxpool", sampler, FLAGS.samples_2, FLAGS.dim_2)]
            for i in range(3, self.depth + 1):
                # layer_infos.append(SAGEInfo("node_maxpool", sampler, FLAGS.samples_1, FLAGS.dim_1))
                layer_infos.append(SAGEInfo("node_hidden-geto_maxpool", sampler,
                                            self.FLAGS.samples_2, self.FLAGS.dim_2))
            layer_infos.append(SAGEInfo("node_hidden-geto_maxpool", sampler,
                                        self.FLAGS.samples_2, self.FLAGS.dim_1))

            model = SupervisedGraphsage(num_classes, placeholders,
                                        features,
                                        adj_info,
                                        minibatch.deg,
                                        geto_elements=self.geto_elements,
                                        geto_adj_info=geto_adj_info,
                                        concat=self.concat,
                                        geto_loss=self.geto_loss,
                                        jumping_knowledge=FLAGS.jumping_knowledge,
                                        jump_type=self.jump_type,
                                        hidden_dim_1_agg=FLAGS.hidden_dim_1_agg,
                                        hidden_dim_2_agg=FLAGS.hidden_dim_2_agg,
                                        positive_class_weight=FLAGS.positive_class_weight,
                                        layer_infos=layer_infos,
                                        aggregator_type="geto-edge",
                                        model_size=FLAGS.model_size,
                                        sigmoid_loss=FLAGS.sigmoid,
                                        identity_dim=FLAGS.identity_dim,
                                        logging=True)
        elif self.FLAGS.model == 'hidden-geto-meanpool_geto-sampling':
            sampler = GeToInformedNeighborSampler(adj_info,
                                                  batch_size=FLAGS.batch_size,
                                                  geto_adj_info=geto_adj_info,
                                                  adj_probs=self.geto_elements,
                                                  resampling_rate=0)
            layer_infos = [SAGEInfo("node_hidden-geto_maxpool", sampler,
                                    self.FLAGS.samples_1, self.FLAGS.dim_1)]  # ,
            # SAGEInfo("node_maxpool", sampler, FLAGS.samples_2, FLAGS.dim_2)]
            for i in range(3, self.depth + 1):
                # layer_infos.append(SAGEInfo("node_maxpool", sampler, FLAGS.samples_1, FLAGS.dim_1))
                layer_infos.append(SAGEInfo("node_hidden-geto_maxpool", sampler,
                                            self.FLAGS.samples_2, self.FLAGS.dim_2))
            layer_infos.append(SAGEInfo("node_hidden-geto_maxpool", sampler,
                                        self.FLAGS.samples_2, self.FLAGS.dim_1))

            model = SupervisedGraphsage(num_classes, placeholders,
                                        features,
                                        adj_info,
                                        minibatch.deg,
                                        geto_elements=self.geto_elements,
                                        geto_adj_info=geto_adj_info,
                                        concat=self.concat,
                                        geto_loss=self.geto_loss,
                                        jumping_knowledge=FLAGS.jumping_knowledge,
                                        jump_type=self.jump_type,
                                        hidden_dim_1_agg=FLAGS.hidden_dim_1_agg,
                                        hidden_dim_2_agg=FLAGS.hidden_dim_2_agg,
                                        positive_class_weight=FLAGS.positive_class_weight,
                                        layer_infos=layer_infos,
                                        aggregator_type="hidden-geto_meanpool",
                                        model_size=FLAGS.model_size,
                                        sigmoid_loss=FLAGS.sigmoid,
                                        identity_dim=FLAGS.identity_dim,
                                        logging=True)
        elif self.FLAGS.model == 'hidden-geto_maxpool':
            sampler = UniformNeighborSampler(adj_info, geto_adj_info=geto_adj_info)
            layer_infos = [SAGEInfo("node_hidden-geto_maxpool", sampler,
                                    self.FLAGS.samples_1, self.FLAGS.dim_1)]  # ,
            # SAGEInfo("node_maxpool", sampler, FLAGS.samples_2, FLAGS.dim_2)]
            for i in range(3, self.depth + 1):
                # layer_infos.append(SAGEInfo("node_maxpool", sampler, FLAGS.samples_1, FLAGS.dim_1))
                layer_infos.append(SAGEInfo("node_hidden-geto_maxpool", sampler,
                                            self.FLAGS.samples_2, self.FLAGS.dim_2))
            layer_infos.append(SAGEInfo("node_hidden-geto_maxpool", sampler,
                                        self.FLAGS.samples_2, self.FLAGS.dim_1))

            model = SupervisedGraphsage(num_classes, placeholders,
                                        features,
                                        adj_info,
                                        minibatch.deg,
                                        geto_loss=self.geto_loss,
                                        geto_elements=self.geto_elements,
                                        geto_adj_info=geto_adj_info,
                                        concat=self.concat,
                                        jumping_knowledge=FLAGS.jumping_knowledge,
                                        jump_type=self.jump_type,
                                        hidden_dim_1_agg=FLAGS.hidden_dim_1_agg,
                                        hidden_dim_2_agg=FLAGS.hidden_dim_2_agg,
                                        positive_class_weight=FLAGS.positive_class_weight,
                                        layer_infos=layer_infos,
                                        aggregator_type="hidden-geto_maxpool",
                                        model_size=FLAGS.model_size,
                                        sigmoid_loss=FLAGS.sigmoid,
                                        identity_dim=FLAGS.identity_dim,
                                        logging=True)
        elif self.FLAGS.model == 'geto-sampling_geto-meanpool':
            sampler = GeToInformedNeighborSampler(adj_info,
                                                  geto_adj_info=geto_adj_info,
                                                  adj_probs=self.geto_elements,
                                                  resampling_rate=0)
            layer_infos = [SAGEInfo("node_geto-sampling_geto-meanpool", sampler,
                                    self.FLAGS.samples_1, self.FLAGS.dim_1)]  # ,
            # SAGEInfo("node_maxpool", sampler, FLAGS.samples_2, FLAGS.dim_2)]
            for i in range(3, self.depth + 1):
                # layer_infos.append(SAGEInfo("node_maxpool", sampler, FLAGS.samples_1, FLAGS.dim_1))
                layer_infos.append(SAGEInfo("node_geto-sampling_geto-meanpool", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2))
            layer_infos.append(SAGEInfo("node_geto-sampling_geto-meanpool", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2))

            model = SupervisedGraphsage(num_classes,placeholders,
                                       features,
                                       adj_info,
                                       minibatch.deg,
                                        geto_loss=self.geto_loss,
                                       geto_elements=self.geto_elements,
                                       geto_weights=self.geto_elements,
                                       geto_adj_info=geto_adj_info,
                                       concat=self.concat,
                                       jumping_knowledge=self.FLAGS.jumping_knowledge,
                                       jump_type=self.jump_type,
                                       hidden_dim_1_agg=self.FLAGS.hidden_dim_1_agg,
                                       hidden_dim_2_agg=self.FLAGS.hidden_dim_2_agg,
                                       positive_class_weight=FLAGS.positive_class_weight,
                                       layer_infos=layer_infos,
                                       aggregator_type="geto-meanpool",
                                       model_size=self.FLAGS.model_size,
                                       identity_dim=self.FLAGS.identity_dim,
                                       logging=True)
        elif self.FLAGS.model == 'geto-sampling_geto-maxpool':
            sampler = GeToInformedNeighborSampler(adj_info,
                                                  geto_adj_info=geto_adj_info,
                                                  adj_probs=self.geto_elements,
                                                  resampling_rate=0)
            layer_infos = [SAGEInfo("node_geto-sampling_geto-maxpool", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1)]  # ,
            # SAGEInfo("node_maxpool", sampler, FLAGS.samples_2, FLAGS.dim_2)]
            for i in range(3, self.depth + 1):
                # layer_infos.append(SAGEInfo("node_maxpool", sampler, FLAGS.samples_1, FLAGS.dim_1))
                layer_infos.append(SAGEInfo("node_geto-sampling_geto-maxpool", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2))
            layer_infos.append(SAGEInfo("node_geto-sampling_geto-maxpool", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2))

            model = SupervisedGraphsage(num_classes,placeholders,
                                       features,
                                       adj_info,
                                       minibatch.deg,
                                        geto_loss=self.geto_loss,
                                       geto_elements=self.geto_elements,
                                       geto_weights=self.geto_elements,
                                       geto_adj_info=geto_adj_info,
                                       concat=self.concat,
                                       jumping_knowledge=self.FLAGS.jumping_knowledge,
                                       jump_type=self.jump_type,
                                       hidden_dim_1_agg=self.FLAGS.hidden_dim_1_agg,
                                       hidden_dim_2_agg=self.FLAGS.hidden_dim_2_agg,
                                       positive_class_weight=FLAGS.positive_class_weight,
                                       layer_infos=layer_infos,
                                       aggregator_type="geto-maxpool",
                                       model_size=self.FLAGS.model_size,
                                       identity_dim=self.FLAGS.identity_dim,
                                       logging=True)
        else:
            print('are equal ', FLAGS.model == 'graphsage_maxpool')
            raise Exception('Error: model name unrecognized.')

        config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = self.GPU_MEM_FRACTION
        config.allow_soft_placement = True

        # Initialize session
        sess = tf.Session(config=config)
        merged = tf.compat.v1.summary.merge_all()
        #summary_writer = tf.summary.FileWriter(self.log_dir(), sess.graph)

        # Init variables
        if self.use_geto:
            sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph: minibatch.adj, geto_adj_info_ph:minibatch.geto_adj})
        else:
            sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph: minibatch.adj})
        print("Training adj graph shape: ", minibatch.adj.shape)
        print("Test trian adj shape: ", minibatch.test_adj.shape)
        # save during training
        #saver = tf.compat.v1.train.Saver(tf.all_variables()) #save_relative_paths=True)
        model_checkpoint = self.log_dir()+'model-session.ckpt'
        # Train model

        total_steps = 0
        avg_time = 0.0
        epoch_val_costs = []

        train_adj_info = tf.assign(adj_info, minibatch.adj)
        val_adj_info = tf.assign(adj_info, minibatch.test_adj,name='adj_assign')
        if self.use_geto:
            train_geto_adj_info = tf.assign(geto_adj_info, minibatch.geto_adj)
            val_geto_adj_info = tf.assign(geto_adj_info, minibatch.test_geto_adj, name='geto_adj_assign')

        start_time = time.time()
        for epoch in range(FLAGS.epochs):
            minibatch.shuffle()

            iter = 0

            epoch_val_costs.append(0)
            while not minibatch.end():
                # Construct feed dictionary
                feed_dict, labels = minibatch.next_minibatch_feed_dict()
                feed_dict.update({placeholders['dropout']: FLAGS.dropout})

                t = time.time()
                # Training step
                outs = sess.run([merged, model.opt_op, model.loss, model.preds],
                                feed_dict=feed_dict)
                if self.geto_loss and "hidden" in self.FLAGS.model:
                    geto_outs = sess.run([merged, model.geto_opt_op, model.geto_loss, model.geto_preds],
                                    feed_dict=feed_dict)
                    geto_train_cost = geto_outs[2]
                train_cost = outs[2]

                preds = self.pred_values(outs[-1])
                minibatch.update_batch_prediction(preds)

                if iter % FLAGS.validate_iter == 0:
                    # Validation
                    sess.run(val_adj_info.op)
                    if FLAGS.validate_batch_size == -1:
                        val_cost, val_f1_mic, val_f1_mac, duration = self.incremental_evaluate(sess, model,
                                                                                               minibatch,
                                                                                          FLAGS.batch_size)
                    else:
                        val_cost, val_f1_mic, val_f1_mac, duration = self.evaluate(sess, model, minibatch,
                                                                              FLAGS.validate_batch_size)
                    sess.run(train_adj_info.op)
                    epoch_val_costs[-1] += val_cost

                    #saver.save(sess, model_checkpoint)#, global_step=total_steps)

                #if total_steps % FLAGS.print_every == 0:
                #    summary_writer.add_summary(outs[0], total_steps)

                # Print results
                avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)

                if total_steps % FLAGS.print_every == 0:
                    print('Epoch: %04d' % (epoch + 1))
                    train_f1_mic, train_f1_mac = self.calc_f1(labels, outs[-1])
                    print("Iter:", '%04d' % iter,
                          "train_loss=", "{:.5f}".format(train_cost),
                          "train_f1_mic=", "{:.5f}".format(train_f1_mic),
                          "train_f1_mac=", "{:.5f}".format(train_f1_mac),
                          "val_loss=", "{:.5f}".format(val_cost),
                          "val_f1_mic=", "{:.5f}".format(val_f1_mic),
                          "val_f1_mac=", "{:.5f}".format(val_f1_mac),
                          "time=", "{:.5f}".format(avg_time))
                    if self.geto_loss and "hidden" in self.FLAGS.model:
                        geto_train_f1_mic, geto_train_f1_mac = self.calc_f1(labels, geto_outs[-1])
                        print("Iter:", '%04d' % iter,
                              "geto_train_loss=", "{:.5f}".format(geto_train_cost),
                              "geto_train_f1_mic=", "{:.5f}".format(geto_train_f1_mic),
                              "geto_train_f1_mac=", "{:.5f}".format(geto_train_f1_mac))
                              #"val_loss=", "{:.5f}".format(val_cost),
                              #"val_f1_mic=", "{:.5f}".format(val_f1_mic),
                              #"val_f1_mac=", "{:.5f}".format(val_f1_mac),
                              #"time=", "{:.5f}".format(avg_time))

                iter += 1
                total_steps += 1

                if total_steps > FLAGS.max_total_steps:
                    break

            if total_steps > FLAGS.max_total_steps:
                break
        end_train_time = time.time()

        print("Optimization Finished!")
        sess.run(val_adj_info.op)
        val_cost, val_f1_mic, val_f1_mac, duration = self.incremental_evaluate(sess, model, minibatch, FLAGS.batch_size)
        print("Full validation stats:",
              "loss=", "{:.5f}".format(val_cost),
              "f1_micro=", "{:.5f}".format(val_f1_mic),
              "f1_macro=", "{:.5f}".format(val_f1_mac),
              "time=", "{:.5f}".format(duration))
        with open(self.log_dir() + "val_stats.txt", "w") as fp:
            fp.write("loss={:.5f} f1_micro={:.5f} f1_macro={:.5f} time={:.5f}".
                     format(val_cost, val_f1_mic, val_f1_mac, duration))



        self.train_time = end_train_time-start_time

        s = time.time()

        val_cost, val_f1_mic, val_f1_mac, duration = self.incremental_evaluate(sess, model, minibatch, FLAGS.batch_size,
                                                                          test=True)
        t = time.time()
        self.pred_time = t-s

        print("Writing test set stats to file (don't peak!)")
        with open(self.log_dir() + "test_stats.txt", "w") as fp:
            fp.write("loss={:.5f} f1_micro={:.5f} f1_macro={:.5f}".
                     format(val_cost, val_f1_mic, val_f1_mac))

        # save session
        print(" >>>>> Saving final session in: ", self.log_dir())
        #saver.save(sess, model_checkpoint)#, global_step=FLAGS.max_total_steps+1)

        self.G = minibatch.get_graph()
        self.model = model
        self.model_path = model_checkpoint
        self.sess = sess
        tf.reset_default_graph()
        sess.close()

    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #




    def get_graph(self):
        return self.G

    def get_graph_prediction(self):
        return self.inference_G




    def get_data(self, train_data=None):
        print("Loading training data..")
        #train_data = load_data(FLAGS.train_prefix)

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
                print(">>>")
                print("first elif")
                print(">>>>")
                train_prefix = 'nNeg-' + str(self.number_negative_samples) + 'nPos-' + str(self.number_positive_samples)
                print("using pre-processed graph data for gnn training")
                self.number_negative_samples = self.number_negative_samples
                number_samples = self.number_negative_samples + self.number_positive_samples
                proportion_negative = int(number_samples / float(self.number_negative_samples))
                train_data = (self.G, self.feats, self.id_map, self.walks, self.class_map, [], [])

            # train from cvt sampled graph and respective in/out arcs as train
            #elif self.positive_arcs and self.negative_arcs:
            #    train_data = load_data(self.positive_arcs, self.negative_arcs, load_walks=self.load_walks, scheme_required=True,
            #                           train_or_test='train')
            #    self.number_negative_samples = len(self.negative_arcs)
            #    number_samples = len(self.positive_arcs) + len(self.negative_arcs)
            #    proportion_negative = int(number_samples / float(self.number_negative_samples))

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


"""    def main(self, argv=None):
        print("Loading training data..")
        #train_data = load_data(FLAGS.train_prefix)

        ### load MSC data
        print('loading msc graph data')

        train_data=self.get_data()
        ## use msc graph data
        # begin training
        print('Begin GNN training')
        print('')
        if self.msc_collection is None:
            self._train(train_data[:-2])
        # else:
        #    self.batch_train(self.msc_collection)
        print("Done loading training data..")
        #_train(train_data)

        if __name__ == '__main__':
            tf.app.run()"""

