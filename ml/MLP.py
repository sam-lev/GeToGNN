from __future__ import print_function

from mlgraph import MLGraph
from proc_manager import experiment_manager
from ml.utils import get_partition_feature_label_pairs
from data_ops.collect_data import collect_training_data, collect_datasets # compute_geomsc,
from ml.features import get_points_from_vertices
from data_ops.utils import tile_region
from metrics.model_metrics import compute_prediction_metrics
from compute_multirun_metrics import multi_run_metrics
from data_ops.utils import plot

import pandas as pd
import tensorflow as tf
import numpy as np
import os
import time
from scipy import ndimage
from ml.features import multiscale_basic_features
from functools import partial
from copy import deepcopy
from sklearn.metrics import f1_score

class mlp(MLGraph):
    def __init__(self,
                 n_classes = 2,
                 model='msc', growth_radius=2,
                 run_num=0, parameter_file_number=None,
                 geomsc_fname_base=None, label_file=None,
                 model_name=None, load_feature_graph_name=False, image=None,
                 flavor = 'msc',  X_BOX=None,Y_BOX=None,
                 BEGIN_LOADING_FEATURES=False, ground_truth_label_file=None, boxes=None,
                 **kwargs):

        self.type = 'mlp'

        self.model_name = model_name

        self.details = "Multi Layer Perceptron Trainer and Classifier"

        self.type = "mlp" + '_' + flavor
        self.parameter_file_number = parameter_file_number

        self.G = None
        self.G_dict = {}


        super(mlp, self).__init__(parameter_file_number=parameter_file_number, run_num=run_num,
                                  name=kwargs['name'], geomsc_fname_base=geomsc_fname_base,
                                  label_file=label_file, image=image, write_folder=kwargs['write_folder'],
                                  model_name=model_name, load_feature_graph_name=load_feature_graph_name)

        self.attributes = self.get_attributes()

        self.run_num = run_num
        self.logger = experiment_manager.experiment_logger(experiment_folder=self.experiment_folder,
                                                           input_folder=self.input_folder)
        self.param_file = os.path.join(self.LocalSetup.project_base_path,
                                       'parameter_list_' + str(parameter_file_number) + '.txt')
        self.topo_image_name = label_file.split('.labels')[0]
        self.logger.record_filename(label_file=label_file,
                                    parameter_list_file=self.param_file,
                                    image_name=image,
                                    topo_image_name=self.topo_image_name)

        self.X_BOX = X_BOX
        self.Y_BOX = Y_BOX
        self.name = kwargs['name']
        self.image_path = image
        #
        # From build
        #
        self.attributes = self.get_attributes()

        if BEGIN_LOADING_FEATURES:
            self.params['load_features'] = True
            self.params['write_features'] = False
            self.params['load_features'] = True
            self.params['write_feature_names'] = False
            self.params['save_filtered_images'] = False
            self.params['collect_features'] = False
            self.params['load_preprocessed'] = True
            self.params['load_geto_attr'] = True
            self.params['load_feature_names'] = True
        else:
            self.params['load_features'] = False
            self.params['write_features'] = True
            self.params['load_features'] = False
            self.params['write_feature_names'] = True
            self.params['save_filtered_images'] = True
            self.params['collect_features'] = True
            self.params['load_preprocessed'] = False
            self.params['load_geto_attr'] = False
            self.params['load_feature_names'] = False

        if self.params['load_geto_attr']:
            if self.params['geto_as_feat']:
                self.load_geto_features()
                self.load_geto_feature_names()

        # features
        if not self.params['load_features']:
            self.compile_features(include_geto=self.params['geto_as_feat'])
            self.write_gnode_features(self.session_name)
            self.write_feature_names()
        else:
            self.load_gnode_features()
            self.load_feature_names()

            # if 'geto' in self.params['aggregator']:
            #    self.load_geto_features()
            #    self.load_geto_feature_names()

        if self.params['write_features']:
            self.write_gnode_features(self.session_name)

        if self.params['write_feature_names']:
            self.write_feature_names()

        if self.params['write_feature_names']:
            if self.params['geto_as_feat']:
                self.write_geto_feature_names()
        if self.params['write_features']:
            if self.params['geto_as_feat']:
                self.write_geto_features(self.session_name)
        # training info, selection, partition train/val/test
        self.read_labels_from_file(file=ground_truth_label_file)

        self.boxes = boxes

        self.data_array, self.data_set = collect_datasets(name=self.name, image=self.image_path,
                                                          dim_invert=self.params['dim_invert'],
                                                          format=self.params['format'])

        self.train_dataloader = collect_training_data(
            dataset=self.data_set,
            data_array=self.data_array,
            params=self.params,
            name=self.name,
            format=format,
            msc_file=None,
            dim_invert=self.params['dim_invert'])

        self.image, self.msc_collection, self.mask, self.segmentation = self.train_dataloader[
            int(self.params['train_data_idx'])]
        self.image = self.image.astype(np.float32)
        max_val = np.max(self.image)
        min_val = np.min(self.image)
        self.image = (self.image - min_val) / (max_val - min_val)

        self.X = self.image.shape[0]
        self.Y = self.image.shape[1] if len(self.image.shape) == 2 else self.image.shape[2]


        #
        # OG #####################################################
        #

        learning_rate = self.params['learning_rate']
        epochs = self.params['epochs']
        batch_size = self.params['batch_size']
        self.dim_hidden_1 = self.params['out_dim_1']
        self.dim_hidden_2 = self.params['out_dim_2']
        self.dim_hidden_3 = self.params['mlp_out_dim_3']
        self.kl_weight = 0.5
        self.xentropy_weight = 0.5

        # Parameters
        self.learning_rate = learning_rate #= 0.001
        self.training_epochs = epochs #= 15
        self.batch_size = batch_size #= 100
        self.display_step = epochs//4

        ###
        self.model_flavor = flavor
        self.growth_radius = growth_radius

        self.update_run_info(batch_multi_run=self.run_num)

        self.build_inputs(flavor=flavor, growth_radius=growth_radius)

        self.build_weights(flavor=flavor)

        self.build()

    def train(self):
        if self.model_flavor=='msc':
            return self._train_priors()
        else:
            return self._train_pixel()

    def next_batch(self, num, data, labels, shuffle=True):
        '''
        Return a total of `num` random samples and labels.
        '''
        idx = np.arange(0, len(data))
        if shuffle:
            np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [data[i] for i in idx]
        labels_shuffle = [labels[i] for i in idx]

        return np.asarray(data_shuffle), np.asarray(labels_shuffle)

    def model(self):
        X , Y = self.X_IN, self.Y_OUT
        # Construct model
        self.logits = self.multilayer_perceptron(X)

        if True:# self.model_flavor == 'pixel':
            a = 1
            self.loss_xentropy = tf.multiply(self.xentropy_weight,
                            tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=Y)))
            self.loss_kl = tf.multiply(self.kl_weight,
                            tf.reduce_mean(self.kl_loss(self.logits, Y)))
            self.loss_op = self.loss_xentropy + self.loss_kl
        # Define loss and optimizer
        if False:#self.model_flavor == 'msc':
            self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=Y))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss_op)
        # Initializing the variables
        self.init = tf.global_variables_initializer()

    def kl_loss(self, true_p, q):
        # #plogp-plogq
        # true_prob = tf.nn.softmax(true_p, axis = 1)
        # loss_1 = -tf.nn.softmax_cross_entropy_with_logits(logits=true_p, labels = true_prob)
        # loss_2 = tf.nn.softmax_cross_entropy_with_logits(logits=q, labels = true_prob)
        # loss = loss_1 + loss_2
        def fixprob(att):
            att = att + 1e-9
            _sum = tf.reduce_sum(att, reduction_indices=1, keep_dims=True)
            att = att / _sum
            att = tf.clip_by_value(att, 1e-9, 1.0, name=None)
            return att

        def kl(x, y):
            x = fixprob(x)
            y = fixprob(y)
            X = tf.distributions.Categorical(probs=x)
            Y = tf.distributions.Categorical(probs=y)
            return tf.distributions.kl_divergence(X, Y)
        return kl(true_p,q)

    def build(self):
        self.model()

    # Create model
    def multilayer_perceptron(self, x):
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.nn.elu(
            tf.add(tf.matmul(x, self.weights['h1']), self.biases['b1'])
        )
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.nn.elu(
            tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        )
        if False:#self.model_flavor == 'pixel':
            # l3
            layer_3 = tf.add(tf.matmul(layer_2, self.weights['h3']), self.biases['b3'])
            layer_4 = tf.add(tf.matmul(layer_3, self.weights['h4']), self.biases['b4'])
            # fully connected layer
            # layer_5 = tf.add(tf.matmul(layer_4, self.weights['h5']), self.biases['b5'])
            # Output fully connected layer with a neuron for each class
            out_encoding = tf.matmul(layer_4, self.weights['out']) + self.biases['out']
        else:
            # Output fully connected layer with a neuron for each class
            out_encoding = tf.matmul(layer_2, self.weights['out']) + self.biases['out']
        # softmax with output R^(number classes X 1)
        return out_encoding

    def build_weights(self, flavor= 'msc'):
        # Store layers weight & bias
        if True:#flavor == 'msc':
            self.weights = {
                'h1': tf.Variable(tf.random_normal([self.dim_input, self.dim_hidden_1])),
                'h2': tf.Variable(tf.random_normal([self.dim_hidden_1, self.dim_hidden_2])),
                # 'h3': tf.Variable(tf.random_normal([dim_hidden_2, dim_hidden_3])),
                # 'h4': tf.Variable(tf.random_normal([dim_hidden_3, dim_hidden_2])),
                # 'h5': tf.Variable(tf.random_normal([dim_hidden_2, dim_hidden_1])),
                'out': tf.Variable(tf.random_normal([self.dim_hidden_2, self.n_classes]))
            }
            self.biases = {
                'b1': tf.Variable(tf.random_normal([self.dim_hidden_1])),
                'b2': tf.Variable(tf.random_normal([self.dim_hidden_2])),
                # 'b3': tf.Variable(tf.random_normal([dim_hidden_3])),
                # 'b4': tf.Variable(tf.random_normal([dim_hidden_2])),
                # 'b5': tf.Variable(tf.random_normal([dim_hidden_1])),
                'out': tf.Variable(tf.random_normal([self.n_classes]))
            }
        else:
            self.weights = {
                'h1': tf.Variable(tf.random_normal([self.dim_input, self.encode1])),
                'h2': tf.Variable(tf.random_normal([self.encode1, self.encode1])),
                'h3': tf.Variable(tf.random_normal([self.encode1, self.decode1])),
                'h4': tf.Variable(tf.random_normal([self.decode1, self.decode1])),
                # 'h5': tf.Variable(tf.random_normal([self.decode1, self.n_classes])),
                'out': tf.Variable(tf.random_normal([self.decode1, self.n_classes])),
            }
            self.biases = {
                'b1': tf.Variable(tf.random_normal([self.encode1])),
                'b2': tf.Variable(tf.random_normal([self.encode1])),
                'b3': tf.Variable(tf.random_normal([self.decode1])),
                'b4': tf.Variable(tf.random_normal([self.decode1])),
                #'b5': tf.Variable(tf.random_normal([self.n_classes])),
                'out': tf.Variable(tf.random_normal([self.n_classes]))
            }

    def build_inputs(self, flavor='msc', growth_radius = 2):
        if flavor == 'msc':
            X_BOX = []
            Y_BOX = []
            box_sets = []
            for box in self.boxes:
                X_BOX.append((box[0], box[1]))
                Y_BOX.append((box[2], box[3]))

            for box_pair in box_sets:
                for box in box_pair:
                    X_BOX.append(box[0])
                    Y_BOX.append(box[1])
            _, _, box_set = self.box_select_geomsc_training(x_range=X_BOX,
                                                            y_range=Y_BOX,
                                                            boxes=None)  #
            self.X_BOX, self.Y_BOX = box_set

            self.get_train_test_val_sugraph_split(collect_validation=False, validation_hops=1,
                                                  validation_samples=1)
            self.box_regions = self.boxes

            num_percent = 0
            for xbox, ybox in zip(self.X_BOX, self.Y_BOX):
                num_percent += float((xbox[1] - xbox[0]) * (ybox[1] - ybox[0]))
            percent = num_percent / float(self.image.shape[0] * self.image.shape[1])
            percent_f = percent * 100
            print("    * ", percent_f)
            percent = int(round(percent_f))
            self.training_size = percent
            self.run_num = percent

            self.update_run_info(batch_multi_run=str(self.training_size))
            out_folder = os.path.join(self.pred_session_run_path)

            if not os.path.exists(out_folder):
                os.makedirs(out_folder)

            self.training_reg_bg = np.zeros(self.image.shape[:2], dtype=np.uint8)
            for region in self.box_regions:
                self.training_reg_bg[region[0]:region[1], region[2]:region[3]] = 1
            partition_label_dict, partition_feat_dict = get_partition_feature_label_pairs(
                self.node_gid_to_partition,
                self.node_gid_to_feature,
                self.node_gid_to_label,
                test_all=True)
            gid_features_dict = partition_feat_dict['all']
            gid_label_dict = partition_label_dict['all']
            train_gid_label_dict = partition_label_dict['train']
            train_gid_feat_dict = partition_feat_dict['train']
            train_labels = train_gid_label_dict
            train_features = train_gid_feat_dict
            test_labels = gid_label_dict
            self.test_gid_features_dict = gid_features_dict

            self.train_features = np.array(list(train_features.values()))
            self.train_pos_labels = np.array(list(train_labels.values()))
            # if len(train_labels[0]) == 1:
            self.train_neg_labels = 1 - np.array(self.train_pos_labels)
            self.train_labels = np.array(
                list(train_labels.values()))  # np.array([[n, p] for n, p in zip(self.train_neg_labels,
            #                                                         self.train_pos_labels)])
            self.test_features = np.array(list(self.test_gid_features_dict.values()))
            self.test_pos_labels = np.array(list(test_labels.values()))
            # if len(test_labels[0]) == 1:
            self.test_neg_labels = 1 - np.array(self.test_pos_labels)
            self.test_labels = np.array(
                list(test_labels.values()))  # np.array([[n, p] for n, p in zip(self.test_neg_labels,
            #                                                        self.test_pos_labels)])

            self.features = list(gid_features_dict)  # np.array(features)
            self.labels = np.array(list(gid_label_dict.values()))
            print("")
            print("    * SAMPLE LABEL")
            print(self.train_labels[0])

            self.num_examples = self.train_labels.shape[0]
            self.total_batch = self.num_examples // self.batch_size

            self.dim_input = int(self.train_features[0].shape[0])
            self.n_classes = 2
            # Network Parameters

            # tf Graph input
            self.X_IN = tf.placeholder("float", [None, self.dim_input])
            self.Y_OUT = tf.placeholder("float", [None, self.n_classes])
            self.X_TEST = self.X_IN  # tf.placeholder("float", [None, dim_input_test])
            self.Y_TEST = self.Y_OUT  # tf.placeholder("float", [None, n_classes])
        else:
            self.n_classes = 2
            self.box_regions = self.boxes
            X_BOX = []
            Y_BOX = []
            box_sets = []
            for box in self.boxes:
                box_set = tile_region(step_X=64, step_Y=64, step=0.5,
                                      Y_START=box[0], Y_END=box[1],
                                      X_START=box[2], X_END=box[3])
                box_sets.append(box_set)

            for box_pair in box_sets:
                for box in box_pair:
                    X_BOX.append(box[0])
                    Y_BOX.append(box[1])
            self.X_BOX = X_BOX
            self.Y_BOX = Y_BOX

            _, _, box_pair = self.box_select_geomsc_training(x_range=self.X_BOX, y_range=self.Y_BOX)
            self.X_BOX, self.Y_BOX = box_pair
            print("    xboxxx", self.X_BOX)
            print("    yboxxx* ", self.Y_BOX)

            X_BOX_all = []
            Y_BOX_all = []
            box_sets_test = []
            for box in [[0, self.image.shape[0], 0, self.image.shape[1]]]:
                box_set_test = tile_region(step_X=64, step_Y=64, step=0.5,
                                           Y_START=0, Y_END=self.image.shape[1],
                                           X_START=0, X_END=self.image.shape[0])
                box_sets_test.append(box_set_test)

            for box_pair in box_sets_test:
                for box in box_pair:
                    X_BOX_all.append(box[0])
                    Y_BOX_all.append(box[1])
            self.X_BOX_all = X_BOX_all
            self.Y_BOX_all = Y_BOX_all
            max_val = np.max(self.image)
            min_val = np.min(self.image)
            self.image = (self.image - min_val) / (max_val - min_val)

            out_folder = os.path.join(self.pred_session_run_path)
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            self.write_selection_bounds(dir=out_folder,
                                        x_box=self.X_BOX, y_box=self.Y_BOX,
                                        mode='w')

            self.train_features, self.train_labels, _ = self.collect_boxes(  # region_list=self.region_list,
                # number_samples=self.training_size,
                training_set=True)

            self.test_features, self.test_labels, self.full_seg = self.collect_boxes(  # region_list = param_lines ,
                training_set=False)


            self.num_examples = self.train_labels[0].shape[0]
            self.total_batch = self.num_examples // self.batch_size
            self.num_test_examples = self.test_labels[0].shape[0]
            self.total_test_batch = self.num_test_examples // self.batch_size

            # aug_ims = []
            # filtered_im_folder = os.path.join(self.experiment_folder, 'filtered_images')
            # df = dataflow()
            # filtered_imgs = df.read_images(filetype='.png', screen='feat-func_', dest_folder=filtered_im_folder)
            # for im in filtered_imgs:
            #     np.max(im)
            #     min_val = np.min(im)
            #     im = (im - min_val) / (max_val - min_val)
            #     im = np.mean(im, axis=2)
            #     im = np.expand_dims(im, axis=-1)
            #     features = np.concatenate((im, features), axis=2)

            dim_input = list(map(int,self.train_features[0].shape))#[0])

            print("    * dim mlp train features: ", dim_input)

            #
            #                          Network Parameters
            #

            self.encode1 = self.dim_hidden_1
            # self.encode2 = self.encode1#int(self.encode1 / 2)
            self.decode1 = self.dim_hidden_1
            dim_im = int(dim_input[0])

            self.dim_hidden_1 = self.dim_hidden_1  # = 256 # 1st layer number of neurons
            self.dim_hidden_ = self.dim_hidden_2  # = 256 # 2nd layer number of neurons
            #list(map(int,self.train_labels[0].shape))

            print("    * label shape: ", len(self.train_labels))

            # tf Graph input
            feat_x , feat_y, in_channels = dim_input[0], dim_input[1], dim_input[-1]

            self.dim_input = in_channels
            self.X_IN = tf.placeholder("float", shape=(None,feat_x , feat_y, in_channels))#[None, in_channels])
            self.Y_OUT = tf.placeholder("float", shape=(None,dim_im , dim_im, self.n_classes))
            self.batch_size_ph = tf.placeholder(tf.int32,shape=(None))


    def get_data_crops(self, image=None, x_range=None,y_range=None, with_range=False,
                       train_set=True,full_img=False,growth_radius = 1):
        seg = np.zeros(self.image.shape[:2]).astype(np.uint8)

        x_full = [0, self.image.shape[0]]
        y_full = [0, self.image.shape[1]]
        full_bounds = (x_full, y_full)
        seg = self.generate_pixel_labeling(full_bounds,
                                           seg_image=seg, growth_radius=growth_radius,
                                           train_set=train_set)

        range_group = zip(x_range, y_range)

        feature_set , labels = [], []
        for x_rng, y_rng in range_group:
            bounds = (x_rng,y_rng)
            train_im_crop = deepcopy(self.image[bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1]])
            seg_crop = deepcopy(seg[bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1]])

            ################## BUILD FEATURES  ################
            sigma_min = 1
            sigma_max = 64
            features_func = partial(multiscale_basic_features,
                                    intensity=True, edges=False, texture=True,
                                    sigma_min=sigma_min, sigma_max=sigma_max,
                                    multichannel=False)
            # features_func.
            features = features_func(train_im_crop)

            if with_range:
                feature_set.append(features)
                pos_labels = seg_crop.reshape((seg_crop.shape[0] * seg_crop.shape[1]))
                label = np.array([[1.0 - p, p] for p in pos_labels])
                label = label.reshape((seg_crop.shape[0], seg_crop.shape[1], self.n_classes))
                labels.append(label)
                #dataset.append((features, seg_crop, (x_rng, y_rng)))
            else:
                feature_set.append(features)
                pos_labels = seg_crop.reshape((seg_crop.shape[0]*seg_crop.shape[1]))
                label = np.array([[1.0-p, p] for p in pos_labels])
                label = label.reshape((seg_crop.shape[0], seg_crop.shape[1], self.n_classes))
                labels.append(label)
        if full_img:
            return seg, feature_set, labels
        return feature_set, labels

    def generate_pixel_labeling(self, train_region, seg_image=None ,growth_radius = 1, train_set=True):
        x_box = train_region[0]
        y_box = train_region[1]
        dim_train_region = (   int(x_box[1] - x_box[0]),int(y_box[1]-y_box[0]) )
        train_im = seg_image#np.zeros(dim_train_region)
        X = train_im.shape[0]
        Y = train_im.shape[1]

        def __box_grow_label(train_im, center_point, label, train_set):
            x = center_point[0]
            y = center_point[1]
            x[x + 1 >= X] = X - 2
            y[y+1>= Y] = Y - 2
            train_im[x , y] = label
            train_im[x-1,y] = label
            train_im[x, y-1] = label
            train_im[x - 1, y-1] = label
            train_im[x+1,y] = label
            train_im[x, y+1] = label
            train_im[x + 1, y+1] = label
            train_im[x - 1, y + 1] = label
            train_im[x + 1, y - 1] = label

        for gid in self.gid_gnode_dict.keys():
            gnode = self.gid_gnode_dict[gid]
            label = self.node_gid_to_label[gid]
            if train_set:
                self.node_gid_to_partition[gid] = 'test'
            label = label
            points = gnode.points
            p1 = points[0]
            p2 = points[-1]
            in_box = False
            not_all = False
            end_points = (p1, p2)
            points = get_points_from_vertices([gnode])

            points = np.reshape(points, (-1, 2))
            interior_points = []

            for p in points:
                if x_box[0] <= p[1] <= x_box[1] and y_box[0] <= p[0] <= y_box[1]:
                    in_box = True
                    interior_points.append(p)
                    mapped_y = points[:, 0]  # - x_box[0]
                    mapped_x = points[:, 1]  # - y_box[0]
                    if int(label[1]) == 1:
                        train_im[mapped_x, mapped_y] = 1
        train_im = ndimage.maximum_filter(train_im, size=growth_radius)
        n_samples = dim_train_region[0] * dim_train_region[1]
        n_classes = 2
        class_bins = np.bincount(train_im.astype(np.int64).flatten())
        self.class_weights = 1.0#n_samples / (n_classes * class_bins)
        return train_im.astype(np.int8)

    def collect_boxes(self, resize=False, run_num=-1, training_set=False):
        if training_set:
            train_features, train_labels = self.get_data_crops(self.image, x_range=self.X_BOX,
                                               with_range = not training_set,
                                                               full_img=False,
                                                    y_range=self.Y_BOX,
                                               growth_radius = self.growth_radius,
                                               train_set=training_set)
            self.pred_run_path = os.path.join(self.LocalSetup.project_base_path, 'datasets',
                                              self.params['write_folder'],
                                              'runs')
            return train_features, train_labels, None

        else:
            full_seg, test_features, test_labels = self.get_data_crops(self.image,
                                                                       x_range=self.X_BOX_all,
                                               with_range=False,
                                                                       full_img=True,
                                               y_range=self.Y_BOX_all,
                                               growth_radius=self.growth_radius,
                                               train_set=training_set)
            # inf_or_train_dataset = dataset(training_tiles, do_transform=False,
            #                                with_hand_seg=False)
            return test_features, test_labels, full_seg




    def _train_priors(self):



        self.write_gnode_partitions(self.pred_session_run_path)
        self.write_selection_bounds(self.pred_session_run_path)

        X , Y = self.X_IN, self.Y_OUT
        X_TEST, Y_TEST = self.X_TEST, self.Y_TEST

        s = time.time()

        with tf.Session() as sess:
            sess.run(self.init)

            # Training cycle
            for epoch in range(self.training_epochs):
                avg_cost = 0.
                # Loop over all batches
                for i in range(self.total_batch):
                    batch_x, batch_y = self.next_batch(self.batch_size, self.train_features, self.train_labels)
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([self.train_op, self.loss_op], feed_dict={X: batch_x,
                                                                              Y: batch_y})
                    # Compute average loss
                    avg_cost += c / self.total_batch
                # Display logs per epoch step
                if epoch % self.display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))

            f = time.time()
            self.record_time(round(f - s, 4), dir=self.pred_session_run_path, type='train')

            print("Optimization Finished!")

            # Test model

            s = time.time()

            pred = tf.nn.softmax(self.logits)  # Apply softmax to logits
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            ac = accuracy.eval({X_TEST: self.test_features, Y_TEST: self.test_labels})

            preds =pred.eval({X_TEST: self.test_features, Y_TEST: self.test_labels})

            f = time.time()
            self.record_time(round(f - s, 4), dir=self.pred_session_run_path, type='pred')

            print("Accuracy:", ac)
            pred_train = pred.eval({X: self.train_features, Y: self.train_labels})
            self.accuracy=ac
            class_preds = []
            true_labels = []
            for gid, pred in zip(self.test_gid_features_dict.keys(), preds):
                self.node_gid_to_prediction[gid] = float(pred[1])
                if self.node_gid_to_partition[gid] != 'train':
                    class_preds.append(pred[1])
                    label = self.node_gid_to_label[gid]
                    true_labels.append(label[1])
            return preds, true_labels, ac #[pred_test, pred_train] , [self.test_labels , self.train_labels] , ac

    def _train_pixel(self, growth_radius=2):

        self.update_run_info(batch_multi_run=self.run_num)

        self.write_gnode_partitions(self.pred_session_run_path)
        self.write_selection_bounds(self.pred_session_run_path)

        X, Y = self.X_IN, self.Y_OUT
        #X_TEST, Y_TEST = self.X_TEST, self.Y_TEST

        s = time.time()

        with tf.Session() as sess:
            sess.run(self.init)

            # Training cycle
            for epoch in range(self.training_epochs):
                avg_cost = 0.
                # Loop over all batches
                for i in range(self.total_batch):
                    batch_x, batch_y = self.next_batch(self.batch_size,
                                                       self.train_features,
                                                       self.train_labels)
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([self.train_op, self.loss_op], feed_dict={X: batch_x,
                                                                              Y: batch_y})
                    # Compute average loss
                    avg_cost += c / self.total_batch
                # Display logs per epoch step
                if epoch % self.display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))

            f = time.time()
            self.record_time(round(f - s, 4), dir=self.pred_session_run_path, type='train')

            print("Optimization Finished!")

            # Test model
            prediction = tf.nn.softmax(self.logits)  # Apply softmax to logits
            correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            accuracies = []
            all_preds = []

            s = time.time()
            for im_slice, slice_label in zip(self.test_features,self.test_labels):
                ac = accuracy.eval({X: im_slice[None],
                                    Y: slice_label[None]})#/self.total_test_batch
                accuracies.append(ac)
                pred = prediction.eval({X: im_slice[None],
                                   Y: slice_label[None]})
                pred = pred[:,:,:,1]#
                all_preds.append(pred)

            f = time.time()
            self.record_time(round(f - s, 4), dir=self.pred_session_run_path, type='pred')

            print("Accuracy:", np.mean(accuracies))
            self.accuracy=ac

            return all_preds, [], ac #[pred_test, pred_train] , [self.test_labels , self.train_labels] , ac

    def compute_metrics(self, pred_images,# scores, pred_labels, pred_thresh,
                        INTERACTIVE=False):#predictions_topo, labels_topo,

        self.training_reg_bg = np.zeros(self.image.shape[:2], dtype=np.uint8)
        for region in self.box_regions:
            self.training_reg_bg[region[0]:region[1], region[2]:region[3]] = 1

        use_average = True

        self.pred_val_im = np.zeros(self.image.shape[:2], dtype=np.float32)
        tile_samples = []
        sample_box = []
        for idx, box_set in enumerate(zip(self.X_BOX_all, self.Y_BOX_all)):
            border_xl = 16
            border_yl = 16
            border_xh = 16
            border_yh = 16

            x_box = box_set[0]
            y_box = box_set[1]
            if x_box[0]==0:
                border_xl = 0
            if y_box[0] == 0:
                border_yl = 0
            if x_box[1]==self.training_reg_bg.shape[0]:
                border_xh = 0
            if y_box[1] == self.training_reg_bg.shape[1]:
                border_yh = 0



            pad_xl = border_xl
            pad_yl = border_yl

            pad_xh = border_xh
            pad_yh = border_yh

            pred_tile = np.squeeze(pred_images[idx], axis=0)
            self.pred_val_im[x_box[0] + pad_xl: x_box[1] - pad_xh,
            y_box[0] + pad_yl: y_box[1] - pad_yh] = pred_tile[pad_xl:pred_tile.shape[0] - pad_xh,
                                                    pad_yl:pred_tile.shape[1] - pad_yh]

            # if idx ==len(self.all_region_boxes)//4:
            #     sample_box = box_set
            #     im_tile = self.image[x_box[0] + pad: x_box[1] - pad, y_box[0] + pad: y_box[1] - pad]
            #     seg_tile = self.training_labels[x_box[0] + pad: x_box[1] - pad, y_box[0] + pad: y_box[1] - pad]
            #     p_tile = pred_tile[pad:pred_tile.shape[0] - pad, pad:pred_tile.shape[1] - pad]
            #     tile_samples = [im_tile,seg_tile,p_tile]


        self.cutoffs = np.arange(0.01, 0.98, 0.01)
        scores = np.zeros((len(self.cutoffs), 4), dtype="int32")

        predictions = []
        true_labels = []
        max_X_bound = max(self.image.shape)
        max_X_bound = min(self.image.shape)

        for gid in self.node_gid_to_label.keys():
            gnode = self.gid_gnode_dict[gid]
            label = self.node_gid_to_label[gid]
            label = label if type(label) != list else label[1]
            line = get_points_from_vertices([gnode])
            # else is fg
            vals = []

            for point in line:

                ly = int(point[0])
                lx = int(point[1])
                vals.append(self.pred_val_im[lx, ly])

            inferred = np.array(vals, dtype="float32")
            infval = np.average(inferred)
            pred_mode = infval
            if not use_average:
                vals, counts = np.unique(inferred, return_counts=True)
                mode_value = np.argwhere(counts == np.max(counts))
                pred_mode = inferred[mode_value].flatten().tolist()[0]

            infval = infval if use_average else pred_mode



            self.node_gid_to_prediction[gid] = [1. - infval, infval]

            for idx, cutoff in enumerate(self.cutoffs):
                if infval >= cutoff:
                    if label == 1:
                        scores[idx, 0] += len(line)  # true positive
                    else:
                        scores[idx, 2] += len(line)  # false positive
                else:
                    if label == 1:
                        scores[idx, 1] += len(line)  # false negative
                    else:
                        scores[idx, 3] += len(line)  # true negative

            for point in line:
                ly = int(point[0])
                lx = int(point[1])
                if self.training_reg_bg[lx, ly] != 1:
                        predictions.append(infval )
                        true_labels.append(label )


        # print("       labelslslslsls", labels_topo)
        # print("After sampling back to lines, using 0.5 cutoff:")
        restab = pd.DataFrame(scores.T)
        restab.columns = [str(x) for x in self.cutoffs]

        self.F1_log = {}
        self.max_f1 = 0
        self.opt_thresh = 0
        for thresh in self.cutoffs:

            threshed_arc_segmentation_logits = [logit > thresh for logit in true_labels]
            threshed_arc_predictions_proba = [logit > thresh for logit in predictions]

            F1_score_topo = f1_score(y_true=threshed_arc_segmentation_logits,
                                     y_pred=threshed_arc_predictions_proba,average=None)[-1]

            #self.F1_log[F1_score_topo] = thresh
            if F1_score_topo >= self.max_f1:
                self.max_f1 = F1_score_topo

                self.F1_log[self.max_f1] = thresh
                self.opt_thresh = thresh


        gt_polyline_labels = []
        self.pred_prob_im = np.zeros(self.image.shape[:2], dtype=np.float32) #* min(0.25, self.opt_thresh/2.)
        pred_labels_conf_matrix = np.zeros(self.image.shape[:2], dtype=np.float32)# * min(0.25,self.opt_thresh/2.)#dtype=np.uint8)
        pred_labels_msc = np.zeros(self.image.shape[:2], dtype=np.float32) #* min(0.25, self.opt_thresh/2.)
        gt_msc = np.zeros(self.image.shape[:2], dtype=np.float32)
        predictions_topo_bool = []
        labels_topo_bool = []
        check=30
        for gid in self.node_gid_to_label.keys():  # zip(mygraph.labels, mygraph.polylines):

            gnode = self.gid_gnode_dict[gid]
            label = self.node_gid_to_label[gid]
            label = label if type(label) != list else label[1]
            line = get_points_from_vertices([gnode])
            # else is fg
            cutoff = self.F1_log[self.max_f1]

            vals = []

            for point in line:
                ly = int(point[0])
                lx = int(point[1])
                pred = self.pred_val_im[lx, ly]
                vals.append(pred)

            if not use_average:
                vals = list(map(lambda x : round(x,2),vals))
            inferred = np.array(vals, dtype="float32")
            infval = np.average(inferred)
            pred_mode = infval
            if not use_average:
                vals, counts = np.unique(inferred, return_counts=True)
                mode_value = np.argwhere(counts == np.max(counts))
                pred_mode = inferred[mode_value].flatten().tolist()[0]

            infval = infval if use_average else pred_mode

            self.node_gid_to_prediction[gid] = [1.-infval, infval]
            if check >= 0:

                check -= 1

            t = 0
            if infval >= self.opt_thresh:
                if label == 1:  # true positive
                    t = 4  # red
                    ## ["lightgray", "blue", "yellow", "cyan", "red", 'mediumspringgreen'])
                else:  # . False positive
                    t = 2  # yellow
            else:
                if label == 1:  # false negative
                    t = 5  # mediumspringgreen
                else:  # True Negatuve
                    t = 1  # blue

            for point in line:
                ly = int(point[0])
                lx = int(point[1])
                gt_msc[lx, ly] = 4 if label == 1 else 1
                pred_labels_conf_matrix[lx, ly] = t
                pred_labels_msc[lx,ly] = 1 if infval >= self.F1_log[self.max_f1] else 0
                self.pred_prob_im[lx, ly] = infval
                if self.training_reg_bg[lx, ly] != 1:
                    self.node_gid_to_partition[gid] = 'test'
                    predictions_topo_bool.append(infval >= cutoff)
                    labels_topo_bool.append(label >= cutoff)
                else:
                    self.node_gid_to_partition[gid] = 'train'


        out_folder = os.path.join(self.pred_session_run_path)#,
        #                          str(self.training_size))
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)



        exp_folder = os.path.join(self.params['experiment_folder'])#, 'runs')

        batch_metric_folder = os.path.join(exp_folder, 'batch_metrics')
        if not os.path.exists(batch_metric_folder):
            os.makedirs(batch_metric_folder)

        self.draw_segmentation(dirpath=out_folder)
        compute_prediction_metrics('unet', predictions_topo_bool,
                                   labels_topo_bool,
                                   out_folder)
        self.write_arc_predictions(dir=out_folder)
        self.write_training_percentages(dir=out_folder,msc_segmentation=self.full_seg)
        self.write_training_percentages(dir=out_folder,train_regions=self.training_reg_bg)
        self.draw_segmentation(dirpath=out_folder)
        self.write_gnode_partitions(dir=out_folder)

        multi_run_metrics(model='mlp', exp_folder=exp_folder,
                           batch_multi_run=True,
                           bins=7, runs='runs',#str(self.training_size),
                           plt_title=exp_folder.split('/')[-1])

        print("MLP_MAX_F1:", self.max_f1)
        print("pthresh:", self.F1_log[self.max_f1], 'opt', self.opt_thresh)#cutoffs[F1_MAX_ID])
        print("Num Pixels:", self.image.shape[0] * self.image.shape[1], "Num Pixels training:",
              np.sum(self.full_seg), "Percent:",
              100.0 * np.sum(self.full_seg) / (self.image.shape[0] * self.image.shape[1]))


        images = [self.image, self.full_seg, self.pred_val_im,
                  self.pred_prob_im]
        names = ["Image", "Ground Truth Segmentation", "Predicted Foreground Segmentation",
                 "Pixel to Lines Foreground Probability"]
        for image, name in zip(images, names):
            plot(image_set=[image, self.training_reg_bg], name=name, type='contour', write_path=out_folder)

        image_set = [pred_labels_msc, self.training_reg_bg, pred_labels_conf_matrix]
        plot(image_set, name="TP FP TF TN Line Prediction",
             type='confidence', write_path=out_folder)

        plot(image_set, name="TP FP TF TN Line Prediction",
             type='zoom', write_path=out_folder)

        image_set = [self.image, self.training_reg_bg, gt_msc]
        plot(image_set, name="Ground Truth MSC",
             type='confidence', write_path=out_folder)

        plot(image_set, name="Ground Truth MSC",
             type='zoom', write_path=out_folder)

        for image, name in zip(images, names):
            plot(image_set=[image, self.training_reg_bg], name=name, type='zoom', write_path=out_folder)






        np.savez_compressed(os.path.join(out_folder,'pred_matrix.npz'),self.pred_val_im)
        np.savez_compressed(os.path.join(out_folder,'training_matrix.npz'),self.training_reg_bg)