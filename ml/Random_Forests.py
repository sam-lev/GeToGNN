from sklearn.ensemble import RandomForestClassifier
import numpy as np

import os
from copy import deepcopy

from mlgraph import MLGraph
from getograph import  Attributes
from proc_manager import experiment_manager
from data_ops import set_parameters
from data_ops.utils import dbgprint
from metrics.prediction_score import f1
from metrics.prediction_score import recall
from metrics.prediction_score import precision
from metrics.prediction_score import compute_quality_metrics

class RandomForest(MLGraph):

    def __init__(self, training_selection_type='box',run_num=0, parameter_file_number = None,
                 geomsc_fname_base = None, label_file=None,
                 model_name=None, load_feature_graph_name=False,image=None,  **kwargs):

        self.model_name = model_name

        self.details = "Random forest classifier with feature importance"

        self.type = "random_forest"



        #if self.params is None:
        self.parameter_file_number = parameter_file_number

        '''self.params = {}
        if parameter_file_number is None:
            self.params = kwargs
        else:
            for param in kwargs:
                self.params[param] = kwargs[param]'''

        #self.write_folder = self.params['write_folder']

        self.G = None
        self.G_dict = {}

        super(RandomForest, self).__init__(parameter_file_number=parameter_file_number, run_num=run_num,
                                      name=kwargs['name'], geomsc_fname_base=geomsc_fname_base,
                                      label_file=label_file, image=image, write_folder=kwargs['write_folder'],
                                      model_name=model_name, load_feature_graph_name=load_feature_graph_name)


        self.attributes = self.get_attributes()
        '''for param in kwargs:
            self.params[param] = kwargs[param]
        # for param in kwargs['params']:
        #    self.params[param] = kwargs['params'][param]
        if 'params' in kwargs.keys():
            param_add_ons = kwargs['params']
            for k, v in param_add_ons.items():
                self.params[k] = v'''



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

        #
        # Training / val /test sets
        #
        self.subgraph_sample_set = {}
        self.subgraph_sample_set_ids = {}
        self.positive_arc_ids = set()
        self.selected_positive_arc_ids = set()
        self.negative_arc_ids = set()
        self.selected_negative_arc_ids = set()
        self.positive_arcs = set()
        self.selected_positive_arcs = set()
        self.negative_arcs = set()
        self.selected_negative_arcs = set()

    def build_random_forest(self,
                 ground_truth_label_file=None, write_path=None, feature_file=None,
                 window_file=None, model_name="GeToGNN"):



        dbgprint(self.params['geto_as_feat'], 'use geto')
        dbgprint(self.params['load_features'], 'load feat')

        if self.run_num > 0:
            #
            # Perform remainder of runs and don't need to read feats again
            #
            # if not UNet.params['load_features']:
            self.params['load_features'] = True
            self.params['write_features'] = False
            self.params['load_features'] = True
            self.params['write_feature_names'] = False
            self.params['save_filtered_images'] = False
            self.params['collect_features'] = False
            self.params['load_preprocessed'] = True
            self.params['load_geto_attr'] = True
            self.params['load_feature_names'] = True

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

            #if 'geto' in self.params['aggregator']:
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

        training_set , test_and_val_set = self.box_select_geomsc_training(x_range=self.params['x_box'], y_range=self.params['y_box'])

        self.get_train_test_val_sugraph_split(collect_validation=False, validation_hops = 1,
                                                 validation_samples = 1)



        #self.attributes = deepcopy(self.get_attributes())
        self.write_gnode_partitions(self.session_name)
        self.write_selection_bounds(self.session_name)

    def classifier(self, node_gid_to_prediction, train_features=None, train_labels=None,
                   test_features=None, test_labels=None, feature_map=False,class_1_weight=1.0,
                   class_2_weight=1.0,
                   n_trees=10, depth=4, weighted_distribution=False):
        print("_____________________________________________________")
        print("                Random_Forest     ")
        print("number trees: ", n_trees)
        print("depth: ", depth)
        # Import the model we are using
        # Instantiate model with 1000 decision trees
        train_gid_feat_dict = train_features
        train_gid_label_dict = train_labels
        test_gid_feat_dict = test_features
        test_gid_label_dict = test_labels

        train_features = np.array(list(train_features.values()))
        train_labels = list(train_labels.values())
        train_labels_binary = [l[1] for l in train_labels]
        train_labels = np.array(train_labels_binary)
        print("RF label sample: ", train_labels[0])
        if test_features is not None:
            test_features = np.array(list(test_features.values()))
            test_labels = list(test_labels.values())
            test_labels_binary = [l[1] for l in test_labels]
            test_labels = np.array(test_labels_binary)

        wn = 1
        wp = 1
        if weighted_distribution:
            wn = float(len(train_labels)) / (2.*(len(train_labels) - np.sum(train_labels)))
            wp = float(len(train_labels)) / (2.* np.sum(train_labels))
            print("Using class weights for negative: ", wn)
            print("Using class weights for positive: ", wp)
            print("total training samples: ", len(train_labels))
            print("total positive samples: ", np.sum(train_labels))


        rf = RandomForestClassifier(max_depth=depth,
                                    n_estimators=n_trees, class_weight={0:wn,1:wp}, random_state=666)

        rf.fit(train_features, train_labels)

        # Use the forest's predict method on the test data
        if test_features is not None:
            pred_proba_test = rf.predict_proba(test_features)

            # pred_proba_train = rf.predict_proba(train_features)
            # train_arcs = [arc for arc in self.msc.arcs if arc.partition == 'train']
            # test_arcs = [arc for arc in self.msc.arcs if arc.partition == 'test']
            # for arc, pred in zip(train_arcs, list(pred_proba_train)):
            #    arc.prediction = pred[1]#[1-pred[0], pred[0]]
            print("Forest pred sample ", pred_proba_test[0])
            for gid, pred in zip(test_gid_feat_dict.keys(),pred_proba_test):
                self.node_gid_to_prediction[gid] = pred[1]

            preds = pred_proba_test  # np.array([[1-p[1],p[1]] for p in pred_proba_test])#
            preds[preds >= 0.5] = 1.
            preds[preds < 0.5] = 0.
            preds = [l[len(l) - 1] for l in preds]

            # errors = abs(self.preds - self.labels)  # Print out the mean absolute error (mae)
            # round(np.mean(errors), 2), 'degrees.')
            print('----------------------------')
            mse = rf.score(test_features, test_labels_binary)  # np.array(list(test_features) + list(train_features)),
            #                                       np.array(list(test_labels) + list(train_labels)))
            print('Mean Absolute Error:', mse)
            #p, r, fs = compute_quality_metrics(preds, test_labels_binary)

            return preds, test_labels, node_gid_to_prediction #,p, r, fs, mse,

