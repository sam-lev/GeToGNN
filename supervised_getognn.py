import numpy as np
import os
from copy import deepcopy

from getognn import GeToGNN
from ml.Random_Forests import RandomForest
from getograph import  Attributes
from localsetup import LocalSetup

LocalSetup = LocalSetup()


class supervised_getognn:
    def __init__(self, model_name):
        self.model_name = model_name
        self.attributes = Attributes()

    def build_getognn(self, sample_idx, experiment_num, experiment_name, window_file_base,
                 parameter_file_number, format = 'raw', run_num=0, experiment_folder=None,
                 name=None, image=None, label_file=None, msc_file=None,
                 ground_truth_label_file=None, write_path=None, feature_file=None,
                 window_file=None, model_name="GeToGNN"):



        self.getognn = GeToGNN(training_selection_type='box',
                          run_num=run_num,
                          parameter_file_number = parameter_file_number,
                          name=name,
                          image=image,
                          feature_file=feature_file,
                          geomsc_fname_base = msc_file,
                          label_file=ground_truth_label_file,
                          write_folder=write_path,
                        experiment_folder=experiment_folder,
                         model_name=model_name,
                          load_feature_graph_name=None,
                          write_json_graph = False)

        # features
        if not self.getognn.params['load_features']:
            self.getognn.compile_features()
        else:
            self.getognn.load_gnode_features(filename=model_name)
        if self.getognn.params['write_features']:
            self.getognn.write_gnode_features(self.getognn.session_name)
        if self.getognn.params['write_feature_names']:
            self.getognn.write_feature_names(self.getognn.session_name)

        # training info, selection, partition train/val/test
        self.getognn.read_labels_from_file(file=ground_truth_label_file)

        training_set , test_and_val_set = self.getognn.box_select_geomsc_training(x_range=self.getognn.params['x_box'], y_range=self.getognn.params['y_box'])

        self.getognn.get_train_test_val_sugraph_split(collect_validation=True, validation_hops = 1,
                                                 validation_samples = 1)



        self.attributes = deepcopy(self.getognn.get_attributes())

        if self.getognn.params['write_json_graph']:
            self.getognn.write_json_graph_data(folder_path=self.getognn.pred_session_run_path, name=model_name + '_' + self.getognn.params['name'])


        self.getognn.write_gnode_partitions(self.getognn.session_name)
        self.getognn.write_selection_bounds(self.getognn.session_name)

        # random walks
        if not self.getognn.params['load_preprocessed_walks']:
            walk_embedding_file = os.path.join(self.getognn.LocalSetup.project_base_path, 'datasets',
                                               self.getognn.params['write_folder'],'walk_embeddings',
                                               'run-'+str(self.getognn.run_num)+'_walks')
            self.getognn.params['load_walks'] = walk_embedding_file
            self.getognn.run_random_walks(walk_embedding_file=walk_embedding_file)



    def train(self, getognn=None):
        if getognn is not None:
            self.getognn = getognn
        #training
        self.getognn.supervised_train()
        G = self.getognn.get_graph()
        self.getognn.equate_graph(G)

        self.getognn.write_arc_predictions(self.getognn.session_name)
        self.getognn.draw_segmentation(dirpath=self.getognn.pred_session_run_path)
        self.getognn = self.getognn
        return self.getognn










