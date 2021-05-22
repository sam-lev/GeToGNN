import os
import shutil
import numpy as np


from supervised_getognn import supervised_getognn
from getograph import Attributes
from ml.Random_Forests import RandomForest
from ml.MLP import MLP
from ml.utils import get_train_test_val_partitions
from ml.utils import get_partition_feature_label_pairs
from proc_manager.run_manager import *
from metrics.model_metrics import compute_getognn_metrics
from metrics.model_metrics import compute_prediction_metrics
from localsetup import LocalSetup
LocalSetup = LocalSetup()

#__all__ = ['run_experiment', 'experiment_logger']

class runner:
    def __init__(self, experiment_name, window_file_base = None
                 , sample_idx=2, multi_run = True):

        # base attributes shared between models
        self.attributes = Attributes()

        # perform one or multiple runs
        self.multi_run = multi_run

        #
        # Experiment resources
        #

        # for naming, reassigned in getognn
        self.sample_idx = sample_idx
        self.experiment_num = 1
        self.experiment_name = experiment_name
        self.window_file_base = window_file_base
        self.run_num = 0
        self.model_name = 'GeToGNN'  # 'experiment'#'input_select_from_1st_inference'

        #
        # input
        #
        self.name = ['retinal',                            # 0
                     'neuron1',                            # 1
                     'neuron2',                            # 2
                     'mat2_lines',                         # 3
                     'berghia',                            # 4
                     'faults_exmouth',                     # 5
                     'transform_tests'][self.sample_idx]   # 6

        self.image = ['im0236_o_700_605.raw',
                 'MAX_neuron_640_640.raw',
                 'MAX__0030_Image0001_01_o_1737_1785.raw',
                 'sub_CMC_example_o_969_843.raw',
                 'berghia_o_891_897.raw',
                 'att_0_460_446_484.raw',
                'diadem16_transforms_o_1000_1000.raw'][self.sample_idx]  # neuron1
        self.label_file = ['im0236_la2_700_605.raw.labels_2.txt',
                      'MAX_neuron_640_640.raw.labels_3.txt',
                      'MAX__0030_Image0001_01_s2_C001Z031_1737_1785.raw.labels_4.txt',
                      'sub_CMC_example_l1_969_843.raw.labels_0.txt',
                      'berghia_prwpr_e4_891_896.raw.labels_3.txt',
                      'att_L3_0_460_446_484.raw.labels_0.txt',
                      'diadem16_transforms_s1_1000_1000.raw.labels_14.txt'][self.sample_idx]  # neuron1
        self.msc_file = os.path.join(LocalSetup.project_base_path, 'datasets', self.name,
                                'input', self.label_file.split('raw')[0] + 'raw')
        self.ground_truth_label_file = os.path.join(LocalSetup.project_base_path, 'datasets',
                                               self.name, 'input', self.label_file)
        self.format = 'raw'
        self.experiment_folder = os.path.join(LocalSetup.project_base_path, 'datasets',
                                              self.name, self.experiment_name)
        if not os.path.exists(self.experiment_folder):
            os.makedirs(self.experiment_folder)

        self.write_path = os.path.join(self.name,self.experiment_name)  # 'experiment_'+str(experiment_num)+'_'+experiment_name

        self.feature_file = os.path.join( self.write_path, 'features', self.model_name)
        self.window_file = os.path.join(LocalSetup.project_base_path, "datasets",
                                   self.write_path,
                                   self.window_file_base)
        self.parameter_file_number = 1


    def start(self, model='getognn'):
        if model == 'getognn':
            self.run_getognn(self.multi_run)
        if model == 'mlp':
            self.run_mlp(self.multi_run)
        if model == 'random_forest':
            self.run_random_forest(self.multi_run)



    def run_getognn(self, multi_run ):
        #
        # Train single run of getognn and obtrain trained model
        #
        sup_getognn = supervised_getognn(model_name=self.model_name)
        sup_getognn.build_getognn( sample_idx=self.sample_idx, experiment_num=self.experiment_num,
                                   experiment_name=self.experiment_name,window_file_base=self.window_file_base,
                                   parameter_file_number=self.parameter_file_number,
                                   format = format, run_num=self.run_num,name=self.name, image=self.image,
                                   label_file=self.label_file, msc_file=self.msc_file,
                                   ground_truth_label_file=self.ground_truth_label_file,
                                   experiment_folder = self.experiment_folder,
                                   write_path=self.write_path, feature_file=self.feature_file,
                                   window_file=None,model_name="GeToGNN")

        self.attributes = sup_getognn.attributes


        getognn = sup_getognn.train()

        #
        # Get inference metrics
        #
        compute_getognn_metrics(getognn=getognn)

        #
        # Feature Importance
        #
        partition_label_dict, partition_feat_dict = get_partition_feature_label_pairs(
            getognn.node_gid_to_partition,
            getognn.node_gid_to_feature,
            getognn.node_gid_to_label,
            test_all=True)
        gid_features_dict = partition_feat_dict['all']
        gid_label_dict = partition_label_dict['all']
        getognn.load_feature_names()
        getognn.feature_importance(feature_names=getognn.feature_names,
                              features=gid_features_dict,
                              labels=gid_label_dict,
                                   plot=True)
        getognn.write_feature_importance()

        #
        # Perform remainder of runs and don't need to read feats again
        #
        if not getognn.params['load_features']:
            getognn.params['load_features'] = False
            getognn.params['write_features'] = False
            getognn.params['load_features'] = True
            getognn.params['write_feature_names'] = False
            getognn.params['save_filtered_images'] = False
            getognn.params['collect_features'] = False

        run_manager = Run_Manager(model=getognn,
                                  training_window_file=self.window_file,
                                  features_file=self.feature_file,
                                  sample_idx=self.sample_idx,
                                  model_name=self.model_name,
                                  format=format)
        if multi_run:
            run_manager.perform_runs()

    def run_mlp(self, multi_run ):
        train_set, test_set, val_set = get_train_test_val_partitions(self.attributes.node_gid_to_partition,
                                                                     self.attributes.gid_gnode_dict,
                                                                     test_all=True)
        partition_label_dict, partition_feat_dict = get_partition_feature_label_pairs(self.attributes.node_gid_to_partition,
                                                                                      self.attributes.node_gid_to_feature,
                                                                                      self.attributes.node_gid_to_label,
                                                                                      test_all=True)
        run_params = self.attributes.params
        gid_features_dict = partition_feat_dict['all']
        gid_label_dict = partition_label_dict['all']
        train_gid_label_dict = partition_label_dict['train']
        train_gid_feat_dict = partition_feat_dict['train']
        mlp_accuracy, mlp_p, mlp_r, mlp_fs = MLP(features=gid_features_dict, labels=gid_label_dict,
                                                 train_labels=train_gid_label_dict,
                                                 train_features=train_gid_feat_dict,
                                                 test_labels=np.array(gid_label_dict),
                                                 test_features=np.array(gid_features_dict),
                                                 learning_rate=run_params['mlp_lr'],
                                                 epochs=run_params['mlp_epochs'],
                                                 batch_size=run_params['mlp_batch_size'],
                                                 dim_hidden_1=run_params['mlp_out_dim_1'],
                                                 dim_hidden_2=run_params['mlp_out_dim_2'],
                                                 dim_hidden_3=run_params['mlp_out_dim_3'],
                                                 feature_map=False)
        manp_mlp = mlp_accuracy[1]
        manr_mlp = mlp_accuracy[2]
        mlp_accuracy = mlp_accuracy[0]

    def run_random_forest(self, multi_run ):


        RF = RandomForest(training_selection_type='box',
                          run_num=self.run_num,
                          parameter_file_number = self.parameter_file_number,
                          name=self.name,
                          image=self.image,
                          feature_file=self.feature_file,
                          geomsc_fname_base = self.msc_file,
                          label_file=self.ground_truth_label_file,
                          write_folder=self.write_path,
                         model_name=self.model_name,
                          load_feature_graph_name=None,
                          write_json_graph = False)

        RF.build_random_forest(ground_truth_label_file=self.ground_truth_label_file)

        self.attributes = RF.attributes

        partition_label_dict, partition_feat_dict = get_partition_feature_label_pairs(
            self.attributes.node_gid_to_partition,
            self.attributes.node_gid_to_feature,
            self.attributes.node_gid_to_label,
            test_all=True)

        run_params = self.attributes.params
        gid_features_dict = partition_feat_dict['all']
        gid_label_dict = partition_label_dict['all']
        train_gid_label_dict = partition_label_dict['train']
        train_gid_feat_dict = partition_feat_dict['train']

        n_trees = run_params['number_forests']
        tree_depth = run_params['forest_depth']

        predictions, labels, node_gid_to_prediction = RF.classifier(self.attributes.node_gid_to_prediction,
                                                                    train_labels=train_gid_label_dict,
                                                                    train_features=train_gid_feat_dict,
                                                                    test_labels=gid_label_dict,
                                                                    test_features=gid_features_dict,
                                                                    n_trees=n_trees,
                                                                    depth=tree_depth)
        out_folder = self.attributes.pred_session_run_path

        RF.write_arc_predictions(RF.session_name)
        RF.draw_segmentation(dirpath=RF.pred_session_run_path)

        compute_prediction_metrics('random_forest', predictions, labels, out_folder)

        RF.load_feature_names()
        RF.feature_importance(feature_names=RF.feature_names,
                              features=gid_features_dict,
                              labels=gid_label_dict)
        RF.write_feature_importance()


        #
        # Perform remainder of runs and don't need to read feats again
        #
        if not RF.params['load_features']:
            RF.params['load_features'] = False
            RF.params['write_features'] = False
            RF.params['load_features'] = True
            RF.params['write_feature_names'] = False
            RF.params['save_filtered_images'] = False
            RF.params['collect_features'] = False

        run_manager = Run_Manager(model=RF,
                                  training_window_file=self.window_file,
                                  features_file=self.feature_file,
                                  sample_idx=self.sample_idx,
                                  model_name=self.model_name,
                                  format=format)
        if multi_run:
            run_manager.perform_runs()

    def update_run_info(self, experiment_folder_name=None):
        self.attributes.update_run_info(write_folder=experiment_folder_name)

class experiment_logger:
    def __init__(self, experiment_folder, input_folder):
        self.experiment_folder = experiment_folder
        self.input_folder = input_folder
        self.dataset_base_path = os.path.basename(experiment_folder)
        self.parameter_list_file = None
        self.window_list_file = None
        self.image_name = None
        self.topo_image_name = None
        self.label_file = None

        self.input_list = [ self.topo_image_name,
                            self.image_name,
                            self.label_file]

    def record_filename(self, **kwargs):
        if 'parameter_list_file' in kwargs.keys():
            self.parameter_list_file = kwargs['parameter_list_file']
        if 'window_list_file' in kwargs.keys():
            self.window_list_file = kwargs['window_list_file']
        if 'label_file' in kwargs.keys():
            self.label_file = kwargs['label_file']
            self.label_file = os.path.split(self.label_file)[1]
        if 'image_name' in kwargs.keys():
            self.image_name = kwargs['image_name']
            self.image_name = os.path.split(self.image_name)[1]
        if 'topo_image_name' in kwargs.keys():
            self.topo_image_name = kwargs['topo_image_name']
            self.topo_image_name = os.path.split(self.topo_image_name)[1]

        self.input_list = [self.topo_image_name,
                           self.image_name,
                           self.label_file]

    def write_experiment_info(self):

        def write_input_info():
            description_file = os.path.join(self.input_folder, 'description.txt')
            print("... Writing bounds file to:", description_file)
            description_file = open(description_file, "w+")
            for fname in self.input_list[:-1]:
                description_file.write(fname + '\n')
            description_file.write(str(self.input_list[-1]))
            description_file.close()

        def write_experiment_info():
            shutil.copy(self.parameter_list_file, self.experiment_folder)
            #shutil.copy(self.window_list_file, self.experiment_folder)

        write_input_info()
        #write_experiment_info()

