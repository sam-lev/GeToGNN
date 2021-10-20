import os
import shutil
import numpy as np


from getognn import supervised_getognn
from getognn import unsupervised_getognn
from getograph import Attributes
from ml.Random_Forests import RandomForest
from ml.MLP import mlp
from ml.UNet import UNetwork, UNet_Trainer, UNet_Classifier
from ml.utils import get_train_test_val_partitions
from ml.utils import get_partition_feature_label_pairs
from proc_manager.run_manager import *
from metrics.model_metrics import compute_getognn_metrics
from metrics.model_metrics import compute_prediction_metrics
from data_ops.set_params import set_parameters
from compute_multirun_metrics import multi_run_metrics

from localsetup import LocalSetup
LocalSetup = LocalSetup()

#__all__ = ['run_experiment', 'experiment_logger']

class runner:
    def __init__(self, experiment_name, window_file_base = None
                 , sample_idx=2, multi_run = True, parameter_file_number = 1):

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
                     'transform_tests',                    # 6
                     'map_border',
                     'foam_cell'][self.sample_idx]        # 7

        self.image = ['im0236_o_700_605.raw',
                 'MAX_neuron_640_640.raw',
                 'MAX__0030_Image0001_01_o_1737_1785.raw',
                 'sub_CMC_example_o_969_843.raw',
                 'berghia_o_891_897.raw',
                 'att_0_460_446_484.raw',
                'diadem16_transforms_o_1000_1000.raw',
                      'border1_636_2372.raw',
                      'foam0235_828_846.raw'][self.sample_idx]  # neuron1
        self.label_file = ['im0236_la2_700_605.raw.labels_2.txt',
                      'MAX_neuron_640_640.raw.labels_3.txt',
                      'MAX__0030_Image0001_01_s2_C001Z031_1737_1785.raw.labels_4.txt',
                      'sub_CMC_example_l1_969_843.raw.labels_0.txt',
                      'berghia_prwpr_e4_891_896.raw.labels_3.txt',
                      'att_L3_0_460_446_484.raw.labels_0.txt',
                      'diadem16_transforms_s1_1000_1000.raw.labels_14.txt',
                           'border1_636x2372.raw.labels_0.txt',
                           'foam0235_fa_828_846.raw.labels_8.txt'][self.sample_idx]  # neuron1
        self.msc_file = os.path.join(LocalSetup.project_base_path, 'datasets', self.name,
                                'input', self.label_file.split('raw')[0] + 'raw')

        print("    * : local project base path:  ", LocalSetup.project_base_path)

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
        self.parameter_file_number = parameter_file_number


    def start(self, model='getognn', learning='supervised'):
        if model == 'getognn':
            if learning == 'supervised':
                self.run_supervised_getognn(self.multi_run)
            else:
                self.run_unsupervised_getognn(self.multi_run)
        if model == 'mlp':
            self.run_mlp(self.multi_run)
        if model == 'random_forest':
            self.run_random_forest(self.multi_run)
        if model == 'unet':
            self.run_unet(self.multi_run)



    def run_supervised_getognn(self, multi_run):
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
        if getognn.params['feature_importance']:
            getognn.feature_importance(feature_names=getognn.feature_names,
                                  features=gid_features_dict,
                                  labels=gid_label_dict,
                                       plot=False)
            getognn.write_feature_importance()

        #
        # Perform remainder of runs and don't need to read feats again
        #
        #if not getognn.params['load_features']:
        getognn.params['write_features'] = False
        getognn.params['load_features'] = True
        getognn.params['write_feature_names'] = False
        getognn.params['save_filtered_images'] = False
        getognn.params['collect_features'] = False
        getognn.params['load_geto_attr'] = True

        run_manager = Run_Manager(model=getognn,
                                  training_window_file=self.window_file,
                                  features_file=self.feature_file,
                                  sample_idx=self.sample_idx,
                                  model_name=self.model_name,
                                  format=format)
        if multi_run:
            run_manager.perform_runs()

    def run_unsupervised_getognn(self, multi_run):
        #
        # Train single run of getognn and obtrain trained model
        #
        unsup_getognn = unsupervised_getognn(model_name=self.model_name)
        unsup_getognn.build_getognn( sample_idx=self.sample_idx, experiment_num=self.experiment_num,
                                   experiment_name=self.experiment_name,window_file_base=self.window_file_base,
                                   parameter_file_number=self.parameter_file_number,
                                   format = format, run_num=self.run_num,name=self.name, image=self.image,
                                   label_file=self.label_file, msc_file=self.msc_file,
                                   ground_truth_label_file=self.ground_truth_label_file,
                                   experiment_folder = self.experiment_folder,
                                   write_path=self.write_path, feature_file=self.feature_file,
                                   window_file=None,model_name="GeToGNN")

        self.attributes = unsup_getognn.attributes

        embedding_name = os.path.join(self.experiment_folder,'embedding')


        getognn = unsup_getognn.train()
        unsup_getognn.classify(embedding_name=embedding_name)

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
                                   plot=False)
        getognn.write_feature_importance()

        #
        # Perform remainder of runs and don't need to read feats again
        #
        if not getognn.params['load_features']:
            getognn.params['write_features'] = False
            getognn.params['load_features'] = True
            getognn.params['write_feature_names'] = False
            getognn.params['save_filtered_images'] = False
            getognn.params['collect_features'] = False
            getognn.params['load_preprocessed'] = True
            getognn.params['load_geto_attr'] = True
            getognn.params['load_feature_names'] = True

        run_manager = Run_Manager(model=getognn,
                                  training_window_file=self.window_file,
                                  features_file=self.feature_file,
                                  sample_idx=self.sample_idx,
                                  model_name=self.model_name,
                                  format=format,
                                  learning_type='unsupervised')
        if multi_run:
            run_manager.perform_runs()

    def run_mlp(self, multi_run ):
        run_params = set_parameters(read_params_from=self.parameter_file_number,
                                    experiment_folder=self.write_path)
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

        MLP = mlp(features=gid_features_dict, labels=gid_label_dict,
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

        preds, labels, accuracy = MLP.train()

        out_folder = self.attributes.pred_session_run_path

        MLP.write_arc_predictions(MLP.session_name)
        MLP.draw_segmentation(dirpath=MLP.pred_session_run_path)

        compute_prediction_metrics('mlp', preds, labels, out_folder)

        MLP.load_feature_names()
        MLP.feature_importance(feature_names=MLP.feature_names,
                              features=gid_features_dict,
                              labels=gid_label_dict)
        MLP.write_feature_importance()

        #
        # Perform remainder of runs and don't need to read feats again
        #
        if not MLP.params['load_features']:
            MLP.params['write_features'] = False
            MLP.params['load_features'] = True
            MLP.params['write_feature_names'] = False
            MLP.params['save_filtered_images'] = False
            MLP.params['collect_features'] = False
            MLP.params['load_preprocessed'] = True
            MLP.params['load_geto_attr'] = True
            MLP.params['load_feature_names'] = True


        run_manager = Run_Manager(model=MLP,
                                  training_window_file=self.window_file,
                                  features_file=self.feature_file,
                                  sample_idx=self.sample_idx,
                                  model_name=self.model_name,
                                  format=format,
                                  expanding_boxes=True,
                                  parameter_file_number=self.parameter_file_number)
        if multi_run:
            run_manager.perform_runs()

        '''mlp_accuracy, mlp_p, mlp_r, mlp_fs = mlp(features=gid_features_dict, labels=gid_label_dict,
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
        mlp_accuracy = mlp_accuracy[0]'''


    #
    #
    #                   *     UNET            *
    #
    #
    #
    def run_unet(self,multi_run):

        f = open(self.window_file, 'r')
        param_lines = f.readlines() if multi_run else None
        f.close()

        el = len(param_lines)-1
        sets = el//10
        sets = sets-1 if sets%2!=0 else sets
        select_subsets = []
        select_subsets = [param_lines[0:2]]
        # [param_lines[0:2] , param_lines[sets:sets+4] ,\
        #               param_lines[2*sets:(2*sets)+6] , param_lines[3*sets:(3*sets)+8],
        #               param_lines[4 * sets:(4 * sets) + 16], param_lines[6 * sets:(6 * sets) + 32],
        #               param_lines[5 * sets:-1]   ,param_lines[3 * sets:-1]                ]
        select_subsets.sort()

        training_size = len(select_subsets) if multi_run else None

        num_training = 1 if training_size is None else training_size
        exp_folder = ''
        input_folder = ''
        label_file = ''
        param_file = ''

        experiment_folder = os.path.join(LocalSetup.project_base_path, 'datasets'
                                              , self.write_path)
        run_folder = os.path.join(experiment_folder, 'runs')
        if os.path.exists(run_folder):
            shutil.rmtree(run_folder)

        #lines = select_subsets
        for num_samp, samp in enumerate(select_subsets):

            print("    * training regions: ")
            print("    * ", samp)


            UNet = UNetwork(in_channel=1, out_channel=1,
                            ground_truth_label_file=self.ground_truth_label_file,
                            skip_connect=True,
                            run_num=self.run_num,
                            parameter_file_number = self.parameter_file_number,
                            geomsc_fname_base = self.msc_file,
                            label_file=self.ground_truth_label_file,
                            model_name=self.model_name,
                            load_feature_graph_name=False,
                            image=self.image,
                            name=self.name,
                            write_folder=self.write_path,
                            compute_features=num_samp!=0,
                            training_size=len(samp),
                            region_list=samp)



            trainer = UNet_Trainer(UNet)
            pred_thresh = 0.5
            results = trainer.launch_training(view_results=False, pred_thresh=pred_thresh)

            train_losses, test_losses, F1_scores, best_f1, \
            val_imgs, val_segs, sample_losses, val_img_preds, running_best_model = results
            UNet.running_best_model = trainer.running_best_model

            #
            # Perform remainder of runs and don't need to read feats again
            #
            #if not UNet.params['load_features']:
            UNet.params['load_features'] = True
            UNet.params['write_features'] = False
            UNet.params['load_features'] = True
            UNet.params['write_feature_names'] = False
            UNet.params['save_filtered_images'] = False
            UNet.params['collect_features'] = False
            UNet.params['load_preprocessed'] = True
            UNet.params['load_geto_attr'] = True
            UNet.params['load_feature_names'] = True

            ### unet_classifier = UNet_Classifier(UNet, self.window_file)
            ### inf_resuts = unet_classifier.infer(running_best_model ,infer_subsets=True, view_results=True)
            inf_results = UNet.infer(running_best_model,training_window_file=self.window_file,
                                     infer_subsets=True, view_results=False,
                                     pred_thresh=pred_thresh,

                                     test=True)

            test_losses, val_imgs, val_segs, val_img_preds, running_val_loss,\
            F1_score, labels, predictions = inf_results

            exp_folder = os.path.join(UNet.params['experiment_folder'])
            input_folder=UNet.input_folder
            # batch_metric_folder = os.path.join(exp_folder, 'batch_metrics') if not multi_run \
            #     else os.path.join(exp_folder,'runs',str(num_samp), 'batch_metrics')
            # if not os.path.exists(batch_metric_folder):
            #     os.makedirs(batch_metric_folder)
            #
            # out_folder = UNet.pred_session_run_path
            #
            # # compute_prediction_metrics('unet', predictions, labels, out_folder)
            # #
            # #UNet.write_arc_predictions(UNet.session_name)
            # # UNet.draw_segmentation(dirpath=UNet.pred_session_run_path)
            #
            # multi_run_metrics(model=UNet.type, exp_folder=exp_folder, batch_multi_run=str(num_samp),
            #                   bins=7, runs=os.path.join('runs',str(num_samp)),
            #                   plt_title=exp_folder.split('/')[-1])



            # if num_samp == num_training:
            #     exp_folder = os.path.join(UNet.params['experiment_folder'])
            #     batch_metric_folder = os.path.join(exp_folder, 'batch_metrics')
            #     if not os.path.exists(batch_metric_folder):
            #         os.makedirs(batch_metric_folder)
            #     multi_run_metrics(model=UNet.type, exp_folder=exp_folder, batch_multi_run=True,
            #                       bins=7, runs='runs', plt_title=exp_folder.split('/')[-1])
            #     self.logger = experiment_logger(experiment_folder=self.experiment_folder,
            #                                                        input_folder=UNet.input_folder)
            #     topo_image_name = self.ground_truth_label_file.split('.labels')[0]
            #     self.logger.record_filename(label_file=UNet.label_file,
            #                                 parameter_list_file=UNet.param_file,
            #                                 image_name=self.image,
            #                                 topo_image_name=topo_image_name)
        base_exp_folder = exp_folder
        exp_folder = os.path.join(exp_folder, 'runs')

        batch_metric_folder = base_exp_folder
        if not os.path.exists(batch_metric_folder):
            os.makedirs(batch_metric_folder)

        # compute_prediction_metrics('unet', predictions, labels, out_folder)
        #
        # UNet.write_arc_predictions(UNet.session_name)
        # UNet.draw_segmentation(dirpath=UNet.pred_session_run_path)
        metric = 'f1'
        print("    * ", "after training loop")
        multi_run_metrics(model='metric_averages', exp_folder=exp_folder,
                          batch_multi_run=False, avg_multi=True,batch_of_batch=True,
                          bins=7, runs='batch_metrics',
                          plt_title=exp_folder.split('/')[-1])



    #
    #
    #                     *  RANDOM FOREST *
    #
    #
    def run_random_forest(self, multi_run ):

        #run_params = set_parameters(read_params_from=self.parameter_file_number,
        #                            experiment_folder=self.write_path)
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

        print("    * :", RF.params)

        RF.build_random_forest(ground_truth_label_file=self.ground_truth_label_file)

        print("   *:   feat names length", len(RF.feature_names))

        partition_label_dict, partition_feat_dict = get_partition_feature_label_pairs(
            RF.node_gid_to_partition,
            RF.node_gid_to_feature,
            RF.node_gid_to_label,
            test_all=True)


        gid_features_dict = partition_feat_dict['all']
        gid_label_dict = partition_label_dict['all']
        train_gid_label_dict = partition_label_dict['train']
        train_gid_feat_dict = partition_feat_dict['train']

        n_trees = RF.params['number_forests']
        tree_depth = RF.params['forest_depth']

        if RF.params['forest_class_weights']:
            class_1_weight = RF.params['class_1_weight']
            class_2_weight = RF.params['class_2_weight']
        else:
            class_1_weight = 1.0
            class_2_weight = 1.0
        predictions, labels, node_gid_to_prediction = RF.classifier(self.attributes.node_gid_to_prediction,
                                                                    train_labels=train_gid_label_dict,
                                                                    train_features=train_gid_feat_dict,
                                                                    test_labels=gid_label_dict,
                                                                    test_features=gid_features_dict,
                                                                    n_trees=n_trees,
                                                                    class_1_weight=class_1_weight,
                                                                    class_2_weight=class_2_weight,
                                                                    weighted_distribution=RF.params['forest_class_weights'],
                                                                    depth=tree_depth)
        out_folder = RF.attributes.pred_session_run_path

        RF.write_arc_predictions(RF.session_name)
        RF.draw_segmentation(dirpath=RF.pred_session_run_path)

        compute_prediction_metrics('random_forest', predictions, labels, out_folder)

        #if RF.params['load_feature_names']:
        #    RF.load_feature_names()
        if RF.params['feature_importance']:
            RF.feature_importance(feature_names=RF.feature_names,
                                  features=gid_features_dict,
                                  labels=gid_label_dict,
                                  plot=False)
            RF.write_feature_importance()


        #
        # Perform remainder of runs and don't need to read feats again
        #
        if not RF.params['load_features']:
            RF.params['write_features'] = False
            RF.params['load_features'] = True
            RF.params['write_feature_names'] = False
            RF.params['save_filtered_images'] = False
            RF.params['collect_features'] = False
            RF.params['load_preprocessed'] = True
            RF.params['load_geto_attr'] = True
            RF.params['load_feature_names'] = True

        run_manager = Run_Manager(model=RF,
                                  training_window_file=self.window_file,
                                  features_file=self.feature_file,
                                  sample_idx=self.sample_idx,
                                  model_name=self.model_name,
                                  format=format,
                                  parameter_file_number=self.parameter_file_number)
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

