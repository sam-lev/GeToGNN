import os
import shutil
import numpy as np
import matplotlib.pyplot as plt


from getograph import Attributes
import getograph
from getofeaturegraph import GeToFeatureGraph
import getograph as ggraph
from data_ops.utils import compute_subgraph_features
from ml.Random_Forests import RandomForest
from ml.MLP import mlp
from ml.UNet import UNetwork
from ml.utils import get_partition_feature_label_pairs
from ml.utils import get_merged_features, pout

from proc_manager.run_manager import *
from metrics.model_metrics import compute_getognn_metrics, compute_prediction_metrics,compute_opt_F1_and_threshold
from data_ops.set_params import set_parameters
from compute_multirun_metrics import multi_run_metrics, multi_model_metrics
from ml.features import get_points_from_vertices
from data_ops.utils import plot
from data_ops.utils import tile_region

from localsetup import LocalSetup
LocalSetup = LocalSetup()

#__all__ = ['run_experiment', 'experiment_logger']

class runner:
    def __init__(self, experiment_name, window_file_base = None, clear_runs=False,
                 percent_train_thresh = 0, break_training_size=50,
                 load_features = True, compute_features = False, load_geto_features = False,
                 feats_independent=False,
                 compute_geto_features = False,
                 sample_idx=2, multi_run = True, parameter_file_number = 1,
                 load_subgraph_labels=False):

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
        self.run_num = 1
        self.model_name = 'GeToGNN'  # 'experiment'#'input_select_from_1st_inference'

        self.load_features = load_features
        self.compute_features = compute_features
        self.load_geto_features = load_geto_features
        self.compute_geto_features = compute_geto_features
        self.feats_independent = feats_independent
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
                     'foam_cell',
                     'diadem_sub1',
                     'berghia_membrane'][self.sample_idx]        # 7

        self.image = ['im0236_o_700_605.raw',
                 'MAX_neuron_640_640.raw',
                 'MAX__0030_Image0001_01_o_1737_1785.raw',
                 'sub_CMC_example_o_969_843.raw',
                 'berghia_o_891_897.raw',
                 'att_0_460_446_484.raw',
                'diadem16_transforms_o_1000_1000.raw',
                      'border1_636_2372.raw',
                      'foam0235_828_846.raw',
                      'maxdiadem_o_1170_1438.raw',
                      'berghia_o_891_896.raw'][self.sample_idx]  # neuron1
        self.label_file = ['im0236_la2_700_605.raw.labels_0.txt', # og lab is _2
                      'MAX_neuron_640_640.raw.labels_3.txt',
                      'MAX__0030_Image0001_01_s2_C001Z031_1737_1785.raw.labels_3.txt',
                      'sub_CMC_example_l1_969_843.raw.labels_0.txt',
                      'berghia_prwpr_e4_891_896.raw.labels_3.txt',
                      'att_L3_0_460_446_484.raw.labels_0.txt',
                      'diadem16_transforms_s1_1000_1000.raw.labels_14.txt',
                           'border1_636x2372.raw.labels_0.txt',
                           'foam0235_fa_828_846.raw.labels_0.txt',
                           'maxdiadem_m1g1_1170_1438.raw.labels_4.txt',
                           'berghia_s5ipr_891_896.raw.labels_4.txt'][self.sample_idx]  # neuron1

        if load_subgraph_labels:
            self.subgraph_labels = ['im0236_la2_700_605.raw.labels_0.txt',  # og lab is _2
                               'MAX_neuron_640_640.raw.labels_3.txt',
                               ['.labels_0.txt'],
                               'sub_CMC_example_l1_969_843.raw.labels_0.txt',
                               'berghia_prwpr_e4_891_896.raw.labels_3.txt',
                               'att_L3_0_460_446_484.raw.labels_0.txt',
                               'diadem16_transforms_s1_1000_1000.raw.labels_14.txt',
                               'border1_636x2372.raw.labels_0.txt',
                               'foam0235_fa_828_846.raw.labels_0.txt',
                               'maxdiadem_m1g1_1170_1438.raw.labels_4.txt',
                               'berghia_s5ipr_891_896.raw.labels_4.txt'][self.sample_idx]
        else:
            self.subgraph_labels = None
        self.labels_subcomplex = self.label_file#'im0236_la2_700_605.raw.labels_0.txt'#][self.sample_idx]

        self.msc_file = os.path.join(LocalSetup.project_base_path, 'datasets', self.name,
                                'input', self.label_file.split('raw')[0] + 'raw')
        self.input_path = os.path.join(LocalSetup.project_base_path, 'datasets', self.name,
                                'input')
        self.image_path = os.path.join(self.input_path, self.label_file.split('raw')[0] + 'raw')#self.image)

        print("    * : local project base path:  ", LocalSetup.project_base_path)

        self.ground_truth_label_file = os.path.join(LocalSetup.project_base_path, 'datasets',
                                               self.name, 'input', self.label_file)
        self.format = 'raw'
        self.experiment_folder = os.path.join(LocalSetup.project_base_path, 'datasets',
                                              self.name, self.experiment_name)
        if not os.path.exists(self.experiment_folder):
            os.makedirs(self.experiment_folder)

        self.write_path = os.path.join(self.name,self.experiment_name)  # 'experiment_'+str(experiment_num)+'_'+experiment_name

        # clear past runs
        experiment_folder = os.path.join(LocalSetup.project_base_path, 'datasets'
                                         , self.write_path)

        if clear_runs:
            run_folder = os.path.join(experiment_folder, 'runs')
            if os.path.exists(run_folder):
                shutil.rmtree(run_folder)
            
        self.feature_file = os.path.join( self.write_path, 'features', self.model_name)
        self.window_file = None    # os.path.join(LocalSetup.project_base_path, "datasets",
                                   # self.write_path,
                                   # self.window_file_base)
        self.parameter_file_number = parameter_file_number

        self.thresh_list = [0.3,0.4,0.5,0.6]

        self.growth_radius = 2 if self.name is not 'berghia_membrane' else 4

        self.percent_train_thresh = percent_train_thresh

        self.break_training_size = break_training_size




    def start(self, model='getognn',
              boxes=None,
              dims=None,
              learning='supervised',
              compute_complex=False,
              persistences = None,
              subgraph_labels=None):


        if model == 'getognn':
            if learning == 'supervised':
                self.run_supervised_getognn( boxes=boxes, dims=dims )
            elif learning == 'unsupervised':
                self.run_unsupervised_getognn(self.multi_run)
            elif learning == 'subcomplex':
                self.run_subcomplex_informed_getognn(boxes=boxes,
                                                     dims=dims,
                                                     compute_complex=compute_complex,
                                                     persistence_subgraphs = persistences,
                                                     subgraph_labels=self.subgraph_labels)
        if model == 'mlp':
            self.run_mlp( boxes=boxes, dims=dims, flavor=learning )
        if model == 'random_forest':
            self.run_random_forest(self.multi_run, boxes=boxes, dims=dims, learning=learning)
        if model == 'unet':
            self.run_unet(self.multi_run, boxes=boxes, dims=dims)

    def multi_model_metrics(self, models, exp_dirs, write_dir, metric='f1', plot_experiments = None):
        experiment_folders = []
        models = []
        for exp in exp_dirs:
            experiment_folder = os.path.join(LocalSetup.project_base_path, 'datasets',
                                                  self.name,exp)

            experiment_folders.append(experiment_folder)
            #m = str(exp).split('/')[-2]
            m = exp.replace('_', ' ')
            models.append(m)
        print(experiment_folders)
        write_dir =  os.path.join(LocalSetup.project_base_path)
        multi_model_metrics(models=models, exp_dirs=experiment_folders, #batchmulti_run=True#
                            data_name=self.name, write_dir=write_dir,metric=metric,
                            plot_experiments=plot_experiments)



    def __group_pairs(self, lst):
        for i in range(0, len(lst), 2):
            yield tuple(lst[i : i + 2])

    def graph_statistics(self, node_gid_dict, gid_edge_dict, gid_split_dict, gid_to_label_dict = None):

        total_number_nodes = len(node_gid_dict.keys())
        total_training_nodes = 0#len(node_gid_dict.keys())
        total_test_nodes = 0
        total_length_positive_training_nodes = 0
        total_length_positive_nodes = 0
        total_length_training_nodes = 0
        total_length_test_nodes = 0
        total_length_nodes = 0
        total_number_edges = len(gid_edge_dict.keys())
        total_foreground_nodes = 0
        total_nodes            = 0
        for gid in node_gid_dict.keys():#gid_split_dict.keys():
            label = gid_to_label_dict[gid]
            node = node_gid_dict[gid]
            if gid_split_dict[gid] == 'test' or gid_split_dict[gid] == 'val':
                total_test_nodes += 1
                total_length_test_nodes += len(node.points)
            else:
                total_training_nodes += 1
                if not label[1] < 1:
                    total_length_positive_training_nodes += len(node.points)
                total_length_training_nodes += len(node.points)

            total_length_nodes += len(node.points)
            total_nodes        += 1
            if not label[1] < 1:
                total_length_positive_nodes += len(node.points)
                total_foreground_nodes      += 1

        return total_number_nodes, total_training_nodes, total_number_edges, total_test_nodes,\
               total_length_training_nodes, total_length_positive_training_nodes, total_length_positive_nodes,\
               total_length_test_nodes, total_length_nodes, total_nodes, total_foreground_nodes


    def grow_box(self, dims, boxes):
        IMG_WIDTH = dims[0]
        IMG_HEIGHT = dims[1]
        growth_regions = []
        add_y = 0
        add_x = 0
        percent_training = IMG_HEIGHT*IMG_WIDTH
        for grow in range(int(percent_training)):
            g_box = []
            for box in boxes:

                add_y = int( 0.05*IMG_WIDTH) * grow
                add_x = int(0.05*IMG_HEIGHT) * grow

                # if add_y + box[1] > IMG_WIDTH or box[0] - add_y < 0:
                #     add_x = int(0.015*IMG_HEIGHT) * grow
                # if add_x + box[3] > IMG_HEIGHT or box[2] - add_x < 0:
                #     add_y = int(0.015 * IMG_WIDTH) * grow



                growth_box = [max(0, box[0] - add_y), min(IMG_WIDTH, box[1] + add_y),
                              max(box[2] - add_x, 0), min(IMG_HEIGHT, box[3] + add_x)]
                g_box.append(growth_box)


            growth_regions.append(g_box)
        g_box = []
        for box in boxes:
            add_y = int(0.42 * IMG_WIDTH)
            add_x = int(0.42 * IMG_HEIGHT)

            growth_box = [max(0, box[0] - add_y), min(IMG_WIDTH, box[1] + add_y),
                          max(box[2] - add_x, 0), min(IMG_HEIGHT, box[3] + add_x)]
            g_box.append(growth_box)
        growth_regions.append(g_box)
        
        return growth_regions

    def get_box_regions(self, boxes, INVERT=False):
        X_BOX = []
        Y_BOX = []

        box_sets = []
        for box in boxes:
            box_set = tile_region(step_X=64, step_Y=64, step=0.5,
                                  Y_START=box[2], Y_END=box[3],
                                  X_START=box[0], X_END=box[1],
                                  INVERT=INVERT)
            box_sets.append(box_set)

        boxes = []
        for box_pair in box_sets:
            boxes += box_pair
            # for box in box_pair:
            #     X_BOX.append(box[0])
            #     Y_BOX.append(box[1])
        return boxes#X_BOX,Y_BOX



    def get_boxes(self, param_lines):
        boxes = []
        for box_set in param_lines:
            boxes = [
                i for i in self.__group_pairs(box_set)
            ]

        current_box_dict = {}

        # self.model.update_run_info(batch_multi_run=str(len(box_set)))
        # self.model.run_num = str(len(box_set))
        for current_box_idx in range(len(boxes)):

            self.active_file = os.path.join(LocalSetup.project_base_path, 'continue_active.txt')
            f = open(self.active_file, 'w')
            f.write('0')
            f.close()
            #     else current_box_idx+1
            current_box = [boxes[current_box_idx]]  # , boxes[current_box_idx+1]]
            # if self.expanding_boxes is \
            #                                        False else boxes[0:current_box_idx]
            # if self.expanding_boxes:
            current_box = [y for x in current_box for y in x]


            for bounds in current_box:

                name_value = bounds.split(' ')


                # window selection(s)
                # for single training box window
                if name_value[0] not in current_box_dict.keys():
                    current_box_dict[name_value[0]] = list(map(int, name_value[1].split(',')))
                # for multiple boxes
                else:
                    current_box_dict[name_value[0]].extend(list(map(int, name_value[1].split(','))))
            self.counter_file = os.path.join(LocalSetup.project_base_path, 'run_count.txt')
        X_BOX = [
            i for i in self.__group_pairs([i for i in current_box_dict['x_box']])
        ]
        Y_BOX = [
            i for i in self.__group_pairs([i for i in current_box_dict['y_box']])
        ]
        #X_BOX =list(set(X_BOX))
        #Y_BOX = list(set(Y_BOX))
        return X_BOX,Y_BOX

    def run_supervised_getognn(self, boxes=None,  dims=None):


        from getognn import supervised_getognn

        IMG_WIDTH = dims[0]
        IMG_HEIGHT = dims[1]

        growth_regions = self.grow_box(dims=dims, boxes=boxes)

        BEGIN_LOADING_FEATURES      = self.load_features
        COMPUTE_FEATURES            = self.compute_features
        BEGIN_LOADING_GETO_FEATURES = self.load_geto_features
        COMPUTE_GETO_FEATURES       = self.compute_geto_features
        FEATS_INDEPENDENT           = self.feats_independent
        run_feat_importance = 1

        for gr in range(len(growth_regions)):



            regions = growth_regions[gr]

            BOXES = self.get_box_regions(regions)
            X_BOX = [b[0] for b in BOXES]
            Y_BOX = [b[1] for b in BOXES]

            num_percent = 0
            for box in regions:
                num_percent += float((box[3] - box[2]) * (box[1] - box[0]))
            percent = num_percent / float(IMG_WIDTH * IMG_HEIGHT)
            percent_float = percent *100
            print("    * percent", percent)
            if percent_float > self.break_training_size:
                break
            #if gr > 1 and percent_float < 1:
            #    continue
            if percent_float < self.percent_train_thresh:
                continue
            percent = int(round(percent, 2) )


            sup_getognn = supervised_getognn(model_name=self.model_name)
            sup_getognn.build_getognn(
                BEGIN_LOADING_FEATURES      = BEGIN_LOADING_FEATURES,
                COMPUTE_FEATURES            = COMPUTE_FEATURES,
                BEGIN_LOADING_GETO_FEATURES = BEGIN_LOADING_GETO_FEATURES,
                COMPUTE_GETO_FEATURES       = COMPUTE_GETO_FEATURES,
                FEATS_INDEPENDENT           = FEATS_INDEPENDENT,
                                       sample_idx=self.sample_idx,
                                       experiment_num=self.experiment_num,
                                       experiment_name=self.experiment_name,
                                       window_file_base=self.window_file_base,
                                       parameter_file_number=self.parameter_file_number,
                                       format = format,
                                       run_num=percent,
                                       name=self.name, image=self.image,
                                       label_file=self.label_file,
                                       msc_file=self.msc_file,
                                       ground_truth_label_file=self.ground_truth_label_file,
                                       experiment_folder = self.experiment_folder,
                                       write_path=self.write_path,
                                       feature_file=self.feature_file,
                                       window_file=None,model_name="GeToGNN",
                                       X_BOX=X_BOX,
                                       Y_BOX=Y_BOX,
                                       regions=regions)

            if BEGIN_LOADING_FEATURES:
                sup_getognn.getognn.params['load_features'] = True
                sup_getognn.getognn.params['write_features'] = False
                sup_getognn.getognn.params['write_feature_names'] = False
                sup_getognn.getognn.params['save_filtered_images'] = False
                sup_getognn.getognn.params['collect_features'] = False
                sup_getognn.getognn.params['load_preprocessed'] = True
                sup_getognn.getognn.params['load_feature_names'] = True
            elif COMPUTE_FEATURES:
                sup_getognn.getognn.params['load_features'] = False
                sup_getognn.getognn.params['write_features'] = True
                sup_getognn.getognn.params['write_feature_names'] = True
                sup_getognn.getognn.params['save_filtered_images'] = True
                sup_getognn.getognn.params['collect_features'] = True
                sup_getognn.getognn.params['load_preprocessed'] = False
                sup_getognn.getognn.params['load_feature_names'] = False
            if BEGIN_LOADING_GETO_FEATURES:
                sup_getognn.getognn.params['load_geto_attr'] = True
                sup_getognn.getognn.params['load_feature_names'] = True
            elif COMPUTE_GETO_FEATURES:
                sup_getognn.getognn.params['geto_as_feat'] = True
                sup_getognn.getognn.params['load_geto_attr'] = False

            sup_getognn.getognn.get_train_test_val_subgraph_split(  collect_validation=False,
                                                                    validation_hops=1,
                                                                    validation_samples=1)

            if sup_getognn.getognn.params['write_json_graph']:
                sup_getognn.getognn.write_json_graph_data(folder_path=sup_getognn.getognn.pred_session_run_path,
                                                          name="GeToGNN" + '_' + sup_getognn.getognn.params['name'])

            # random walks
            # if not sup_getognn.getognn.params['load_preprocessed_walks']:
            #     walk_embedding_file = os.path.join(sup_getognn.getognn.LocalSetup.project_base_path, 'datasets',
            #                                        sup_getognn.getognn.params['write_folder'], 'walk_embeddings',
            #                                        'gnn')
            #     sup_getognn.getognn.params['load_walks'] = walk_embedding_file
            #     sup_getognn.getognn.run_random_walks(walk_embedding_file=walk_embedding_file)
            # else:
            #     walk_embedding_file = os.path.join(sup_getognn.getognn.LocalSetup.project_base_path, 'datasets',
            #                                        sup_getognn.getognn.params['write_folder'], 'walk_embeddings',
            #                                        'gnn')
            #     sup_getognn.getognn.params['load_walks'] = walk_embedding_file

            sup_getognn.getognn.params['load_walks'] = False


            sup_getognn.compute_features()


            if BEGIN_LOADING_FEATURES or COMPUTE_FEATURES:
                BEGIN_LOADING_FEATURES      = True
                COMPUTE_FEATURES            = False

            if BEGIN_LOADING_GETO_FEATURES or COMPUTE_GETO_FEATURES:
                BEGIN_LOADING_GETO_FEATURES = True
                COMPUTE_GETO_FEATURES       = False




            getognn = sup_getognn.train(run_num=str(percent))

            #getognn.update_run_info(batch_multi_run=str(percent))
            getognn.run_num = percent

            out_folder = os.path.join(getognn.pred_session_run_path)
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)


            #

            #
            # Feature Importance
            #
            if BEGIN_LOADING_GETO_FEATURES and BEGIN_LOADING_FEATURES:
                node_gid_to_feature, node_gid_to_feat_idx, features = get_merged_features(getognn)
            else:
                node_gid_to_feature = getognn.node_gid_to_standard_feature

            partition_label_dict, partition_feat_dict = get_partition_feature_label_pairs(
                getognn.node_gid_to_partition,
                node_gid_to_feature,
                getognn.node_gid_to_label,
                test_all=True)
            gid_features_dict = partition_feat_dict['all']
            gid_label_dict = partition_label_dict['all']
            #getognn.load_feature_names()
            if run_feat_importance:     #getognn.params['feature_importance'] and

                names = []

                if BEGIN_LOADING_FEATURES or COMPUTE_FEATURES:
                    names += getognn.load_feature_names()
                if BEGIN_LOADING_GETO_FEATURES or COMPUTE_GETO_FEATURES:
                    names += getognn.load_geto_feature_names()


                getognn.feature_importance(feature_names=names,#getognn.feature_names,
                                      features=node_gid_to_feature,#gid_features_dict,
                                      labels=gid_label_dict,
                                           plot=False)
                getognn.write_feature_importance()
                getognn.params['feature_importance'] = False
                run_feat_importance = 0

            training_reg_bg = np.zeros(getognn.image.shape[:2], dtype=np.uint8)
            for x_b, y_b in zip(getognn.x_box, getognn.y_box):
                x_box = x_b
                y_box = y_b
                training_reg_bg[x_box[0]:x_box[1], y_box[0]:y_box[1]] = 1
            # getognn.write_training_percentages(dir=getognn.pred_session_run_path, train_regions=training_reg_bg)

            #
            # Perform remainder of runs and don't need to read feats again
            #


            getognn.supervised_train()
            getognn.record_time(round(getognn.train_time, 4),
                                   dir=getognn.pred_session_run_path,
                                   type='train')
            getognn.record_time(round(getognn.pred_time, 4),
                                   dir=getognn.pred_session_run_path,
                                   type='pred')
            G = getognn.get_graph()
            getognn.equate_graph(G)
            # For computing the line graph for visualisation
            # getognn.draw_priors_graph(G)





            predictions, labels, opt_thresh = compute_getognn_metrics(getognn=getognn)


            getognn.write_arc_predictions(dir=getognn.pred_session_run_path)
            getognn.draw_segmentation(dirpath=getognn.pred_session_run_path)
            getognn.write_gnode_partitions(dir=getognn.pred_session_run_path)  # self.getognn.session_name)
            getognn.write_selection_bounds(dir=getognn.pred_session_run_path)  # self.getognn.session_name)

            # update newly partitioned/infered graoh
            G = getognn.get_graph()
            getognn.equate_graph(G)

            total_number_nodes, total_training_nodes, total_number_edges, total_test_nodes, total_length_training_nodes,\
            total_length_positive_training_nodes,  total_length_positive_nodes,\
            total_length_test_nodes, total_length_nodes, total_nodes, total_foreground_nodes = self.graph_statistics( getognn.gid_gnode_dict,
                                                                                 getognn.gid_edge_dict,
                                                                                 getognn.node_gid_to_partition,
                                                                                 getognn.node_gid_to_label)
            getognn.write_graph_statistics(total_number_nodes,
                                           total_training_nodes,
                                           total_number_edges,
                                           total_test_nodes,
                                           total_length_training_nodes,
                                           total_length_positive_training_nodes,
                                           total_length_positive_nodes,
                                           total_length_test_nodes,
                                           total_length_nodes,
                                           total_nodes, total_foreground_nodes,
                                           fname='region_percents')

            # getognn.write_training_graph_percentages(dir=getognn.pred_session_run_path,
            #                                          graph_orders=(total_number_nodes,
            #                                                        total_training_nodes))

            pred_labels_conf_matrix = np.zeros(getognn.image.shape[:2], dtype=np.float32) #* min(0.25,opt_thresh/2.) # dtype=np.uint8)
            pred_labels_msc = np.zeros(getognn.image.shape[:2], dtype=np.float32) #* min(0.25, opt_thresh/2.)
            gt_labels_msc = np.zeros(getognn.image.shape[:2], dtype=np.float32) #* min(.25,opt_thresh/2.)
            pred_prob_im = np.zeros(getognn.image.shape[:2], dtype=np.float32)
            gt_msc = np.zeros(getognn.image.shape[:2], dtype = np.float32)
            predictions_topo_bool = []
            labels_topo_bool = []
            check = 30
            for gid in getognn.node_gid_to_label.keys():  # zip(mygraph.labels, mygraph.polylines):

                gnode = getognn.gid_gnode_dict[gid]
                label = getognn.node_gid_to_label[gid]
                label = label if type(label) != list else label[1]
                line = get_points_from_vertices([gnode])
                # else is fg
                cutoff = opt_thresh

                vals = []

                for point in line:
                    ly = int(point[0])
                    lx = int(point[1])
                    pred = getognn.node_gid_to_prediction[gid]
                    vals.append(pred)


                inferred = np.array(vals, dtype="float32")
                infval = np.average(inferred)
                pred_mode = infval

                getognn.node_gid_to_prediction[gid] = [1. - infval, infval]



                getognn.node_gid_to_prediction[gid] = [1. - infval, infval]
                if check >= 0:

                    check -= 1

                t = 0
                if infval >= opt_thresh:
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
                    pred_labels_msc[lx, ly] = 1 if infval >= opt_thresh else 0
                    gt_labels_msc[lx,ly] = label
                    pred_prob_im[lx, ly] = infval
                    if training_reg_bg[lx, ly] != 1:
                        getognn.node_gid_to_partition[gid] = 'test'
                        #predictions_topo_bool.append(infval >= cutoff)
                        #gt_label = seg_whole[lx, ly]
                        labels_topo_bool.append(label >= cutoff)


            out_folder = getognn.pred_session_run_path





            images = [getognn.image, gt_labels_msc, pred_labels_msc,
                      pred_prob_im]
            names = ["Image", "Ground Truth Segmentation", "Predicted Foreground Segmentation",
                     "Line Foreground Probability"]
            for image, name in zip(images, names):
                plot(image_set=[image, training_reg_bg], name=name, type='contour', write_path=out_folder)

            image_set = [pred_labels_msc, training_reg_bg, pred_labels_conf_matrix]
            plot(image_set, name="TP FP TF TN Line Prediction",
                 type='confidence', write_path=out_folder)

            plot(image_set, name="TP FP TF TN Line Prediction",
                 type='zoom', write_path=out_folder)


            image_set = [getognn.image, training_reg_bg, gt_msc]
            plot(image_set, name="Ground Truth MSC",
                 type='confidence', write_path=out_folder)

            plot(image_set, name="Ground Truth MSC",
                 type='zoom', write_path=out_folder)

            for image, name in zip(images, names):
                plot(image_set=[image, training_reg_bg], name=name, type='zoom', write_path=out_folder)




            # batch_folder = os.path.join(self.params['experiment_folder'],'batch_metrics', 'prediction')
            # if not os.path.exists(batch_folder):
            #    os.makedirs(batch_folder)
            #

            del getognn
            del sup_getognn



    #
    #
    #                   *     UNET            *
    #
    #
    #
    def run_unet(self,multi_run=False, dims=None, boxes=None):




        IMG_WIDTH = dims[0]
        IMG_HEIGHT = dims[1]

        growth_regions = self.grow_box(dims=dims, boxes=boxes)

        all_boxes = self.get_box_regions([[0, IMG_WIDTH, 0, IMG_HEIGHT]])
        X_BOX_all = [b[0] for b in all_boxes]
        Y_BOX_all = [b[1] for b in all_boxes]

        exp_folder = ''


        for gr in range(len(growth_regions)):


            BEGIN_LOADING_FEATURES = True

            regions = growth_regions[gr]

            BOXES = self.get_box_regions(regions)
            X_BOX = [b[0] for b in BOXES]
            Y_BOX = [b[1] for b in BOXES]


            num_percent = 0
            for box in regions:
                num_percent += float((box[3] - box[2]) * (box[1] - box[0]))
            percent = num_percent / float(IMG_WIDTH * IMG_HEIGHT)
            percent_float=percent *100

            if percent_float > self.break_training_size:
                break
            #if gr > 1 and percent_float < 1:
            #    continue
            if percent_float < self.percent_train_thresh:
                continue
            print("    * percent", percent)
            # if percent_float > 60:
            #     break

            percent = int(round(percent_float,2))



            UNet = UNetwork(
                            write_folder = self.write_path,
                            ground_truth_label_file=self.ground_truth_label_file,
                            parameter_file_number = self.parameter_file_number,
                            geomsc_fname_base = self.msc_file,
                            label_file=self.ground_truth_label_file,
                            load_feature_graph_name=False,
                            image=self.image,
                            name=self.name
                            )
            UNet.set_attributes(
                BEGIN_LOADING_FEATURES=BEGIN_LOADING_FEATURES,
                            in_channel=1, out_channel=1,
                            ground_truth_label_file=self.ground_truth_label_file,
                            skip_connect=True,
                            run_num=percent,
                            parameter_file_number = self.parameter_file_number,
                            geomsc_fname_base = self.msc_file,
                            label_file=self.ground_truth_label_file,
                            model_name=self.model_name,
                            load_feature_graph_name=False,
                            image=self.image,
                            name=self.name,
                            growth_radius=self.growth_radius,
                            # compute_features=num_samp!=0,
                            training_size=percent,
                            # region_list=samp,
                            X_BOX=X_BOX,
                            Y_BOX=Y_BOX,
                            X_BOX_all=X_BOX_all,
                            Y_BOX_all=Y_BOX_all,
                            boxes=regions,
                            all_boxes=all_boxes,
                            )
            # UNet.update_run_info(batch_multi_run=str(percent))
            # UNet.run_num = percent
            # UNet.training_size = percent

            debug = False
            INTERACTIVE = False


            trained_model = UNet.train( test = debug)


            pred_thresh = 0.5
            preds = UNet.infer(model=trained_model, training_window_file=self.window_file, load_pretrained=False,
                        pred_thresh=pred_thresh, test=debug, INTERACTIVE=INTERACTIVE)

            train_region_labeling = np.multiply(np.array(UNet.training_reg_bg), np.array(UNet.training_labels))
            total_positive_training_pixels = np.sum(train_region_labeling)
            total_positive_pixels = np.sum(UNet.training_labels)
            UNet.write_training_percentages(dir=UNet.pred_session_run_path,
                                          train_regions=UNet.training_reg_bg,
                                          total_positive_training_pixels=total_positive_training_pixels,
                                          total_positive_pixels=total_positive_pixels)

            total_number_nodes, total_training_nodes, total_number_edges, total_test_nodes, \
            total_length_training_nodes, total_length_positive_training_nodes,  total_length_positive_nodes,\
            total_length_test_nodes, total_length_nodes,total_nodes, total_foreground_nodes = self.graph_statistics(UNet.gid_gnode_dict,
                                                                                UNet.gid_edge_dict,
                                                                                UNet.node_gid_to_partition,
                                                                                UNet.node_gid_to_label)
            UNet.write_graph_statistics(total_number_nodes, total_training_nodes, total_number_edges, total_test_nodes,
                                      total_length_training_nodes, total_length_positive_training_nodes, total_length_positive_nodes,
                                      total_length_test_nodes, total_length_nodes,
                                        total_nodes, total_foreground_nodes)

            UNet.compute_metrics( pred_images=preds, # scores=scores, pred_labels=pred_labels, pred_thresh = pred_thresh,
                                 # predictions_topo=predictions_topo, labels_topo=labels_topo,
                                 INTERACTIVE= INTERACTIVE)




            del UNet



        # multi_run_metrics(model='unet', exp_folder=exp_folder, bins=7,
        #
        #                   # data_folder = self.model.params['write_folder'].split('/')[0],
        #                   runs='runs', plt_title=exp_folder.split('/')[-1],
        #                   batch_multi_run=True)



    #
    #
    #                     *  RANDOM FOREST *
    #
    #
    def run_random_forest(self, multi_run , boxes=None, dims= None, learning='pixel'):
        if multi_run:

            IMG_WIDTH = dims[0]
            IMG_HEIGHT = dims[1]

            growth_regions = self.grow_box(dims=dims, boxes=boxes)

            BEGIN_LOADING_FEATURES = self.load_features
            COMPUTE_FEATURES = self.compute_features
            BEGIN_LOADING_GETO_FEATURES = self.load_geto_features
            COMPUTE_GETO_FEATURES = self.compute_geto_features
            FEATS_INDEPENDENT = self.feats_independent
            run_feat_importance = 1

            for gr in  range(len(growth_regions)):




                regions = growth_regions[gr]

                BOXES = self.get_box_regions(regions)
                X_BOX = [b[0] for b in BOXES]
                Y_BOX = [b[1] for b in BOXES]

                num_percent = 0
                for box in regions:
                    num_percent += float((box[3] - box[2]) * (box[1] - box[0]))
                percent = num_percent / float(IMG_WIDTH * IMG_HEIGHT)
                percent_float = percent * 100
                if percent_float > self.break_training_size:
                    break
                #if gr > 1 and percent_float < 1:
                #    continue
                if percent_float < self.percent_train_thresh:
                    continue
                percent = int(round(percent_float, 2))





                RF = RandomForest(training_selection_type='box',
                                  classifier = learning,
                                  run_num=percent,
                                  parameter_file_number = self.parameter_file_number,
                                  name=self.name,
                                  image=self.image,
                                  feature_file=self.feature_file,
                                  geomsc_fname_base = self.msc_file,
                                  label_file=self.ground_truth_label_file,
                                  write_folder=self.write_path,
                                 model_name=self.model_name,
                                  load_feature_graph_name=None,
                                  write_json_graph = False,
                                  X_BOX=X_BOX,
                                  Y_BOX=Y_BOX,
                                  boxes=regions)





                RF.build_random_forest(
                    BEGIN_LOADING_FEATURES=BEGIN_LOADING_FEATURES,
                    COMPUTE_FEATURES=COMPUTE_FEATURES,
                    BEGIN_LOADING_GETO_FEATURES=BEGIN_LOADING_GETO_FEATURES,
                    COMPUTE_GETO_FEATURES=COMPUTE_GETO_FEATURES,
                    FEATS_INDEPENDENT=FEATS_INDEPENDENT,
                                       ground_truth_label_file=self.ground_truth_label_file,
                                       boxes=regions)

                if BEGIN_LOADING_FEATURES:
                    RF.params['load_features'] = True
                    RF.params['write_features'] = False
                    RF.params['write_feature_names'] = False
                    RF.params['save_filtered_images'] = False
                    RF.params['collect_features'] = False
                    RF.params['load_preprocessed'] = True
                    RF.params['load_feature_names'] = True
                elif COMPUTE_FEATURES:
                    RF.params['load_features'] = False
                    RF.params['write_features'] = True
                    RF.params['write_feature_names'] = True
                    RF.params['save_filtered_images'] = True
                    RF.params['collect_features'] = True
                    RF.params['load_preprocessed'] = False
                    RF.params['load_feature_names'] = False
                if BEGIN_LOADING_GETO_FEATURES:
                    RF.params['load_geto_attr'] = True
                    RF.params['load_feature_names'] = True
                elif COMPUTE_GETO_FEATURES:
                    RF.params['geto_as_feat'] = True
                    RF.params['load_geto_attr'] = False

                #sup_getognn.compute_features()

                if BEGIN_LOADING_FEATURES or COMPUTE_FEATURES:
                    BEGIN_LOADING_FEATURES = True
                    COMPUTE_FEATURES = False

                if BEGIN_LOADING_GETO_FEATURES or COMPUTE_GETO_FEATURES:
                    BEGIN_LOADING_GETO_FEATURES = True
                    COMPUTE_GETO_FEATURES = False



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

                if learning=='pixel':
                    predictions, labels, node_gid_to_prediction = RF.pixel_classifier(self.attributes.node_gid_to_prediction,
                                                                                train_labels=train_gid_label_dict,
                                                                                train_features=train_gid_feat_dict,
                                                                                test_labels=gid_label_dict,
                                                                                test_features=gid_features_dict,
                                                                                n_trees=n_trees,
                                                                                class_1_weight=class_1_weight,
                                                                                      growth_radius=self.growth_radius,
                                                                                class_2_weight=class_2_weight,
                                                                                weighted_distribution=RF.params[
                                                                                    'forest_class_weights'],
                                                                                depth=tree_depth)
                else:
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
                    ###




                    predictions = np.array(predictions)
                    labels = np.array(labels)

                    # compute_opt_f1(self.model.type, predictions=predictions, labels=labels,
                     #               out_folder=self.model.pred_session_run_path)

                    predictions, labels, opt_thresh = compute_getognn_metrics(getognn=RF)

                    RF.write_arc_predictions(dir=RF.pred_session_run_path)
                    RF.draw_segmentation(dirpath=RF.pred_session_run_path)

                    # update newly partitioned/infered graoh
                    #RF.equate_graph(G)

                    pred_labels_conf_matrix = np.zeros(RF.image.shape[:2], dtype=np.float32)# * min(0.25, opt_thresh/2.) # dtype=np.uint8)
                    pred_labels_msc = np.zeros(RF.image.shape[:2], dtype=np.float32)# * min(0.25, opt_thresh/2.)
                    gt_labels_msc = np.zeros(RF.image.shape[:2], dtype=np.float32) #* min(0.25, opt_thresh/2.)
                    pred_prob_im = np.zeros(RF.image.shape[:2], dtype=np.float32)
                    gt_msc = np.zeros(RF.image.shape[:2], dtype=np.float32)
                    training_reg_bg = np.zeros(RF.image.shape[:2], dtype=np.uint8)
                    for x_b, y_b in zip(RF.x_box, RF.y_box):  # RF.x_box, RF.y_box):
                        x_box = x_b
                        y_box = y_b
                        training_reg_bg[x_box[0]:x_box[1], y_box[0]:y_box[1]] = 1
                    predictions_topo_bool = []
                    labels_topo_bool = []
                    check = 30
                    for gid in RF.node_gid_to_label.keys():  # zip(mygraph.labels, mygraph.polylines):

                        gnode = RF.gid_gnode_dict[gid]
                        label = RF.node_gid_to_label[gid]
                        label = label if type(label) != list else label[1]
                        line = get_points_from_vertices([gnode])
                        # else is fg
                        cutoff = RF.opt_thresh

                        vals = []

                        for point in line:
                            ly = int(point[0])
                            lx = int(point[1])
                            pred = RF.node_gid_to_prediction[gid]
                            vals.append(pred)

                        inferred = np.array(vals, dtype="float32")
                        infval = np.average(inferred)
                        pred_mode = infval

                        RF.node_gid_to_prediction[gid] = [1. - infval, infval]

                        RF.node_gid_to_prediction[gid] = [1. - infval, infval]
                        if check >= 0:
                            check -= 1

                        t = 0
                        if infval >= RF.opt_thresh:
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
                            pred_labels_msc[lx, ly] = 1 if infval >= RF.opt_thresh else 0
                            gt_labels_msc[lx, ly] = label
                            pred_prob_im[lx, ly] = infval
                            if training_reg_bg[lx, ly] != 1:
                                RF.node_gid_to_partition[gid] = 'test'
                                # predictions_topo_bool.append(infval >= cutoff)
                                # gt_label = seg_whole[lx, ly]
                                labels_topo_bool.append(label >= cutoff)
                            else:
                                RF.node_gid_to_partition[gid] = 'train'

                    out_folder = RF.pred_session_run_path




                    ###############
                    images = [RF.image, gt_labels_msc, pred_labels_msc,
                              pred_prob_im]
                    names = ["Image", "Ground Truth Segmentation", "Predicted Foreground Segmentation",
                             "Line Foreground Probability"]
                    for image, name in zip(images, names):
                        plot(image_set=[image,training_reg_bg], name=name, type='contour', write_path=out_folder)

                    image_set = [pred_labels_msc, training_reg_bg, pred_labels_conf_matrix]
                    plot(image_set, name="TP FP TF TN Line Prediction",
                         type='confidence', write_path=out_folder)

                    plot(image_set, name="TP FP TF TN Line Prediction",
                         type='zoom', write_path=out_folder)

                    image_set = [RF.image, training_reg_bg, gt_msc]
                    plot(image_set, name="Ground Truth MSC",
                         type='confidence', write_path=out_folder)

                    plot(image_set, name="Ground Truth MSC",
                         type='zoom', write_path=out_folder)

                    for image, name in zip(images, names):
                        plot(image_set=[image, training_reg_bg], name=name, type='zoom', write_path=out_folder)

                    np.savez_compressed(os.path.join(out_folder, 'pred_matrix.npz'), pred_prob_im)
                    np.savez_compressed(os.path.join(out_folder, 'training_matrix.npz'), training_reg_bg)



                    ####################################################################################3
                out_folder = RF.attributes.pred_session_run_path

                RF.write_arc_predictions(RF.pred_session_run_path)
                RF.draw_segmentation(dirpath=RF.pred_session_run_path)

                compute_prediction_metrics('random_forest', predictions, labels, out_folder)
                #predictions, labels, opt_thresh = compute_getognn_metrics(getognn=RF)

                # training_reg_bg = np.zeros(RF.image.shape[:2], dtype=np.uint8)
                # for x_b, y_b in zip(RF.X_BOX,RF.Y_BOX):
                #     x_box = x_b
                #     y_box = y_b
                #     training_reg_bg[x_box[0]:x_box[1], y_box[0]:y_box[1]] = 1


                # training_reg_bg = np.zeros(RF.image.shape[:2], dtype=np.uint8)
                # for region in RF.box_regions:
                #     training_reg_bg[region[0]:region[1], region[2]:region[3]] = 1
                total_number_nodes, total_training_nodes, total_number_edges, total_test_nodes,\
                total_length_training_nodes, total_length_positive_training_nodes, total_length_positive_nodes,\
                total_length_test_nodes, total_length_nodes, total_nodes, total_foreground_nodes = self.graph_statistics(RF.gid_gnode_dict,
                                                                                    RF.gid_edge_dict,
                                                                                    RF.node_gid_to_partition,
                                                                                    RF.node_gid_to_label)
                fname = 'graph_statistics' if learning == 'pixel' else 'region_percents'
                total_number_edges = RF.edge_count
                total_nodes        = RF.vertex_count
                total_number_nodes = RF.vertex_count
                RF.write_graph_statistics(total_number_nodes, total_training_nodes, total_number_edges, total_test_nodes,
                                          total_length_training_nodes, total_length_positive_training_nodes,
                                          total_length_positive_nodes,
                                          total_length_test_nodes, total_length_nodes,
                                          total_nodes, total_foreground_nodes,
                                          fname=fname)

                if learning=='pixel':
                    train_region_labeling = np.multiply(np.array(RF.training_reg_bg), np.array(RF.ground_seg))
                    total_positive_training_pixels = np.sum(RF.train_region_only)
                    total_positive_pixels = np.sum(RF.ground_seg)
                    RF.write_training_percentages(dir=RF.pred_session_run_path,
                                                  train_regions=RF.training_reg_bg,
                                                  total_positive_training_pixels=total_positive_training_pixels,
                                                  total_positive_pixels=total_positive_pixels)


                #if RF.params['load_feature_names']:
                #    RF.load_feature_names()
                if RF.params['feature_importance']:
                    RF.feature_importance(feature_names=RF.feature_names,
                                          features=gid_features_dict,
                                          labels=gid_label_dict,
                                          plot=False)
                    RF.write_feature_importance()
                    RF.params['feature_importance'] = False

                BEGIN_LOADING_FEATURES = True

                del RF









    def run_unsupervised_getognn(self, multi_run):
        from getognn import unsupervised_getognn
        #
        # Train single run of getognn and obtrain trained model
        #
        unsup_getognn = unsupervised_getognn(model_name=self.model_name)
        unsup_getognn.run_num = 0
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

    def run_mlp(self, flavor='msc' , boxes=None, dims=None):
        # run_params = set_parameters(read_params_from=self.parameter_file_number,
        #                             experiment_folder=self.write_path)
        # train_set, test_set, val_set = get_train_test_val_partitions(self.attributes.node_gid_to_partition,
        #                                                              self.attributes.gid_gnode_dict,
        #                                                              test_all=True)
        print("    * model flavor: ", flavor)

        IMG_WIDTH = dims[0]
        IMG_HEIGHT = dims[1]

        growth_regions = self.grow_box(dims=dims, boxes=boxes)

        start_exp = 1

        for gr in range(len(growth_regions)):

            BEGIN_LOADING_FEATURES = True

            regions = growth_regions[gr]

            BOXES = self.get_box_regions(regions)
            X_BOX = [b[0] for b in BOXES]
            Y_BOX = [b[1] for b in BOXES]

            num_percent = 0
            for box in regions:
                num_percent += float((box[3] - box[2]) * (box[1] - box[0]))
            percent = num_percent / float(IMG_WIDTH * IMG_HEIGHT)
            percent_float = percent * 100
            if percent_float > self.break_training_size:
                break
            # if gr > 1 and percent_float < 1:
            #    continue
            if percent_float < self.percent_train_thresh:
                continue
            percent = int(round(percent_float, 2))



            partition_label_dict, partition_feat_dict = get_partition_feature_label_pairs(self.attributes.node_gid_to_partition,
                                                                                          self.attributes.node_gid_to_feature,
                                                                                          self.attributes.node_gid_to_label,
                                                                                          test_all=True)


            # run_params = self.attributes.params

            train_gid_label_dict = partition_label_dict['train']
            train_gid_feat_dict = partition_feat_dict['train']

            MLP = mlp(flavor=flavor,
                      feature_map=False,
                      # added from rf
                      training_selection_type='box',
                      classifier=flavor,
                      run_num=percent,
                      parameter_file_number=self.parameter_file_number,
                      name=self.name,
                      image=self.image,
                      feature_file=self.feature_file,
                      geomsc_fname_base=self.msc_file,
                      label_file=self.ground_truth_label_file,
                      write_folder=self.write_path,
                      model_name=self.model_name,
                      load_feature_graph_name=None,
                      write_json_graph=False,
                      X_BOX=X_BOX,
                      Y_BOX=Y_BOX,
                      boxes=regions,
                      BEGIN_LOADING_FEATURES=BEGIN_LOADING_FEATURES,
                      type=flavor,
                      ground_truth_label_file=self.ground_truth_label_file
                      )

            predictions, labels, accuracy = MLP.train()

            if flavor == 'pixel':
                MLP.compute_metrics(predictions)
                # predictions, labels, f1_score, opt_thresh = compute_opt_F1_and_threshold(MLP)
                #exp_folder = os.path.join(MLP.params['experiment_folder'])  # , 'runs')
                #multi_run_metrics(model='mlp', exp_folder=exp_folder,
                #                  batch_multi_run=True,
                #                  bins=7, runs='runs',  # str(self.training_size),
                #                  plt_title=exp_folder.split('/')[-1])
            else:


                ###############################################################################

                predictions = np.array(predictions)
                labels = np.array(labels)

                # compute_opt_f1(self.model.type, predictions=predictions, labels=labels,
                #               out_folder=self.model.pred_session_run_path)

                predictions, labels, f1_score, opt_thresh = compute_opt_F1_and_threshold(MLP)
                # compute_getognn_metrics(getognn=MLP)

                # update newly partitioned/infered graoh
                # RF.equate_graph(G)

                pred_labels_conf_matrix = np.zeros(MLP.image.shape[:2],
                                                   dtype=np.float32)  # * min(0.25, opt_thresh/2.) # dtype=np.uint8)
                pred_labels_msc = np.zeros(MLP.image.shape[:2], dtype=np.float32)  # * min(0.25, opt_thresh/2.)
                gt_labels_msc = np.zeros(MLP.image.shape[:2], dtype=np.float32)  # * min(0.25, opt_thresh/2.)
                pred_prob_im = np.zeros(MLP.image.shape[:2], dtype=np.float32)
                gt_msc = np.zeros(MLP.image.shape[:2], dtype=np.float32)
                training_reg_bg = np.zeros(MLP.image.shape[:2], dtype=np.uint8)
                for x_b, y_b in zip(MLP.x_box, MLP.y_box):  # RF.x_box, RF.y_box):
                    x_box = x_b
                    y_box = y_b
                    training_reg_bg[x_box[0]:x_box[1], y_box[0]:y_box[1]] = 1
                #predictions_topo_bool = []
                labels_topo_bool = []
                check = 30
                for gid in MLP.node_gid_to_label.keys():  # zip(mygraph.labels, mygraph.polylines):

                    gnode = MLP.gid_gnode_dict[gid]
                    label = MLP.node_gid_to_label[gid]
                    label = label if type(label) != list else label[1]
                    line = get_points_from_vertices([gnode])
                    # else is fg
                    cutoff = opt_thresh

                    vals = []

                    for point in line:
                        ly = int(point[0])
                        lx = int(point[1])
                        pred = MLP.node_gid_to_prediction[gid]
                        vals.append(pred)

                    inferred = np.array(vals, dtype="float32")
                    infval = np.average(inferred)
                    pred_mode = infval

                    MLP.node_gid_to_prediction[gid] = [1. - infval, infval]

                    MLP.node_gid_to_prediction[gid] = [1. - infval, infval]
                    if check >= 0:
                        check -= 1

                    t = 0
                    if infval >= opt_thresh:
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
                        pred_labels_msc[lx, ly] = 1 if infval >= opt_thresh else 0
                        gt_labels_msc[lx, ly] = label
                        pred_prob_im[lx, ly] = infval
                        if training_reg_bg[lx, ly] != 1:
                            MLP.node_gid_to_partition[gid] = 'test'
                            # predictions_topo_bool.append(infval >= cutoff)
                            # gt_label = seg_whole[lx, ly]
                            labels_topo_bool.append(label >= cutoff)
                        else:
                            MLP.node_gid_to_partition[gid] = 'train'

                out_folder = MLP.pred_session_run_path

                ###############
                images = [MLP.image, gt_labels_msc, pred_labels_msc,
                          pred_prob_im]
                names = ["Image", "Ground Truth Segmentation", "Predicted Foreground Segmentation",
                         "Line Foreground Probability"]
                for image, name in zip(images, names):
                    plot(image_set=[image, training_reg_bg], name=name, type='contour', write_path=out_folder)

                image_set = [pred_labels_msc, training_reg_bg, pred_labels_conf_matrix]
                plot(image_set, name="TP FP TF TN Line Prediction",
                     type='confidence', write_path=out_folder)

                plot(image_set, name="TP FP TF TN Line Prediction",
                     type='zoom', write_path=out_folder)

                image_set = [MLP.image, training_reg_bg, gt_msc]
                plot(image_set, name="Ground Truth MSC",
                     type='confidence', write_path=out_folder)

                plot(image_set, name="Ground Truth MSC",
                     type='zoom', write_path=out_folder)

                for image, name in zip(images, names):
                    plot(image_set=[image, training_reg_bg], name=name, type='zoom', write_path=out_folder)

                np.savez_compressed(os.path.join(out_folder, 'pred_matrix.npz'), pred_prob_im)
                np.savez_compressed(os.path.join(out_folder, 'training_matrix.npz'), training_reg_bg)

                ####################################################################################


            MLP.write_arc_predictions(MLP.pred_session_run_path)
            MLP.draw_segmentation(dirpath=MLP.pred_session_run_path)

            # compute_prediction_metrics('mlp', predictions, labels, out_folder)

            total_number_nodes, total_training_nodes, total_number_edges, total_test_nodes, \
            total_length_training_nodes, total_length_positive_training_nodes, total_length_positive_nodes, \
            total_length_test_nodes, total_length_nodes, total_nodes, total_foreground_nodes = self.graph_statistics(MLP.gid_gnode_dict,
                                                                                MLP.gid_edge_dict,
                                                                                MLP.node_gid_to_partition,
                                                                                MLP.node_gid_to_label)
            fname = 'graph_statistics' if flavor == 'pixel' else 'region_percents'
            MLP.write_graph_statistics(total_number_nodes, total_training_nodes, total_number_edges, total_test_nodes,
                                      total_length_training_nodes, total_length_positive_training_nodes,
                                      total_length_positive_nodes,
                                      total_length_test_nodes, total_length_nodes,
                                       total_nodes, total_foreground_nodes,
                                      fname=fname)

            #
            #  Not Implemented !
            #
            if flavor == 'pixel':
                train_region_labeling = np.multiply(np.array(MLP.training_reg_bg), np.array(MLP.full_seg))
                total_positive_training_pixels = np.sum(train_region_labeling)
                total_positive_pixels = np.sum(MLP.full_seg)
                MLP.write_training_percentages(dir=MLP.pred_session_run_path,
                                              train_regions=MLP.training_reg_bg,
                                              total_positive_training_pixels=total_positive_training_pixels,
                                              total_positive_pixels=total_positive_pixels)

            # if RF.params['load_feature_names']:
            #    RF.load_feature_names()
            if MLP.params['feature_importance']:
                gid_features_dict = partition_feat_dict['all']
                gid_label_dict = partition_label_dict['all']
                MLP.feature_importance(feature_names=MLP.feature_names,
                                      features=gid_features_dict,
                                      labels=gid_label_dict,
                                      plot=False)
                MLP.write_feature_importance()

            del MLP


#
#     Simply Informed
#
    def run_subcomplex_informed_getognn(self,
                                        boxes=None,
                                        dims=None,
                                        compute_complex = False,
                                        persistence_subgraphs = None,
                                        subgraph_labels=None):



        def compute_subgraph(sub_persistence, supgraph, subgraph_id, num_sublevel_sets, subgraph_labels=None):



            map_labels = True
            if subgraph_labels is not None:
                f_path = self.image_path
                seg_folder = os.path.dirname(os.path.abspath(f_path))
                if not os.path.exists(os.path.join(seg_folder, 'geomsc')):
                    os.makedirs(os.path.join(seg_folder, 'geomsc'))
                if not os.path.exists(os.path.join(seg_folder, 'geomsc', str(sub_persistence))):
                    os.makedirs(os.path.join(seg_folder, 'geomsc', str(sub_persistence)))
                write_path = os.path.join(seg_folder, 'geomsc', str(sub_persistence))

                msc_fname = os.path.join(write_path, self.label_file.split('raw')[0] + 'raw' )
                msc_label_fname = msc_fname + subgraph_labels[int(subgraph_id)]
                # subgraph.read_labels_from_file(file=msc_label_fname)
                map_labels = False
                subgraph = getograph.GeToGraph(geomsc_fname_base=msc_fname,
                                                                label_file=msc_label_fname)
            else:
                subgraph, msc_fname = supgraph.compute_morse_smale_complex(
                    fname_base=self.image_path,
                    persistence=[sub_persistence],
                    sigma=[2],
                    X=supgraph.X,
                    Y=supgraph.Y
                )

            pout(("MAX PERSISTENCE FOR SUBCOMPLEX:", subgraph.persistence,
                  "__ In read Geom file__",
                  "TOTAL NUMBER EDGES:",
                  subgraph.edge_count,
                  "TOTAL NUMBER NODES:",
                  subgraph.vertex_count,
                  "TOTAL NUMBER ISOLATED(-1 gid) VERTICES:",
                  subgraph.isolated_node_count))

            # sublevel_priors_graph_max = getograph.GeToGraph(geomsc_fname_base=msc_fname)

            # compute nx graph and idx info
            subgraph.G, subgraph.graph_idx_to_gid, subgraph.node_gid_to_graph_idx = supgraph.build_subgraph(
                gid_gnode_dict=subgraph.gid_gnode_dict,
                gid_edge_dict=subgraph.gid_edge_dict)

            subgraph.params['collect_features'] = COMPUTE_FEATURES
            subgraph.params['load_features'] = BEGIN_LOADING_FEATURES

            if supgraph.params['load_features']:
                supgraph.load_feature_images()
                supgraph.load_feature_image_names()

            if subgraph_labels is not None:
                subgraph.features, subgraph.node_gid_to_feature, subgraph.node_gid_to_feat_idx, subgraph.node_gid_to_graph_idx, subgraph.graph_idx_to_gid = compute_subgraph_features(
                    gid_gnode_dict=subgraph.gid_gnode_dict,
                    node_gid_to_graph_idx=subgraph.node_gid_to_graph_idx,
                    subgraph_name=subgraph_id,
                    collect_features=subgraph.params['collect_features'],
                    model=supgraph)

            supgraph.gid_gnode_dict, supgraph.gid_edge_dict, subgraph.node_gid_to_label, subgraph.sup_gid_to_sub_dict = supgraph.mark_sublevel_set(
                sublevel_nodes=subgraph.gid_gnode_dict,
                sublevel_edges=subgraph.gid_edge_dict,
                X=supgraph.X,
                Y=supgraph.Y,
                sublevel_label_dict=subgraph.node_gid_to_label,
                union_radius=None,
                union_thresh=union_thresh,
                map_labels=map_labels,
                level_id=sub_persistence)


            #
            #     ......................     EROR_R_R_R   NEED TO BUILD SUBGRAPH USING SUPGRAPH IDS ..........
            #
            subgraph.G, subgraph.graph_idx_to_gid, subgraph.node_gid_to_graph_idx = supgraph.build_subgraph(
                gid_gnode_dict=supgraph.gid_gnode_dict,  # changed this!
                #                                        #                     #   err; used to be sub
                gid_edge_dict=supgraph.gid_edge_dict)    # changed this

            subgraph.params['collect_features'] = COMPUTE_FEATURES
            subgraph.params['load_features'] = BEGIN_LOADING_FEATURES

            if supgraph.params['load_features']:
                supgraph.load_feature_images()
                supgraph.load_feature_image_names()

            #
            #   End untested addition
            #


            superlevel_training_set, sublevel_training_set, subgraph.G = supgraph.complex_sublevel_training_set(
                level_id=sub_persistence,
                subgraph=subgraph,
                map_labels=map_labels,
                num_sublevel_sets=num_sublevel_sets)


            # subgraph.homophily_ratio, homophily_train_ratio = supgraph.compute_homophily(G=subgraph.G,
            #                                                                              graph_idx_to_gid=subgraph.graph_idx_to_gid,
            #                                                                              gid_to_label_dict=subgraph.node_gid_to_label,
            #                                                                              sup_gids_of_sublevel_training_set=sublevel_training_set,
            #                                                                              subgraph=subgraph)
            subgraph.homophily_ratio, homophily_train_ratio, sublevel_homophily_ratio, sublevel_homophily_train_ratio = supgraph.compute_homophily(
                G=supgraph.G,
                graph_idx_to_gid=supgraph.graph_idx_to_gid,
                gid_to_label_dict=supgraph.node_gid_to_label,
                sup_gids_of_sublevel_set=sublevel_training_set,
                subgraph=subgraph)

            # class_stats = supgraph.compute_class_balance(G=subgraph.G,
            #                                              graph_idx_to_gid=subgraph.graph_idx_to_gid,
            #                                              gid_to_label_dict=subgraph.node_gid_to_label)
            class_stats = supgraph.compute_class_balance(G=supgraph.G,
                                                         graph_idx_to_gid=supgraph.graph_idx_to_gid,
                                                         gid_to_label_dict=supgraph.node_gid_to_label,
                                                         sup_gids_of_sublevel_set=sublevel_training_set)
            subgraph.class_statistics['positive'] = class_stats[0]
            subgraph.class_statistics['negative'] = class_stats[1]
            subgraph.class_statistics['total'] = class_stats[2]

            pout(("____ HOMOPHILY RATIO ____",
                  subgraph.homophily_ratio))
            pout(("____ CLASS BALANCE STATISTICS _____ ",
                  "positive", class_stats[0],
                  "negative", class_stats[1],
                  "total", class_stats[2]))

            supgraph.write_homophily_stats(homo_ratio = subgraph.homophily_ratio, homo_train_ratio = homophily_train_ratio, sublevel_homophily_ratio=sublevel_homophily_ratio, sublevel_homophily_train_ratio=sublevel_homophily_train_ratio )
            supgraph.write_class_stats(positive = class_stats[0],
                                       negative = class_stats[1],
                                       pos_neg_ratio = class_stats[0]/class_stats[1],
                                       positive_train=class_stats[3],
                                       negative_train=class_stats[4],
                                       #
                                       positive_sublevel=class_stats[5],
                                       negative_sublevel=class_stats[6],
                                       pos_neg_sublevel_ratio=class_stats[5] / class_stats[6],
                                       positive_sublevel_train=class_stats[8],
                                       negative_sublevel_train=class_stats[9],
                                       )

            pout(["len subgraph "+subgraph_id, len(sublevel_training_set)])

            supgraph.draw_segmentation(dirpath=os.path.join(self.input_path, 'geomsc'),
                                                  draw_sublevel_set=True, name='persistence_' + str(
                    sub_persistence))
            # subgraph_dict['subgraph_0'] = subgraph
            return subgraph, supgraph, superlevel_training_set, sublevel_training_set

        from getognn import supervised_getognn



        IMG_WIDTH = dims[0]
        IMG_HEIGHT = dims[1]

        growth_regions = self.grow_box(dims=dims, boxes=boxes)

        BEGIN_LOADING_FEATURES      = self.load_features
        COMPUTE_FEATURES            = self.compute_features
        BEGIN_LOADING_GETO_FEATURES = self.load_geto_features
        COMPUTE_GETO_FEATURES       = self.compute_geto_features
        FEATS_INDEPENDENT           = self.feats_independent
        run_feat_importance = 1

        if COMPUTE_FEATURES:
            write_feat_comp_time = True
        else:
            write_feat_comp_time = False
        # neuron2 45. foam 220
        persistence_subcomplex_max = persistence_subgraphs[0]#w/ lr 1e-3, sub/100 wd 1e-6 comb loss sep weights sep/shared mlp good results for low
        persistence_subcomplex_med = 0
        persistence_subcomplex_maxypad = 0

        sublevel_persistence_values = [persistence_subcomplex_max]#,
                                       # persistence_subcomplex_med,
                                       # persistence_subcomplex_maxypad]

        union_thresh = 0

        for gr in range(len(growth_regions)):





            regions = growth_regions[gr]

            BOXES = self.get_box_regions(regions)
            X_BOX = [b[0] for b in BOXES]
            Y_BOX = [b[1] for b in BOXES]

            num_percent = 0
            for box in regions:
                num_percent += float((box[3] - box[2]) * (box[1] - box[0]))
            percent = num_percent / float(IMG_WIDTH * IMG_HEIGHT)
            percent_float = percent *100
            print("    * percent", percent)
            if percent_float > self.break_training_size:
                break
            #if gr > 1 and percent_float < 1:
            #    continue
            if percent_float < self.percent_train_thresh:
                continue
            percent = int(round(percent, 2) )


            sup_getognn = supervised_getognn(model_name=self.model_name)
            sup_getognn.build_getognn(
                BEGIN_LOADING_FEATURES      = BEGIN_LOADING_FEATURES,
                COMPUTE_FEATURES            = COMPUTE_FEATURES,
                BEGIN_LOADING_GETO_FEATURES = BEGIN_LOADING_GETO_FEATURES,
                COMPUTE_GETO_FEATURES       = COMPUTE_GETO_FEATURES,
                FEATS_INDEPENDENT           = FEATS_INDEPENDENT,
                                       sample_idx=self.sample_idx,
                                       experiment_num=self.experiment_num,
                                       experiment_name=self.experiment_name,
                                       window_file_base=self.window_file_base,
                                       parameter_file_number=self.parameter_file_number,
                                       format = format,
                                       run_num=percent,
                                       name=self.name, image=self.image,
                                       label_file=self.label_file,
                                       msc_file=self.msc_file,
                                       ground_truth_label_file=self.ground_truth_label_file,
                                       experiment_folder = self.experiment_folder,
                                       write_path=self.write_path,
                                       feature_file=self.feature_file,
                                       window_file=None,model_name="GeToGNN",
                                       X_BOX=X_BOX,
                                       Y_BOX=Y_BOX,
                                       regions=regions)

            # persistence_subcomplex_med = sup_getognn.getognn.thirty_percent_image_intensity_range

            pout(("ten percent image intensity", persistence_subcomplex_maxypad))

            if BEGIN_LOADING_FEATURES:
                sup_getognn.getognn.params['load_features'] = True
                sup_getognn.getognn.params['write_features'] = False
                sup_getognn.getognn.params['write_feature_names'] = False
                sup_getognn.getognn.params['save_filtered_images'] = False
                sup_getognn.getognn.params['collect_features'] = False
                sup_getognn.getognn.params['load_preprocessed'] = True
                sup_getognn.getognn.params['load_feature_names'] = True
            elif COMPUTE_FEATURES:
                sup_getognn.getognn.params['load_features'] = False
                sup_getognn.getognn.params['write_features'] = True
                sup_getognn.getognn.params['write_feature_names'] = True
                sup_getognn.getognn.params['save_filtered_images'] = True
                sup_getognn.getognn.params['collect_features'] = True
                sup_getognn.getognn.params['load_preprocessed'] = False
                sup_getognn.getognn.params['load_feature_names'] = False
            if BEGIN_LOADING_GETO_FEATURES:
                sup_getognn.getognn.params['load_geto_attr'] = True
                sup_getognn.getognn.params['load_feature_names'] = True
            elif COMPUTE_GETO_FEATURES:
                sup_getognn.getognn.params['geto_as_feat'] = True
                sup_getognn.getognn.params['load_geto_attr'] = False

            subgraph_dict = {}
            sup_gid_to_sub_dict = {}
            sublevel_training_sets = []
            # featureGraph = GeToFeatureGraph()
            if compute_complex:
                pout(["COMPUTING SUBCOMPLEX"])
                #pout(['sublevel graph epoch growth ', sup_getognn.getognn.params['sublevel_init_epochs']])

                subgraph, supgraph, superlevel_training_set, sublevel_training_set = compute_subgraph(sub_persistence=persistence_subcomplex_max,
                                                      supgraph=sup_getognn.getognn,
                                                      subgraph_id='0',
                                                      subgraph_labels=subgraph_labels,
                                                                                                      num_sublevel_sets = len(sublevel_persistence_values))
                sup_getognn.getognn = supgraph
                subgraph_dict['subgraph_0'] = subgraph
                sublevel_training_sets.append(sublevel_training_set)

                '''sublevel_priors_graph_max, msc_fname = sup_getognn.getognn.compute_morse_smale_complex(fname_base=self.image_path,
                                                                               persistence=[persistence_subcomplex_max],
                                                                               sigma=[2],
                                                                               X=sup_getognn.getognn.X,
                                                                               Y=sup_getognn.getognn.Y
                                                                               )

                pout(("MAX PERSISTENCE FOR SUBCOMPLEX:" , sublevel_priors_graph_max.persistence,
                      "__ In read Geom file__",
                      "TOTAL NUMBER EDGES:",
                      sublevel_priors_graph_max.edge_count,
                      "TOTAL NUMBER NODES:",
                      sublevel_priors_graph_max.vertex_count,
                      "TOTAL NUMBER ISOLATED(-1 gid) VERTICES:",
                      sublevel_priors_graph_max.isolated_node_count))

                # sublevel_priors_graph_max = getograph.GeToGraph(geomsc_fname_base=msc_fname)

                # compute nx graph and idx info
                sublevel_priors_graph_max.G, sublevel_priors_graph_max.graph_idx_to_gid, sublevel_priors_graph_max.node_gid_to_graph_idx = sup_getognn.getognn.build_subgraph(
                    gid_gnode_dict = sublevel_priors_graph_max.gid_gnode_dict,
                    gid_edge_dict = sublevel_priors_graph_max.gid_edge_dict)



                sublevel_priors_graph_max.params['collect_features'] = COMPUTE_FEATURES
                sublevel_priors_graph_max.params['load_features'] = BEGIN_LOADING_FEATURES

                if sup_getognn.getognn.params['load_features']:
                    sup_getognn.getognn.load_feature_images()
                    sup_getognn.getognn.load_feature_image_names()



                sublevel_priors_graph_max.features, sublevel_priors_graph_max.node_gid_to_feature, sublevel_priors_graph_max.node_gid_to_feat_idx, sublevel_priors_graph_max.node_gid_to_graph_idx = compute_subgraph_features(
                    gid_gnode_dict=sublevel_priors_graph_max.gid_gnode_dict,
                    node_gid_to_graph_idx=sublevel_priors_graph_max.node_gid_to_graph_idx,
                    subgraph_name='0',
                    collect_features=sublevel_priors_graph_max.params['collect_features'],
                    model=sup_getognn.getognn)

                # if node_gid_to_graph_idx is not None:
                #     sublevel_priors_graph_max.node_gid_to_graph_idx = node_gid_to_graph_idx
                subgraph_dict['subgraph_0'] = sublevel_priors_graph_max
                '''

                if persistence_subcomplex_med:
                    sublevel_priors_graph_med = sup_getognn.getognn.compute_morse_smale_complex(fname_base=self.image_path,
                                                                                            persistence=[
                                                                                                persistence_subcomplex_med],
                                                                                            sigma=[2],
                                                                                            X=sup_getognn.getognn.X,
                                                                                            Y=sup_getognn.getognn.Y
                                                                                            )
                    subgraph_dict['subgraph_1'] = sublevel_priors_graph_med
                if persistence_subcomplex_maxypad:
                    sublevel_priors_graph_maxypad = sup_getognn.getognn.compute_morse_smale_complex(fname_base=self.image_path,
                                                                                            persistence=[
                                                                                                persistence_subcomplex_maxypad],
                                                                                            sigma=[2],
                                                                                            X=sup_getognn.getognn.X,
                                                                                            Y=sup_getognn.getognn.Y
                                                                                            )
                    subgraph_dict['subgraph_2'] = sublevel_priors_graph_maxypad
            # elif persistence_subcomplex_max:
            #     pout(["LOADING PRECOMPUTED/LABELED SUBCOMPLEX"])
            #     f_path = self.image_path
            #     seg_folder = os.path.dirname(os.path.abspath(f_path))
            #     if not os.path.exists(os.path.join(seg_folder, 'geomsc')):
            #         os.makedirs(os.path.join(seg_folder, 'geomsc'))
            #     write_path = os.path.join(seg_folder, 'geomsc')
            #     if '/' in self.image_path:
            #         image_no_path_name = self.image_path.split(os.path.dirname(self.image_path))[-1].split('/')[-1]
            #     else:
            #         image_no_path_name = self.image_path
            #     fname_raw = os.path.join(write_path, image_no_path_name)
            #     label_subcomplex = os.path.join(write_path, self.labels_subcomplex)
            #     pout(["labels files", label_subcomplex])
            #     #fname_raw += 'PERS'+str(persistence_subcomplex)+'_smoothed.raw'
            #     sublevel_priors_graph_max = ggraph.GeToGraph(geomsc_fname_base=fname_raw,
            #                                        label_file=label_subcomplex)
            #     # if using old msc (.nodes.txt file and .arcs.txt)
            #     # gid_gnode_dict, gid_edge_dict = sup_getognn.getognn.map_to_priors_graph(msc=sublevel_msc)
            # else:
            #     sublevel_priors_graph_max = sup_getognn.getognn
            #     sublevel_priors_graph_max.node_gid_to_label = sup_getognn.getognn.node_gid_to_label
            #     sublevel_priors_graph_max.gid_gnode_dict = sup_getognn.getognn.gid_gnode_dict
            #     sublevel_priors_graph_max.gid_edge_dict =  sup_getognn.getognn.gid_edge_dict

            #mark sublevel
            '''sup_getognn.getognn.gid_gnode_dict,sup_getognn.getognn.gid_edge_dict, sublevel_priors_graph_max.node_gid_to_label, sublevel_priors_graph_max.sup_gid_to_sub_dict = sup_getognn.getognn.mark_sublevel_set(
                sublevel_nodes = sublevel_priors_graph_max.gid_gnode_dict,
                sublevel_edges = sublevel_priors_graph_max.gid_edge_dict,
                X=sup_getognn.getognn.X,
                Y=sup_getognn.getognn.Y,
                sublevel_label_dict=subgraph_dict['subgraph_0'].node_gid_to_label,
                union_radius=None,
                union_thresh=union_thresh,
                sublevel_labels=False,
                level_id=persistence_subcomplex_max)



            superlevel_training_set_max, sublevel_training_set_max, sublevel_priors_graph_max.G = sup_getognn.getognn.complex_sublevel_training_set(level_id=persistence_subcomplex_max, subgraph=sublevel_priors_graph_max)

            subgraph_dict['subgraph_0'] = sublevel_priors_graph_max


            if not persistence_subcomplex_max:
                sublevel_training_set_max = superlevel_training_set_max

            pout(["len sub max", len(sublevel_training_set_max)])

            sup_getognn.getognn.draw_segmentation(dirpath=os.path.join(self.input_path, 'geomsc'),
                                                  draw_sublevel_set=True, name='persistence_' + str(
                    persistence_subcomplex_max))'''

            if persistence_subcomplex_med:
                gid_gnode_dict_med = sublevel_priors_graph_med.gid_gnode_dict
                gid_edge_dict_med = sublevel_priors_graph_med.gid_edge_dict
                sublevel_priors_graph_labels = False # sup_getognn.getognn.node_gid_to_label#sublevel_priors_graph_max.node_gid_to_label if not compute_complex else False
                # mark sublevel
                sup_getognn.getognn.gid_gnode_dict, sup_getognn.getognn.gid_edge_dict, subgraph_dict['subgraph_1'].node_gid_to_label = sup_getognn.getognn.mark_sublevel_set(
                    gid_edge_dict_med,
                    gid_gnode_dict_med,
                    X=sup_getognn.getognn.X,
                    Y=sup_getognn.getognn.Y,
                    sublevel_label_dict=subgraph_dict['subgraph_1'].node_gid_to_label,
                    union_radius=None,
                    union_thresh=union_thresh, sublevel_labels=sublevel_priors_graph_labels,
                    level_id=persistence_subcomplex_med)

                superlevel_training_set_med, sublevel_training_set_med = sup_getognn.getognn.complex_sublevel_training_set(level_id=persistence_subcomplex_med)

                sup_getognn.getognn.draw_segmentation(dirpath=os.path.join(self.input_path, 'geomsc'),
                                                      draw_sublevel_set=True, name='persistence_' + str(
                        persistence_subcomplex_med))

                pout(["len sub med", len(sublevel_training_set_med)])
            if persistence_subcomplex_maxypad:
                gid_gnode_dict_maxypad = sublevel_priors_graph_maxypad.gid_gnode_dict
                gid_edge_dict_maxypad = sublevel_priors_graph_maxypad.gid_edge_dict
                sublevel_priors_graph_labels = False #sup_getognn.getognn.node_gid_to_label#sublevel_priors_graph_max.node_gid_to_label if not compute_complex else False
                # mark sublevel
                sup_getognn.getognn.gid_gnode_dict,sup_getognn.getognn.gid_edge_dict,subgraph_dict['subgraph_2'].node_gid_to_label = sup_getognn.getognn.mark_sublevel_set(
                    gid_edge_dict_maxypad,
                    gid_gnode_dict_maxypad,
                    X=sup_getognn.getognn.X,
                    Y=sup_getognn.getognn.Y,
                    sublevel_label_dict=subgraph_dict['subgraph_2'].node_gid_to_label,
                    union_radius=None,
                    union_thresh=union_thresh, sublevel_labels=sublevel_priors_graph_labels,
                    level_id=persistence_subcomplex_maxypad)

                superlevel_training_set_maxypad, sublevel_training_set_maxypad = sup_getognn.getognn.complex_sublevel_training_set(level_id=persistence_subcomplex_maxypad)

                pout(["len sub maxy", len(sublevel_training_set_maxypad)])


                sup_getognn.getognn.draw_segmentation(dirpath=os.path.join(self.input_path,'geomsc'),
                                                      draw_sublevel_set=True, name='persistence_'+str(persistence_subcomplex_maxypad))




            sup_getognn.getognn.get_complex_informed_subgraph_split(sublevel_training_sets=sublevel_training_sets,
                                                                    collect_validation=False,
                                                                    validation_hops=1,
                                                                    validation_samples=1)

            if sup_getognn.getognn.params['write_json_graph']:
                sup_getognn.getognn.write_json_graph_data(folder_path=sup_getognn.getognn.pred_session_run_path,
                                                   name="GeToGNN" + '_' + sup_getognn.getognn.params['name'])

            # random walks
            # if not sup_getognn.getognn.params['load_preprocessed_walks']:
            #     walk_embedding_file = os.path.join(sup_getognn.getognn.LocalSetup.project_base_path, 'datasets',
            #                                        sup_getognn.getognn.params['write_folder'], 'walk_embeddings',
            #                                        'gnn')
            #     sup_getognn.getognn.params['load_walks'] = walk_embedding_file
            #     sup_getognn.getognn.run_random_walks(walk_embedding_file=walk_embedding_file)
            # else:
            #     walk_embedding_file = os.path.join(sup_getognn.getognn.LocalSetup.project_base_path, 'datasets',
            #                                        sup_getognn.getognn.params['write_folder'], 'walk_embeddings',
            #                                        'gnn')
            #     sup_getognn.getognn.params['load_walks'] = walk_embedding_file

            sup_getognn.getognn.params['load_walks'] = False
            sup_getognn.compute_features()


            if BEGIN_LOADING_FEATURES or COMPUTE_FEATURES:
                BEGIN_LOADING_FEATURES      = True
                COMPUTE_FEATURES            = False

            if BEGIN_LOADING_GETO_FEATURES or COMPUTE_GETO_FEATURES:
                BEGIN_LOADING_GETO_FEATURES = True
                COMPUTE_GETO_FEATURES       = False



            subgraph_dict=None
            getognn = sup_getognn.train(run_num=str(percent), sublevel_sets=True, subgraph_dict=subgraph_dict)

            #getognn.update_run_info(batch_multi_run=str(percent))
            getognn.run_num = percent

            out_folder = os.path.join(getognn.pred_session_run_path)
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)

            if write_feat_comp_time:
                getognn.write_feature_comp_time()
                write_feat_comp_time = False
            #

            #
            # Feature Importance
            #
            if BEGIN_LOADING_GETO_FEATURES and BEGIN_LOADING_FEATURES:
                node_gid_to_feature, node_gid_to_feat_idx, features = get_merged_features(getognn)
            else:
                node_gid_to_feature = getognn.node_gid_to_standard_feature
            partition_label_dict, partition_feat_dict = get_partition_feature_label_pairs(
                getognn.node_gid_to_partition,
                node_gid_to_feature,
                getognn.node_gid_to_label,
                test_all=True)


            gid_features_dict = partition_feat_dict['all']
            gid_label_dict = partition_label_dict['all']
            #getognn.load_feature_names()
            if run_feat_importance:     #getognn.params['feature_importance'] and
                names = []

                if BEGIN_LOADING_FEATURES or COMPUTE_FEATURES:
                    names += getognn.load_feature_names()
                if BEGIN_LOADING_GETO_FEATURES or COMPUTE_GETO_FEATURES:
                    names += getognn.load_geto_feature_names()

                pout(['len featname', len(names), 'shape feat', np.array(list(node_gid_to_feature.values())).shape])

                getognn.feature_importance(feature_names=names,#getognn.feature_names,
                                      features=node_gid_to_feature,#gid_features_dict,
                                      labels=gid_label_dict,
                                           plot=False)
                getognn.write_feature_importance()
                getognn.params['feature_importance'] = False
                run_feat_importance = 0

            training_reg_bg = np.zeros(getognn.image.shape[:2], dtype=np.uint8)
            for x_b, y_b in zip(getognn.x_box, getognn.y_box):
                x_box = x_b
                y_box = y_b
                training_reg_bg[x_box[0]:x_box[1], y_box[0]:y_box[1]] = 1
            # getognn.write_training_percentages(dir=getognn.pred_session_run_path, train_regions=training_reg_bg)

            #
            # Perform remainder of runs and don't need to read feats again
            #

            # !!! called twice
            # getognn.supervised_train()

            getognn.record_time(round(getognn.train_time, 4),
                                   dir=getognn.pred_session_run_path,
                                   type='train')
            getognn.record_time(round(getognn.pred_time, 4),
                                   dir=getognn.pred_session_run_path,
                                   type='pred')
            G = getognn.get_graph()
            getognn.equate_graph(G)
            # For computing the line graph for visualisation
            # getognn.draw_priors_graph(G)





            predictions, labels, opt_thresh = compute_getognn_metrics(getognn=getognn)


            getognn.write_arc_predictions(dir=getognn.pred_session_run_path)
            getognn.draw_segmentation(dirpath=getognn.pred_session_run_path)
            getognn.write_gnode_partitions(dir=getognn.pred_session_run_path)  # self.getognn.session_name)
            getognn.write_selection_bounds(dir=getognn.pred_session_run_path)  # self.getognn.session_name)

            # update newly partitioned/infered graoh
            G = getognn.get_graph()
            getognn.equate_graph(G)

            total_number_nodes, total_training_nodes, total_number_edges, total_test_nodes, total_length_training_nodes,\
            total_length_positive_training_nodes,  total_length_positive_nodes,\
            total_length_test_nodes, total_length_nodes, total_nodes, total_foreground_nodes = self.graph_statistics( getognn.gid_gnode_dict,
                                                                                 getognn.gid_edge_dict,
                                                                                 getognn.node_gid_to_partition,
                                                                                 getognn.node_gid_to_label)
            getognn.write_graph_statistics(total_number_nodes,
                                           total_training_nodes,
                                           total_number_edges,
                                           total_test_nodes,
                                           total_length_training_nodes,
                                           total_length_positive_training_nodes,
                                           total_length_positive_nodes,
                                           total_length_test_nodes,
                                           total_length_nodes,
                                           total_nodes, total_foreground_nodes,
                                           fname='region_percents')

            # getognn.write_training_graph_percentages(dir=getognn.pred_session_run_path,
            #                                          graph_orders=(total_number_nodes,
            #                                                        total_training_nodes))

            pred_labels_conf_matrix = np.zeros(getognn.image.shape[:2], dtype=np.float32) #* min(0.25,opt_thresh/2.) # dtype=np.uint8)
            pred_labels_msc = np.zeros(getognn.image.shape[:2], dtype=np.float32) #* min(0.25, opt_thresh/2.)
            gt_labels_msc = np.zeros(getognn.image.shape[:2], dtype=np.float32) #* min(.25,opt_thresh/2.)
            pred_prob_im = np.zeros(getognn.image.shape[:2], dtype=np.float32)
            gt_msc = np.zeros(getognn.image.shape[:2], dtype = np.float32)
            predictions_topo_bool = []
            labels_topo_bool = []
            check = 30
            for gid in getognn.node_gid_to_label.keys():  # zip(mygraph.labels, mygraph.polylines):

                gnode = getognn.gid_gnode_dict[gid]
                label = getognn.node_gid_to_label[gid]
                label = label if type(label) != list else label[1]
                line = get_points_from_vertices([gnode])
                # else is fg
                cutoff = opt_thresh

                vals = []

                for point in line:
                    ly = int(point[0])
                    lx = int(point[1])
                    pred = getognn.node_gid_to_prediction[gid]
                    vals.append(pred)


                inferred = np.array(vals, dtype="float32")
                infval = np.average(inferred)
                pred_mode = infval

                getognn.node_gid_to_prediction[gid] = [1. - infval, infval]



                getognn.node_gid_to_prediction[gid] = [1. - infval, infval]
                if check >= 0:

                    check -= 1

                t = 0
                if infval >= opt_thresh:
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
                    pred_labels_msc[lx, ly] = 1 if infval >= opt_thresh else 0
                    gt_labels_msc[lx,ly] = label
                    pred_prob_im[lx, ly] = infval
                    if training_reg_bg[lx, ly] != 1:
                        getognn.node_gid_to_partition[gid] = 'test'
                        #predictions_topo_bool.append(infval >= cutoff)
                        #gt_label = seg_whole[lx, ly]
                        labels_topo_bool.append(label >= cutoff)


            out_folder = getognn.pred_session_run_path





            images = [getognn.image, gt_labels_msc, pred_labels_msc,
                      pred_prob_im]
            names = ["Image", "Ground Truth Segmentation", "Predicted Foreground Segmentation",
                     "Line Foreground Probability"]
            for image, name in zip(images, names):
                plot(image_set=[image, training_reg_bg], name=name, type='contour', write_path=out_folder)

            image_set = [pred_labels_msc, training_reg_bg, pred_labels_conf_matrix]
            plot(image_set, name="TP FP TF TN Line Prediction",
                 type='confidence', write_path=out_folder)

            plot(image_set, name="TP FP TF TN Line Prediction",
                 type='zoom', write_path=out_folder)


            image_set = [getognn.image, training_reg_bg, gt_msc]
            plot(image_set, name="Ground Truth MSC",
                 type='confidence', write_path=out_folder)

            plot(image_set, name="Ground Truth MSC",
                 type='zoom', write_path=out_folder)

            for image, name in zip(images, names):
                plot(image_set=[image, training_reg_bg], name=name, type='zoom', write_path=out_folder)




            # batch_folder = os.path.join(self.params['experiment_folder'],'batch_metrics', 'prediction')
            # if not os.path.exists(batch_folder):
            #    os.makedirs(batch_folder)
            #

            del getognn
            del sup_getognn




#
#             LOGGING
#
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

