import numpy as np
import os

from ml.MLP import mlp
from ml.utils import get_partition_feature_label_pairs
from metrics.model_metrics import compute_prediction_metrics
from metrics.model_metrics import compute_getognn_metrics
from compute_multirun_metrics import multi_run_metrics
from data_ops.set_params import set_parameters

from localsetup import LocalSetup
LocalSetup = LocalSetup()

__all__ = ['Run_Manager']

class Run_Manager:
    def __init__(self, model, training_window_file, features_file,
                 sample_idx, model_name, format, learning_type='supervised',
                 parameter_file_number=1, expanding_boxes=True,
                 collect_validation=False):

        self.collect_validation = collect_validation

        self.expanding_boxes = expanding_boxes

        f = open(training_window_file, 'r')
        self.box_dict = {}
        param_lines = f.readlines()
        f.close()

        el = len(param_lines) - 1
        sets = el // 10
        sets = sets - 1 if sets % 2 != 0 else sets
        self.param_lines = [param_lines[0:2] , param_lines[sets:sets+4] ,\
                      param_lines[2*sets:(2*sets)+6] , param_lines[3*sets:(3*sets)+8],
                      param_lines[4 * sets:(4 * sets) + 16], param_lines[6 * sets:(6 * sets) + 32],
                      param_lines[5 * sets:-1]   ,param_lines[3 * sets:-1]]
        self.model = model
        self.model.logger.record_filename(window_list_file=os.path.join(LocalSetup.project_base_path,
                                                                        training_window_file))
        # data / run attributes
        self.sample_idx = sample_idx
        self.model_name = model_name
        self.features_file = features_file
        self.format = format

        # path information
        self.input_folder = None
        self.experiment_folder = None

        self.learning_type=learning_type

        self.parameter_file_number = parameter_file_number


    def __group_pairs(self, lst):
        for i in range(0, len(lst), 2):
            yield tuple(lst[i : i + 2])

    def perform_runs(self):
        # reads from file where x bounds line followed by y bounds line e.g
        # x_box 1,2
        # y_box 3,4
        # multi x bounds and y bounds added to used x boxes and y boxes e.g.
        # reading all lines
        # x_box 1,2
        # y_box 3,4
        # x_box 5,6
        # y_box 7,8
        # produces { 'x_box' : [[1,2],[5,6]] , 'y_box' : [[3,4],[7,8]] }

        for box_set in self.param_lines:
            boxes = [
                i for i in self.__group_pairs(box_set)
            ]
            self.model.update_run_info(batch_multi_run=str(len(box_set)))
            self.model.run_num=0

            for current_box_idx in range(len(boxes)):

                self.model.update_run_info()
                #self.model.pred_session_run_path = os.path.join(LocalSetup.project_base_path,str(self.model.run))
                self.counter_file = os.path.join(LocalSetup.project_base_path, 'run_count.txt')
                f = open(self.counter_file, 'r')
                c = f.readlines()
                self.run_num = int(c[0]) + 1
                f.close()
                f = open(self.counter_file, 'w')
                f.write(str(self.run_num))
                f.close()
                print("&&&& run num ", self.run_num)

                self.active_file = os.path.join(LocalSetup.project_base_path, 'continue_active.txt')
                f = open(self.active_file, 'w')
                f.write('0')
                f.close()

                current_box_dict = {}
                current_box_idx = None if current_box_idx==(len(boxes)-1)\
                    else current_box_idx+1
                current_box = boxes[current_box_idx] if self.expanding_boxes is \
                                                        False else boxes[0:current_box_idx]
                if self.expanding_boxes:
                    current_box = [y for x in current_box for y in x]
                print('    * : box set', current_box)

                for bounds in current_box:

                    name_value = bounds.split(' ')
                    print(name_value)

                    # window selection(s)
                    # for single training box window
                    if name_value[0] not in current_box_dict.keys():
                        current_box_dict[name_value[0]] = list(map(int, name_value[1].split(',')))
                    # for multiple boxes
                    else:
                        current_box_dict[name_value[0]].extend(list(map(int, name_value[1].split(','))))

                X_BOX = [
                        i for i in self.__group_pairs([i for i in current_box_dict['x_box']])
                    ]
                Y_BOX = [
                        i for i in self.__group_pairs([i for i in current_box_dict['y_box']])
                    ]
                # features
                if self.model.params['load_features'] and self.model.features is None:
                    self.model.load_gnode_features()
                    if self.model.getoelms is None and (self.model.params['geto_as_feat'] or self.model.params['load_geto_attr']):
                        self.model.load_geto_features()
                else:
                    if self.model.features is None:
                        self.model.compile_features(include_geto=self.model.params['geto_as_feat'])
                    if self.model.getoelms is None and (not self.model.params['geto_as_feat'] or not self.model.params['load_geto_attr']):
                        self.model.build_geto_adj_list(influence_type=self.model.params['geto_influence_type'])

                if self.model.params['write_features']:
                    self.model.write_gnode_features(self.model.session_name)
                    self.model.write_geto_features(self.model.session_name)

                if self.model.params['write_feature_names']:
                    self.model.write_feature_names()
                    self.model.write_geto_feature_names()




                #if self.getognn.params['write_features']:
                #    self.getognn.write_gnode_features(self.getognn.session_name)

                # train/test/val collection
                training_class_sets , test_set = self.model.box_select_geomsc_training(x_range=X_BOX,
                                                                                y_range=Y_BOX)

                #
                # ensure selected training is reasonable
                #
                flag_class_empty = False
                cardinality_training_sets = 0
                for i, t_class in enumerate(training_class_sets):
                    flag_class_empty = len(t_class) == 0 if not flag_class_empty else flag_class_empty
                    cardinality_training_sets += len(t_class)
                    print("LENGTH .. Training Set",i,'length:', len(t_class))
                print(".. length test: ", len(test_set))
                # skip box if no training arcs present in region
                if cardinality_training_sets <= 1 or flag_class_empty:
                    removed_file = os.path.join(self.model.LocalSetup.project_base_path,
                                                         'datasets', self.model.params['write_folder'],
                                                         'removed_windows.txt')
                    if not os.path.exists(removed_file):
                        open(removed_file, 'w').close()
                    removed_box_file = open(os.path.join(self.model.LocalSetup.project_base_path,
                                                         'datasets', self.model.params['write_folder'],
                                                         'removed_windows.txt'),'a+')
                    removed_box_file.write(str(self.model.run_num)+' x_box '+str(X_BOX[0][0])+','+str(X_BOX[0][1])+'\n')
                    removed_box_file.write(str(self.model.run_num)+' y_box ' + str(Y_BOX[0][0]) + ',' + str(Y_BOX[0][1]) + '\n')
                    continue

                # must do cvt before assigning class
                # to ensure validation set doesn't remove all training nodes
                if self.collect_validation:
                    print("    * : PERFORMING cvt sampling")
                    validation_hops = self.model.params['validation_hops']
                    validation_samples = self.model.params['validation_samples']
                    validation_set, validation_set_ids,\
                    _, _ = self.model.cvt_sample_validation_set(hops=validation_hops,
                                                            samples=validation_samples)
                    # all_validation = self.model.validation_set_ids["positive"].union(self.model.validation_set_ids["negative"])
                    # all_selected = [self.model.selected_positive_arc_ids, self.model.selected_negative_arc_ids]
                    # #self.model.selected_positive_arc_ids.union(self.model.selected_negative_arc_ids)
                    # if not self.model.check_valid_partitions(all_selected, all_validation):
                    #     removed_box_file = open(os.path.join(self.model.LocalSetup.project_base_path,
                    #                                       'datasets', self.model.params['write_folder'],
                    #                                          'removed_windows.txt'), 'a+')
                    #     removed_box_file.write(str(self.model.run_num)+' x_box ' + str(X_BOX[0][0]) + ',' + str(X_BOX[0][1]) + '\n')
                    #     removed_box_file.write(str(self.model.run_num)+' y_box ' + str(Y_BOX[0][0]) + ',' + str(Y_BOX[0][1]) + '\n')
                    #     continue
                    #
                    # for gid in all_validation:
                    #     self.model.node_gid_to_partition[gid] = 'val'

                self.model.get_train_test_val_sugraph_split(collect_validation=False, validation_hops=1,
                                                            validation_samples=1)

                all_validation = self.model.validation_set_ids["positive"].union(self.model.validation_set_ids["negative"])
                all_selected = [self.model.selected_positive_arc_ids, self.model.selected_negative_arc_ids]
                # self.model.selected_positive_arc_ids.union(self.model.selected_negative_arc_ids)
                if not self.model.check_valid_partitions(all_selected, all_validation):
                    removed_box_file = open(os.path.join(self.model.LocalSetup.project_base_path,
                                                         'datasets', self.model.params['write_folder'],
                                                         'removed_windows.txt'), 'a+')
                    removed_box_file.write(
                        str(self.model.run_num) + ' x_box ' + str(X_BOX[0][0]) + ',' + str(X_BOX[0][1]) + '\n')
                    removed_box_file.write(
                        str(self.model.run_num) + ' y_box ' + str(Y_BOX[0][0]) + ',' + str(Y_BOX[0][1]) + '\n')
                    continue

                for gid in all_validation:
                    self.model.node_gid_to_partition[gid] = 'val'

                #if self.getognn.params['write_partitions']:
                self.model.write_gnode_partitions(self.model.pred_session_run_path)
                self.model.write_selection_bounds(self.model.pred_session_run_path)

                if self.model.type == "getognn" and self.model.params['write_json_graph']:
                    self.model.write_json_graph_data(folder_path=self.model.pred_session_run_path,
                                                     name=self.model_name + '_' + self.model.params['name'])

                # random walks
                if self.model.type == "getognn" and not self.model.params['load_preprocessed_walks'] and self.model.params['random_context']:
                    walk_embedding_file = os.path.join(self.model.LocalSetup.project_base_path, 'datasets',
                                                       self.model.params['write_folder'], 'walk_embeddings',
                                                       'run-' + str(self.model.run_num) + '_walks')
                    self.model.params['load_walks'] = walk_embedding_file
                    self.model.run_random_walks(walk_embedding_file=walk_embedding_file)

                #training
                if self.model.type == "getognn":
                    self.model.supervised_train()
                    G = self.model.get_graph()
                    self.model.equate_graph(G)

                    self.model.write_arc_predictions(dir= self.model.pred_session_run_path)
                    self.model.draw_segmentation(dirpath=self.model.pred_session_run_path)

                    # get inference metrics
                    compute_getognn_metrics(getognn=self.model)

                    # update newly partitioned/infered graoh
                    G = self.model.get_graph()
                    self.model.equate_graph(G)

                if self.model.type == "random_forest":

                    partition_label_dict, partition_feat_dict = get_partition_feature_label_pairs(
                        self.model.node_gid_to_partition,
                        self.model.node_gid_to_feature,
                        self.model.node_gid_to_label,
                        test_all=True)

                    run_params = self.model.params
                    run_params = set_parameters(read_params_from=self.parameter_file_number,
                                                experiment_folder=self.model.params['write_folder'])
                    gid_features_dict = partition_feat_dict['all']
                    gid_label_dict = partition_label_dict['all']
                    train_gid_label_dict = partition_label_dict['train']
                    train_gid_feat_dict = partition_feat_dict['train']

                    n_trees = run_params['number_forests']
                    tree_depth = run_params['forest_depth']

                    if run_params['forest_class_weights']:
                        class_1_weight = run_params['class_1_weight']
                        class_2_weight = run_params['class_2_weight']
                    else:
                        class_1_weight = 1.0
                        class_2_weight = 1.0

                    predictions, labels, node_gid_to_prediction = self.model.classifier(self.model.node_gid_to_prediction,
                                                                                train_labels=train_gid_label_dict,
                                                                                train_features=train_gid_feat_dict,
                                                                                test_labels=gid_label_dict,
                                                                                test_features=gid_features_dict,
                                                                                n_trees=n_trees,
                                                                        class_1_weight=class_1_weight,
                                                                        class_2_weight=class_2_weight,
                                                                        weighted_distribution=run_params['forest_class_weights'],
                                                                                depth=tree_depth)
                    out_folder = self.model.pred_session_run_path

                    self.model.write_arc_predictions(self.model.pred_session_run_path)
                    self.model.draw_segmentation(dirpath=self.model.pred_session_run_path)

                    # need to read features by gid or idx
                    #self.model.feature_importance(gid_features_dict, feature_names, gid_label_dict, n_informative = 3, plot=False)

                    compute_prediction_metrics('random_forest', predictions, labels, out_folder)
                if self.model.type == 'mlp':
                    partition_label_dict, partition_feat_dict = get_partition_feature_label_pairs(
                        self.model.attributes.node_gid_to_partition,
                        self.model.attributes.node_gid_to_feature,
                        self.model.attributes.node_gid_to_label,
                        test_all=True)

                    run_params = self.model.attributes.params
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

                    out_folder = self.model.attributes.pred_session_run_path

                    MLP.write_arc_predictions(MLP.session_name)
                    MLP.draw_segmentation(dirpath=MLP.pred_session_run_path)

                    compute_prediction_metrics('mlp', preds, labels, out_folder)

                self.model.write_arc_predictions(self.model.pred_session_run_path)
                self.model.draw_segmentation(dirpath=os.path.join(self.model.pred_session_run_path)) # , invert=True)

        exp_folder = os.path.join(self.model.params['experiment_folder'] )
        batch_metric_folder = os.path.join(exp_folder, 'batch_metrics')
        if not os.path.exists(batch_metric_folder):
            os.makedirs(batch_metric_folder)

        multi_run_metrics(model=self.model.type, exp_folder=exp_folder,bins=7,
                          runs='runs', plt_title=exp_folder.split('/')[-1])
        self.model.logger.write_experiment_info()

