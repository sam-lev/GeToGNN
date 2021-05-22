import numpy as np
import os

from ml.utils import get_partition_feature_label_pairs
from metrics.model_metrics import compute_prediction_metrics
from metrics.model_metrics import compute_getognn_metrics
from localsetup import LocalSetup
LocalSetup = LocalSetup()

__all__ = ['Run_Manager']

class Run_Manager:
    def __init__(self, model, training_window_file, features_file, sample_idx, model_name, format):
        f = open(training_window_file, 'r')
        self.box_dict = {}
        self.param_lines = f.readlines()
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
        boxes = [
            i for i in self.__group_pairs(self.param_lines)
            ]
        for current_box_idx in range(len(boxes)):

            self.model.run_num += 1

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

            self.model.update_run_info()

            current_box_dict = {}
            current_box = boxes[current_box_idx]

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
            if self.model.params['collect_features']:
                self.model.compile_features()
            elif self.model.params['load_features']:
                self.model.load_gnode_features(self.features_file)

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
                removed_box_file = open(os.path.join(self.model.LocalSetup.project_base_path,
                                                     'datasets', self.model.params['write_folder'],
                                                     'removed_windows.txt'),'w+')
                removed_box_file.write('x_box '+str(X_BOX[0][0])+','+str(X_BOX[0][1])+'\n')
                removed_box_file.write('y_box ' + str(Y_BOX[0][0]) + ',' + str(Y_BOX[0][1]) + '\n')
                continue

            # must do cvt before assigning class
            # to ensure validation set doesn't remove all training nodes
            validation_hops = 1
            validation_samples = 1
            validation_set, validation_set_ids,\
            _, _ = self.model.cvt_sample_validation_set(hops=validation_hops,
                                                        samples=validation_samples)
            all_validation = self.model.validation_set_ids["positive"].union(self.model.validation_set_ids["negative"])
            all_selected = self.model.selected_positive_arc_ids.union(self.model.selected_negative_arc_ids)
            if not self.model.check_valid_partitions(all_selected, all_validation):
                continue 

            for gid in all_validation:
                self.model.node_gid_to_partition[gid] = 'val'
            self.model.get_train_test_val_sugraph_split(collect_validation=True, validation_hops=1,
                                                        validation_samples=1)
            #if self.getognn.params['write_partitions']:
            self.model.write_gnode_partitions(self.model.session_name)
            self.model.write_selection_bounds(self.model.session_name)

            if self.model.type == "Getognn" and self.model.params['write_json_graph']:
                self.model.write_json_graph_data(folder_path=self.model.pred_session_run_path,
                                                 name=self.model_name + '_' + self.model.params['name'])

            # random walks
            if self.model.type == "Getognn" and not self.model.params['load_preprocessed_walks']:
                walk_embedding_file = os.path.join(self.model.LocalSetup.project_base_path, 'datasets',
                                                   self.model.params['write_folder'], 'walk_embeddings',
                                                   'run-' + str(self.model.run_num) + '_walks')
                self.model.params['load_walks'] = walk_embedding_file
                self.model.run_random_walks(walk_embedding_file=walk_embedding_file)

            #training
            if self.model.type == "Getognn":
                self.model.supervised_train()
                G = self.model.get_graph()
                self.model.equate_graph(G)

                self.model.write_arc_predictions(self.model.session_name)
                self.model.draw_segmentation(dirpath=self.model.pred_session_run_path)

                # get inference metrics
                compute_getognn_metrics(getognn=self.model)

                # update newly partitioned/infered graoh
                G = self.model.get_graph()
                self.model.equate_graph(G)

            if self.model.type == "Random Forest":

                partition_label_dict, partition_feat_dict = get_partition_feature_label_pairs(
                    self.model.node_gid_to_partition,
                    self.model.node_gid_to_feature,
                    self.model.node_gid_to_label,
                    test_all=True)

                run_params = self.model.params
                gid_features_dict = partition_feat_dict['all']
                gid_label_dict = partition_label_dict['all']
                train_gid_label_dict = partition_label_dict['train']
                train_gid_feat_dict = partition_feat_dict['train']

                n_trees = run_params['number_forests']
                tree_depth = run_params['forest_depth']

                predictions, labels, node_gid_to_prediction = self.model.classifier(self.model.node_gid_to_prediction,
                                                                            train_labels=train_gid_label_dict,
                                                                            train_features=train_gid_feat_dict,
                                                                            test_labels=gid_label_dict,
                                                                            test_features=gid_features_dict,
                                                                            n_trees=n_trees,
                                                                            depth=tree_depth)
                out_folder = self.model.pred_session_run_path

                self.model.write_arc_predictions(self.model.session_name)
                self.model.draw_segmentation(dirpath=self.model.pred_session_run_path)

                # need to read features by gid or idx
                #self.model.feature_importance(gid_features_dict, feature_names, gid_label_dict, n_informative = 3, plot=False)

                compute_prediction_metrics('random_forest', predictions, labels, out_folder)

            self.model.write_arc_predictions(self.model.session_name)
            self.model.draw_segmentation(dirpath=os.path.join(self.model.pred_session_run_path)) # , invert=True)

        self.model.logger.write_experiment_info()

