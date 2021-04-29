import numpy as np
import os

from getognn import *
from localsetup import LocalSetup
LocalSetup = LocalSetup(env='slurm')

class Run_Manager:
    def __init__(self, getognn, training_window_file, features_file, sample_idx, model_name, format):
        f = open(training_window_file, 'r')
        self.box_dict = {}
        self.param_lines = f.readlines()
        self.getognn = getognn
        self.getognn.logger.record_filename(window_list_file=os.path.join(LocalSetup.project_base_path,
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

            self.getognn.run_num += 1

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

            self.getognn.update_run_info()

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
            if not self.getognn.params['load_features']:
                self.getognn.compile_features()
            else:
                self.getognn.load_gnode_features(self.features_file)

            #if self.getognn.params['write_features']:
            #    self.getognn.write_gnode_features(self.getognn.session_name)

            # train/test/val collection
            training_set , test_set = self.getognn.box_select_geomsc_training(x_range=X_BOX,
                                                    y_range=Y_BOX)
            # skip box if no training arcs present in region
            if len(training_set) <= 1:
                removed_box_file = open(os.path.join(self.getognn.LocalSetup.project_base_path,
                                                     'datasets', self.getognn.params['write_folder'],
                                                     'removed_windows.txt'),'w+')
                removed_box_file.write('x_box '+str(X_BOX[0][0])+','+str(X_BOX[0][1])+'\n')
                removed_box_file.write('x_box ' + str(Y_BOX[0][0]) + ',' + str(Y_BOX[0][1]) + '\n')
                continue

            self.getognn.get_train_test_val_sugraph_split(collect_validation=True, validation_hops=1,
                                                     validation_samples=1)
            #if self.getognn.params['write_partitions']:
            self.getognn.write_gnode_partitions(self.getognn.session_name)
            self.getognn.write_selection_bounds(self.getognn.session_name)

            if self.getognn.params['write_json_graph']:
                self.getognn.write_json_graph_data(folder_path=self.getognn.pred_session_run_path,
                                              name=self.model_name + '_' + self.getognn.params['name'])

            # random walks
            if not self.getognn.params['load_preprocessed_walks']:
                walk_embedding_file = os.path.join(self.getognn.LocalSetup.project_base_path, 'datasets',
                                                   self.getognn.params['write_folder'], 'walk_embeddings',
                                                   'run-' + str(self.getognn.run_num) + '_walks')
                self.getognn.params['load_walks'] = walk_embedding_file
                self.getognn.run_random_walks(walk_embedding_file=walk_embedding_file)

            #training
            self.getognn.supervised_train()
            G = self.getognn.get_graph()
            self.getognn.equate_graph(G)

            self.getognn.write_arc_predictions(self.getognn.session_name)
            self.getognn.draw_segmentation(dirpath=os.path.join(self.getognn.pred_session_run_path)) # , invert=True)

        self.getognn.logger.write_experiment_info()

