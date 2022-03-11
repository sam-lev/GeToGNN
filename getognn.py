import os
import time
import numpy as np
from copy import deepcopy
import json
from networkx.readwrite import json_graph
import datetime

from ml import supervised_gnn
from ml import unsupervised_gnn
from ml.utils import format_data
from ml.LinearRegression import LinearRegression
from ml.utils import random_walk_embedding
from mlgraph import MLGraph
from getograph import  Attributes
from ui.arcselector import ArcSelector
from proc_manager import experiment_manager

from localsetup import LocalSetup

class GeToGNN(MLGraph):
    def __init__(self,training_selection_type='box', parameter_file_number = None,
                 geomsc_fname_base = None, label_file=None,run_num=1, X_BOX=None,Y_BOX=None,
                 model_name=None, load_feature_graph_name=False,image=None, **kwargs):

        self.type = "getognn"

        self.params = {}
        if parameter_file_number is None:
            self.params = kwargs
        else:
            for param in kwargs:
                self.params[param] = kwargs[param]


        self.G = None
        self.G_dict = {}


        self.LocalSetup = LocalSetup()
        self.run_num=run_num


        # self.X_BOX = X_BOX
        # self.Y_BOX = Y_BOX

        super(GeToGNN, self).__init__(parameter_file_number=parameter_file_number,
                                      run_num=run_num,
                                      name=self.params['name'],geomsc_fname_base=geomsc_fname_base,
                                      label_file=label_file,image=image,write_folder=self.params['write_folder'],
                                      model_name=model_name,load_feature_graph_name=load_feature_graph_name)

        #                                     params=self.params)

        for param in kwargs:
            self.params[param] = kwargs[param]
        #for param in kwargs['params']:
        #    self.params[param] = kwargs['params'][param]
        if 'params' in kwargs.keys():
            param_add_ons = kwargs['params']
            for k, v in param_add_ons.items():
                self.params[k] = v
        self.run_num=run_num

        self.logger = experiment_manager.experiment_logger(experiment_folder=self.experiment_folder,
                                        input_folder=self.input_folder)
        self.param_file = os.path.join(self.LocalSetup.project_base_path,
                                       'parameter_list_'+str(parameter_file_number)+'.txt')
        self.topo_image_name = label_file.split('.labels')[0]
        self.logger.record_filename(label_file=label_file,
                                    parameter_list_file=self.param_file,
                                    image_name=image,
                                    topo_image_name=self.topo_image_name)

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

        self.train_time = 0
        self.pred_time = 0

        #self.update_run_info(batch_multi_run=run_num)






    def unsupervised_train(self, **kwargs):
        for arg in kwargs:
            self.params[arg] = kwargs[arg]

        if self.params['getofeaturegraph_file']:
            print("loading")
            cwd = './'
            walks = self.params['load_walks']
            train_path = os.path.join(cwd, 'data', 'json_graphs', self.params['getofeaturegraph_name'])

            if not self.params['embedding_file_out']:
                embedding_name = datetime.datetime.now()

            self.gnn.train(train_prefix=self.params['getofeaturegraph_name'],
                           learning_rate=self.params['learning_rate'],
                           load_walks=self.params['load_walks'],
                           embedding_file_out=self.params['embedding_name'],
                           epochs=self.params['epochs'],
                           weight_decay=self.params['weight_decay'],
                           polarity=self.params['polarity'],
                           depth=self.params['depth'],
                           gpu=self.params['gpu'],
                           val_model=self.params['val_model'],
                           env=self.params['env']
                           , max_degree=self.params['max_node_degree']
                           ,degree_l1=self.params['degree_l1']
                           , degree_l2=self.params['degree_l2']
                           , degree_l3=self.params['degree_l3'],
                       hidden_dim_1=self.params['hidden_dim_1'],
                       hidden_dim_2=self.params['hidden_dim_2'],
                       concat=self.params['concat'],
                       jumping_knowledge=self.params['jumping_knowledge'],
                       jump_type=self.params['jump_type'])

        # format networkx idx to features and labels
        nx_idx_to_feat_idx = {self.node_gid_to_graph_idx[gid]: feat for gid, feat
                              in self.node_gid_to_feat_idx.items()}
        nx_idx_to_label_idx = {self.node_gid_to_graph_idx[gid]: label for gid, label
                               in self.node_gid_to_label.items()}
        nx_idx_to_getoelm_idx = {self.node_gid_to_graph_idx[gid]: getoelm_idx for gid, getoelm_idx
                                 in self.gid_to_getoelm_idx.items()}

        G, features, nx_idx_to_feat_idx, _, nx_idx_to_label_idx , _ \
        , _ = format_data(dual=self.G,
                          features=self.features,
                          node_id=nx_idx_to_feat_idx ,
                          id_map=nx_idx_to_feat_idx ,
                          node_classes=nx_idx_to_label_idx,
                          train_or_test='',
                          scheme_required=True,
                          load_walks=False)

        self.gnn = unsupervised_gnn.gnn(aggregator=self.params['aggregator'], env=self.params['env'])

        self.gnn.train(G=G,
                       learning_rate=self.params['learning_rate'],
                       load_walks=self.params['load_walks'],
                       feats=features,
                       id_map=nx_idx_to_feat_idx,
                       nx_idx_to_getoelm_idx= self.lin_adj_idx_to_getoelm_idx,
                       geto_elements=  self.getoelms,
                       class_map=nx_idx_to_label_idx,
                       embedding_file_out='embedding',
                       base_log_dir=self.params['experiment_folder'],
                       epochs=self.params['epochs'],
                       batch_size=self.params['batch_size'],
                       weight_decay=self.params['weight_decay'],
                       polarity=self.params['polarity'],
                       depth=self.params['depth'],
                       gpu=self.params['gpu'],
                       env=self.params['env'],
                       use_embedding=None,
                       val_model=self.params['val_model']
                       , max_degree=self.params['max_node_degree']
                       , degree_l1=self.params['degree_l1']
                       , degree_l2=self.params['degree_l2']
                       , degree_l3=self.params['degree_l3']
                       , model_size=self.params['model_size']
                       , dim_1=self.params['out_dim_1']
                       , dim_2=self.params['out_dim_2'],
                       hidden_dim_1=self.params['hidden_dim_1'],
                       hidden_dim_2=self.params['hidden_dim_2'],
                       concat=self.params['concat'],
                       jumping_knowledge=self.params['jumping_knowledge'],
                       jump_type=self.params['jump_type'],
                       geto_loss=self.params['geto_loss'])

    def supervised_train(self, **kwargs):
        for arg in kwargs:
            self.params[arg] = kwargs[arg]

        wp=1.0
        if self.params['getognn_class_weights']:
            positive_sample_count = 0
            negative_sample_count = 0
            for node in self.G.nodes():
                if self.G.node[node]["label"][0] > 0 and self.G.node[node]['train']:
                    negative_sample_count += 1
                if self.G.node[node]["label"][1] > 0 and self.G.node[node]['train']:
                    positive_sample_count += 1
            train_size = positive_sample_count + negative_sample_count
            wn = float(train_size) / (2. * negative_sample_count)
            wp = float(train_size) / (2. * positive_sample_count)
            print("Using class weights for negative: ", wn)
            print("Using class weights for positive: ", wp)
            print("total training samples: ", train_size)
            print("total positive samples: ", positive_sample_count)

        if self.params['getofeaturegraph_file']:
            print("loading")
            cwd = './'
            walks = self.params['load_walks']
            train_path = os.path.join(cwd, 'data', 'json_graphs', self.params['getofeaturegraph_name'])

            if not self.params['embedding_file_out']:
                embedding_name = datetime.datetime.now()



            self.gnn.train(train_prefix=self.params['getofeaturegraph_name'],
                           learning_rate=self.params['learning_rate'],
                           load_walks=self.params['load_walks'],
                           embedding_file_out=self.params['embedding_name'],
                           epochs=self.params['epochs'],
                           weight_decay=self.params['weight_decay'],
                           polarity=self.params['polarity'],
                           depth=self.params['depth'],
                           gpu=self.params['gpu'],
                           val_model=self.params['val_model'],
                           env=self.params['env']
                           , max_degree=self.params['max_node_degree']
                           ,degree_l1=self.params['degree_l1']
                           , degree_l2=self.params['degree_l2']
                           , degree_l3=self.params['degree_l3'])
            self.train_time = self.gnn.train_time
            self.ped_time = self.gnn.pred_time
        else:
            # format networkx idx to features and labels
            nx_idx_to_feat_idx = {self.node_gid_to_graph_idx[gid]: feat for gid, feat
                                  in self.node_gid_to_feat_idx.items()}
            nx_idx_to_label_idx = {self.node_gid_to_graph_idx[gid]: label for gid, label
                                   in self.node_gid_to_label.items()}
            nx_idx_to_getoelm_idx = {self.node_gid_to_graph_idx[gid]: getoelm_idx for gid, getoelm_idx
                                     in self.gid_to_getoelm_idx.items()}

            G, features, nx_idx_to_feat_idx, _ , nx_idx_to_label_idx, _ , _ \
                = format_data(dual=self.G.copy(),
                              features=self.features,
                              node_id=nx_idx_to_feat_idx,
                              id_map=nx_idx_to_feat_idx,
                              node_classes=nx_idx_to_label_idx,
                              train_or_test='',
                              scheme_required=True,
                              load_walks=False)


            self.gnn = supervised_gnn.gnn(aggregator=self.params['aggregator'],
                                          env=self.params['env'],
                                          msc_collection=None,
                                          model_path=None)#self.params['model_path'])
            print("    * : " )
            print("    * : depth", self.params['depth'])
            self.gnn.train(G=G,
                           learning_rate=self.params['learning_rate'],
                           load_walks=self.params['load_walks'],
                           random_context=self.params['random_context'],
                           feats=features,
                           id_map=nx_idx_to_feat_idx,#node_gid_to_feat_idx ,
                           class_map=nx_idx_to_label_idx,#node_gid_to_label,
                           nx_idx_to_getoelm_idx=self.lin_adj_idx_to_getoelm_idx,
                           geto_elements=self.getoelms,
                           epochs=self.params['epochs'],
                           batch_size=self.params['batch_size'],
                           weight_decay=self.params['weight_decay'],
                           polarity=self.params['polarity'],
                           model_size=self.params['model_size'],
                           depth=self.params['depth'],
                           gpu=self.params['gpu'],
                           env=self.params['env'],
                           val_model = self.params['val_model'],
                           max_degree=self.params['max_node_degree'],
                           hidden_dim_1=self.params['hidden_dim_1'],
                           hidden_dim_2=self.params['hidden_dim_2'],
                           use_embedding=None,
                           sigmoid=False,
                           positive_class_weight=wp,
                           degree_l1=self.params['degree_l1'],
                           degree_l2=self.params['degree_l2'],
                           degree_l3=self.params['degree_l3'],
                           dim_1=self.params['out_dim_1'],
                           dim_2=self.params['out_dim_2'],
                           concat=self.params['concat'],
                           jumping_knowledge=self.params['jumping_knowledge'],
                           jump_type=self.params['jump_type'])
            self.train_time = self.gnn.train_time
            self.ped_time = self.gnn.pred_time

    def embedding_regression_classifier(self, MSCGNN_infer=None, test_prefix=None, trained_prefix=None
                                        , embedding_prefix=None, embedding_path_and_name=None, aggregator=None
                                        , learning_rate=None, MSCGNN=None, supervised=False, size='small'):
        cwd = './'
        # embedding_path =  os.path.join(cwd,'log-dir',embedding_prefix+'-unsup-json_graphs','graphsage_mean_small_'+'0.100000')
        #if embedding_path_name is None and learning_rate is not None:
        #    embedding_p = embedding_prefix + '-unsup-json_graphs' + '/' + aggregator + '_' + size if not supervised else embedding_prefix + '/' + aggregator + '_' + 'small'
        #    embedding_p += ("_{lr:0.6f}").format(lr=learning_rate)
        #else:
        #    embedding_p = embedding_path_name
        if test_prefix is not None and trained_prefix is not None and not self.G:
            trained_p = os.path.join(cwd, 'data', 'json_graphs', trained_prefix)
            test_p = os.path.join(cwd, 'data', 'json_graphs', test_prefix)
            trained_prfx = trained_prefix
            test_prfx = test_prefix
            G_infered = LinearRegression(test_path=test_p, MSCGNN_infer=MSCGNN_infer
                                            , test_prefix=test_prfx, trained_path=trained_p
                                            , trained_prefix=trained_prfx, MSCGNN=self
                                            , embedding_path=self.params['experiment_folder']).run()
            self.G = G_infered
        elif self.G:
            G, feats, id_map, walks \
                , class_map, number_negative_samples, number_positive_samples = format_data(dual=self.G.copy(),
                                                                                            features=self.features,
                                                                                            node_id=self.node_gid_to_feat_idx,
                                                                                            id_map=self.node_gid_to_feat_idx,
                                                                                            node_classes=self.node_gid_to_label,
                                                                                            train_or_test='',
                                                                                            scheme_required=True,
                                                                                            load_walks=False)

            G_infered = LinearRegression(G=G,
                                            MSCGNN_infer=MSCGNN_infer,
                                            features=feats,
                                            labels=class_map,
                                            num_neg=10,  # len(self.negative_arcs),
                                            id_map=id_map,
                                            embedding_path=embedding_path_and_name).run()
            self.G = G_infered
        return self.G

    def run_random_walks(self, walk_embedding_file):

        print("... Computing walks")
        random_walk_embedding(self.G, walk_length=self.params['walk_length'],
                              number_walks=self.params['number_walks'], out_file=walk_embedding_file)

    def get_graph(self):
        return self.gnn.get_graph()



    def write_json_graph_data(self, folder_path, name):
        print('.writing graph family data')
        s = time.time()
        #if not os.path.exists(os.path.join(folder_path, 'json_graphs')):
        #    os.makedirs(os.path.join(folder_path, 'json_graphs'))
        graph_file_path = os.path.join(folder_path, 'json_graphs')
        # group_name = 'left'
        for graph_data, f_name in zip([json_graph.node_link_data(self.G), self.node_gid_to_feat_idx , self.node_gid_to_label, self.features],
                                      [name + '-G', name + '-id_map', name + '-class_map']):

            if not os.path.exists(os.path.join(graph_file_path, f_name + '.json')):
                open(os.path.join(graph_file_path, f_name + '.json'), 'w').close()
            with open(os.path.join(graph_file_path, f_name + '.json'), 'w') as graph_file:
                json.dump(graph_data, graph_file)

        if not os.path.exists(os.path.join(graph_file_path, name + '-feats.npy')):
            open(os.path.join(graph_file_path, name + '-feats.npy'), 'w').close()
        np.save(os.path.join(graph_file_path, name + '-feats.npy'), self.features)

        f = time.time()
        print('graph family written in & w/ prefix ', graph_file_path + '/' + name, '(', f - s, ')')

    """
    def update_training_from_inference(self):
        self.selection_type = 'active'

        if self.kdtree is None:
            self.build_kdtree()
        selector = ArcSelector(image=self.image, gid_edge_dict=self.gid_edge_dict,
                               gid_gnode_dict=self.gid_gnode_dict, kdtree=self.kdtree)

        in_arc_ids, in_arcs, out_arc_ids, out_arcs, out_pixels, \
        selected_test_arcs, selected_test_arc_ids = selector.launch_ui( use_inference=True)  # xlims=X, ylims=Y)

        self.new_selected_positive_arc_ids = set(in_arc_ids)
        self.new_selected_negative_arc_ids = set(out_arc_ids)
        self.new_selected_positive_arcs = set(in_arcs)  # _ids
        self.new_selected_negative_arcs = set(out_arcs)  # _ids
        self.new_selected_test_arcs = set(selected_test_arcs)
        self.new_selected_test_arc_ids = set(selected_test_arc_ids)


        self.selected_positive_arc_ids = self.new_selected_positive_arc_ids.union(self.selected_positive_arc_ids)
        self.selected_negative_arc_ids = self.new_selected_negative_arc_ids.union(self.selected_negative_arc_ids)
        self.selected_positive_arcs = self.new_selected_positive_arcs.union(self.selected_positive_arcs)
        self.selected_negative_arcs = self.new_selected_negative_arcs.union(self.selected_negative_arcs)
        self.selected_test_arcs = self.new_selected_test_arcs.union(self.selected_test_arcs)
        self.selected_test_arc_ids = self.new_selected_test_arc_ids.union(self.selected_test_arc_ids)
        if self.params['use_ground_truth']:
            selected_arcs = self.selected_positive_arcs.union(self.selected_negative_arcs)

            selected_arc_ids = self.selected_positive_arc_ids.union(self.selected_negative_arc_ids)

            selected_positive_arcs = set()
            selected_negative_arcs = set()
            selected_positive_arc_ids = set()
            selected_negative_arc_ids = set()
            for arc in selected_arcs:
                arc.partition = 'train'
                self.node_gid_to_partition[arc.gid] = 'train'
                if arc.label > 0:
                    selected_positive_arcs.add(arc)
                else:
                    selected_negative_arcs.add(arc)
            for idx in selected_arc_ids:
                if self.gid_gnode_dict[idx].label[1] > 0:
                    selected_positive_arc_ids.add(idx)
                else:
                    selected_negative_arc_ids.add(idx)
            self.selected_positive_arcs = selected_positive_arcs
            self.selected_negative_arcs = selected_negative_arcs
            self.selected_positive_arc_ids = selected_positive_arc_ids
            self.selected_negative_arc_ids = selected_negative_arc_ids
        self.positive_arc_ids = self.selected_positive_arc_ids
        self.negative_arc_ids = self.selected_negative_arc_ids
        self.positive_arcs = self.selected_positive_arcs
        self.negative_arcs = self.selected_negative_arcs

    def select_geomsc_training(self, image=None, X=None, Y=None, fname_selection=None):
        self.selection_type = 'manual'

        start_time = time.time()
        if self.kdtree is None:
            self.build_kdtree()
        selector = ArcSelector(image=self.image, gid_edge_dict=self.gid_edge_dict,
                               gid_gnode_dict=self.gid_gnode_dict, kdtree=self.kdtree)

        in_arc_ids, in_arcs, out_arc_ids, out_arcs, out_pixels, \
        selected_test_arcs, selected_test_arc_ids = selector.launch_ui()
        self.selected_positive_arc_ids = set(in_arc_ids)
        self.selected_negative_arc_ids = set(out_arc_ids)
        self.selected_positive_arcs = set(in_arcs)#_ids
        self.selected_negative_arcs = set(out_arcs)#_ids
        self.selected_test_arcs = set(selected_test_arcs)
        self.selected_test_arc_ids = set(selected_test_arcs)
        if self.params['use_ground_truth']:
            selected_arcs = self.selected_positive_arcs.union(self.selected_negative_arcs)
            selected_arc_ids = self.selected_positive_arc_ids.union(self.selected_negative_arc_ids)
            selected_positive_arcs = set()
            selected_negative_arcs = set()
            selected_positive_arc_ids = set()
            selected_negative_arc_ids = set()
            for gnode in selected_arcs:
                gnode.partition = 'train'
                self.node_gid_to_partition[gnode.gid] = 'train'
                if gnode.label[1] > 0:
                    selected_positive_arcs.add(gnode)
                else:
                    selected_negative_arcs.add(gnode)
            for idx in selected_arc_ids:
                if self.gid_gnode_dict[idx].label[1] > 0:
                    selected_positive_arc_ids.add(idx)
                else:
                    selected_negative_arc_ids.add(idx)
            self.selected_positive_arcs = selected_positive_arcs
            self.selected_negative_arcs = selected_negative_arcs
            self.selected_positive_arc_ids = selected_positive_arc_ids
            self.selected_negative_arc_ids = selected_negative_arc_ids

        self.positive_arc_ids = self.selected_positive_arc_ids
        self.negative_arc_ids = self.selected_negative_arc_ids
        self.positive_arcs = self.selected_positive_arcs
        self.negative_arcs = self.selected_negative_arcs

        end_time = time.time()

        print('..Time to make selection: ', end_time-start_time)

        return (in_arcs, out_arcs, out_pixels)

    def box_select_geomsc_training(self, x_range, y_range, image=None):
        self.selection_type = 'box'

        print("&&&& doing selection")
        print("&&&& Box X Range_",x_range)
        print("&&&& Box Y Range_", y_range)
        self.x_box = x_range
        self.y_box = y_range

        self.selected_positive_arcs = set()
        self.selected_negative_arcs = set()
        self.selected_positive_arc_ids = set()
        self.selected_negative_arc_ids = set()
        self.selected_test_arcs = set()
        self.selected_test_arc_ids = set()

        if len(x_range) == 1:#np.array(x_range).shape == np.array([6,9]).shape:
            x_range = x_range[0]
            y_range = y_range[0]
            for gnode in self.gid_gnode_dict.values():
                #idx = self.make_arc_id(arc)
                gid = gnode.gid
                self.node_gid_to_partition[gnode.gid] = ''
                p1 = gnode.points[0]
                p2 = gnode.points[-1]
                in_box = False
                points = (p1,p2)
                for p in points:
                    if x_range[0] <= p[0] <= x_range[1] and y_range[0] <= p[1] <= y_range[1]:
                        in_box = True
                if in_box and gnode.z != 1:
                    gnode.box = 1
                    gnode.partition = 'train'
                    self.node_gid_to_partition[gnode.gid] = 'train'
                    if gnode.label[1] > 0:
                        self.selected_positive_arcs.add(gnode)
                        self.selected_positive_arc_ids.add(gid)
                    else:
                        self.selected_negative_arcs.add(gnode)
                        self.selected_negative_arc_ids.add(gid)
                else:
                    gnode.box = 0
                    gnode.partition = 'test'
                    self.node_gid_to_partition[gnode.gid] = 'test'
                    self.selected_test_arcs.add(gnode)
                    self.selected_test_arc_ids.add(gid)
        else:
            range_group = zip(x_range, y_range)
            for x_rng , y_rng in range_group:

                for gnode in self.gid_gnode_dict.values():#self.msc.arcs:
                    gid = gnode.gid
                    self.node_gid_to_partition[gnode.gid] = ''
                    p1 = gnode.points[0]
                    p2 = gnode.points[-1]
                    in_box = False
                    for p in (p1, p2):
                        if x_rng[0] <= p[0] <= x_rng[1] and y_rng[0] <= p[1] <= y_rng[1]:
                            in_box = True
                    if in_box and gnode.z != 1:
                        gnode.box = 1
                        gnode.partition = 'train'
                        self.node_gid_to_partition[gnode.gid] = 'train'
                        if gnode.label[1] > 0:
                            self.selected_positive_arcs.add(gnode)
                            self.selected_positive_arc_ids.add(gid)
                        else:
                            self.selected_negative_arcs.add(gnode)
                            self.selected_negative_arc_ids.add(gid)
                    else:
                        gnode.box = 0
                        gnode.partition = 'test'
                        self.node_gid_to_partition[gnode.gid] = 'test'
                        self.selected_test_arcs.add(gnode)
                        self.selected_test_arc_ids.add(gid)
        self.positive_arc_ids = self.selected_positive_arc_ids
        self.negative_arc_ids = self.selected_negative_arc_ids
        self.positive_arcs = self.selected_positive_arcs
        self.negative_arcs = self.selected_negative_arcs
        self.all_train = self.positive_arc_ids.union(self.negative_arc_ids)
        self.all_test_and_val = self.selected_test_arcs
        return self.all_train , self.all_test_and_val

    def edge_map_overlap(self, arc, labeled_segmentation,  labeled_mask= None, msc=None, geomsc=None,
                         invert=True):
        arc_accuracy = 0
        percent_interior = 0
        for point in arc.points:
            x = 0
            y = 1
            if invert:
                x = 1
                y = 0
            if labeled_segmentation[int(point[x]), int(point[y])] > 0:
                arc_accuracy += 1.
            #if labeled_mask[int(point[x]), int(point[y])] > 0:
            #    percent_interior += 1.
        label_accuracy = arc_accuracy / float(len(arc.points))
        #interior_percent_arc = percent_interior / float(len(arc.points))
        #if interior_percent_arc < .85:
        #    arc.exterior = 1
        if label_accuracy == 0.:
            label_accuracy = 1e-4
        return label_accuracy

    def map_labeling(self, labeled_segmentation, labeled_mask=None,
                     invert=False):

        for arc in self.gid_gnode_dict.values():
            arc.label_accuracy = self.edge_map_overlap(arc=arc
                                                       , labeled_segmentation=labeled_segmentation
                                                       , labeled_mask=labeled_mask
                                                       , invert=invert)

    def cvt_sample_validation_set(self, hops, samples):
        all_selected_arcs = []
        if self.selection_type == 'manual' or self.selection_type == 'box':  #
            for i in self.selected_positive_arcs:
                all_selected_arcs.append(i)
            for i in self.selected_negative_arcs:
                all_selected_arcs.append(i)
        self.subgraph_sample_set, self.subgraph_sample_set_ids, \
        subgraph_positive_arcs, subgraph_negative_arcs= self.cvt_sample_subgraphs(sample_gnode_set=all_selected_arcs,

                                                                                   count=samples,
                                                                                   hops=hops)
        for gnode in subgraph_negative_arcs:
            v = self.gid_gnode_dict[gnode.gid]
            v.partition  = 'val'
            self.node_gid_to_partition[v.gid] = 'val'
        for gnode in subgraph_positive_arcs:
            v = self.gid_gnode_dict[gnode.gid]
            v.partition  = 'val'
            self.node_gid_to_partition[v.gid] = 'val'
        return self.subgraph_sample_set, self.subgraph_sample_set_ids, subgraph_positive_arcs, subgraph_negative_arcs

    def cvt_sample_test_set(self, samples, hops):
        self.subgraph_sample_set, self.subgraph_sample_set_ids, subgraph_positive_arcs, subgraph_negative_arcs = self.cvt_sample_subgraphs(sample_gnode_set=self.gid_gnode_dict.values(),
                                                                                   count=samples,
                                                                                   hops=hops)
        return self.subgraph_sample_set, self.subgraph_sample_set_ids, subgraph_positive_arcs, subgraph_negative_arcs

    def cvt_sample_subgraphs(self, sample_gnode_set, accuracy_threshold=0.1, count=1, hops=1, seed=666):
        X = self.image.shape[0]
        Y = self.image.shape[1] if len(self.image.shape) == 2 else self.image.shape[2]
        np.random.seed(seed)
        hypercube = samply.hypercube.cvt(count, 2)
        hypercube[:, 0] *= X
        hypercube[:, 1] *= Y

        self.build_select_kdtree(sample_gnode_set)
        seed_gnode_ids = list()

        for x in hypercube:
            print(" >>>> hypercube x", x)
            gnode_gid = self.get_closest_selected_gnode(x)  # self.msc.get_closest_arc_index(x)
            seed_gnode_ids.append(gnode_gid)

        hop = 0
        hop_index = 0
        ring_count = len(seed_gnode_ids)

        while hop <= hops:
            next_ring = seed_gnode_ids[hop_index:(hop_index + ring_count)]

            ring_count = 0
            for gnode_gid in next_ring:
                gnode = self.gid_gnode_dict[gnode_gid]
                for adj_edge_gid in gnode.edge_gids:
                    adj_edge = self.gid_edge_dict[adj_edge_gid]
                    for adj_gnode_gid in adj_edge.gnode_gids:
                        if adj_gnode_gid not in seed_gnode_ids:
                            seed_gnode_ids.append(adj_gnode_gid)
                            ring_count += 1
                hop_index += 1
            hop += 1
        print('size seed arc keys ', len(seed_gnode_ids))
        seed_gnode_ids= set(seed_gnode_ids)

        print("map output arc ")
        sample_set_ids = (
        self.positive_arc_ids.intersection(seed_gnode_ids), self.negative_arc_ids.intersection(seed_gnode_ids))
        subgraph_positive_arcs = set()
        subgraph_negative_arcs = set()
        for gnode_gid in sample_set_ids[0]:
            subgraph_positive_arcs.add(self.gid_gnode_dict[gnode_gid])
        for gnode_gid in sample_set_ids[1]:
            subgraph_negative_arcs.add(self.gid_gnode_dict[gnode_gid])

        self.subgraph_sample_set["positive"] = subgraph_positive_arcs
        self.subgraph_sample_set["negative"] = subgraph_negative_arcs
        self.subgraph_sample_set_ids["positive"] = sample_set_ids[0]
        self.subgraph_sample_set_ids["negative"] = sample_set_ids[1]
        return self.subgraph_sample_set, self.subgraph_sample_set_ids, subgraph_positive_arcs, subgraph_negative_arcs

    def check_valid_partitions(self, train, val):
        train_copy = train.copy()
        for vl in train.intersection(val):
            train_copy.remove(vl)
        return not len(train_copy) == 0

    def get_train_test_val_sugraph_split(self, validation_hops = 1, validation_samples = 1,
                                         test_hops = None, test_samples = None, multiclass= False
                                         , collect_validation=True):
        if collect_validation:
            self.validation_set , self.validation_set_ids, _ , _ = self.cvt_sample_validation_set(hops = validation_hops,
                                                                                                  samples= validation_samples)
            for gid in self.validation_set_ids["positive"].union(self.validation_set_ids["negative"]):

                self.node_gid_to_partition[gid] = 'val'

        all_validation = self.validation_set_ids["positive"].union(self.validation_set_ids["negative"])
        all_selected = self.selected_positive_arc_ids.union(self.selected_negative_arc_ids)

        if test_samples is not None:
            self.test_set, self.test_set_ids, _, _ =self.cvt_sample_test_set(samples=test_samples
                                            , hops=test_hops)
            all_test = self.test_set_ids["positive"].union(self.test_set_ids["negative"])
        #-- dict mapping node ids to label
        #-- dict mapping node ids to index in feature tensor
        gid_feat_idx = 0

        for gnode in self.gid_gnode_dict.values():


            partition = self.node_gid_to_partition[gnode.gid]
            features = self.node_gid_to_feature[gnode.gid].tolist()

            if gnode.label is not None and gnode.gid not in all_selected:
                label = self.node_gid_to_label[gnode.gid]# [0 , 1] if gnode.label > 0 else [1, 0]#[
            else:
                label = [
                        int(gnode.gid in self.negative_arc_ids),
                        int(gnode.gid in self.positive_arc_ids)
                ]
                self.node_gid_to_label[gnode.gid] = label
            if multiclass:
                label = [
                    int(gnode.gid in self.negative_arc_ids),
                   int(gnode.gid in self.positive_arc_ids),
                    0
                ]
            nx_gid = self.node_gid_to_nx_idx[gnode.gid]
            node = self.G.node[nx_gid]
            node["features"] =  features#gnode.features.tolist()
            node["gid"] = gnode.gid
            node["key"] = gnode.key
            node["box"] = gnode.box
            node["partition"] = partition
            # assign partition to node
            node["train"] = partition == 'train'
            node["test"] = partition == 'test'
            node["val"] = partition == 'val'
            node["label"] = label
            if self.selection_type == 'map':
                node["label_accuracy"] = gnode.label_accuracy
            node["prediction"] = []
            self.node_gid_to_prediction[gnode.gid] = []

            # labeled nodes assigned as train, test, or val
            if self.params['union_space']:
                node["label"] =  label
                if self.selection_type == 'map':
                    node["label_accuracy"] = gnode.label_accuracy
                node["prediction"] = []
                modified = 0
                group = None
                if gnode.z == 1:
                    modified = 1
                    node["train"] = False
                    node["test"] = True
                    node["val"] = False
                    gnode.partition = 'test'
                    self.node_gid_to_partition[gnode.gid] = 'test'
                    continue
                if gnode.gid in all_validation:
                    modified = 1
                    node["train"] = False
                    node["test"] = False
                    node["val"] = True
                    gnode.partition = 'val'
                    self.node_gid_to_partition[gnode.gid] = 'val'
                elif test_samples is not None and gnode.gid in all_test:
                    node["test"] = True
                    node["val"] = False
                    node["train"] = False
                    gnode.partition = 'test'
                    self.node_gid_to_partition[gnode.gid] = 'test'
                else:  # and  i_val < val_count:
                    modified = 1
                    node["train"] = True
                    node["test"] = False
                    node["val"] = False
                    gnode.partition = 'train'
                    self.node_gid_to_partition[gnode.gid] = 'train'
            elif False:
                node["label"] = label  # arc.label_accuracy
                node["label_accuracy"] = gnode.label_accuracy
                node["prediction"] = []
                group = ''
                if partition == 'val':
                    modified = 1
                    node["train"] = False
                    node["test"] = False
                    node["val"] = True
                    gnode.partition = 'val'
                    self.node_gid_to_partition[gnode.gid] = 'val'
                    continue
                elif gnode.gid not in all_validation and (gnode.gid in self.selected_negative_arc_ids or gnode.gid
                                                     in self.selected_positive_arc_ids):

                    node["test"] = False
                    node["val"] = False
                    node["train"] = True
                    gnode.partition = 'train'
                    self.node_gid_to_partition[gnode.gid] = 'train'
                else:
                    modified = 1
                    node["train"] = False
                    node["test"] = True
                    node["val"] = False
                    gnode.partition = 'test'
                    self.node_gid_to_partition[gnode.gid] = 'test'
            gid_feat_idx += 1

        for nx_gid, node in list(self.G.nodes_iter(data=True)):

            for adj_edge in node["edges"]:
                for adj_gnode_gid in self.gid_edge_dict[adj_edge].gnode_gids:
                    adj_gnode_nx_id = self.node_gid_to_nx_idx[adj_gnode_gid]
                    #if adj_gnode_nx_id != nx_gid:
                    self.G.add_edge(nx_gid, adj_gnode_nx_id)

        self.G_dict = json_graph.node_link_data(self.G)
        s1 = json.dumps(self.G_dict)  # input graph
        s2 = json.dumps(self.node_gid_to_feat_idx)  # dict: nodes to ints
        s3 = json.dumps(self.node_gid_to_label)  # dict: node_id to class

        end_time = time.time()
        return (self.G_dict, self.node_gid_to_feat_idx , self.node_gid_to_label, self.features)

    def write_arc_predictions(self, filename, path=None, msc=None):
        msc_pred_file = os.path.join(self.pred_session_run_path, "preds.txt")
        print("&&&& writing predictions in: ", msc_pred_file)
        pred_file = open(msc_pred_file,"w+")
        for gnode in self.gid_gnode_dict.values():
            pred_file.write(str(gnode.gid)+ ' '+str(gnode.prediction) + "\n")
        pred_file.close()

    # write to file per line 'gid partition'
    # where partition = {train:0, test:1, val:2}
    def write_gnode_partitions(self, name):
        partitions_file = os.path.join(self.pred_session_run_path, 'partitions.txt')
        print("... Writing partitions file to:", partitions_file)
        partitions_file = open(partitions_file, "w+")
        for gid in self.gid_gnode_dict.keys():
            node_partition = self.node_gid_to_partition[gid]
            p = str(0) if node_partition == 'train' else str(2)
            p = str(1) if node_partition == 'test' else str(2)
            partitions_file.write(str(gid) + ' ' + str(p) + "\n")
        partitions_file.close()


    def write_selection_bounds(self, name):
        window_file = os.path.join(self.pred_session_run_path, 'window.txt')
        print("... Writing bounds file to:", window_file)
        window_file = open(window_file, "w+")
        window_file.write('x_box' + ' ' + str(self.x_box) + "\n")
        window_file.write('y_box' + ' ' + str(self.y_box) )
        window_file.close()
    """

class supervised_getognn:
    def __init__(self, model_name):
        self.model_name = model_name
        #self.attributes = Attributes()


    def build_getognn(self, sample_idx, experiment_num, experiment_name, window_file_base,
                 parameter_file_number, format = 'raw', run_num=2, experiment_folder=None,
                 name=None, image=None, label_file=None, msc_file=None, X_BOX=None,Y_BOX=None,regions =None,
                 ground_truth_label_file=None, write_path=None, feature_file=None, boxes = None,
                 window_file=None, model_name="GeToGNN", BEGIN_LOADING_FEATURES=False):




        self.getognn = GeToGNN(training_selection_type='box',X_BOX=X_BOX,Y_BOX=Y_BOX,
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
                          write_json_graph = False,
                               BEGIN_LOADING_FEATURES=BEGIN_LOADING_FEATURES)

        #self.attributes = deepcopy(self.getognn.get_attributes())

        if BEGIN_LOADING_FEATURES:
            self.getognn.params['load_features'] = True
            self.getognn.params['write_features'] = False
            self.getognn.params['load_features'] = True
            self.getognn.params['write_feature_names'] = False
            self.getognn.params['save_filtered_images'] = False
            self.getognn.params['collect_features'] = False
            self.getognn.params['load_preprocessed'] = True
            self.getognn.params['load_geto_attr'] = True
            self.getognn.params['load_feature_names'] = True
        else:
            self.getognn.params['load_features'] = False
            self.getognn.params['write_features'] = True
            self.getognn.params['load_features'] = False
            self.getognn.params['write_feature_names'] = True
            self.getognn.params['save_filtered_images'] = True
            self.getognn.params['collect_features'] = True
            self.getognn.params['load_preprocessed'] = False
            self.getognn.params['load_geto_attr'] = False
            self.getognn.params['load_feature_names'] = False


        X_BOX = []
        Y_BOX = []
        box_sets = []
        for box in regions:
            X_BOX.append((box[0], box[1]))
            Y_BOX.append((box[2], box[3]))



        self.getognn.box_regions = boxes

        num_percent = 0
        for xbox, ybox in zip(X_BOX, Y_BOX):
            num_percent += float((xbox[1] - xbox[0]) * (ybox[1] - ybox[0]))
        percent = num_percent / float(self.getognn.image.shape[0] * self.getognn.image.shape[1])
        percent_f = percent *100
        print("    * ", percent_f)
        percent = int(round(percent_f))
        self.training_size = percent
        self.run_num = percent
        # self.X_BOX = X_BOX
        # self.Y_BOX = Y_BOX

        self.getognn.update_run_info(batch_multi_run=str(self.training_size))
        out_folder = os.path.join(self.getognn.pred_session_run_path)

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        if self.getognn.params['load_geto_attr']:
            if 'geto' in self.getognn.params['aggregator']:
                self.getognn.load_geto_features()
                self.getognn.load_geto_feature_names()
        else:
            if 'geto' in self.getognn.params['aggregator']:
                self.getognn.build_geto_adj_list(influence_type=self.getognn.params['geto_influence_type'])
                self.getognn.write_geto_features(self.getognn.session_name)
                self.getognn.write_geto_feature_names()

        # features
        if not self.getognn.params['load_features']:
            self.getognn.compile_features(include_geto=self.getognn.params['geto_as_feat'])
            self.getognn.write_gnode_features(self.getognn.session_name)
            self.getognn.write_feature_names()
        else:
            self.getognn.load_gnode_features()
            self.getognn.load_feature_names()
            if 'geto' in self.getognn.params['aggregator'] or self.getognn.params['geto_as_feat']:
                self.getognn.load_geto_features()
                self.getognn.load_geto_feature_names()

        if self.getognn.params['write_features']:
            self.getognn.write_gnode_features(self.getognn.session_name)

        if self.getognn.params['write_feature_names']:
            self.getognn.write_feature_names()


        if self.getognn.params['write_feature_names']:
            if 'geto' in self.getognn.params['aggregator'] or self.getognn.params['geto_as_feat']:
                self.getognn.write_geto_feature_names()
        if self.getognn.params['write_features']:
            if 'geto' in self.getognn.params['aggregator'] or self.getognn.params['geto_as_feat']:
                self.getognn.write_geto_features(self.getognn.session_name)

        # training info, selection, partition train/val/test
        self.getognn.read_labels_from_file(file=ground_truth_label_file)



        training_set , test_and_val_set, box_set = self.getognn.box_select_geomsc_training(x_range=X_BOX,
                                                                                  y_range=Y_BOX,
                                                                                  boxes=None)
        # self.getognn.X_BOX = box_set[0]
        # self.getognn.Y_BOX = box_set[1]
        # self.X_BOX, self.Y_BOX = box_set
        print("BOX SET")
        print(box_set)

        self.getognn.get_train_test_val_sugraph_split(collect_validation=False, validation_hops = 1,
                                                 validation_samples = 1)





        if self.getognn.params['write_json_graph']:
            self.getognn.write_json_graph_data(folder_path=self.getognn.pred_session_run_path, name=model_name + '_' + self.getognn.params['name'])




        # random walks
        if not self.getognn.params['load_preprocessed_walks']:
            walk_embedding_file = os.path.join(self.getognn.LocalSetup.project_base_path, 'datasets',
                                               self.getognn.params['write_folder'],'walk_embeddings',
                                               'gnn')
            self.getognn.params['load_walks'] = walk_embedding_file
            self.getognn.run_random_walks(walk_embedding_file=walk_embedding_file)
        else:
            walk_embedding_file = os.path.join(self.getognn.LocalSetup.project_base_path, 'datasets',
                                               self.getognn.params['write_folder'], 'walk_embeddings',
                                               'gnn')
            self.getognn.params['load_walks'] = walk_embedding_file


    def compute_features(self):
        if self.getognn.params['load_geto_attr']:
            if 'geto' in self.getognn.params['aggregator']:
                self.getognn.load_geto_features()
                self.getognn.load_geto_feature_names()
        else:
            if 'geto' in self.getognn.params['aggregator']:
                self.getognn.build_geto_adj_list(influence_type=self.getognn.params['geto_influence_type'])
                self.getognn.write_geto_features(self.getognn.session_name)
                self.getognn.write_geto_feature_names()

        # features
        if not self.getognn.params['load_features']:
            self.getognn.compile_features(include_geto=self.getognn.params['geto_as_feat'])
            self.getognn.write_gnode_features(self.getognn.session_name)
            self.getognn.write_feature_names()
        else:
            self.getognn.load_gnode_features()
            self.getognn.load_feature_names()
            if 'geto' in self.getognn.params['aggregator'] or self.getognn.params['geto_as_feat']:
                self.getognn.load_geto_features()
                self.getognn.load_geto_feature_names()

        if self.getognn.params['write_features']:
            self.getognn.write_gnode_features(self.getognn.session_name)

        if self.getognn.params['write_feature_names']:
            self.getognn.write_feature_names()


        if self.getognn.params['write_feature_names']:
            if 'geto' in self.getognn.params['aggregator'] or self.getognn.params['geto_as_feat']:
                self.getognn.write_geto_feature_names()
        if self.getognn.params['write_features']:
            if 'geto' in self.getognn.params['aggregator'] or self.getognn.params['geto_as_feat']:
                self.getognn.write_geto_features(self.getognn.session_name)

    def train(self, getognn=None, run_num=1):
        if getognn is not None:
            self.getognn = getognn
        # self.X_BOX = self.getognn.X_BOX
        #
        # self.Y_BOX = self.getognn.Y_BOX
        #training

        self.getognn.supervised_train()

        self.pred_time = self.getognn.pred_time
        self.train_time = self.getognn.train_time

        G = self.getognn.get_graph()
        self.getognn.equate_graph(G)

        #self.getognn.update_run_info(batch_multi_run=run_num)

        #self.getognn.write_arc_predictions(dir=self.getognn.pred_session_run_path)
        #self.getognn.draw_segmentation(dirpath=self.getognn.pred_session_run_path)
        self.getognn = self.getognn
        return self.getognn

class unsupervised_getognn:
    def __init__(self, model_name):
        self.model_name = model_name
        self.attributes = Attributes()

    def build_getognn(self, sample_idx, experiment_num, experiment_name, window_file_base,
                 parameter_file_number, format = 'raw',  experiment_folder=None,
                 name=None, image=None, label_file=None, msc_file=None,
                 ground_truth_label_file=None, write_path=None, feature_file=None,
                 window_file=None, model_name="GeToGNN"):



        self.getognn = GeToGNN(training_selection_type='box',
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
            self.getognn.load_gnode_features()
        if self.getognn.params['write_features']:
            self.getognn.write_gnode_features(self.getognn.session_name)
        if self.getognn.params['write_feature_names']:
            self.getognn.write_feature_names()

        self.getognn.build_geto_adj_list()

        # training info, selection, partition train/val/test
        self.getognn.read_labels_from_file(file=ground_truth_label_file)

        training_set , test_and_val_set, empty_set = self.getognn.box_select_geomsc_training(x_range=self.getognn.params['x_box'], y_range=self.getognn.params['y_box'])

        self.getognn.get_train_test_val_sugraph_split(collect_validation=True, validation_hops = 1,
                                                 validation_samples = 1)



        #self.attributes = deepcopy(self.getognn.get_attributes())

        if self.getognn.params['write_json_graph']:
            self.getognn.write_json_graph_data(folder_path=self.getognn.pred_session_run_path, name=model_name + '_' + self.getognn.params['name'])


        self.getognn.write_gnode_partitions(self.getognn.session_name)
        #self.getognn.write_selection_bounds(self.getognn.session_name)

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
        self.getognn.unsupervised_train()
        #G = self.getognn.get_graph()
        #self.getognn.equate_graph(G)

        #self.getognn.write_arc_predictions(self.getognn.session_name)
        #self.getognn.draw_segmentation(dirpath=self.getognn.pred_session_run_path)
        self.getognn = self.getognn
        return self.getognn

    def classify(self, embedding_name = None):
        #
        G = self.getognn.embedding_regression_classifier(embedding_path_and_name = embedding_name)
        self.getognn.equate_graph(G)
        #self.getognn.write_arc_predictions(self.getognn.session_name)
        #self.getognn.draw_segmentation(dirpath=self.getognn.pred_session_run_path)
