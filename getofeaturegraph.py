import os
import time
import networkx as nx
import numpy as np
#import math as m
import copy
import scipy.stats
import numpy.linalg as linalg
#import imageio
import matplotlib.pyplot as plt
#from PIL import Image
from sklearn.preprocessing import PowerTransformer, RobustScaler, QuantileTransformer
from functools import partial
import umap
from umap.umap_ import nearest_neighbors
from memory_profiler import profile

from ml.features import multiscale_basic_features
from getograph import GeToGraph
from data_ops.collect_data import collect_training_data, compute_geomsc, collect_datasets
from data_ops.utils import dbgprint
from data_ops.set_params import set_parameters
from ml.features import (
    mean_filter,
    variance_filter,
    median_filter,
    minimum_filter,
    maximum_filter,
    gaussian_blur_filter,
    difference_of_gaussians_filter,
    sobel_filter,
    cosine_similarity,
    hyperbolic_distance,
    hyperbolic_distance_line,
    get_points_from_vertices,
    get_pixel_values_from_vertices,
    translate_points_by_centroid,
    get_centroid,
    laplacian_filter,
    sum_euclid,
    gaussian_fit,
    slope,
    end_to_end_euclid,
    manhattan_distance,
    mahalanobis_distance_arc,
    cumulative_distance_from_centroid,
    cos_sim_2d,
)


from localsetup import LocalSetup
from ml.utils import load_data
from ml.utils import pout

class GeToFeatureGraph(GeToGraph):
    def __init__(self, image=None, geomsc_fname_base = None, label_file=None,
                 parameter_file_number = None, run_num=2,
                 dataset_group='neuron',write_folder="results",
                 model_name='ndlas',
                 name='neuron2', format='raw', load_feature_graph_name = None,
                 msc_file=None, dim_invert=False, map_labels=False,
                 reset_run = False, **kwargs):

        self.LocalSetup = LocalSetup()


        #if parameter_file_number is not None:
        #    self.params = set_parameters(read_params_from=parameter_file_number)
        #else:
        #self.params = kwargs['params']
        #for param in kwargs:
        #    self.params[param] = kwargs[param]
        #for param in kwargs['params']:
        #    self.params[param] = kwargs['params'][param]

        self.run_num=run_num
        super(GeToFeatureGraph, self).__init__(parameter_file_number=parameter_file_number,
                                               geomsc_fname_base=geomsc_fname_base,
                                               write_folder= write_folder,
                                               label_file=label_file, run_num=run_num)



        #
        # Write Paths
        #



        #if parameter_file_number is not None:
        #    self.params = set_parameters(read_params_from=parameter_file_number)
        #else:
        #    self.params = kwargs['params']

        '''for param in kwargs:
            self.params[param] = kwargs[param]
        #for param in kwargs['params']:
        #    self.params[param] = kwargs['params'][param]
        if 'params' in kwargs.keys():
            param_add_ons = kwargs['params']
            for k, v in param_add_ons.items():
                self.params[k] = v'''



        self.data_name = name
        self.params['write_folder'] = write_folder
        self.experiment_folder = os.path.join(self.LocalSetup.project_base_path, 'datasets'
                                              , self.params['write_folder'])
        self.input_folder = os.path.join(self.LocalSetup.project_base_path, 'datasets', name, 'input')
        self.params['experiment_folder'] = self.experiment_folder
        self.params['input_folder'] = self.input_folder
        self.params['data_name'] = self.data_name

        self.params['format'] = format
        self.pred_run_path = os.path.join(self.LocalSetup.project_base_path, 'datasets',
                                          self.params['write_folder'], 'runs')
        #if not os.path.exists(self.pred_run_path):
        #    os.makedirs(os.path.join(self.pred_run_path))
        #self.segmentation_path = self.LocalSetup.neuron_training_segmentation_path
        #self.msc_write_path = self.LocalSetup.neuron_training_base_path

        self.model_name = model_name
        #self.segmentation_path = self.LocalSetup.neuron_training_segmentation_path
        #self.msc_write_path = self.LocalSetup.neuron_training_base_path


        self.session_name = str(self.run_num)

        model_type = os.path.join(self.LocalSetup.project_base_path, 'model_type.txt')
        logged_model = open(model_type, 'r')
        m = logged_model.readlines()
        logged_model.close()

        self.inference_target = m[0]

        print("    * read model", m)
        #if m[0] != 'unet':
        #    self.pred_session_run_path = os.path.join(self.pred_run_path, self.session_name)
        #    if not os.path.exists(self.pred_session_run_path):
        #        os.makedirs(os.path.join(self.pred_session_run_path))


        #
        # Data Collection
        #
        self.data_array, self.data_set = collect_datasets(name=name,image=image,
                                                          dim_invert=self.params['dim_invert'],
                                                          format=self.params['format'])

        if geomsc_fname_base is None:
            compute_geomsc(self.params, self.data_array, self.pred_run_path, self.segmentation_path,
                           self.msc_write_path, map_labels=False)

        #
        # Collect data flow with im, msc, mask-(if avail/used)
        #
        self.train_dataloader = collect_training_data(
            dataset=self.data_set,
            data_array=self.data_array,
            params=self.params,
            name=name,
            format=format,
            msc_file=None,
            dim_invert=self.params['dim_invert'])

        self.image, self.msc_collection, self.mask, self.segmentation = self.train_dataloader[
            int(self.params['train_data_idx'])]
        self.image = self.image.astype(np.float32)
        max_val = np.max(self.image)
        min_val = np.min(self.image)
        self.image = (self.image - min_val) / (max_val - min_val)

        self.thirty_percent_image_intensity_range = 0.33*(max_val - min_val)

        #self.image = self.image if len(self.image.shape) == 2 else np.transpose(np.mean(self.image, axis=1), (1, 0))

        self.X = self.image.shape[0]
        self.Y = self.image.shape[1] if len(self.image.shape) == 2 else self.image.shape[2]

        #
        #
        self.features = None
        self.feature_ops = {}

        self.feature_comp_time = 0

        #
        # build feature graph
        #
        #if load_feature_graph_name is None:
        #    self.msc_graph_name = 'geomsc_feature_graph-' + str(self.params['number_features']) + '_dataset-' + name + model_name
        #else:
        #    self.msc_graph_name = load_feature_graph_name
        #    self.load_feature_graph()
        """if self.G is None:
            self.G = self.build_graph()
        """
        if self.G is None:
            self.G = self.build_graph()
        #if self.features is None and not self.params['load_features']:
        #    self.compiled_data = self.compile_features()
        #    self.features = self.compiled_data
        #elif self.params['load_features']:
        #    self.load_gnode_features(filename = self.params['feature_file'])
        #else:
        #    self.compiled_data = self.features



    def build_graph(self):
        self.G = nx.Graph()
        nx_node_idx = 0
        for gnode in self.gid_gnode_dict.values():
            gid = gnode.gid
            self.G.add_node(
                nx_node_idx, edges=gnode.edge_gids, size=len(gnode.points)
            )
            self.node_gid_to_graph_idx[gid] = nx_node_idx
            self.graph_idx_to_gid[nx_node_idx] = gid
            nx_node_idx += 1

        for nx_gid, node in list(self.G.nodes_iter(data=True)):

            for adj_edge in node["edges"]:
                for adj_gnode_gid in self.gid_edge_dict[adj_edge].gnode_gids:
                    node_gid = self.graph_idx_to_gid[nx_gid]
                    adj_gnode_nx_id = self.node_gid_to_graph_idx[adj_gnode_gid]
                    if adj_gnode_gid != node_gid:
                        self.G.add_edge(nx_gid, adj_gnode_nx_id)

        return self.G.copy()

    def build_subgraph(self, gid_gnode_dict, gid_edge_dict):
        G = nx.Graph()
        nx_node_idx = 0
        node_gid_to_graph_idx = {}
        graph_idx_to_gid = {}
        for gnode in gid_gnode_dict.values():
            #
            #                                CHECK THIS!
            #                added to skip supergraph 8/31/22
            #
            # if gnode.level_id == 0:
            #     continue


            gid = gnode.gid
            G.add_node(
                nx_node_idx, edges=gnode.edge_gids, size=len(gnode.points)
            )
            node_gid_to_graph_idx[gid] = nx_node_idx
            graph_idx_to_gid[nx_node_idx] = gid
            nx_node_idx += 1

        for nx_gid, node in list(G.nodes_iter(data=True)):

            for adj_edge in node["edges"]:
                for adj_gnode_gid in gid_edge_dict[adj_edge].gnode_gids:

                    node_gid = graph_idx_to_gid[nx_gid]
                    adj_gnode_nx_id = node_gid_to_graph_idx[adj_gnode_gid]
                    if adj_gnode_gid != node_gid:
                        G.add_edge(nx_gid, adj_gnode_nx_id)

        return G.copy(), graph_idx_to_gid, node_gid_to_graph_idx

    def compute_homophily(self, G, graph_idx_to_gid, gid_to_label_dict, sup_gids_of_sublevel_set, subgraph):
        homophily_count = 0.
        homophily_train_count = 0.
        total_edges = G.size()
        total_train_edges = 0.

        sublevel_homophily_count = 0.
        sublevel_homophily_train_count = 0.
        total_sublevel_edges = 0
        total_sublevel_train_edges = 0.

        sup_G = self.G

        # sub_gid = subgraph.sup_gid_to_sub_dict[sup_gid]
        sub_gid_to_sup_gid_dict ={v: k for k, v in subgraph.sup_gid_to_sub_dict.items()}
        # dict([(subgid,supgid) for supgid,subgid in subgraph.sup_gid_to_sub_dict.items()])

        seen_nodes = []
        for nodeid in G.nodes():
            node_gid = graph_idx_to_gid[nodeid]

            seen_nodes.append(node_gid)
            self_label = gid_to_label_dict[node_gid]
            # neighbor_nxids = [neighbor for neighbor in G.neighbors(nodeid)]
            neighbor_gids = [(neighbor, graph_idx_to_gid[neighbor]) for neighbor in G.neighbors(nodeid)]
            neighbor_labels = [gid_to_label_dict[neighbor[1]] for neighbor in neighbor_gids]
            for nl, n_ids in zip(neighbor_labels, neighbor_gids):
                n_gid = n_ids[1]
                n_nxid = n_ids[0]
                # nbr_node = G.node[n_nxid]
                # sup_gid = sub_gid_to_sup_gid_dict[n_gid]    ## !!! why doesnt this work
                # nbr_is_train = sup_gid in sup_gids_of_sublevel_training_set
                nbr_is_train = self.node_gid_to_partition[n_gid] == 'train'

                # sup_partition = self.node_gid_to_partition[sup_gid]
                # nx_gid = subgraph.node_gid_to_graph_idx[n_gid]
                # node = subgraph.G.node[nx_gid]
                if n_gid in sup_gids_of_sublevel_set:
                    total_sublevel_edges += 1

                if n_gid not in seen_nodes and nbr_is_train:
                    total_train_edges += 1
                    if n_gid in sup_gids_of_sublevel_set:
                        total_sublevel_train_edges += 1

                if n_gid not in seen_nodes and self_label == nl:
                    homophily_count += 1.
                    if n_gid in sup_gids_of_sublevel_set:
                        sublevel_homophily_count += 1
                    if nbr_is_train:
                        homophily_train_count += 1
                        if n_gid in sup_gids_of_sublevel_set:
                            sublevel_homophily_train_count += 1
        homophily_ratio = homophily_count/float(total_edges)
        homophily_train_ratio = homophily_train_count/float(total_train_edges)

        sublevel_homophily_ratio = sublevel_homophily_count/float(total_sublevel_edges)
        sublevel_homophily_train_ratio = sublevel_homophily_train_count/float(total_sublevel_train_edges)
        return homophily_ratio, homophily_train_ratio, sublevel_homophily_ratio, sublevel_homophily_train_ratio

    def compute_class_balance(self, G, graph_idx_to_gid, gid_to_label_dict, sup_gids_of_sublevel_set):
        positive_count = 0.
        negative_count = 0.

        positive_count_train = 0.
        negative_count_train = 0.

        positive_sublevel_count = 0.
        negative_sublevel_count = 0.
        positive_sublevel_count_train = 0.
        negative_sublevel_count_train = 0.

        total_nodes = G.number_of_nodes()
        seen_nodes = []

        check = 0

        for nodeid in G.nodes():
            node_gid = graph_idx_to_gid[nodeid]
            seen_nodes.append(node_gid)
            self_label = gid_to_label_dict[node_gid]
            # neighbor_gids = np.array([graph_idx_to_gid[neighbor]
            #                          for neighbor in G.neighbors(nodeid)])
            # neighbor_labels = np.array([gid_to_label_dict[neighbor]
            #                             for neighbor in neighbor_gids])
            # for nl, n_gid in zip(neighbor_labels, neighbor_gids):
            node_is_train = self.node_gid_to_partition[node_gid] == 'train'

            node = G.node[nodeid]
            if self_label[1] > 0:#n_gid not in seen_nodes and self_label == nl:
                check+=1
                positive_count += 1
                if node_gid in sup_gids_of_sublevel_set:
                    positive_sublevel_count += 1
                if node_is_train:
                    positive_count_train += 1
                    if node_gid in sup_gids_of_sublevel_set:
                        positive_sublevel_count_train += 1
            else:
                negative_count += 1
                if node_gid in sup_gids_of_sublevel_set:
                    negative_sublevel_count += 1
                if node_is_train:
                    negative_count_train += 1
                    if node_gid in sup_gids_of_sublevel_set:
                        negative_sublevel_count_train += 1
            # if check < 10:
            #     pout(("label observed", self_label))
        #                0             1            2                  3                    4                     5                       6                          7                                    8                             9
        return positive_count, negative_count, total_nodes, positive_count_train, negative_count_train,  positive_sublevel_count, negative_sublevel_count, len(sup_gids_of_sublevel_set), positive_sublevel_count_train, negative_sublevel_count_train

    #@profile
    def build_geto_adj_list(self, influence_type='weighted'):
        print("    * : influence type", influence_type)


        #
        # compute gepometric / topo attributes
        # for geto adjacency matrix
        def row_major_index(row, col , num_col):
            lin_index = row*num_col + col
            return lin_index


        self.getoelms = []#np.ones((num_nodes,num_nodes))
        self.geto_attr_names = []
        self.lin_adj_idx_to_getoelm_idx = {}
        #self.lin_adj_idx_to_getoelm_idx = []
        lin_getoelm_idx = 0

        check = 10

        point_set = get_points_from_vertices(self.gid_gnode_dict.values())

        inverse_covariance_points = np.linalg.inv(np.cov(point_set.T))
        centroid = get_centroid(point_set)

        # add geto attributes to neighbors geto_attr_vec
        def add_name(name, geto_attr_names):
            if len(self.getoelms) == 0:
                geto_attr_names.append(name)
            return geto_attr_names

        def add_geto_attributes_to_self_vec(attribute_set, names, all_geto_attr, geto_attr_names):
            for attr, name in zip(attribute_set, names):
                all_geto_attr.append(attr)
                geto_attr_names = add_name(name, geto_attr_names)
            return all_geto_attr , geto_attr_names

        prod_nbr_attr = []
        sum_nbr_attr = []
        nbr_attr_names = []
        computed_self_attributes = []
        self_attr_names = []

        # build sampled points matrix
        """
                        DEAL WITH THIS LATER
        """
        sphere_file = os.path.join(self.experiment_folder,'features', "spherical_coordinates.txt")
        if os.path.isfile(sphere_file):
            spherical_embedding, gid_to_sampled_point = self.read_gnode_spherical_coordinates()
        else:
            gid_to_sampled_point = {}
            sample_idx = 0
            sampled_points = []
            #sampled_points.append(list(centroid)[0])
            for gnode in self.gid_gnode_dict.values():
                sampled_arc_points = np.array(get_points_from_vertices([gnode],sampled=True))
                x1, y1 = sampled_arc_points[0]
                x2, y2 = sampled_arc_points[-1]
                gnode_rows = [sample_idx]
                sampled_points.append(sampled_arc_points[0])
                sample_idx += 1
                gnode_rows.append(sample_idx)
                gid_to_sampled_point[gnode.gid] = gnode_rows
                sampled_points.append(sampled_arc_points[-1])
                sample_idx += 1
            sampled_points = np.array(sampled_points)
            # compute K nearest neighbors to save time with spherical
            # embedding
            knn = nearest_neighbors(sampled_points,
                                    n_neighbors=350,
                                    metric="haversine",
                                    angular=True,
                                    metric_kwds=None,
                                    random_state=None,
                                    low_memory=True,
                                    verbose=True)
            spherical_embedding = umap.UMAP(output_metric='haversine',
                                            n_neighbors=350,
                                            n_components=2,
                                            min_dist=0.01,
                                            precomputed_knn=knn,
                                            angular_rp_forest=True,
                                            transform_seed=801,
                                            low_memory=True,
                                            set_op_mix_ratio=1.0,
                                            target_metric='l1',
                                            negative_sample_rate = 2,
                                            target_n_neighbors = 4,
                                            n_epochs=20,
                                            random_state=801,
                                            verbose=True).fit(sampled_points)
            spherical_embedding = spherical_embedding.embedding_
            self.write_gnode_spherical_coordinates(spherical_embedding, gid_to_sampled_point)



        #projection onto R3 sphere
        sphere_x = np.sin(spherical_embedding[:, 0]) * np.cos(spherical_embedding[:, 1])
        sphere_y = np.sin(spherical_embedding[:, 0]) * np.sin(spherical_embedding[:, 1])
        sphere_z = np.cos(spherical_embedding[:, 0])

        proj_2d_x = np.arctan2(sphere_x,sphere_y)
        proj_2d_y = -np.arccos(sphere_z)

        # for nodes with one incident edge
        # one_edge_gnode = list(self.gid_gnode_dict.values())[0].copy()
        # one_edge_gnode.points = [(1., 1.), (1., 1.), (1., 1.)]
        # one_edge_gnode.gid = -1

        max_deg = 0
        isolated = 0

        for gnode in self.gid_gnode_dict.values():

            geto_attr_vec = []



            num_nodes = len(self.gid_gnode_dict.values())
            sampled_arc_points = np.array(get_points_from_vertices([gnode],sampled=True))
            node_translated_vec = translate_points_by_centroid([gnode], centroid)
            length_gnode = len(gnode.points)

            gnode_nx_idx = self.node_gid_to_graph_idx[gnode.gid]

            num_geto_features = 0

            #
            # selg geto attributes
            #
            node_degree = gnode.degree

            hyperbolic_distance_arc = hyperbolic_distance_line(gnode.points)
            end_to_end_hyperbolic_grad, end_to_end_hyperbolic_dist = hyperbolic_distance(gnode.points[0],
                                                                                         gnode.points[-1])
            p1_centroid_hyperbolic_grad, p1_centroid_hyperbolic_dist = hyperbolic_distance(gnode.points[0],
                                                                                         centroid[0])
            p2_centroid_hyperbolic_grad, p2_centroid_hyperbolic_dist = hyperbolic_distance(gnode.points[-1],
                                                                                           centroid[0])

            # cosim
            centroid_norm_vec = np.array([0.0, linalg.norm(np.array(centroid), axis=0)])
            cos_sim_numerator = linalg.norm(sampled_arc_points, axis=0) @ centroid_norm_vec
            cos_sim_denominator = linalg.norm(sampled_arc_points) * linalg.norm(centroid_norm_vec)
            cos_sim_centroid = cos_sim_numerator / (cos_sim_denominator+1e-9)
            cos_sim_centroid_max = np.max(cos_sim_2d(sampled_arc_points,centroid))
            cos_sim_centroid_min = np.min(cos_sim_2d(sampled_arc_points,centroid))

            # velocity
            velocity_arc_x = np.gradient(sampled_arc_points[:,0])
            velocity_arc_y = np.gradient(sampled_arc_points[:,1])
            velocity_arc = list(np.array([[velocity_arc_x[i], velocity_arc_y[i]] for i in range(velocity_arc_y.size)]).flatten())

            euclidean_sum_self = sum_euclid(sampled_arc_points)#gnode.points)
            euclidean_dist_arc = end_to_end_euclid(sampled_arc_points)#gnode.points)
            manhattan_distance_arc = manhattan_distance(gnode.points)
            manhattan_distance_centroid = manhattan_distance(sampled_arc_points, centroid)
            length_self = len(gnode.points)
            slope_self = slope(gnode.points)
            self_distance_from_centroid = cumulative_distance_from_centroid(sampled_arc_points,
                                                                            centroid)
            mahalanobis_distance_self = mahalanobis_distance_arc(gnode.points[0],
                                                                gnode.points[-1],
                                                                inverse_covariance_points) if len(gnode.points) >= 2 else 0
            mahalanobis_distance_self0_centroid = mahalanobis_distance_arc(sampled_arc_points[0],
                                                                 centroid[0],
                                                                 inverse_covariance_points)
            mahalanobis_distance_self_mid_centroid = mahalanobis_distance_arc(sampled_arc_points[1],
                                                                           centroid[0],
                                                                           inverse_covariance_points)
            mahalanobis_distance_self_end_centroid = mahalanobis_distance_arc(sampled_arc_points[-1],
                                                                           centroid[0],
                                                                           inverse_covariance_points)
            centroid_translated_points_self = translate_points_by_centroid([gnode],
                                                                           centroid)

            computed_self_attributes = []
            #self_attr_names = []
            # for i, p in enumerate(centroid_translated_points_self.flatten()):
            #     computed_self_attributes.append(p)
            #     if len(self.getoelms) == 0:
            #         self_attr_names += ['geom-self_centroid_coord_' + str(i)]
            # for i, p in enumerate(sampled_arc_points):
            #     computed_self_attributes.append(np.sqrt((p[0]-centroid[0][0])**2+(p[1]-centroid[0][1])**2))
            #     if len(self.getoelms) == 0:
            #         self_attr_names += ['geom-sampled_points_euclid_dist_centroid_' + str(i)]
            #
            # # new
            # for i, p in enumerate(sampled_arc_points.flatten()):
            #     computed_self_attributes.append(p)
            #     if len(self.getoelms) == 0:
            #         self_attr_names += ['sampled_points_' + str(i)]

            '''computed_self_attributes += list(end_to_end_hyperbolic_grad.flatten())
            computed_self_attributes += list(p1_centroid_hyperbolic_grad.flatten())
            computed_self_attributes += list(p2_centroid_hyperbolic_grad.flatten())'''

            computed_self_attributes += list(end_to_end_hyperbolic_dist.flatten())
            # computed_self_attributes += list(p1_centroid_hyperbolic_dist.flatten())
            # computed_self_attributes += list(p2_centroid_hyperbolic_dist.flatten())
            #
            if len(self.getoelms) == 0:
                '''self_attr_names += ['hyperbolic-grad_e2e'+str(i) for i in
                                    range(len(list(end_to_end_hyperbolic_grad.flatten())))]
            #     self_attr_names += ['hyperbolic-grad_p1'+str(i) for i in
            #                         range(len(list(p1_centroid_hyperbolic_grad.flatten())))]
            #     self_attr_names += ['hyperbolic-grad_p2'+str(i) for i in
            #                         range(len(list(p2_centroid_hyperbolic_grad.flatten())))]'''
                self_attr_names += ['geom-hyperbolic-dist_e2e'+str(i) for i in
                                     range(len(list(end_to_end_hyperbolic_dist.flatten())))] #new
            #     self_attr_names += ['geom-hyperbolic-dist_p1' + str(i) for i in
            #                         range(len(list(p1_centroid_hyperbolic_dist.flatten())))] #0
            #     self_attr_names += ['geom-hyperbolic-dist_p2' + str(i) for i in
            #                          range(len(list(p2_centroid_hyperbolic_dist.flatten())))] #0

            for i,attr_vec in enumerate(velocity_arc):

                attr_vec = list(attr_vec.flatten())
                computed_self_attributes += attr_vec
                if len(self.getoelms) == 0:
                    pout(["velo length", len(attr_vec)])
                    self_attr_names += ['geom-velocity_arc_' + str(i)]


            # angle between unit vectors
            x1, y1 = sampled_arc_points[0]
            x2, y2 = sampled_arc_points[-1]
            vp = np.array([x2 - x1, y2 - y1])
            veclength = np.sqrt(vp[0] * vp[0] + vp[1] * vp[1])
            thetax = np.arccos(vp[0] / (veclength+1e-9))

            # project onto S1
            vec_scaled = (1.0/(veclength+1e-9)) * vp

            for i, attr_vec in enumerate(vec_scaled):
                computed_self_attributes.append( attr_vec )
                if len(self.getoelms) == 0:
                    self_attr_names.append('geom-vec_scaled_' + '_' + str(i))

            # hyperbolic dist vec
            vec_centroid_hyperbolic_grad, vec_centroid_hyperbolic_dist = hyperbolic_distance(vp,
                                                                                           centroid[0])
            # computed_self_attributes += list(vec_centroid_hyperbolic_dist.flatten())
            # if len(self.getoelms) == 0:
            #     self_attr_names += ['geom-hyperbolic-dist_vec' + str(i) for i in
            #                         range(len(list(vec_centroid_hyperbolic_dist.flatten())))]

            # whitney embedding M^n -> R^(2n+1)
            # taking each 1-d coord t_i of (t_x,t_y) and mapping to (1/(1+t^2)) , t-(2t/(1+t^2))
            whitney_embed_p0_p1 = []
            for t in [x1,y1,x2,y2]:
                whit_p = [1./(1.+t**2) , t-((2.*t)/(1.+t**2))]
                whitney_embed_p0_p1 += whit_p
            # computed_self_attributes += whitney_embed_p0_p1
            # if len(self.getoelms)==0:
            #     self_attr_names += ['geom-whiteney_embedding_p'+str(i) for i in range(len(whitney_embed_p0_p1))]

            #whitney embedding R2 -> R5
            whitney_embed_R5 = []
            for t in [(x1,y1),(x2,y2)]:
                whit_p = [1./((1.+t[0]**2)*(1.+t[1]**2)) , t[0]-((2.*t[0])/((1.+t[0]**2)*(1.+t[1]**2))),
                          (t[0]*t[1])/((1.+t[0]**2)*(1.+t[1]**2)),t[1]]
                whitney_embed_R5 += whit_p
            #computed_self_attributes += whitney_embed_R5
            #if len(self.getoelms) == 0:
            #    self_attr_names += ['whiteney_embedding_R5_p' + str(i) for i in range(len(whitney_embed_R5))]

            # location of gnode in spherical embedding of all gnodes
            gnode_points_row = gid_to_sampled_point[gnode.gid]
            sphere_emb_p0 = spherical_embedding[gnode_points_row[0]]
            sphere_emb_p1 = spherical_embedding[gnode_points_row[1]]
            sphere_emb_gnode = list(sphere_emb_p0)+list(sphere_emb_p1)
            # projection of spherical embedding into R3
            row1 = gnode_points_row[0]
            row2 = gnode_points_row[1]
            sphere_embedding_R3 = [sphere_x[row1], sphere_y[row1],sphere_z[row1],
                                   sphere_x[row2], sphere_y[row2],sphere_z[row2]]
            proj_2d_x_gnode_1 = proj_2d_x[row1]
            proj_2d_y_gnode_1 = proj_2d_y[row1]
            proj_2d_x_gnode_2 = proj_2d_x[row2]
            proj_2d_y_gnode_2 = proj_2d_y[row2]

            # computed_self_attributes += sphere_emb_gnode
            # if len(self.getoelms) == 0:
            #     self_attr_names += ['geom-sphere_embedding_p'+str(i) for i in range(len(sphere_emb_gnode))]
            # computed_self_attributes += sphere_embedding_R3
            # if len(self.getoelms) == 0:
            #     self_attr_names += ['geom-R3_sphere_embedding_p' + str(i) for i in range(len(sphere_embedding_R3))]

            #print
            if check > 5:
                pout(["vp", vp, 'x1', x1, 'y1', y1, 'thetax', thetax, 'veclength', veclength,
                      'spherical embedding',sphere_emb_gnode])
                pout(['hyperbolic dist arc',hyperbolic_distance_arc])
                check -= 1

            computed_self_attributes += [
                                    cos_sim_centroid_max,
                                    # cos_sim_centroid_min,
                                   hyperbolic_distance_arc,
                                   #node_degree,
                                   length_self,
                                   # slope_self,
                                   # euclidean_sum_self,
                                   # euclidean_dist_arc,
                                   # self_distance_from_centroid,
                                   manhattan_distance_arc,
                                   # manhattan_distance_centroid,
                                   mahalanobis_distance_self,
                mahalanobis_distance_self0_centroid,
                mahalanobis_distance_self_mid_centroid,
                mahalanobis_distance_self_end_centroid,
                # proj_2d_x_gnode_1,
                # proj_2d_y_gnode_1,
                # proj_2d_x_gnode_2,
                # proj_2d_y_gnode_2,
                                   thetax,
                                   vp[0],
                                   vp[1],
                                veclength]

            if len(self.getoelms) == 0:
                self_attr_names += [
                    'geom-cos_sim_centroid_max', #0
                    # 'geom-cos_sim_centroid_min',
                         'geom-hyperbolic_distance_self_arc', #0
                         #'node_degree', #0
                         'geom-length_self', #0
                         # 'geom-slope_self',
                         # 'geom-euclidean_sum_self', #0
                         # 'geom-euclidean_dist_arc', # 0
                         # 'geom-self_distance_from_centroid', #0
                         'geom-manhattan_distance_arc',
                         # 'geom-manhattan_distance_centroid',
                         'geom-mahalanobis_distance_self',
                    'geom-mahalanobis_distance_self0_centroid',
                    'geom-mahalanobis_distance_self_mid_centroid',
                    'geom-mahalanobis_distance_self_end_centroid',
                    # 'proj_2d_x_gnode_1',
                    # 'proj_2d_y_gnode_1',
                    # 'proj_2d_x_gnode_2',
                    # 'proj_2d_y_gnode_2',
                    'geom-thetax',
                    'geom-vec_x',
                    'geom-vec_y',
                    'geom-vec_length'
                         ]
                #pout(["sample of getoelms", computed_self_attributes,"names",self_attr_names])
                #self.geto_attr_names = self_attr_names
            prod_nbr_attr = []
            sum_nbr_attr = []
            #nbr_attr_names = []





            #
            #  Adjacency features
            #
            sanity = 0
            use_adj_edges = True
            if use_adj_edges:

                gedge_list = list(gnode.edge_gids)
                gedge_id = gedge_list[-1]
                adj_gnode_gids = self.gid_edge_dict[gedge_id].gnode_gids
                if len(adj_gnode_gids) < 2:
                    isolated += 1
                    if sanity<10:
                        pout(['adj gids', adj_gnode_gids, 'gedge_list',gedge_list, 'for gnode',gnode.gid])
                        sanity+=1
                #    adj_gnode_gids = [-1,-1]#[gnode.gid, gnode.gid]


                adj_gnode_gid = adj_gnode_gids[0] if adj_gnode_gids[0] != gnode.gid else adj_gnode_gids[1]
                if True:#for adj_edge in gnode.edge_gids:
                    #adj_gnode_gids = self.gid_edge_dict[adj_edge].gnode_gids
                    #deg = len(gnode.edge_gids)
                    #if deg > max_deg:
                    #    max_deg = deg
                    #first_neigh = 0

                    if True:#for adj_gnode_gid in adj_gnode_gids:
                        #if adj_gnode_gid == gnode.gid:
                        #    continue

                        nbr_geto_attr_vec = []

                        '''# geto feature vector attributes for learning hidden
                        # weighted representation
                        adj_gnode_nx_id = self.node_gid_to_graph_idx[adj_gnode_gid]
                        seen = (gnode_nx_idx,adj_gnode_nx_id) in self.lin_adj_idx_to_getoelm_idx.keys() or (adj_gnode_nx_id, gnode_nx_idx) in self.lin_adj_idx_to_getoelm_idx.keys()
                        if seen:
                            continue
                        if adj_gnode_nx_id == gnode_nx_idx:
                            continue'''
                        adj_gnode = self.gid_gnode_dict[adj_gnode_gid]
                        # if adj_gnode_gid != -1:
                        #     adj_gnode = self.gid_gnode_dict[adj_gnode_gid]
                        # else:
                        #     adj_gnode = one_edge_gnode

                        sampled_nbr_points = np.array(get_points_from_vertices([adj_gnode], sampled=True))

                        # get vector of adj edge
                        # angle between unit vectors
                        x1_n, y1_n = sampled_nbr_points[0]
                        x2_n, y2_n = sampled_nbr_points[-1]
                        vp_n = np.array([x2_n - x1_n, y2_n - y1_n])
                        veclength_n = np.sqrt(vp_n[0] * vp_n[0] + vp_n[1] * vp_n[1])
                        theta_nbr = np.arccos((vp[0]*vp_n[0] + vp[1]*vp_n[1]) / (veclength * veclength_n + 1e-9))

                        # cosine similarity
                        cos_sim = np.cos(theta_nbr)

                        nbr_translated_vec = translate_points_by_centroid([adj_gnode],centroid)

                        #cos_sim_numerator = linalg.norm(node_translated_vec,axis=0) @ linalg.norm(nbr_translated_vec,axis=0)
                        #cos_sim_denominator = linalg.norm(node_translated_vec)*linalg.norm(nbr_translated_vec)
                        #cos_sim = cos_sim_numerator/cos_sim_denominator if cos_sim_denominator != 0 else 0.0

                        end_to_adj_hyperbolic_grad, end_to_adj_hyperbolic_dist = hyperbolic_distance(gnode.points[0],
                                                                                                     adj_gnode.points[-1])
                        p1_adj_hyperbolic_grad, p1_adj_hyperbolic_dist = hyperbolic_distance(gnode.points[-1],
                                                                                             adj_gnode.points[-1])
                        p3_adj_hyperbolic_grad, p3_adj_hyperbolic_dist = hyperbolic_distance(gnode.points[0],
                                                                                             adj_gnode.points[-1])


                        def twod_dot(x, y):
                            dot = 0
                            for i,j in zip(x,y):
                                dot += np.dot(i,j)
                            return dot

                        dot_vecs = twod_dot(np.array(gnode.points), np.array(adj_gnode.points))
                        #denom_prod_norm = linalg.norm(np.array(gnode.points)) * linalg.norm(np.array(adj_gnode.points).T)
                        #inv_cos = np.clip(cos_sim_numerator/cos_sim_denominator,-1,1) if cos_sim_denominator != 0  else 0.0
                        #angle_adj = m.degrees(np.arccos(inv_cos))
                        #angle_adj = 90 if np.isnan(angle_adj) else angle_adj

                        triangle_area = 0.5 * linalg.norm(np.array(gnode.points)) * linalg.norm(np.array(adj_gnode.points)) * np.sin((theta_nbr*np.pi)/180.)
                        #triangle_area2 = 0.5 * linalg.norm(np.array(centroid_translated_points_self) @ np.array(nbr_translated_vec).T)

                        # print
                        if check > 5:
                            pout(['triangle area', triangle_area, 'theta', theta_nbr,
                                  'hyperbolic_dist_nbr', p3_adj_hyperbolic_dist])

                        euclidean_dist_btwn_arcs_adj1_self1 = end_to_end_euclid([gnode.points[-1],
                                                                                 adj_gnode.points[-1]])
                        euclidean_dist_btwn_arcs_adj1_self0 = end_to_end_euclid([gnode.points[0],
                                                                                 adj_gnode.points[-1]])
                        euclidean_dist_btwn_arcs_adj0_self1 = end_to_end_euclid([gnode.points[-1],
                                                                                 adj_gnode.points[0]])

                        mahalanobis_distance_adj = mahalanobis_distance_arc(adj_gnode.points[0],
                                                                           adj_gnode.points[-1],
                                                                           inverse_covariance_points) if len(adj_gnode.points) >= 2 else 0
                        mahalanobis_distance_adj = mahalanobis_distance_adj #- mahalanobis_distance_self
                        mahalanobis_distance_adj1_self0 = mahalanobis_distance_arc(gnode.points[0],
                                                                             adj_gnode.points[-1],
                                                                             inverse_covariance_points) if len(adj_gnode.points) >= 2 else 0
                        mahalanobis_distance_adj1_self1 = mahalanobis_distance_arc(gnode.points[-1],
                                                                             adj_gnode.points[-1],
                                                                             inverse_covariance_points) if len(adj_gnode.points) >= 2 and len(gnode.points) >= 2 else 0
                        # #mahalanobis_distance_adj0_self0 = mahalanobis_distance_arc(gnode.points[0],
                        # #                                                           adj_gnode.points[0],
                        # #                                                           inverse_covariance_points) if len(adj_gnode.points) >= 2 else 0
                        mahalanobis_distance_adj0_self1 = mahalanobis_distance_arc(gnode.points[-1],
                                                                                    adj_gnode.points[0],
                                                                                    inverse_covariance_points) if len(adj_gnode.points) >= 2 and len(gnode.points) >= 2 else 0


                        computed_nbr_attributes = [end_to_adj_hyperbolic_dist,
                                                   p1_adj_hyperbolic_dist,
                                                   p3_adj_hyperbolic_dist,
                                                ##dot_vecs, #fi = 0
                                                theta_nbr,
                                                cos_sim, #fi = 0
                                                #cos_sim_numerator, #fi = 0
                                                #cos_sim_denominator, #fi = 0
                                                #triangle_area,#fi = 0
                                                #triangle_area2, #fi = 0
                                                #angle_adj, #fi = 0
                                                euclidean_dist_btwn_arcs_adj1_self1,
                                                euclidean_dist_btwn_arcs_adj1_self0,#fi = 0
                                                euclidean_dist_btwn_arcs_adj0_self1,
                                                mahalanobis_distance_adj1_self0, #fi = 0
                                                mahalanobis_distance_adj1_self1, #fi = 0
                                                mahalanobis_distance_adj0_self1
                        ]
                        if len(self.getoelms) == 0:# and first_neigh == 0:
                            self_attr_names += ['geom-end_to_adj_hyperbolic_dist',
                                                'geom-p1_adj_hyperbolic_dist',
                                                'geom-p3_adj_hyperbolic_dist',
                                               ##'geom-dot_vecs',
                                               'geom-theta',
                                               'geom-cos_sim',
                                               #'cos_sim_numerator',
                                               #'cos_sim_denominator',
                                               #'triangle_area',
                                               #'triangle_area2',
                                               #'angle_adj',
                                               'geom-euclidean_dist_btwn_arcs_adj1_self1',
                                               'geom-euclidean_dist_btwn_arcs_adj1_self0',
                                               'geom-euclidean_dist_btwn_arcs_adj0_self1',
                                               'geom-mahalanobis_distance_adj1_self0',
                                               'geom-mahalanobis_distance_adj1_self1',
                                               'geom-mahalanobis_distance_adj0_self1'
                                ]
                            #self.geto_attr_names += nbr_attr_names
                        #first_neigh += 1


                        prod_nbr_attr = []
                        sum_nbr_attr += computed_nbr_attributes


                #nbr_edge_attributes =  sum_nbr_attr #+ prod_nbr_attr
                computed_self_attributes += computed_nbr_attributes#sum_nbr_attr
                #self_attr_names += nbr_attr_names


            geto_attr_vec = computed_self_attributes# + nbr_edge_attributes
            #nbr_attributes =  nbr_attributes.flatten()




            if influence_type == 'weighted':
                geto_attr_vec = [np.mean(geto_attr_vec)]

            #geto_attr_vec = list(-1 * np.log(geto_attr_vec))

            self.node_gid_to_geom_feature[gnode.gid] = geto_attr_vec

            self.getoelms.append(geto_attr_vec)
            node_graph_idx = self.node_gid_to_graph_idx[gnode.gid]
            self.gid_to_getoelm_idx[gnode.gid] = lin_getoelm_idx
            self.lin_adj_idx_to_getoelm_idx[gnode_nx_idx] = lin_getoelm_idx
            lin_getoelm_idx += 1


            num_geto_features = len(geto_attr_vec)

        self.geto_attr_names = self_attr_names


        pout(["Number of Single Nodes", isolated])

        #
        # padding and scaling of features to be more uniform due to large outliers
        #
        attr_idx = 0
        variable_feat_lengths = []
        variable_feat_lengths_only = []
        for f_num, geto_attr in enumerate(self.getoelms):
            variable_feat_lengths.append((attr_idx, len(geto_attr) - 1))
            variable_feat_lengths_only.append(len(geto_attr))
            attr_idx += len(geto_attr)-1

        max_feat_vec_length = np.max(variable_feat_lengths_only)
        getoelms = []


        self.getoelms = np.array(self.getoelms).astype(dtype=np.float32)

        pad_for_neighbors = False
        scale_feat_distribution = False
        # scale for wide range / outlier features
        # scaler = QuantileTransformer(n_quantiles=len(self.getoelms[0]), output_distribution='normal')
        # scaler = RobustScaler(with_scaling=True, with_centering=True,unit_variance=False)
        if scale_feat_distribution:
            scaler = RobustScaler()#with_scaling=True, with_centering=True, unit_variance=False)
            #for getoelm in self.getoelms:

            self.getoelms = scaler.fit_transform(self.getoelms)#np.array(getoelm).reshape(-1, 1))
            # getoelm = list(np.array(getoelm).flatten())

            # if pad_for_neighbors:
            #     pad_size = max_feat_vec_length - len(getoelm)
            #     for i,z in enumerate(range(pad_size)):
            #         print("    *")
            #         print("    * : ADDING PADDING")
            #         print("    *")
            #         getoelm.append(1.0)
            #
            # # getoelms.append(getoelm)
            #
            # if pad_for_neighbors:
            #     pad_size_names = max_feat_vec_length - len(self.geto_attr_names)
            #     for i, z in enumerate(range(pad_size_names)):
            #         self.geto_attr_names.append(nbr_attr_names[i % len(nbr_attr_names)])
            # self.getoelms = getoelms


        # self.getoelms = np.array(self.getoelms).astype(dtype=np.float32)


        '''getoelms = []
        self.getoelms = scaler.fit_transform(self.getoelms)
        for getoelm in self.getoelms:
            pad_size =  max_feat_vec_length - len(getoelm)
            for i,z in enumerate(range(pad_size)):
                reverse_idx = -1-i
                getoelm[reverse_idx]=1.
            getoelms.append(getoelm)
            if check < 6:
                print("scaled:")
                print(getoelm)
                print("")
                #print("sanity")
                #print(sanity_check[check])
                #print('og:')
                #print(geto_copy[check])
                check += 1
        self.getoelms = getoelms
        self.getoelms = np.array(self.getoelms).astype(dtype=np.float32)
        #self.getoelms[np.isnan(self.getoelms)] = 0.'''
        return self.getoelms

    def compile_features(self, image=None, return_labels=False, include_geto=False,
                         include_generic_feat = True,
                         save_filtered_images=False,
                         min_number_features=1, number_features=5, selection=None):

        start_time = time.time()

        self.set_default_features()

        if include_geto:
            if self.getoelms is None:
                self.getoelms = self.build_geto_adj_list(influence_type=self.params['geto_influence_type'])


        gnode_features = []
        feature_names = []
        feature_idx = 0
        feature_order = 0

        check = 0
        # skip powers
        skip_exp = ['feat-func_19_var','variance_2_mode','feat-func_19_mode',
                    'feat-func_17_var','feat-func_17_mode','feat-func_16_var',
                    'feat-func_16_mode','feat-func_20_var','feat-func_14_var',
                    'feat-func_13_var','feat-func_13_mode','feat-func_11_mode',
                    'feat-func_10_mode','feat-func_8_mode','feat-func_14_mode',
                    'feat-func_20_mode']

        start_time = time.time()

        for gnode in self.gid_gnode_dict.values():

            gnode_feature_row = []

            if include_generic_feat:
                # pout((gnode.points, "gnodepoints"))
                gnode_points = np.round(np.array(gnode.points, dtype=float)).astype(int)
                # pout((gnode_points[:,1], "gnode   sdgf 1 points"))
                # pout((self.feature_image.shape, "feat im shape"))
                gnode_aug_imstack = self.feature_image[gnode_points[:, 1], gnode_points[:, 0], :]
                # pout((gnode_aug_im_pixels.shape, "gnode_aug im shape"))
                # aug_im_names = list(self.images.keys())
                for im_idx in range(self.feature_image.shape[2]):
                    # for image_name, im in self.images.items():
                    image_name = self.feature_im_names[im_idx]

                    gnode_pixels = gnode_aug_imstack[:, im_idx]  # get_pixel_values_from_vertices([gnode], im)

                    for function_name, foo in self.feature_ops.items():
                        aug_name = image_name + "_" + function_name
                        if aug_name not in skip_exp:
                            attribute = foo(gnode_pixels)
                            gnode_feature_row.append(attribute)
                            if len(gnode_features) == 0:
                                feature_names.append(image_name + "_" + function_name)
                                self.fname_to_featidx[image_name + "_" + function_name] = feature_order
                                feature_order += 1
                self.node_gid_to_standard_feature[gnode.gid] = gnode_feature_row

            # if include_geto:
            #    gnode_feature_row = np.array(gnode_feature_row)
            #    gnode_nx_idx = self.node_gid_to_graph_idx[gnode.gid]
            #    gnode_feature_row = np.hstack((gnode_feature_row, self.getoelms[gnode_nx_idx,:]))
            gnode_features.append(gnode_feature_row)
            gnode.features = np.array(gnode_feature_row).astype(dtype=np.float32)
            self.node_gid_to_feature[gnode.gid] = np.array(gnode_feature_row).astype(dtype=np.float32)
            self.node_gid_to_feat_idx[gnode.gid] = feature_idx

            feature_idx += 1

        end_time = time.time()

        self.feature_comp_time += end_time - start_time

        gnode_features = np.array(gnode_features).astype(dtype=np.float32)

        # if include_geto:
        #    gnode_features = np.hstack((gnode_features, self.getoelms))

        self.number_features = len(feature_names)

        self.feature_names = feature_names
        self.features = gnode_features

        if return_labels:
            print(">>>> returning with labels")
            return gnode_features, feature_names
        else:
            print(" returning without labels")
            self.features = gnode_features
            self.compiled_data = gnode_features
            return gnode_features

    def construct_feature_ops(self):
        self.feature_ops = {}
        self.feature_ops["min"] = lambda pixels: np.min(pixels)
        self.feature_ops["max"] = lambda pixels: np.max(pixels)
        self.feature_ops["median"] = lambda pixels: np.median(pixels)
        # self.feature_ops["mode"] = lambda pixels: scipy.stats.mode(np.round(pixels, 2))[0][0]
        # self.feature_ops["mean"] = lambda pixels: gaussian_fit(pixels)[0]

        self.feature_ops["std"] = lambda pixels: gaussian_fit(pixels)[1]
        # self.feature_ops["var"] = lambda pixels: gaussian_fit(pixels)[2]

        # self.feature_ops["skew"] = lambda pixels: scipy.stats.skew(pixels)
        # self.feature_ops["kurtosis"] = lambda pixels: scipy.stats.kurtosis(pixels)
        # self.feature_ops["range"] = lambda pixels: np.max(pixels) - np.min(pixels)

    def set_default_features(self):

        image = self.image
        min_number_features = self.params['min_number_features']
        number_features = self.params['number_features']

        image_og = image  # copy.deepcopy(image)
        # image_c = copy.deepcopy(image)
        self.images = {}
        self.feature_image = np.expand_dims(image, axis=-1)
        self.feature_im_names = []
        self.feature_im_names.append('identity')

        def add_feat_im(image, name):
            image = np.expand_dims(image, axis=-1)
            self.feature_image = np.concatenate((self.feature_image, image), axis=2)
            self.feature_im_names.append(name)
            return self.feature_image, self.feature_im_names

        feature_scope = [2 ** nf for nf in range(min_number_features, number_features + 1)]

        print("Feature scales: ", feature_scope)

        # Functions to apply to the pixels of an arc
        self.construct_feature_ops()

        start_time = time.time()

        # self.images["identity"] = image

        sigma_min = 1
        sigma_max = 64
        features_func = partial(multiscale_basic_features,
                                intensity=True, edges=True, texture=True,
                                sigma_min=sigma_min, sigma_max=sigma_max,
                                multichannel=False)

        image_c = image_og  # copy.deepcopy(image_og)

        features = features_func(image_c)
        self.feature_image = np.concatenate((self.feature_image, features), axis=2)

        # skip_exp = ['feat-func_20', 'feat-func_17', 'feat-func_14', 'feat-func_19',
        #             'feat-func_11', 'feat-func_13', 'feat-func_16']
        for i in range(features.shape[2]):
            # skip powers which have 0 feature importance (20, 17, 14, 19, 11, 13, 16)
            # if i not in skip_exp:
            self.feature_im_names.append('feat-func_' + str(i))
            # self.images['feat-func_'+str(i)] = features[:,:,i]

        image_c = image_og  # copy.deepcopy(image_og)
        # self.images["sobel"] = sobel_filter(image_c)
        sob_im = sobel_filter(image_c)
        self.feature_image, self.feature_im_names = add_feat_im(sob_im, 'sobel')
        image_c = image_og  # copy.deepcopy(image_og)

        for i in feature_scope:
            pow1 = 2 * i  # 2 ** (i - 1)
            pow2 = i

            # self.images["variance_{}".format(i)] = \
            var_filt_i = variance_filter(
                image_c, i
            )
            self.feature_image, self.feature_im_names = add_feat_im(var_filt_i, 'variance_' + str(i))
            # image_c = image_og#copy.deepcopy(image_og)

            # self.images["median_{}".format(i)] = \
            med_i = median_filter(image_c, i)
            self.feature_image, self.feature_im_names = add_feat_im(med_i, 'median_' + str(i))
            # image_c = image_og#copy.deepcopy(image_og)

            # self.images["min_{}".format(i)] = \
            min_i = minimum_filter(image_c, i)
            # image_c = image_og#copy.deepcopy(image_og)
            self.feature_image, self.feature_im_names = add_feat_im(min_i, 'min_' + str(i))

            # self.images["max_{}".format(i)] = \
            max_i = maximum_filter(image_c, i)
            self.feature_image, self.feature_im_names = add_feat_im(max_i, 'max_' + str(i))
            # image_c = image_og#copy.deepcopy(image_og)
            '''
            # self.images["gauss_{}".format(pow2)] = \
            gauss_i = gaussian_blur_filter(
                image_c, pow2
            )
            self.feature_image, self.feature_im_names = add_feat_im(gauss_i, 'gauss_'+str(i))
            # image_c = image_og#copy.deepcopy(image_og)

            # self.images[
            #     "delta_gauss_{}_{}".format(pow1, pow2)
            # ] = \
            delta_gauss_i = difference_of_gaussians_filter(image_c, pow1, pow2)
            image_c = image_og#copy.deepcopy(image_og)
            self.feature_image, self.feature_im_names = add_feat_im(delta_gauss_i, 'delta_gauss_'+str(i))
            '''

        end_time = time.time()

        self.feature_comp_time += end_time - start_time

        self.save_feature_images()
        self.write_feature_image_names()

        # for name, im in self.images.items():
        #     # pout(("image shape",im.shape))
        #     if name == 'identity':
        #         continue
        #     im = np.expand_dims(im, axis=-1)
        #     self.feature_image = np.concatenate((self.feature_image,im), axis=2)

        if self.params['save_filtered_images']:
            for name, image in self.images.items():

                if not os.path.exists(os.path.join(self.experiment_folder, 'filtered_images')):
                    os.makedirs(os.path.join(self.experiment_folder, 'filtered_images'))
                #Img.save(os.path.join(self.experiment_folder, 'filtered_images', name+'.tif'), quality=90)
                im_name = os.path.join(self.experiment_folder, 'filtered_images', name+'.png')

                plt.figure()
                plt.title("name")
                import matplotlib as mplt
                plt.imsave(im_name, image,cmap=mplt.cm.Greys_r)#astype('uint8'),
                #           cmap=mplt.cm.Greys_r)

                plt.close()

                # image = np.array(image).astype(np.uint8)
                #
                # #Img = Image.fromarray(image)
                # if not os.path.exists(os.path.join(self.experiment_folder, 'filtered_images')):
                #     os.makedirs(os.path.join(self.experiment_folder, 'filtered_images'))
                # #Img.save(os.path.join(self.experiment_folder, 'filtered_images', name+'.tif'), quality=90)
                # imageio.imsave(os.path.join(self.experiment_folder, 'filtered_images', name+'.tif'), image)
    def compile_subgraph_features(self, image=None, return_labels=False, include_geto=False,
                         include_generic_feat = True, gid_gnode_dict = None,
                         save_filtered_images=False,
                         min_number_features=1, number_features=5, selection=None):

        start_time = time.time()

        # self.set_default_features()

        # if include_geto:
        #     if self.getoelms is None:
        #         self.getoelms = self.build_geto_adj_list(influence_type=self.params['geto_influence_type'])


        gnode_features = []
        feature_names = []
        feature_idx = 0
        feature_order = 0

        node_gid_to_standard_feature = {}
        node_gid_to_feature = {}
        node_gid_to_feat_idx = {}

        # Functions to apply to the pixels of an arc
        self.construct_feature_ops()

        pout(("FEATRE IMAGE NAMES:", self.feature_im_names))



        check = 0
        # skip powers
        skip_exp = ['feat-func_19_var','variance_2_mode','feat-func_19_mode',
                    'feat-func_17_var','feat-func_17_mode','feat-func_16_var',
                    'feat-func_16_mode','feat-func_20_var','feat-func_14_var',
                    'feat-func_13_var','feat-func_13_mode','feat-func_11_mode',
                    'feat-func_10_mode','feat-func_8_mode','feat-func_14_mode',
                    'feat-func_20_mode']

        start_time = time.time()

        for gnode in gid_gnode_dict.values():

            gnode_feature_row = []

            if include_generic_feat:
                # pout((gnode.points, "gnodepoints"))
                gnode_points = np.round(np.array(gnode.points, dtype=float)).astype(int)
                # pout((gnode_points[:,1], "gnode   sdgf 1 points"))
                # pout((self.feature_image.shape, "feat im shape"))
                gnode_aug_imstack = self.feature_image[gnode_points[:, 1], gnode_points[:, 0], :]
                # pout((gnode_aug_im_pixels.shape, "gnode_aug im shape"))
                # aug_im_names = list(self.images.keys())
                for im_idx in range(self.feature_image.shape[2]):
                    pout((im_idx))
                    # for image_name, im in self.images.items():

                    image_name = self.feature_im_names[im_idx]

                    gnode_pixels = gnode_aug_imstack[:, im_idx]  # get_pixel_values_from_vertices([gnode], im)

                    for function_name, foo in self.feature_ops.items():
                        aug_name = image_name + "_" + function_name
                        if aug_name not in skip_exp:
                            attribute = foo(gnode_pixels)
                            gnode_feature_row.append(attribute)
                            if len(gnode_features) == 0:
                                feature_names.append(image_name + "_" + function_name)
                                # self.fname_to_featidx[image_name + "_" + function_name] = feature_order
                                feature_order += 1
                node_gid_to_standard_feature[gnode.gid] = gnode_feature_row

            # if include_geto:
            #    gnode_feature_row = np.array(gnode_feature_row)
            #    gnode_nx_idx = self.node_gid_to_graph_idx[gnode.gid]
            #    gnode_feature_row = np.hstack((gnode_feature_row, self.getoelms[gnode_nx_idx,:]))
            gnode_features.append(gnode_feature_row)
            gnode.features = np.array(gnode_feature_row).astype(dtype=np.float32)
            node_gid_to_feature[gnode.gid] = np.array(gnode_feature_row).astype(dtype=np.float32)
            node_gid_to_feat_idx[gnode.gid] = feature_idx

            feature_idx += 1

        end_time = time.time()

        # self.feature_comp_time += end_time - start_time

        gnode_features = np.array(gnode_features).astype(dtype=np.float32)

        # if include_geto:
        #    gnode_features = np.hstack((gnode_features, self.getoelms))

        # self.number_features = len(feature_names)

        # self.feature_names = feature_names
        # self.features = gnode_features

        # if return_labels:
        #     print(">>>> returning with labels")
        #     return gnode_features, feature_names
        # else:
        #     # print(" returning without labels")
        #     # self.features = gnode_features
        #     # self.compiled_data = gnode_features
        return gnode_features, node_gid_to_feature, node_gid_to_feat_idx


    def load_json_feature_graph(self):
        graph_path = os.path.join(self.pred_run_path, self.msc_graph_name)
        self.G, self.features, self.node_idx, self.walks, self.node_gid_to_label \
            , negative_sample_count \
            , positive_sample_count = load_data(prefix=graph_path)

    def write_gnode_features(self, filename='', gid_gnode_dict = None,
                             node_gid_to_graph_idx=None, node_gid_to_standard_feature=None):
        if gid_gnode_dict is None:
            gid_gnode_dict = self.gid_gnode_dict
            #if node_gid_to_graph_idx is None:
            node_gid_to_graph_idx = self.node_gid_to_graph_idx
            #if node_gid_to_standard_feature is None:
            node_gid_to_standard_feature = self.node_gid_to_standard_feature

        if not os.path.exists(os.path.join(self.experiment_folder,'features')):
            os.makedirs(os.path.join(self.experiment_folder,'features'))
        msc_feats_file = os.path.join(self.experiment_folder,'features', filename+"feats.txt")
        msc_gid_to_feats_file = os.path.join(self.experiment_folder, 'features', filename+"gid_graph_idx.txt")#"gid_feat_idx.txt")
        print("&&&& writing features in: ", msc_feats_file)
        feats_file = open(msc_feats_file, "w+")
        gid_feats_file = open(msc_gid_to_feats_file, "w+")
        lines = 1

        check = len(gid_gnode_dict.keys())
        for gnode in gid_gnode_dict.values():
            nl = '\n' if lines != len(gid_gnode_dict) else ''
            gid_feats_file.write(str(gnode.gid)+' '+str( node_gid_to_graph_idx[gnode.gid])+nl)
            #str(self.node_gid_to_feat_idx[gnode.gid])+nl)
            feature_vec = node_gid_to_standard_feature[gnode.gid]
            feats_file.write(str(gnode.gid)+ ' ')
            for f in feature_vec[0:-1]:
                feats_file.write(str(f)+' ')
            feats_file.write(str(feature_vec[-1]) + '\n')
            lines += 1

            # if check < 3:
            #     pout(["feat write end", feature_vec])
            # elif check > len(self.gid_gnode_dict.keys()) - 2:
            #     pout(["feat write begin", feature_vec])
            check -= 1

        gid_feats_file.close()
        feats_file.close()

    def write_geto_features(self, filename):
        if self.getoelms is not None:
            if not os.path.exists(os.path.join(self.experiment_folder,'features')):
                os.makedirs(os.path.join(self.experiment_folder,'features'))
            msc_feats_file = os.path.join(self.experiment_folder,'features', "geto_feats.txt")
            msc_gid_to_feats_file = os.path.join(self.experiment_folder, 'features', "gid_graph_idx.txt")#"idx_to_getoelm_idx.txt")
            print("&&&& writing geto features in: ", msc_feats_file)
            feats_file = open(msc_feats_file, "w+")
            gid_to_graph_idx = open(msc_gid_to_feats_file, "w+")
            lines = 1
            check = len(self.gid_gnode_dict.keys())
            feat_count = 0
            for gid in self.gid_gnode_dict.keys():
                gnode = self.gid_gnode_dict[gid]
                gnode_nx_idx = self.node_gid_to_graph_idx[gnode.gid]
                nl = '\n' if lines != len(self.gid_gnode_dict) else ''
                gid_to_graph_idx.write(str(gnode.gid)+' '+#str(gnode_nx_idx)+#' '+nl)#
                                     str(gnode_nx_idx)+nl)#str(self.lin_adj_idx_to_getoelm_idx[gnode_nx_idx])+nl)
                feature_vec = self.node_gid_to_geom_feature[gid]
                #self.getoelms[feat_count]#gnode_nx_idx]
                feats_file.write(str(gnode.gid)+' ')#gnode_nx_idx)+ ' ')
                for f in feature_vec[0:-1]:
                    feats_file.write(str(f)+' ')
                feats_file.write(str(feature_vec[-1]) + nl)
                lines += 1

                # if check < 3:
                #     pout(["geto feat write end", feature_vec])
                # elif check > len(self.gid_gnode_dict.keys()) - 2:
                #     pout(["geto feat write begin", feature_vec])
                check -= 1
                feat_count+=1
            gid_to_graph_idx.close()
            feats_file.close()

    def write_feature_names(self):
        msc_feats_file = os.path.join(self.experiment_folder,'features', "featnames.txt")
        print("&&&& writing feature namesin: ", msc_feats_file)
        feats_file = open(msc_feats_file, "w+")
        for fname in self.feature_names:
            feats_file.write(fname + "\n")
        feats_file.close()
    def write_feature_image_names(self):
        msc_feats_file = os.path.join(self.experiment_folder,'features', "feature_image_names.txt")
        print("&&&& writing feature namesin: ", msc_feats_file)
        feats_file = open(msc_feats_file, "w+")
        for fname in self.feature_im_names:
            feats_file.write(fname + "\n")
        feats_file.close()

    def write_geto_feature_names(self):
        if self.getoelms is not None:
            msc_feats_file = os.path.join(self.experiment_folder,'features', "geto_featnames.txt")
            print("&&&& writing geto feature namesin: ", msc_feats_file)
            feats_file = open(msc_feats_file, "w+")
            for fname in self.geto_attr_names:
                feats_file.write(fname + "\n")
            feats_file.close()

    def load_feature_names(self):
        msc_feats_file = os.path.join(self.experiment_folder, 'features', "featnames.txt")
        print("&&&& Reading feature names from: ", msc_feats_file)
        feats_file = open(msc_feats_file, "r")
        feat_lines = feats_file.readlines()
        self.feature_names = []
        for f in feat_lines:
            self.feature_names.append(f)
        #print(self.feature_names)
        feats_file.close()
        return self.feature_names
    def load_feature_image_names(self):
        msc_feats_file = os.path.join(self.experiment_folder, 'features', "feature_image_names.txt")
        print("&&&& Reading feature names from: ", msc_feats_file)
        feats_file = open(msc_feats_file, "r")
        feat_lines = feats_file.readlines()
        self.feature_im_names = []
        for f in feat_lines:
            self.feature_names.append(f)
        #print(self.feature_names)
        feats_file.close()
        return self.feature_im_names

    def load_geto_feature_names(self):
        msc_feats_file = os.path.join(self.experiment_folder, 'features', "geto_featnames.txt")
        print("&&&& Reading geto feature names from: ", msc_feats_file)
        feats_file = open(msc_feats_file, "r")
        feat_lines = feats_file.readlines()
        self.geto_attr_names = []
        for f in feat_lines:
            self.geto_attr_names.append(f)
        #print(self.feature_names)
        feats_file.close()
        return self.geto_attr_names

    def load_gnode_features(self, filename=None, gid_gnode_dict=None,
                            node_gid_to_standard_feature={},
                            node_gid_to_feat_idx={},
                            node_gid_to_graph_idx = {},
                            features=None):
        if gid_gnode_dict is None:
            gid_gnode_dict = self.gid_gnode_dict
            # if not node_gid_to_standard_feature:
            node_gid_to_standard_feature = self.node_gid_to_standard_feature
            # if not node_gid_to_feat_idx:
            node_gid_to_feat_idx = self.node_gid_to_feat_idx
            # if features is None:
            features = self.features
            # if not node_gid_to_graph_idx:
            node_gid_to_graph_idx = self.node_gid_to_graph_idx
        if filename is None:
            filename = ''



        msc_feats_file = os.path.join( self.experiment_folder,'features', filename+"feats.txt")
        print("&&&& Reading features from: ", msc_feats_file)
        feats_file = open(msc_feats_file, "r")
        feat_lines = feats_file.readlines()
        feats_file.close()
        features = []
        check = len(feat_lines)
        feat_idx = 0
        for v in feat_lines:
            gid_feats = v.split(' ')
            gid = int(gid_feats[0])
            # gnode = gid_gnode_dict[gid]
            # gnode.features = np.array(gid_feats[1:])
            node_gid_to_standard_feature[gid] = gid_feats[1:]
            #self.node_gid_to_feature[gid] = np.array(gid_feats[1:])
            features.append(np.array(gid_feats[1:]))
            # if check < 3:
            #     pout(["feat load end", gid_feats[1:]])
            # elif check > len(feat_lines) - 2:
            #     pout(["feat load begin", gid_feats[1:]])
            check -= 1
            node_gid_to_feat_idx[gid] = feat_idx
            feat_idx += 1
        features = np.array(features).astype(dtype=np.float32)

        # nx_idx_to_feat_idx = {self.node_gid_to_graph_idx[gid]: feat for gid, feat
        #                       in self.node_gid_to_feat_idx.items()}
        # nx_idx_to_label_idx = {self.node_gid_to_graph_idx[gid]: label for gid, label
        #                        in self.node_gid_to_label.items()}
        # nx_idx_to_getoelm_idx = {self.node_gid_to_graph_idx[gid]: getoelm_idx for gid, getoelm_idx
        #                          in self.gid_to_getoelm_idx.items()}

        feats_file.close()

        gid_to_feats_file = os.path.join(self.experiment_folder, 'features',filename+"gid_graph_idx.txt")
        print("&&&& reading gid to graph id in: ", gid_to_feats_file)
        feats_file = open(gid_to_feats_file, "r")
        feat_lines = feats_file.readlines()
        feats_file.close()
        for l in feat_lines:
            gid_featidx = l.split(' ')
            node_gid_to_graph_idx[int(gid_featidx[0])] = int(gid_featidx[1])
            #self.node_gid_to_feat_idx[int(gid_featidx[0])] = int(gid_featidx[1])

        return features, node_gid_to_standard_feature, node_gid_to_feat_idx, node_gid_to_graph_idx

    def load_geto_features(self):
        self.getoelms = []
        self.lin_adj_idx_to_getoelm_idx = {}
        msc_feats_file = os.path.join( self.experiment_folder,'features', "geto_feats.txt")
        print("&&&& Reading geto features from: ", msc_feats_file)
        feats_file = open(msc_feats_file, "r")
        feat_lines = feats_file.readlines()
        feats_file.close()
        features = []
        getofeatures = []
        geto_idx = 0
        check = len(feat_lines)
        check2=3
        for v in feat_lines:
            gid_feats = v.split(' ')
            #gnode_nx_idx = int(gid_feats[0])
            gid=int(gid_feats[0])
            gnode = self.gid_gnode_dict[gid]
            gid_feats_arr = np.array(gid_feats[1:])
            gid_feats_list = list(gid_feats_arr)

            gnode.features = gid_feats_arr
            self.node_gid_to_geom_feature[gid] = gid_feats_list
            #self.node_gid_to_feature[gid] = gid_feats_arr # changed in call to collect feats if combined
            #self.lin_adj_idx_to_getoelm_idx[gid] = geto_idx#gnode_nx_idx] = geto_idx

            #self.node_gid_to_feat_idx[gid] = geto_idx#features.shape[0]
            features.append(gnode.features)



            getofeatures.append(np.array(gid_feats[1:]))
            self.gid_to_getoelm_idx[gid] = geto_idx
            geto_idx += 1

            # if check < 3:
            #     pout(["geto feat load end", gid_feats[1:]])
            # elif check > len(feat_lines) - 2:
            #     pout(["geto feat load begin", gid_feats[1:]])
            check -= 1
        #self.features = np.array(features).astype(dtype=np.float32)
        self.getoelms = np.array(getofeatures).astype(dtype=np.float32)
        feats_file.close()

        gid_to_feats_file = os.path.join(self.experiment_folder, 'features', "gid_graph_idx.txt")
        print("&&&& reading gid to graph id in: ", gid_to_feats_file)
        feats_file = open(gid_to_feats_file, "r")
        feat_lines = feats_file.readlines()
        feats_file.close()
        for l in feat_lines:
            gid_featidx = l.split(' ')
            self.node_gid_to_graph_idx[int(gid_featidx[0])] = int(gid_featidx[1])



    def write_gnode_spherical_coordinates(self, spherical_coordinates, gid_spherical_dict):
        if not os.path.exists(os.path.join(self.experiment_folder,'features')):
            os.makedirs(os.path.join(self.experiment_folder,'features'))
        msc_feats_file = os.path.join(self.experiment_folder,'features', "spherical_coordinates.txt")
        print("&&&& writing features in: ", msc_feats_file)
        feats_file = open(msc_feats_file, "w+")
        lines = 1

        for gnode in self.gid_gnode_dict.values():
            gid = gnode.gid
            r1 = gid_spherical_dict[gid][0]

            r2 = gid_spherical_dict[gid][1]
            nl = '\n' if lines != len(self.gid_gnode_dict) else ''
            feature_vec = list(spherical_coordinates[r1]) + list(spherical_coordinates[r2])
            feats_file.write(str(gnode.gid)+ ' ')
            for f in feature_vec[0:-1]:
                feats_file.write(str(f)+' ')
            feats_file.write(str(feature_vec[-1]) + '\n')
            lines += 1
        feats_file.close()

    def save_feature_images(self):
        if not os.path.exists(os.path.join(self.experiment_folder,'features')):
            os.makedirs(os.path.join(self.experiment_folder,'features'))
        feat_im_file = os.path.join(self.experiment_folder,'features', 'feature_image.npy')
        feat_im_file_shape = os.path.join(self.experiment_folder, 'features', 'feature_image_shape.npy')
        np.save(feat_im_file_shape, self.feature_image.shape)
        np.save(feat_im_file, self.feature_image)
    def load_feature_images(self):
        feat_im_file = os.path.join(self.experiment_folder, 'features', 'feature_image.npy')
        feat_im_file_shape = os.path.join(self.experiment_folder, 'features', 'feature_image_shape.npy')
        self.feature_image = np.load(feat_im_file)
        feat_im_shape = np.load(feat_im_file_shape)
        self.feature_image = self.feature_image.reshape(feat_im_shape)

    def read_gnode_spherical_coordinates(self):
        #if not os.path.exists(os.path.join(self.experiment_folder,'features')):
        #    os.makedirs(os.path.join(self.experiment_folder,'features'))
        msc_feats_file = os.path.join(self.experiment_folder,'features', "spherical_coordinates.txt")
        print("&&&& reading spherical coords in: ", msc_feats_file)
        feats_file = open(msc_feats_file, "r")
        s_lines = feats_file.readlines()
        lines = 1
        embedding_coords = []
        gid_sphereical_embed_row = {}
        rcount = 0
        for line in s_lines:
            line = line.split(' ')
            gid = int(line[0])

            scoords = line[1:]#np.array(line[1:], dtype=np.float32)

            coord_sep = len(scoords)//2
            gid_sphereical_embed_row[gid] = [rcount]
            embedding_coords.append(scoords[0:coord_sep])
            rcount += 1
            embedding_coords.append(scoords[coord_sep:len(scoords)])
            # if rcount <4:
            #     pout(["line", line[1:]])
            #     pout(['lenscoords', len(scoords)])
            #     pout(['r1', scoords[0:coord_sep]])
            #     pout(['r2',scoords[coord_sep:len(scoords)]])
            #     pout(['scoords', scoords])
            gid_sphereical_embed_row[gid].append(rcount)
            rcount+=1
        feats_file.close()
        embedding_coords = np.array(embedding_coords).astype(dtype=np.float32)
        return embedding_coords, gid_sphereical_embed_row

    def write_feature_comp_time(self):
        msc_feats_file = os.path.join(self.pred_session_run_path, 'featcomp_time.txt')
        feats_time_file = open(msc_feats_file, "w+")
        feats_time_file.write(str(self.feature_comp_time)+'\n')
        feats_time_file.close()

        msc_feats_file = os.path.join(self.experiment_folder, 'features', 'featcomp_time.txt')
        feats_time_file = open(msc_feats_file, "w+")
        feats_time_file.write(str(self.feature_comp_time) + '\n')
        feats_time_file.close()

