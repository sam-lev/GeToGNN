import os
import time
import networkx as nx
import numpy as np
import math as m
import copy
import samply
import scipy.stats
import numpy.linalg as linalg
import imageio
from PIL import Image
from sklearn.preprocessing import PowerTransformer, RobustScaler, QuantileTransformer

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
)


from localsetup import LocalSetup
from ml.utils import load_data

class GeToFeatureGraph(GeToGraph):
    def __init__(self, image=None, geomsc_fname_base = None, label_file=None,
                 parameter_file_number = None, run_num=0,
                 dataset_group='neuron',write_folder="results",
                 model_name='ndlas',
                 name='neuron2', format='raw', load_feature_graph_name = None,
                 msc_file=None, dim_invert=False, map_labels=False,
                 reset_run = False, **kwargs):

        #self.LocalSetup = LocalSetup(env='slurm')


        #if parameter_file_number is not None:
        #    self.params = set_parameters(read_params_from=parameter_file_number)
        #else:
        #self.params = kwargs['params']
        #for param in kwargs:
        #    self.params[param] = kwargs[param]
        #for param in kwargs['params']:
        #    self.params[param] = kwargs['params'][param]


        super(GeToFeatureGraph, self).__init__(parameter_file_number=parameter_file_number,
                                               geomsc_fname_base=geomsc_fname_base,
                                               write_folder= write_folder,
                                               label_file=label_file, run_num=run_num)


        #
        # Write Paths
        #
        self.run_num = run_num


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
        if not os.path.exists(self.pred_run_path):
            os.makedirs(os.path.join(self.pred_run_path))
        self.segmentation_path = self.LocalSetup.neuron_training_segmentation_path
        self.msc_write_path = self.LocalSetup.neuron_training_base_path

        self.model_name = model_name
        self.segmentation_path = self.LocalSetup.neuron_training_segmentation_path
        self.msc_write_path = self.LocalSetup.neuron_training_base_path
        self.session_name = str(self.run_num)
        self.pred_session_run_path = os.path.join(self.pred_run_path, self.session_name)
        if not os.path.exists(self.pred_session_run_path):
            os.makedirs(os.path.join(self.pred_session_run_path))


        #
        # Data
        #
        self.data_array, self.data_set = collect_datasets(name=name,image=image,
                                                          dim_invert=self.params['dim_invert'],
                                                          format=self.params['format'])

        if geomsc_fname_base is None:
            compute_geomsc(self.params, self.data_array, self.pred_run_path, self.segmentation_path,
                           self.msc_write_path, map_labels=False)
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
        #self.image = self.image if len(self.image.shape) == 2 else np.transpose(np.mean(self.image, axis=1), (1, 0))

        self.X = self.image.shape[0]
        self.Y = self.image.shape[1] if len(self.image.shape) == 2 else self.image.shape[2]

        #
        #
        self.features = None
        self.feature_ops = {}

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

        max_deg = 0
        for gnode in self.gid_gnode_dict.values():

            geto_attr_vec = []



            num_nodes = len(self.gid_gnode_dict.values())
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
                                                                                         centroid)
            p2_centroid_hyperbolic_grad, p2_centroid_hyperbolic_dist = hyperbolic_distance(gnode.points[-1],
                                                                                           centroid)

            velocity_arc_x = np.gradient(np.array(gnode.points)[:,0])
            velocity_arc_y = np.gradient(np.array(gnode.points)[:,1])
            velocity_arc = list(np.array([[velocity_arc_x[i], velocity_arc_y[i]] for i in range(velocity_arc_y.size)]).flatten())

            euclidean_sum_self = sum_euclid(gnode.points)
            euclidean_dist_arc = end_to_end_euclid(gnode.points)
            manhattan_distance_arc = manhattan_distance(gnode.points)
            length_self = len(gnode.points)
            slope_self = slope(gnode.points)
            self_distance_from_centroid = cumulative_distance_from_centroid(gnode.points, centroid)
            mahalanobis_distance_self = mahalanobis_distance_arc(gnode.points[0],
                                                                gnode.points[-1],
                                                                inverse_covariance_points) if len(gnode.points) >= 2 else 0
            centroid_translated_points_self = translate_points_by_centroid([gnode],
                                                                           centroid)

            computed_self_attributes = []
            #self_attr_names = []
            for i, p in enumerate(centroid_translated_points_self.flatten()):
                computed_self_attributes.append(p)
                if len(self.getoelms) == 0:
                    self_attr_names += add_name('self_centroid_coord_' + str(i), self_attr_names)
            '''computed_self_attributes += list(end_to_end_hyperbolic_grad.flatten())
            computed_self_attributes += list(p1_centroid_hyperbolic_grad.flatten())
            computed_self_attributes += list(p2_centroid_hyperbolic_grad.flatten())'''
            computed_self_attributes += list(end_to_end_hyperbolic_dist.flatten())
            computed_self_attributes += list(p1_centroid_hyperbolic_dist.flatten())
            computed_self_attributes += list(p2_centroid_hyperbolic_dist.flatten())

            if len(self.getoelms) == 0:
                '''self_attr_names += ['hyperbolic-grad_e2e'+str(i) for i in
                                    range(len(list(end_to_end_hyperbolic_grad.flatten())))]
                self_attr_names += ['hyperbolic-grad_p1'+str(i) for i in
                                    range(len(list(p1_centroid_hyperbolic_grad.flatten())))]
                self_attr_names += ['hyperbolic-grad_p2'+str(i) for i in
                                    range(len(list(p2_centroid_hyperbolic_grad.flatten())))]'''
                self_attr_names += ['hyperbolic-dist_e2e'+str(i) for i in
                                    range(len(list(end_to_end_hyperbolic_dist.flatten())))]
                self_attr_names += ['hyperbolic-dist_p1' + str(i) for i in
                                    range(len(list(p1_centroid_hyperbolic_dist.flatten())))]
                self_attr_names += ['hyperbolic-dist_p2' + str(i) for i in
                                    range(len(list(p2_centroid_hyperbolic_dist.flatten())))]

            for attr_vec in velocity_arc:
                computed_self_attributes += list(attr_vec.flatten())
            if len(self.getoelms) == 0:
                self_attr_names += ['velocity_arc_' + '_' + str(i) for i in
                                      range(len(velocity_arc))]

            computed_self_attributes += [
                                   hyperbolic_distance_arc,
                                   node_degree,
                                   length_self,
                                   slope_self,
                                   euclidean_sum_self,
                                   euclidean_dist_arc,
                                   self_distance_from_centroid,
                                   manhattan_distance_arc,
                                   mahalanobis_distance_self]
            if len(self.getoelms) == 0:
                self_attr_names += [
                         'hyperbolic_distance_arc',
                         'node_degree',
                         'length_self',
                         'slope_self',
                         'euclidean_sum_self',
                         'euclidean_dist_arc',
                         'self_distance_from_centroid',
                         'manhattan_distance_arc',
                         'mahalanobis_distance_self'
                         ]
                #self.geto_attr_names = self_attr_names
            prod_nbr_attr = []
            sum_nbr_attr = []
            #nbr_attr_names = []
            if check < 1:
                dbgprint(len(self_attr_names), 'len self geto attr')

            for adj_edge in gnode.edge_gids:
                adj_gnode_gids = self.gid_edge_dict[adj_edge].gnode_gids
                deg = len(gnode.edge_gids)
                if deg > max_deg:
                    max_deg = deg
                first_neigh = 0
                for adj_gnode_gid in adj_gnode_gids:
                    if adj_gnode_gid == gnode.gid:
                        continue
                    nbr_geto_attr_vec = []
                    # geto feature vector attributes for learning hidden
                    # weighted representation
                    adj_gnode_nx_id = self.node_gid_to_graph_idx[adj_gnode_gid]
                    seen = (gnode_nx_idx,adj_gnode_nx_id) in self.lin_adj_idx_to_getoelm_idx.keys() or (adj_gnode_nx_id, gnode_nx_idx) in self.lin_adj_idx_to_getoelm_idx.keys()
                    if seen:
                        continue
                    if adj_gnode_nx_id == gnode_nx_idx:
                        continue
                    adj_gnode = self.gid_gnode_dict[adj_gnode_gid]

                    nbr_translated_vec = translate_points_by_centroid([adj_gnode],centroid)

                    cos_sim_numerator = linalg.norm(node_translated_vec,axis=0) @ linalg.norm(nbr_translated_vec,axis=0)
                    cos_sim_denominator = linalg.norm(node_translated_vec)*linalg.norm(nbr_translated_vec)
                    cos_sim = cos_sim_numerator/cos_sim_denominator if cos_sim_denominator != 0 else 0.0

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
                    inv_cos = np.clip(cos_sim_numerator/cos_sim_denominator,-1,1) if cos_sim_denominator != 0  else 0.0
                    angle_adj = m.degrees(np.arccos(inv_cos))
                    angle_adj = 90 if np.isnan(angle_adj) else angle_adj

                    triangle_area = 0.5 * linalg.norm(np.array(gnode.points)) * linalg.norm(np.array(adj_gnode.points)) * np.sin((angle_adj*np.pi)/180.)
                    triangle_area2 = 0.5 * linalg.norm(np.array(centroid_translated_points_self) @ np.array(nbr_translated_vec).T)


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




                    computed_nbr_attributes = []
                    #nbr_attr_names = []
                    attr_vecs = [#end_to_adj_hyperbolic_grad,p1_adj_hyperbolic_grad,
                                 #p3_adj_hyperbolic_grad,
                                 end_to_adj_hyperbolic_dist,p1_adj_hyperbolic_dist,
                                 p3_adj_hyperbolic_dist]
                    attr_vec_names = [#'end_to_adj_hyperbolic_grad','p1_adj_hyperbolic_grad',
                                 #'p3_adj_hyperbolic_grad',
                                 'end_to_adj_hyperbolic_dist','p1_adj_hyperbolic_dist',
                                 'p3_adj_hyperbolic_dist']
                    for attr_vec_name, attr_vec in zip(attr_vec_names, attr_vecs):
                        computed_nbr_attributes += list(attr_vec.flatten())
                        if len(self.getoelms) == 0 and first_neigh==0 :
                            nbr_attr_names += [attr_vec_name+'_' + str(i) for i in
                                               range(len(list(attr_vec.flatten())))]

                    computed_nbr_attributes += [dot_vecs, cos_sim, cos_sim_numerator, cos_sim_denominator,
                                           triangle_area,
                                           triangle_area2,
                                           angle_adj,
                                           euclidean_dist_btwn_arcs_adj1_self1,
                                           euclidean_dist_btwn_arcs_adj1_self0,
                                           euclidean_dist_btwn_arcs_adj0_self1,
                                           mahalanobis_distance_adj1_self0,
                                           mahalanobis_distance_adj1_self1,
                                           mahalanobis_distance_adj0_self1]
                    if len(self.getoelms) == 0 and first_neigh == 0:
                        nbr_attr_names += ['dot_vecs', 'cos_sim','cos_sim_numerator','cos_sim_denominator',
                             'triangle_area',
                                 'triangle_area2',
                                 'angle_adj',
                                           'euclidean_dist_btwn_arcs_adj1_self1',
                                 'euclidean_dist_btwn_arcs_adj1_self0',
                                           'euclidean_dist_btwn_arcs_adj0_self1',
                                           'mahalanobis_distance_adj1_self0',
                                 'mahalanobis_distance_adj1_self1',
                                           'mahalanobis_distance_adj0_self1']
                        #self.geto_attr_names += nbr_attr_names
                    first_neigh += 1

                    if check < 1:
                        print(len(nbr_attr_names), 'len nbr geto attr names')
                        print(deg, 'deg')
                        check+=1
                    prod_nbr_attr = []
                    sum_nbr_attr += computed_nbr_attributes


            nbr_edge_attributes = prod_nbr_attr + sum_nbr_attr
            geto_attr_vec = computed_self_attributes + nbr_edge_attributes
            #nbr_attributes =  nbr_attributes.flatten()




            if influence_type == 'weighted':
                geto_attr_vec = [np.mean(geto_attr_vec)]

            #geto_attr_vec = list(-1 * np.log(geto_attr_vec))

            if check < 3:
                print(geto_attr_vec)
                check += 1
            self.getoelms.append(geto_attr_vec)
            self.lin_adj_idx_to_getoelm_idx[gnode_nx_idx] = lin_getoelm_idx
            lin_getoelm_idx += 1


            num_geto_features = len(geto_attr_vec)

        self.geto_attr_names = self_attr_names
        #for i in range(2*(max_deg)):
        #    self.geto_attr_names +=  nbr_attr_names

        dbgprint(len(self.geto_attr_names), "len_geto_attr_names")
        dbgprint(max_deg,'max degree')
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

        scaler = QuantileTransformer(n_quantiles=1, output_distribution='normal')

        #scaler = RobustScaler(with_scaling=True, with_centering=True,unit_variance=False)
        '''geto_copy= self.getoelms.copy()
        self.getoelms = np.array(sum(self.getoelms,[])).astype(dtype=np.float32)
        getoelms_scaled = scaler.fit_transform(self.getoelms.reshape(-1, 1))
        getoelms_scaled = getoelms_scaled.flatten()
        getoelms_scaled = [list(getoelms_scaled[variable_feat_lengths[idx][0]+1:idx+variable_feat_lengths[idx][1]]) for
                                idx in range(len(variable_feat_lengths))]
        sanity_check = [list(self.getoelms[variable_feat_lengths[idx][0] + 1:idx + variable_feat_lengths[idx][1]])
                           for
                           idx in range(len(variable_feat_lengths))]'''

        for getoelm in self.getoelms:
            pad_size =  max_feat_vec_length - len(getoelm)
            getoelm = scaler.fit_transform(np.array(getoelm).reshape(-1, 1))
            getoelm = list(getoelm.flatten())
            for i,z in enumerate(range(pad_size)):
                getoelm.append(1.0)

            getoelms.append(getoelm)

        pad_size_names = max_feat_vec_length - len(self.geto_attr_names)
        for i, z in enumerate(range(pad_size_names)):
            self.geto_attr_names.append(nbr_attr_names[i % len(nbr_attr_names)])



        self.getoelms = getoelms
        self.getoelms = np.array(self.getoelms).astype(dtype=np.float32)


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


        for gnode in self.gid_gnode_dict.values():

            gnode_feature_row = []



            for image_name, im in self.images.items():

                #if i not in arc_pixel_map:
                #    arc_pixel_map[i] = get_pixel_values_from_arcs([arc], im, sampled=False)

                gnode_pixels = get_pixel_values_from_vertices([gnode], im)

                for function_name, foo in self.feature_ops.items():
                    attribute = foo(gnode_pixels)
                    gnode_feature_row.append(attribute)
                    if len(gnode_features) == 0:
                        feature_names.append(image_name + "_" + function_name)
                        self.fname_to_featidx[image_name + "_" + function_name] = feature_order
                        feature_order += 1


            #deg = gnode.degree
            #gnode_feature_row.append(deg)
            #if len(gnode_features) == 0:
            #    feature_names.append("degree")
            #    self.fname_to_featidx["degree"] = feature_order
            #    feature_order += 1


                #dbgprint(len(gnode_feature_row), "num features after geto")

                #dbgprint(len(feature_names), 'len names b4')

                #if len(gnode_features) == 0:
                #    feature_names += self.geto_attr_names

                #dbgprint(len(feature_names), 'len names')
            if include_geto:
                gnode_feature_row = np.array(gnode_feature_row)
                gnode_nx_idx = self.node_gid_to_graph_idx[gnode.gid]
                gnode_feature_row = np.hstack((gnode_feature_row, self.getoelms[gnode_nx_idx,:]))
            gnode_features.append(gnode_feature_row)
            gnode.features = np.array(gnode_feature_row).astype(dtype=np.float32)
            self.node_gid_to_feature[gnode.gid] = np.array(gnode_feature_row).astype(dtype=np.float32)
            self.node_gid_to_feat_idx[gnode.gid] = feature_idx




            feature_idx += 1



        gnode_features = np.array(gnode_features).astype(dtype=np.float32)

        #if include_geto:
        #    gnode_features = np.hstack((gnode_features, self.getoelms))

        self.number_features = len(feature_names)

        dbgprint(len(feature_names), 'len names')
        if self.getoelms is not None:
            dbgprint(self.getoelms.shape, 'geto elm shape')
        dbgprint(gnode_features.shape, 'gnode shape')

        self.feature_names = feature_names if not include_geto else feature_names + self.geto_attr_names

        dbgprint(len(self.feature_names), 'len names after')
        self.features = gnode_features
        end_time = time.time()

        print('..Time to compute features: ', end_time - start_time)

        if return_labels:
            print(">>>> returning with labels")
            return gnode_features, feature_names
        else:
            print(" returning without labels")
            self.features = gnode_features
            self.compiled_data = gnode_features
            return gnode_features

    def set_default_features(self):

        image = self.image
        min_number_features = self.params['min_number_features']
        number_features = self.params['number_features']

        image_og = copy.deepcopy(image)
        image_c = copy.deepcopy(image)
        self.images = {}

        feature_scope = [2**nf for nf in range(min_number_features, number_features + 1)]

        print("Feature scales: ", feature_scope)

        # Functions to apply to the pixels of an arc
        self.feature_ops["min"] = lambda pixels: np.min(pixels)
        self.feature_ops["max"] = lambda pixels: np.max(pixels)
        self.feature_ops["median"] = lambda pixels: np.median(pixels)
        #self.feature_ops["mode"] = lambda pixels: scipy.stats.mode(np.round(pixels, 2))[0][0]
        #self.feature_ops["mean"] = lambda pixels: gaussian_fit(pixels)[0]

        self.feature_ops["std"] = lambda pixels: gaussian_fit(pixels)[1]
        self.feature_ops["var"] = lambda pixels: gaussian_fit(pixels)[2]

        #self.feature_ops["skew"] = lambda pixels: scipy.stats.skew(pixels)
        #self.feature_ops["kurtosis"] = lambda pixels: scipy.stats.kurtosis(pixels)
        #self.feature_ops["range"] = lambda pixels: np.max(pixels) - np.min(pixels)



        self.images["identity"] = image
        image_c = copy.deepcopy(image_og)
        self.images["sobel"] = sobel_filter(image_c)
        image_c = copy.deepcopy(image_og)

        for i in feature_scope:
            pow1 = 2*i #2 ** (i - 1)
            pow2 = i
            #if i >= 3:
            #    self.images["laplacian"] = laplacian_filter(image_c, size=i)
            #    image_c = copy.deepcopy(image_og)
            #self.images["mean_{}".format(i)] = mean_filter(image_c, i)
            #image_c = copy.deepcopy(image_og)
            self.images["variance_{}".format(i)] = variance_filter(
                image_c, i
            )
            image_c = copy.deepcopy(image_og)
            self.images["median_{}".format(i)] = median_filter(image_c, i)
            image_c = copy.deepcopy(image_og)
            self.images["min_{}".format(i)] = minimum_filter(image_c, i)
            image_c = copy.deepcopy(image_og)
            self.images["max_{}".format(i)] = maximum_filter(image_c, i)
            image_c = copy.deepcopy(image_og)
            self.images["gauss_{}".format(pow2)] = gaussian_blur_filter(
                image_c, pow2
            )
            image_c = copy.deepcopy(image_og)
            self.images[
                "delta_gauss_{}_{}".format(pow1, pow2)
            ] = difference_of_gaussians_filter(image_c, pow1, pow2)
            image_c = copy.deepcopy(image_og)

        if self.params['save_filtered_images']:
            for name, image in self.images.items():
                image = np.array(image).astype(np.int8)
                #Img = Image.fromarray(image)
                if not os.path.exists(os.path.join(self.experiment_folder, 'filtered_images')):
                    os.makedirs(os.path.join(self.experiment_folder, 'filtered_images'))
                #Img.save(os.path.join(self.experiment_folder, 'filtered_images', name+'.tif'), quality=90)
                imageio.imsave(os.path.join(self.experiment_folder, 'filtered_images', name+'.tif'), image)

    def load_json_feature_graph(self):
        graph_path = os.path.join(self.pred_run_path, self.msc_graph_name)
        self.G, self.features, self.node_idx, self.walks, self.node_gid_to_label \
            , negative_sample_count \
            , positive_sample_count = load_data(prefix=graph_path)

    def write_gnode_features(self, filename):
        if not os.path.exists(os.path.join(self.experiment_folder,'features')):
            os.makedirs(os.path.join(self.experiment_folder,'features'))
        msc_feats_file = os.path.join(self.experiment_folder,'features', "feats.txt")
        msc_gid_to_feats_file = os.path.join(self.experiment_folder, 'features', "gid_feat_idx.txt")
        print("&&&& writing features in: ", msc_feats_file)
        feats_file = open(msc_feats_file, "w+")
        gid_feats_file = open(msc_gid_to_feats_file, "w+")
        lines = 1
        for gnode in self.gid_gnode_dict.values():
            nl = '\n' if lines != len(self.gid_gnode_dict) else ''
            gid_feats_file.write(str(gnode.gid)+' '+str(self.node_gid_to_feat_idx[gnode.gid])+nl)
            feature_vec = self.node_gid_to_feature[gnode.gid]
            feats_file.write(str(gnode.gid)+ ' ')
            for f in feature_vec[0:-1]:
                feats_file.write(str(f)+' ')
            feats_file.write(str(feature_vec[-1]) + nl)
            lines += 1
        gid_feats_file.close()
        feats_file.close()

    def write_geto_features(self, filename):
        if self.getoelms is not None:
            if not os.path.exists(os.path.join(self.experiment_folder,'features')):
                os.makedirs(os.path.join(self.experiment_folder,'features'))
            msc_feats_file = os.path.join(self.experiment_folder,'features', "geto_feats.txt")
            msc_gid_to_feats_file = os.path.join(self.experiment_folder, 'features', "idx_to_getoelm_idx.txt")
            print("&&&& writing features in: ", msc_feats_file)
            feats_file = open(msc_feats_file, "w+")
            gid_feats_file = open(msc_gid_to_feats_file, "w+")
            lines = 1
            for gnode in self.gid_gnode_dict.values():
                gnode_nx_idx = self.node_gid_to_graph_idx[gnode.gid]
                nl = '\n' if lines != len(self.gid_gnode_dict) else ''
                gid_feats_file.write(str(gnode_nx_idx)+' '+
                                     str(self.lin_adj_idx_to_getoelm_idx[gnode_nx_idx])+nl)
                feature_vec = self.getoelms[gnode_nx_idx]
                feats_file.write(str(gnode_nx_idx)+ ' ')
                for f in feature_vec[0:-1]:
                    feats_file.write(str(f)+' ')
                feats_file.write(str(feature_vec[-1]) + nl)
                lines += 1
            gid_feats_file.close()
            feats_file.close()

    def write_feature_names(self):
        msc_feats_file = os.path.join(self.experiment_folder,'features', "featnames.txt")
        print("&&&& writing feature namesin: ", msc_feats_file)
        feats_file = open(msc_feats_file, "w+")
        for fname in self.feature_names:
            feats_file.write(fname + "\n")
        feats_file.close()

    def write_geto_feature_names(self):
        if self.getoelms is not None:
            msc_feats_file = os.path.join(self.experiment_folder,'features', "geto_featnames.txt")
            print("&&&& writing feature namesin: ", msc_feats_file)
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

    def load_geto_feature_names(self):
        msc_feats_file = os.path.join(self.experiment_folder, 'features', "geto_featnames.txt")
        print("&&&& Reading feature names from: ", msc_feats_file)
        feats_file = open(msc_feats_file, "r")
        feat_lines = feats_file.readlines()
        self.geto_attr_names = []
        for f in feat_lines:
            self.geto_attr_names.append(f)
        #print(self.feature_names)
        feats_file.close()

    def load_gnode_features(self):
        msc_feats_file = os.path.join( self.experiment_folder,'features', "feats.txt")
        print("&&&& Reading features from: ", msc_feats_file)
        feats_file = open(msc_feats_file, "r")
        feat_lines = feats_file.readlines()
        feats_file.close()
        features = []
        for v in feat_lines:
            gid_feats = v.split(' ')
            gid = int(gid_feats[0])
            gnode = self.gid_gnode_dict[gid]
            gnode.features = np.array(gid_feats[1:])
            self.node_gid_to_feature[gid] = np.array(gid_feats[1:])
            features.append(np.array(gid_feats[1:]))
        self.features = np.array(features)
        feats_file.close()

        gid_to_feats_file = os.path.join(self.experiment_folder, 'features',"gid_feat_idx.txt")
        print("&&&& writing features in: ", gid_to_feats_file)
        feats_file = open(gid_to_feats_file, "r")
        feat_lines = feats_file.readlines()
        feats_file.close()
        for l in feat_lines:
            gid_featidx = l.split(' ')
            self.node_gid_to_feat_idx[int(gid_featidx[0])] = int(gid_featidx[1])

    def load_geto_features(self):
        self.getoelms = []
        self.lin_adj_idx_to_getoelm_idx = {}
        msc_feats_file = os.path.join( self.experiment_folder,'features', "geto_feats.txt")
        print("&&&& Reading features from: ", msc_feats_file)
        feats_file = open(msc_feats_file, "r")
        feat_lines = feats_file.readlines()
        feats_file.close()
        features = []
        geto_idx = 0
        for v in feat_lines:
            gid_feats = v.split(' ')
            gnode_nx_idx = int(gid_feats[0])
            self.lin_adj_idx_to_getoelm_idx[gnode_nx_idx] = geto_idx
            geto_idx += 1
            features.append(np.array(gid_feats[1:]))
        self.getoelms = np.array(features)
        feats_file.close()

