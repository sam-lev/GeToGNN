import os
import time
import networkx as nx
import numpy as np
import copy
import samply
import scipy.stats
import imageio
from PIL import Image

from getograph import GeToGraph
from data_ops.collect_data import collect_training_data, compute_geomsc, collect_datasets
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
    laplacian_filter,
    sum_euclid,
    gaussian_fit,
)

from topology.utils import (
    get_pixel_values_from_vertices,
    get_centroid,
    translate_points_by_centroid,
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


        """if parameter_file_number is not None:
            self.params = set_parameters(read_params_from=parameter_file_number)
        else:
            self.params = kwargs['params']
        """
        for param in kwargs:
            self.params[param] = kwargs[param]
        #for param in kwargs['params']:
        #    self.params[param] = kwargs['params'][param]
        if 'params' in kwargs.keys():
            param_add_ons = kwargs['params']
            for k, v in param_add_ons.items():
                self.params[k] = v



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
        self.data_array, self.data_set = collect_datasets(name=self.params['name'],image=image,
                                                          dim_invert=self.params['dim_invert'],
                                                          format=self.params['format'])

        if geomsc_fname_base is None:
            compute_geomsc(self.params, self.data_array, self.pred_run_path, self.segmentation_path,
                           self.msc_write_path, map_labels=False)
        self.train_dataloader = collect_training_data(
            dataset=self.data_set,
            data_array=self.data_array,
            params=self.params,
            name=self.params['name'],
            format=format,
            msc_file=None,
            dim_invert=self.params['dim_invert'])

        self.image, self.msc_collection, self.mask, self.segmentation = self.train_dataloader[
            int(self.params['train_data_idx'])]
        self.image = self.image if len(self.image.shape) == 2 else np.transpose(np.mean(self.image, axis=1), (1, 0))

        self.X = self.image.shape[0]
        self.Y = self.image.shape[1] if len(self.image.shape) == 2 else self.image.shape[2]

        #
        #
        self.features = None
        self.feature_ops = {}

        #
        # build feature graph
        #
        if load_feature_graph_name is None:
            self.msc_graph_name = 'geomsc_feature_graph-' + str(self.params['number_features']) + '_dataset-' + name + model_name
        else:
            self.msc_graph_name = load_feature_graph_name
            self.load_feature_graph()
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
            self.node_gid_to_nx_idx[gid] = nx_node_idx
            nx_node_idx += 1

        for nx_gid, node in list(self.G.nodes_iter(data=True)):

            for adj_edge in node["edges"]:
                for adj_gnode_gid in self.gid_edge_dict[adj_edge].gnode_gids:
                    adj_gnode_nx_id = self.node_gid_to_nx_idx[adj_gnode_gid]
                    #if adj_gnode_nx_id != nx_gid:
                    self.G.add_edge(nx_gid, adj_gnode_nx_id)
        return self.G.copy()

    def compile_features(self, image=None, return_labels=False, save_filtered_images=False,
                         min_number_features=1, number_features=5, selection=None):

        start_time = time.time()

        self.set_default_features()

        gnode_features = []
        feature_names = []
        feature_idx = 0
        feature_order = 0

        for gnode in self.gid_gnode_dict.values():

            gnode_feature_row = []

            euclidean_sum = sum_euclid(gnode.points)
            gnode_feature_row.append(euclidean_sum)
            if len(gnode_features) == 0:
                feature_names.append('euclidean_sum_length')
                self.fname_to_featidx['euclidean_sum_length'] = feature_order
                feature_order += 1

            centroid = get_centroid(gnode)
            centroid_translated_points = translate_points_by_centroid([gnode], centroid)
            for i, p in enumerate(centroid_translated_points.flatten()):
                gnode_feature_row.append(p)
                if len(gnode_features) == 0:
                    feature_names.append('centroid_coord_'+str(i))
                    self.fname_to_featidx['centroid'] = feature_order
                    feature_order += 1

            length = len(gnode.points)
            gnode_feature_row.append(length)
            if len(gnode_features) == 0:
                feature_names.append("length_line")
                self.fname_to_featidx['length'] = feature_order
                feature_order += 1

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


            deg = gnode.degree
            gnode_feature_row.append(deg)
            if len(gnode_features) == 0:
                feature_names.append("degree")
                self.fname_to_featidx["degree"] = feature_order
                feature_order += 1



            gnode_features.append(gnode_feature_row)
            gnode.features = np.array(gnode_feature_row).astype(dtype=np.float32)
            self.node_gid_to_feature[gnode.gid] = np.array(gnode_feature_row).astype(dtype=np.float32)
            self.node_gid_to_feat_idx[gnode.gid] = feature_idx
            feature_idx += 1

        gnode_features = np.array(gnode_features).astype(dtype=np.float32)

        self.number_features = len(feature_names)
        self.feature_names = feature_names
        print("number features: ", self.number_features)
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
        self.feature_ops["mode"] = lambda pixels: scipy.stats.mode(np.round(pixels, 2))[0][0]
        self.feature_ops["mean"] = lambda pixels: gaussian_fit(pixels)[0]

        self.feature_ops["std"] = lambda pixels: gaussian_fit(pixels)[1]
        self.feature_ops["var"] = lambda pixels: gaussian_fit(pixels)[2]

        self.feature_ops["skew"] = lambda pixels: scipy.stats.skew(pixels)
        self.feature_ops["kurtosis"] = lambda pixels: scipy.stats.kurtosis(pixels)
        self.feature_ops["range"] = lambda pixels: np.max(pixels) - np.min(pixels)



        self.images["identity"] = image
        image_c = copy.deepcopy(image_og)
        self.images["sobel"] = sobel_filter(image_c)
        image_c = copy.deepcopy(image_og)

        for i in feature_scope:
            pow1 = 2*i #2 ** (i - 1)
            pow2 = i
            if i >= 3:
                self.images["laplacian"] = laplacian_filter(image_c, size=i)
                image_c = copy.deepcopy(image_og)
            self.images["mean_{}".format(i)] = mean_filter(image_c, i)
            image_c = copy.deepcopy(image_og)
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
                image = np.array(image).astype('uint8')
                #Img = Image.fromarray(image)
                if not os.path.exists(os.path.join(self.experiment_folder, 'filtered_images')):
                    os.makedirs(os.path.join(self.experiment_folder, 'filtered_images'))
                #Img.save(os.path.join(self.experiment_folder, 'filtered_images', name+'.tif'))
                imageio.imsave(os.path.join(self.experiment_folder, 'filtered_images', name+'.png'), image)

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

    def write_feature_names(self, filename):
        msc_feats_file = os.path.join(self.experiment_folder,'features', "featnames.txt")
        print("&&&& writing feature namesin: ", msc_feats_file)
        feats_file = open(msc_feats_file, "w+")
        for fname in self.feature_names:
            feats_file.write(fname + "\n")
        feats_file.close()

    def load_gnode_features(self, filename):
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