import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy
import copy
import imageio
from PIL import Image
import networkx as nx

from localsetup import LocalSetup
from attributes import Attributes
from topology.utils import (
    get_centroid,
    translate_points_by_centroid,
)

class GeToElement:
    def __init__(self, dim = None, gid = None, points = None):
        self.dim = dim
        self.gid = gid
        self.centroid = None
        self.vec = None
        if dim is not None:
            self.points = points

    def __group_xy(self, lst):
        for i in range(0, len(lst), 2):
            yield tuple(lst[i : i + 2])

    def read_line(self, line, labeled=False):
        tmplist = line.split(" ")
        self.gid = int(tmplist[0])
        self.dim = int(tmplist[1])
        self.points = [
                i for i in self.__group_xy([float(i) for i in tmplist[2:]])
            ] #read the rest of the the points in the arc as xy tuples
        if self.dim == 1:
            self.centroid = get_centroid(self)
            self.vec = translate_points_by_centroid([self], self.centroid)

    def make_id(self):
        pass

class GeTognode(GeToElement):
    def __init__(self, getoelm):
        super().__init__(dim = getoelm.dim , gid = getoelm.gid, points = getoelm.points)
        # topological / geometric attributes
        self.degree = None
        # ml attributes
        self.label = None
        self.label_accuracy = None
        self.partition = None
        self.prediction = None
        self.features = None
        self.box = None
        # indexing/hashing attributes
        self.key = None
        self.z = 0
        self.edge_gids = []

    def make_id(self):
        self.key = (self.gid,) + tuple(self.edge_gids) + (len(self.points),)

    def add_edge(self, edge):
        edge_gids = set(self.edge_gids)
        edge_gids.add(edge)
        self.edge_gids = list(edge_gids)
        self.degree = len(self.edge_gids)


class GeToEdge(GeToElement):
    def __init__(self, getoelm=None):
        if getoelm is not None:
            super().__init__(dim = getoelm.dim , gid = getoelm.gid, points = getoelm.points)
        else:
            super().__init__()
        # indexing/hashing attributes
        self.z = 0
        self.gnode_gids = []
        self.key = None

    def make_id(self):
        self.key = (self.gid,) + tuple(self.gnode_gids)

    def add_vertices(self, v1, v2):
        self.gnode_gids = [v1 , v2]
        self.gnode_gids = [v for v in self.gnode_gids if v!=-1]
        self.make_id()


class GeToGraph(Attributes):
    def __init__(self, geomsc_fname_base = None, label_file = None,
                 experiment_folder = None, parameter_file_number = None,**kwargs):
        # indexing and hashing
        self.gid_gnode_dict = {}
        self.gid_edge_dict   = {}
        self.gid_geto_attr_dict = {'spatial_points':None, 'dim':None}
        self.gid_geto_elm_dict = {}
        self.key_arc_dict    = {}
        self.run_num=kwargs['run_num']

        super(GeToGraph, self).__init__(parameter_file_number=parameter_file_number,
                                        write_folder = kwargs['write_folder'])

        # for kdtree for sampling nearest point
        # in graph and retrieving node gid
        self.points         = []
        self.select_points  = []
        self.key_map        = []
        self.select_key_map = []
        self.kdtree = None
        self.select_kdtree = None

        # topological / geometric information
        self.max_degree = 0

        # ml attribute information
        self.positive_training_size = 0
        self.negative_training_size = 0
        self.test_size = 0
        self.val_size = 0
        self.polyline_point_training_size = 0



        if geomsc_fname_base is not None:
            self.read_from_geo_file(fname_base=geomsc_fname_base)
        if label_file is not None:
            self.read_labels_from_file(file = label_file)



    def update_max_degree(self, gnode1, gnode2):
        if gnode1.degree > self.max_degree:
            self.max_degree = gnode1.degree
        if gnode2.degree > self.max_degree:
            self.max_degree = gnode2.degree


    def read_from_geo_file(self, fname_base, labeled=False):
        nodesname = fname_base + ".mlg_nodes.txt"
        arcsname = fname_base + ".mlg_edges.txt"
        geoname = fname_base + ".mlg_geom.txt"

        geo_file = open(geoname, "r")
        geo_lines = geo_file.readlines()
        geo_file.close()

        edge_file = open(arcsname, "r")
        edge_lines = edge_file.readlines()
        edge_file.close()

        node_file = open(nodesname, "r")
        node_lines = node_file.readlines()
        node_file.close()

        #getoelm_idx = 0
        for l in geo_lines:
            elm = GeToElement()
            elm.read_line(l)
            if elm.dim == 0:
                edge = GeToEdge(getoelm=elm)
                self.gid_edge_dict[edge.gid] = edge
            if elm.dim == 1:
                gnode = GeTognode(getoelm=elm)
                self.gid_gnode_dict[gnode.gid] = gnode
                #self.gid_geto_elm_dict[elm.gid] = elm
                #self.getoelms.append(elm.vec)
                #self.gid_to_getoelm_idx[elm.gid] = getoelm_idx
                #getoelm_idx += 1

        for l in edge_lines:
            tmplist = l.split(' ')
            gid_v1 = int(tmplist[1])
            gid_v2 = int(tmplist[2])
            gid_edge = int(tmplist[0])
            if gid_v1 != -1:
                v1 = self.gid_gnode_dict[gid_v1]
                v1.add_edge(gid_edge)
            if gid_v2 != -1:
                v2 = self.gid_gnode_dict[gid_v2]
                v2.add_edge(gid_edge)
            v2 = v1 if gid_v2 == -1 else v2
            v1 = v2 if gid_v1 == -1 else v1
            self.update_max_degree(v1,v2)

            edge = self.gid_edge_dict[gid_edge]
            edge.add_vertices(gid_v1, gid_v2)


    def write_getograph(self, fname_base, labeled=False):
        nodesname = fname_base + ".mlg_nodes.txt"
        arcsname = fname_base + ".mlg_edges.txt"
        geoname = fname_base + ".mlg_geom.txt"

        geo_file = open(geoname, "w+")

        edge_file = open(arcsname, "w+")

        node_file = open(nodesname, "w+")

        for gid,gnode in zip(self.gid_gnode_dict.keys()[0:-1] , self.gid_gnode_dict.values()[0:-1]):
            geo_file.write(str(gid)+' '+str(gnode.dim)+' ')
            for point in gnode.points[0:-1]:
                geo_file.write(str(point)+' ')
            geo_file.write(str(gnode.points[-1])+'\n')
        last_gnode_gid = self.gid_gnode_dict.keys()[-1]
        last_gnode = self.gid_gnode_dict.values()[-1]
        geo_file.write(str(last_gnode_gid)+' '+str(last_gnode.dim)+' ')
        for point in last_gnode.points[0:-1]:
            geo_file.write(str(point) + ' ')
        geo_file.write(str(last_gnode.points[-1]) + '\n')

        for gid, edge in zip(self.gid_edge_dict.keys()[0:-1] , self.gid_edge_dict.values()[0:-1]):
            adj_vertices = edge.gnode_gids
            edge_file.write(str(gid)+' '+str(adj_vertices[0])+' '+str(adj_vertices[1])+'\n')
            geo_file.write(str(gid)+' '+str(edge.dim)+' '+str(edge.points[0])+' '+str(edge.points[1])+'\n')

        last_edge_gid = self.gid_edge_dict.keys()[-1]
        last_edge = self.gid_edge_dict.values()[-1]
        last_adj_vertices = last_edge.gnode_gids
        edge_file.write(str(last_edge_gid) + ' ' + str(last_adj_vertices[0]) + ' ' + str(last_adj_vertices[1]) + '\n')
        geo_file.write(str(last_edge_gid) + ' ' + str(last_edge.dim) + ' ' + str(last_edge.points[0]) + ' ' + str(last_edge.points[1]) + '\n')

        edge_file.close()
        geo_file.close()

    def read_labels_from_file(self, file = None):
        label_file = open(file, "r")
        self.label_lines = label_file.readlines()
        label_file.close()

        for gid, l in enumerate(self.label_lines):
            tmplist = l.split(' ')
            label = [0., 1.] if int(tmplist[0]) == 1 else [1. , 0.]
            gnode = self.gid_gnode_dict[gid]
            gnode.label = label
            self.node_gid_to_label[gid] = label
            gnode.ground_truth = label

    def build_kdtree(self):
        sorted_vertices = sorted(self.gid_gnode_dict.values(), key=lambda gnode: len(gnode.points))
        for gnode in sorted_vertices:
            gid = gnode.gid
            self.points.extend(gnode.points)
            self.key_map.extend([gid] * len(gnode.points))
        # only needed for selection ui to choose neighboring arcs
        # can cause error with sparse MSC
        self.kdtree = scipy.spatial.KDTree(self.points, leafsize=10000)
        return self.kdtree

    # kdtree for sampling only selected edges
    def build_select_kdtree(self, vertices):
        self.select_points = []
        self.select_key_map = []
        sorted_vertices = sorted(vertices, key=lambda gnode: len(gnode.points))
        for gnode in sorted_vertices:
            gid = gnode.gid
            if len(np.shape(gnode.points)) > 1:
                self.select_points.extend(gnode.points)
                self.select_key_map.extend([gid] * len(gnode.points))
        # only needed for selection ui to choose neighboring arcs
        # can cause error with sparse MSC
        self.select_kdtree = scipy.spatial.KDTree(self.select_points, leafsize=10000)
        return self.select_kdtree

    def get_closest_gnode(self, point):
        distance, index = self.kdtree.query(point)
        return self.key_map[index]

    def get_closest_selected_gnode(self, point):
        distance, index = self.select_kdtree.query(point)
        return self.select_key_map[index]

    def draw_segmentation(self, dirpath, ridge=True, valley=True, invert=False):
        X = self.X #if not invert else self.Y
        Y = self.Y #if not invert else self.X
        original_image = self.image

        self.use_ridge_arcs = ridge
        self.use_valley_arcs = valley

        black_box = np.zeros((X, Y)) if not invert else np.zeros(
            (Y, X))
        cmap = cm.get_cmap('bwr')
        cmap.set_under('black')
        cmap.set_bad('black')
        plt.set_cmap(cmap)
        fig = plt.imshow(black_box, cmap=cmap, alpha=None, vmin=0)
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        original_image = np.stack((original_image,) * 3, axis=-1)

        if original_image.shape[0] == 3:
            mapped_image = np.transpose(original_image, (2, 1, 0))
        elif original_image.shape[1] == 3:
            mapped_image = np.transpose(original_image, (0, 2, 1))
        else:
            mapped_image = original_image

        label_map_image = copy.deepcopy(mapped_image)

        c = 0

        for gnode in self.gid_gnode_dict.values():
            gid = gnode.gid
            prediction = self.node_gid_to_prediction[gid]
            partition = self.node_gid_to_partition[gid]
            label =self.node_gid_to_label[gid]
            gnode = self.gid_gnode_dict[gid]
            if isinstance(prediction,
                          (int, np.integer)) or isinstance(prediction, (float, np.float)):
                label_color = cmap(float(prediction))
            else:
                if len(prediction) == 3:
                    label_color = cmap(0.56) if float(prediction[2]) > 0.5 else cmap(float(prediction[1]))
                else:
                    if prediction == []:
                        #print('perd ' , prediction)
                        #print(gnode.gid)
                        #print(partition)
                        #print(self.node_gid_to_feature[gnode.gid])
                        continue
                    label_color = cmap(float(prediction[len(prediction) - 1]))

            if original_image is not None:
                x = 1#  if invert else 0
                y = 0#0  if invert else 1
                scale = 255.
                if partition == 'train':
                    label_color = [51, 255, 51]
                    scale = 1
                if partition == 'val':
                    label_color = [255, 51, 255]
                    scale = 1
                if len(mapped_image.shape) == 2:
                    for p in np.array(gnode.points):
                        mapped_image[int(p[x]), int(p[y])] = int(label_color[0] * scale)
                        mapped_image[int(p[x]), int(p[y])] = int(label_color[1] * scale)
                        mapped_image[int(p[x]), int(p[y])] = int(label_color[2] * scale)

                        msc_ground_seg_color = cmap(float(label[1]))
                        label_map_image[int(p[x]), int(p[y])] = int(msc_ground_seg_color[0] * 255)
                        label_map_image[int(p[x]), int(p[y])] = int(msc_ground_seg_color[1] * 255)
                        label_map_image[int(p[x]), int(p[y])] = int(msc_ground_seg_color[2] * 255)
                else:
                    for p in np.array(gnode.points):
                        mapped_image[int(p[x]), int(p[y]), 0] = int(label_color[0] * scale)
                        mapped_image[int(p[x]), int(p[y]), 1] = int(label_color[1] * scale)
                        mapped_image[int(p[x]), int(p[y]), 2] = int(label_color[2] * scale)

                        msc_ground_seg_color = cmap(float(label[1]))
                        label_map_image[int(p[x]), int(p[y]), 0] = int(msc_ground_seg_color[0] * 255)
                        label_map_image[int(p[x]), int(p[y]), 1] = int(msc_ground_seg_color[1] * 255)
                        label_map_image[int(p[x]), int(p[y]), 2] = int(msc_ground_seg_color[2] * 255)


        if original_image is not None:
            if len(mapped_image.shape) != 2:
                map_im = np.transpose(mapped_image, (0, 1, 2))
                lab_map_im = np.transpose(label_map_image, (0, 1, 2))
            else:
                map_im = mapped_image
                lab_map_im = label_map_image
            Img = Image.fromarray(map_im.astype('uint8'))  # .astype(np.float32))#mapped_img)
            Img.save(os.path.join(dirpath, 'inference.png'))

            Img = Image.fromarray(
                lab_map_im.astype('uint8'))  # .astype(np.float32))#mapped_img)
            Img.save(os.path.join(dirpath , 'groundseg.png'))

        plt.close()

    def get_set_sizes(self):
        return np.array([self.test_size, self.val_size, self.negative_training_size, self.positive_training_size, self.polyline_point_training_size])

    def equate_graph(self, G, clear_preds=False):
        nx_idx_to_gid = self.graph_idx_to_gid #{v: k for k, v in  self.node_gid_to_nx_idx.items()}
        for nx_node_idx,node in enumerate(G.nodes_iter()):
            gid = nx_idx_to_gid[nx_node_idx]
            #arc = self.gid_map[gid] #self.arc_order[arcidx]#
            gnode = self.gid_gnode_dict[gid]
            if G.node[node]['train']:
                gnode.partition = 'train'
                self.node_gid_to_partition[gid] = 'train'
                gnode.box = 1
            if G.node[node]['val']:
                gnode.partition = 'val'
                self.node_gid_to_partition[gid] = 'val'
                gnode.box = 0
            if G.node[node]['test']:
                gnode.partition = 'test'
                self.node_gid_to_partition[gid] = 'test'
                gnode.box = 0
            if G.node[node]["prediction"] is not None and not clear_preds:
                if G.node[node]["prediction"] == []:
                    #print(node)
                    continue
                gnode.prediction = G.node[node]["prediction"]
                self.node_gid_to_prediction[gid] = G.node[node]["prediction"]
                if isinstance(gnode.prediction,
                                  (int, np.integer)) or isinstance(gnode.prediction,
                                                                   (float, np.float)):
                    gnode.prediction = float(gnode.prediction)
                    self.node_gid_to_prediction[gid] = float(gnode.prediction)
                else:
                    gnode.prediction = float(gnode.prediction[1])
                    self.node_gid_to_prediction[gid] = float(gnode.prediction)
            if clear_preds:
                gnode.prediction = None
            if G.node[node]["label"] is not None:
                gnode.label = np.array(G.node[node]["label"])
            gnode.features = np.array(G.node[node]["features"])

            if gnode.partition == 'train':
                self.polyline_point_training_size += len(gnode.points)
                if gnode.label[1] > 0.5:
                    self.positive_training_size += 1
                else:
                    self.negative_training_size += 1
            elif gnode.partition == 'test':
                self.test_size += 1
            elif gnode.partition == 'val':
                self.val_size +=1
            nx_node_idx += 1