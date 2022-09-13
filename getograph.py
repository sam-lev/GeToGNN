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
from ml.features import (
    get_centroid,
    translate_points_by_centroid,
    get_points_from_vertices,
)
from topology.geomsc import compute_geomsc, MSC
from ml.utils import pout

class GeToElement:
    def __init__(self, dim = None, gid = None, points = None):
        self.dim = dim
        self.gid = gid
        self.centroid = None
        self.vec = None
        self.global_isolated_nodes = 0
        if dim is not None:
            self.points = points

    def __group_xy(self, lst):
        for i in range(0, len(lst), 2):
            yield tuple(lst[i : i + 2])

    def read_line(self, line, labeled=False, isolated_node_count=0):
        tmplist = line.split(" ")
        self.gid = int(tmplist[0])
        self.global_isolated_nodes = isolated_node_count
        if self.gid == -1:
            self.global_isolated_nodes += 1
        # if self.gid == -1:
        #     self.gid += self.multiplicity_count
        self.dim = int(tmplist[1])
        self.points = [
                i for i in self.__group_xy([float(i) for i in tmplist[2:]])
            ] #read the rest of the the points in the arc as xy tuples
        if self.dim == 1:
            self.centroid = get_centroid(self.points)
            self.vec = translate_points_by_centroid([self], self.centroid)

    def make_id(self):
        pass

class GeTogNode(GeToElement):
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
        self.sublevel_set = False
        self.level_id = 0

    def make_id(self):
        self.key = (self.gid,) + tuple(self.edge_gids) + (len(self.points),)

    def add_edge(self, edge):
        #edge_gids = set(self.edge_gids)
        #edge_gids.add(edge)
        if edge not in self.edge_gids:
            self.edge_gids.append(edge)# = list(edge_gids)
        self.degree = len(self.edge_gids)

    def copy(self):
        getoelm = GeToElement(dim=1, gid = self.gid, points=self.points)
        return GeTogNode(getoelm)

    def is_sublevel_set(self, level_id):
        self.sublevel_set = True
        self.level_id = level_id


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
        self.sublevel_set = False
        self.level_id = 0

    def make_id(self):
        self.key = (self.gid,) + tuple(self.gnode_gids)

    def add_vertices(self, v1, v2):
        self.gnode_gids = [v1 , v2]
        self.gnode_gids = [v for v in self.gnode_gids]
        ''' if v!=-1]'''
        self.make_id()

    def add_node(self,v):
        self.gnode_gids.append(v)

    def is_sublevel_set(self, level_id):
        self.sublevel_set = True
        self.level_id = level_id



class GeToGraph(Attributes):
    def __init__(self, geomsc_fname_base = None, label_file = None,
                 experiment_folder = None, parameter_file_number = None,write_folder=None,**kwargs):
        # indexing and hashing
        self.gid_gnode_dict = {}
        self.gid_edge_dict   = {}
        self.gid_geto_attr_dict = {'spatial_points':None, 'dim':None}
        self.gid_geto_elm_dict = {}
        self.key_arc_dict    = {}
        if 'write_folder' not in kwargs.keys():
            kwargs['write_folder'] = write_folder
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

        self.geomsc_fname_base = geomsc_fname_base
        self.label_file = label_file

        self.persistence = None

        self.homophily_ratio = None
        self.class_statistics = {}
        self.class_statistics['positive'] = 0
        self.class_statistics['negative'] = 0
        self.class_statistics['total'] = 0
        #
        # read pre-computed MSC
        #
        if geomsc_fname_base is not None:
            self.read_from_geo_file(fname_base=geomsc_fname_base)
        if label_file is not None:
            self.read_labels_from_file(file = label_file)

    def compute_morse_smale_complex(self, fname_base, polyline_graph = True,
                                    persistence = [0.01], sigma=[2], X=None,Y=None):

        self.persistence = persistence[0]

        f_path      = fname_base
        seg_folder = os.path.dirname(os.path.abspath(f_path))
        if not os.path.exists(os.path.join(seg_folder,'geomsc')):
            os.makedirs(os.path.join(seg_folder,'geomsc'))
        if not os.path.exists(os.path.join(seg_folder,'geomsc',str(persistence[0]))):
            os.makedirs(os.path.join(seg_folder,'geomsc',str(persistence[0])))
        write_path = os.path.join(seg_folder,'geomsc',str(persistence[0]))

        msc, msc_fname= compute_geomsc( persistence_values=persistence, blur_sigmas=sigma,
                data_buffer = None, data_path = f_path, segmentation_path=seg_folder,
                write_path =write_path , labeled_segmentation=None, label=False,
                save=False, save_binary_seg=False, number_images=None,
                              X=X,Y=Y,
                persistence_cardinality = None, valley=polyline_graph, ridge=polyline_graph)
        return msc, msc_fname

    def map_to_priors_graph(self, msc):
        gid_gnode_dict = {}
        gid_edge_dict  = {}
        for arc in msc.arcs:
            gelm = GeToElement()
            gelm.points = arc.points
            gelm.dim = 1
            gelm.gid = arc.id
            gnode = GeTogNode(getoelm=gelm)
            for incident in arc.node_ids:
                gnode.add_edge(incident)
            gid_gnode_dict[gnode.gid] = gnode
        for id, node in msc.nodes.items():
            gelm = GeToElement()
            gelm.points = node.points
            gelm.dim = 0
            gelm.gid = node.id
            gedge = GeToEdge(getoelm=gelm)
            for arc in node.arcs:
                gedge.add_node(arc.id)
            gid_edge_dict[gedge.gid] = gedge
        return gid_gnode_dict, gid_edge_dict

    def mark_sublevel_set(self, sublevel_edges, sublevel_nodes,
                          X, Y, sublevel_label_dict, level_id=1,
                          union_radius=None, union_thresh=0,
                          map_labels=False):


        point_map_sub = np.ones(self.image.shape) * -2
        i_x = 1
        i_y = 0

        sup_gid_to_sub_dict = {}

        for gid, gedge in sublevel_edges.items():
            gid_temp = gid if gid is not None else 0
            gid_temp = 0 if gid < -1 else 0
            gedge.is_sublevel_set(level_id)
            gedge.sublevel_set = True
            sub_points = gedge.points
            sub_points = [tuple(map(round, p)) for p in sub_points]
            for xy in sub_points:
                xy = (xy[i_x], xy[i_y])

                point_map_sub[xy] = gid#_temp

        arc_point_to_gid_sub = {}
        for gid, gnode_sub in sublevel_nodes.items():
            gid_temp = gid if gid is not None else 0
            gid_temp = 0 if gid < -1 else 0
            gnode_sub.is_sublevel_set(level_id)
            gnode_sub.sublevel_set = True
            sub_points = gnode_sub.points
            sub_points = [tuple(map(round, p)) for p in sub_points]
            for xy in sub_points:
                xy = (xy[i_x], xy[i_y])  # if xy[i_x] <= X-1 and xy[i_y] <= Y-1 else (X-1,Y-1)
                point_map_sub[xy] = gid
                arc_point_to_gid_sub[xy] = gid_temp

        for gid, gedge_super in self.gid_edge_dict.items():
            sup_points = gedge_super.points
            sup_points = [tuple(map(round, p)) for p in sup_points]


            cardinality_intersect = 0
            xy_init = None
            for xy in sup_points:
                xy = (xy[i_x],xy[i_y])

                for ngid in gedge_super.gnode_gids:
                    sup_gid_to_sub_dict[ngid] = point_map_sub[xy]#_init]

                if point_map_sub[xy] >= -1:
                    if xy_init is None:
                        xy_init = xy
                    cardinality_intersect += 1
                    if union_thresh == 0:
                        intersects = cardinality_intersect > union_thresh * len(sup_points)
                    else:
                        intersects = cardinality_intersect >= union_thresh * len(sup_points)
                    if intersects:
                        gedge_super.is_sublevel_set(level_id)
                        gedge_super.sublevel_set = True
                        #if sublevel_labels:
                        #    sub_node_label = sublevel_labels[gid_sub]
                        #    self.node_gid_to_label[gid] = sub_node_label
                if union_radius is not None:
                    x, y = xy
                    for factor in [1,-1]:
                        for r_y in range(1,union_radius):
                            for r_x in range(1,union_radius):
                                r_x *= factor
                                r_y *= factor
                                if x + r_x >= X-1:
                                    r_x = 0
                                if y + r_y >= Y-1:
                                    r_y = 0
                                if x + r_x < 0:
                                    r_x = 0
                                if y + r_y < 0:
                                    r_y = 0
                                if point_map_sub[(x + r_x, y + r_y)] >= -1:
                                    cardinality_intersect += 1
                                if point_map_sub[(x, y + r_y)] >= -1:
                                    cardinality_intersect += 1
                                if point_map_sub[(x + r_x, y)] >= -1:
                                    cardinality_intersect += 1
            if union_thresh == 0:
                intersects = cardinality_intersect > union_thresh * len(sup_points)
            else:
                intersects = cardinality_intersect >= union_thresh * len(sup_points)
            if intersects:
                gedge_super.is_sublevel_set(level_id)
                gedge_super.sublevel_set = True

                #sup_gid_to_sub_dict[gid] = point_map_sub[xy_init]
                for ngid in gedge_super.gnode_gids:
                    gn = self.gid_gnode_dict[ngid]
                    gn.is_sublevel_set(level_id)
                    # sub_gid = point_map_sub[xy_init]
                    # sublevel_label_dict[sub_gid] = self.node_gid_to_label[ngid]
                    # if sublevel_labels:
                    #     self.node_gid_to_label[ngid] = sublevel_labels[gid_sub]
                    sup_gid_to_sub_dict[ngid] = point_map_sub[xy_init]
            self.gid_edge_dict[gid] = gedge_super




        for gid, gnode_super in self.gid_gnode_dict.items():
            sup_points = gnode_super.points
            sup_points = [tuple(map(round, p)) for p in sup_points]

            cardinality_intersect = 0
            gid_sub = -2
            xy_init = None
            for xy in sup_points:
                xy = (xy[i_x],xy[i_y]) #if xy[i_x] <= X-1 and xy[i_y] <= Y-1 else (X-1,Y-1)
                # if xy_init is None:
                #     xy_init = xy
                sub_gid = point_map_sub[xy]
                sup_gid_to_sub_dict[gid] = sub_gid

                # map label of sup to sub

                # sublevel_label_dict[sub_gid] = self.node_gid_to_label[gid]

                if point_map_sub[xy] >= -1:
                    gid_sub = point_map_sub[xy] #take gid of interesecting node
                    if xy_init is None:
                        xy_init = xy
                if point_map_sub[xy] >= -1:
                    cardinality_intersect += 1
                    # if union_thresh == 0:
                    #     intersects = cardinality_intersect > union_thresh * len(sup_points)
                    # else:
                    #     intersects = cardinality_intersect >= union_thresh * len(sup_points)
                    # if intersects:
                    #     gnode_super.is_sublevel_set()
                    #     gnode_super.sublevel_set = True
                    #     if sublevel_labels:
                    #         self.node_gid_to_label[gid] = sublevel_labels[gid_sub]
                if union_radius is not None:
                    x, y = xy
                    for factor in (1,-1):
                        for r_y in range(1,union_radius):
                            for r_x in range(1,union_radius):
                                r_x *= factor
                                r_y *= factor
                                if x + r_x >= X - 1:
                                    r_x = 0
                                if y + r_y >= Y - 1:
                                    r_y = 0
                                if x + r_x < 0:
                                    r_x = 0
                                if y + r_y < 0:
                                    r_y = 0
                                if point_map_sub[(x + r_x, y + r_y)] >= -1:
                                    cardinality_intersect += 1
                                    gid_sub = point_map_sub[(x + r_x, y + r_y)]
                                if point_map_sub[(x, y + r_y)] >= -1:
                                    cardinality_intersect += 1
                                    gid_sub = point_map_sub[(x, y + r_y)]
                                if point_map_sub[(x + r_x, y)] >= -1:
                                    cardinality_intersect += 1
                                    gid_sub = point_map_sub[(x + r_x, y)]
            if union_thresh == 0:
                intersects = cardinality_intersect > union_thresh * len(sup_points)
            else:
                intersects = cardinality_intersect >= union_thresh * len(sup_points)
            if intersects:
                gnode_super.is_sublevel_set(level_id)
                gnode_super.sublevel_set = True
                # if sublevel_labels:
                #     self.node_gid_to_label[gid] = sublevel_labels[gid_sub]

                sup_gid_to_sub_dict[gid] = point_map_sub[xy_init]

                incident_edge = gnode_super.edge_gids
                for egid in incident_edge:
                    edge = self.gid_edge_dict[egid]
                    edge.is_sublevel_set(level_id)

                    #sup_gid_to_sub_dict[egid] = point_map_sub[xy_init]
                observed_subgids = []
                last_p = sup_points[-1]
                for xy in sup_points:
                    xy_og = xy
                    xy = (xy[i_x], xy[i_y])
                    sub_gid =point_map_sub[xy]
                    observed_subgids.append(sub_gid)
                    if sub_gid <= -1:
                        if last_p == xy_og and -1 in observed_subgids:
                            sub_gid = -1
                        else:
                            continue
                    sup_gid_to_sub_dict[gid] = sub_gid
                    # sublevel_label_dict[sub_gid] = self.node_gid_to_label[gid]

            self.gid_gnode_dict[gid] = gnode_super
        return self.gid_gnode_dict, self.gid_edge_dict, sublevel_label_dict, sup_gid_to_sub_dict


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
        self.edge_count = 0
        self.vertex_count = 0

        self.isolated_node_count = 1
        #getoelm_idx = 0
        for l in geo_lines:
            elm = GeToElement()
            elm.read_line(l, isolated_node_count = self.isolated_node_count)
            self.isolated_node_count = elm.global_isolated_nodes
            if elm.gid == -1:
                elm.gid = elm.gid/(2**self.isolated_node_count)
            if elm.dim == 0:
                #self.edge_count  +=  1
                edge = GeToEdge(getoelm=elm)
                self.gid_edge_dict[edge.gid] = edge
            if elm.dim == 1:
                self.vertex_count += 1
                gnode = GeTogNode(getoelm=elm)
                self.gid_gnode_dict[gnode.gid] = gnode
                #self.gid_geto_elm_dict[elm.gid] = elm
                #self.getoelms.append(elm.vec)
                #self.gid_to_getoelm_idx[elm.gid] = getoelm_idx
                #getoelm_idx += 1
        # add isolated placement holder gnode due to single
        # vertice edges
        # isolated = GeToElement()
        # isolated.dim = 1
        # isolated.gid = -1
        # isolated.points = [(1., 1.), (1., 1.), (1., 1.)]
        # phantom_gnode = GeTogNode(getoelm=isolated)
        # self.gid_gnode_dict[phantom_gnode.gid] = phantom_gnode

        for l in edge_lines:
            self.edge_count += 1
            tmplist = l.split(' ')
            gid_v1 = int(tmplist[1])
            gid_v2 = int(tmplist[2])
            gid_edge = int(tmplist[0])
            # add self loop if vertex isolated
            if gid_v1 == -1:
                gid_v1 = gid_v2
            if gid_v2 == -1:
                gid_v2 = gid_v1

            v1 = self.gid_gnode_dict[gid_v1]
            v1.add_edge(gid_edge)

            v2 = self.gid_gnode_dict[gid_v2]
            v2.add_edge(gid_edge)
            '''v2 = v1 if gid_v2 == -1 else v2
            v1 = v2 if gid_v1 == -1 else v1'''
            self.update_max_degree(v1,v2)

            edge = self.gid_edge_dict[gid_edge]
            edge.add_vertices(gid_v1, gid_v2)

        pout(("__ In read Geom file__",
              "TOTAL NUMBER EDGES:",
              self.edge_count,
              "TOTAL NUMBER NODES:",
              self.vertex_count,
              "TOTAL NUMBER ISOLATED(-1 gid) VERTICES:",
              self.isolated_node_count))


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
        # self.node_gid_to_label[-1] = [1., 0.] # !!!!!!!1   !!   !!!! !   !! !  ! !  !!!!!!! !! !!

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

    def draw_priors_graph(self, G=None):
        plt.close()
        if G is None:
             G = self.G
        positions = []

        subgraph = []
        for gid in self.gid_gnode_dict.keys():
            gnode = self.gid_gnode_dict[gid]
            label = self.node_gid_to_label[gid]
            line = get_points_from_vertices([gnode])
            # else is
            for point in line:
                ly = int(point[0])
                lx = int(point[1])
                if 0 < lx < 550 and 0 < ly < 550:
                    node = self.node_gid_to_graph_idx[gid]
                    subgraph.append(node)
        H = G.subgraph(subgraph)
        nx_idx_to_gid = self.graph_idx_to_gid
        for nx_node_idx,node in enumerate(G.nodes_iter()):
            #for node in self.gid_edge_dict.keys():
            gid = nx_idx_to_gid[nx_node_idx]
            gnode = self.gid_gnode_dict[gid]#[0]]


            line = gnode.points

            point = line[0]
            ly = int(point[0])
            lx = int(point[1])
            node = self.node_gid_to_graph_idx[node]
            G.node[node]['pos'] = (lx, ly)
            positions.append([lx,ly])
        L = nx.line_graph(H)
        nnodes = L.number_of_nodes()
        print("     * : numbner line graph nodes",nnodes)
        pos = nx.spring_layout(L,weight=.000001)
        #selected1 = [i for i in range(1000, 1200)] + [i for i in range(1800, 1900)]
        #selected2 = [i for i in range(1900, 2100)] + [i for i in range(1200, 1300)]

        nearest_hundred = 784# 0.2*len(self.gid_edge_dict.keys()) - (0.2*len(self.gid_edge_dict.keys())) % 100
        blobs = np.arange(nearest_hundred).reshape((28,28))
        #    (int((0.2*len(self.gid_edge_dict.keys()))) // 100, int((0.2*len(self.gid_edge_dict.keys()))) // 100))
        selected1 = [blobs[i, :] for i in range(0, 28) if i%2==0]#(0.2*len(self.gid_edge_dict.keys())) // 100) if i % 2 == 0]
        selected2 = [blobs[i, :] for i in range(0, 28) if i%2!=0]#range(0, (0.2*len(self.gid_edge_dict.keys())) // 100) if i % 2 != 0]
        selected1 = [i for ar in selected1 for i in ar]
        selected2 = [i for ar in selected2 for i in ar]
        node_color = [.3 if idx in selected1 else .7 if idx in selected2 else .5 for idx,n in enumerate(L.nodes_iter()) ]
        nx.draw(L, node_size=60,cmap=plt.get_cmap('Spectral_r'), node_color=node_color, pos=pos)#np.array(positions))
        plt.savefig(os.path.join(self.pred_session_run_path,'priors_graph_labeled.png'))
        plt.show()
        nearest_hundred = 2401#len(self.gid_edge_dict.keys()) - len(self.gid_edge_dict.keys())%100
        blobs = np.arange(nearest_hundred).reshape(49,49)#(len(self.gid_edge_dict.keys())//100,len(self.gid_edge_dict.keys())//100))
        selected1 = [blobs[i,:] for i in range(0, 49) if i%2==0]+[blobs[i,:] for i in range(0, 10)]#len(self.gid_edge_dict.keys())//100) if i%2==0]
        selected2 = [blobs[i,:] for i in range(10, 49)  if i%2!=0]#len(self.gid_edge_dict.keys())//100) if i%2 != 0]
        selected1 = [i for ar in selected1 for i in ar]
        selected2 = [i for ar in selected2 for i in ar]
        node_color = [.3 if idx in selected1  else .7 if idx in selected2 else .7 for idx, n in enumerate(L.nodes_iter())]
        nx.draw(L, node_size=60, cmap=plt.get_cmap('Spectral_r'), node_color=node_color, pos=pos)  # np.array(positions))
        plt.show()
        plt.savefig(os.path.join(self.pred_session_run_path, 'priors_graph_predicted.png'))
        plt.close()

    def draw_segmentation(self, dirpath, ridge=True, valley=True,
                          draw_sublevel_set = False,invert=False, name=None):
        X = self.X #if not invert else self.Y
        Y = self.Y #if not invert else self.X
        original_image = self.image

        self.use_ridge_arcs = ridge
        self.use_valley_arcs = valley

        # black_box = np.zeros((X, Y)) if not invert else np.zeros(
        #     (Y, X))
        cmap = cm.get_cmap('bwr')
        # cmap.set_under('black')
        # cmap.set_bad('black')
        plt.set_cmap(cmap)
        # fig = plt.imshow(black_box, cmap=cmap, alpha=None, vmin=0)
        # plt.axis('off')
        # fig.axes.get_xaxis().set_visible(False)
        # fig.axes.get_yaxis().set_visible(False)

        original_image = np.stack((original_image.astype(np.float32),) * 3, axis=-1)

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
            if not draw_sublevel_set:
                prediction = self.node_gid_to_prediction[gid]
                partition = self.node_gid_to_partition[gid]
                label =self.node_gid_to_label[gid]
                gnode = self.gid_gnode_dict[gid]
            else:
                prediction = [0.,1.]#self.node_gid_to_prediction[gid]
                partition = 'test'#self.node_gid_to_partition[gid]
                label = self.node_gid_to_label[gid]
            if isinstance(prediction,
                          (int, np.integer)) or isinstance(prediction, (float, np.float)):
                label_color = cmap(float(prediction))
            else:
                if len(prediction) == 3:
                    label_color = cmap(0.56) if float(prediction[2]) > 0.5 else cmap(float(prediction[1]))
                else:
                    if type(prediction) == list:
                        if prediction == []:
                            #print('perd ' , prediction)
                            #print(gnode.gid)
                            #print(partition)
                            #print(self.node_gid_to_feature[gnode.gid])
                            continue
                        label_color = cmap(float(prediction[len(prediction) - 1]))
                    else:
                        label_color = cmap(prediction)

            if original_image is not None:
                x = 1#1#1#1#1#  if invert else 0
                y = 0#0#0#0#0#0  if invert else 1
                scale = 255.
                if partition == 'train':
                    label_color = [255, 51, 255]
                    scale = 1
                if partition == 'val':
                    label_color = [51, 255, 51]
                    scale = 1
                if draw_sublevel_set:
                    label_color = [190,0, 0]  if gnode.sublevel_set else [0,0, 255]
                    scale=1
                if len(mapped_image.shape) == 2:
                    for p in np.array(gnode.points):
                        if len(p) == 1:
                            continue
                        mapped_image[int(p[x]), int(p[y])] = int(label_color[0] * scale)
                        mapped_image[int(p[x]), int(p[y])] = int(label_color[1] * scale)
                        mapped_image[int(p[x]), int(p[y])] = int(label_color[2] * scale)

                        msc_ground_seg_color = cmap(float(label[1]))
                        label_map_image[int(p[x]), int(p[y])] = int(msc_ground_seg_color[0] * 255)
                        label_map_image[int(p[x]), int(p[y])] = int(msc_ground_seg_color[1] * 255)
                        label_map_image[int(p[x]), int(p[y])] = int(msc_ground_seg_color[2] * 255)
                else:
                    for p in np.array(gnode.points):
                        if len(p) == 1:
                            continue
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
            if name is None:
                inf_name = 'inference'
                gs_name = 'groundseg'
            else:
                inf_name = name
                gs_name = name + '_grndseg'

            plt.figure()
            plt.title("Input Image")
            import matplotlib as mplt
            max_val = np.max(mapped_image)
            min_val = np.min(mapped_image)
            mapped_image = (mapped_image.astype(np.float32) - min_val) / (max_val - min_val)
            plt.imsave(os.path.join(dirpath, inf_name+'.png'),mapped_image.astype(np.float32))

            max_val = np.max(label_map_image)
            min_val = np.min(label_map_image)
            label_map_image = (label_map_image.astype(np.float32) - min_val) / (max_val - min_val)
            plt.imsave(os.path.join(dirpath , gs_name+'.png'),label_map_image.astype(np.float32))




            # Img = Image.fromarray(map_im.astype('uint8'))  # .astype(np.float32))#mapped_img)
            # Img.save(os.path.join(dirpath, 'inference.png'))
            #
            # Img = Image.fromarray(
            #     lab_map_im.astype('uint8'))  # .astype(np.float32))#mapped_img)
            # Img.save(os.path.join(dirpath , 'groundseg.png'))

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
            #gnode.features = np.array(G.node[node]["features"])

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