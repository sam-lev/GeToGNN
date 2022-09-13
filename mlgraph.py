import os
import numpy as np
import time
import samply
import json
from networkx.readwrite import json_graph
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

from ui.arcselector import ArcSelector
from getofeaturegraph import GeToFeatureGraph
from ml.features import get_points_from_vertices
from data_ops.utils import grow_box
from data_ops.utils import tile_region
from ml.utils import pout

class MLGraph(GeToFeatureGraph):
    def __init__(self,**kwargs):
        self.run_num = kwargs['run_num']

        '''self.params = {}
        if parameter_file_number is None:
            self.params = kwargs
        else:
            for param in kwargs:
                self.params[param] = kwargs[param]
        run_num=0, parameter_file_number = None,
                 geomsc_fname_base = None, label_file=None,
                 model_name=None, load_feature_graph_name=False,image=None,

                '''

        super(MLGraph, self).__init__(parameter_file_number=kwargs['parameter_file_number'],
                                      name=kwargs['name'], geomsc_fname_base=kwargs['geomsc_fname_base'],
                                      label_file=kwargs['label_file'], image=kwargs['image'],
                                      write_folder=kwargs['write_folder'],
                                      model_name=kwargs['model_name'],
                                      run_num=kwargs['run_num'],
                                      load_feature_graph_name=kwargs['load_feature_graph_name'])
        # super(MLGraph, self).__init__(parameter_file_number=parameter_file_number, run_num=run_num,
        #                               name=kwargs['name'],geomsc_fname_base=geomsc_fname_base,
        #                               label_file=label_file,image=image,
        #                               write_folder=kwargs['write_folder'],
        #                               model_name=model_name,
        #                               load_feature_graph_name=load_feature_graph_name)
        #                              params=self.params)


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

    def box_select_geomsc_training(self, x_range, y_range, boxes=None, image=None):

        # model_type = os.path.join(self.LocalSetup.project_base_path, 'model_type.txt')
        # logged_model = open(model_type, 'r')
        # m = logged_model.readlines()
        # logged_model.close()
        #
        # print("    * read model", m)
        # if m[0] != 'unet':
        if boxes is not None:

            X_BOX = []
            Y_BOX = []

            box_sets = []
            for box in boxes:
                box_set = tile_region(step_X=64, step_Y=64, step=0.5,
                                      Y_START=box[2],Y_END=box[3],
                                      X_START=box[0],X_END=box[1])
                box_sets.append(box_set)


            for box_pair in box_sets:
                for box in box_pair:
                    X_BOX.append(box[0])
                    Y_BOX.append(box[1])


            self.X_BOX = X_BOX
            self.Y_BOX = Y_BOX
            x_range_select = X_BOX
            y_range_select = Y_BOX
        else:
            x_range_select = x_range
            y_range_select = y_range

        self.selection_type = 'box'
        xs = np.array(x_range_select).shape

        # if len(xs) == 1:
        #     x_range = [x_range]
        #     y_range = [y_range]


        self.x_box = x_range_select
        self.y_box = y_range_select

        self.selected_positive_arcs = set()
        self.selected_negative_arcs = set()
        self.selected_positive_arc_ids = set()
        self.selected_negative_arc_ids = set()
        self.selected_test_arcs = set()
        self.selected_test_arc_ids = set()

        self.node_gid_to_partition = {}

        # deal with priors with incident edge and without adjacent prior( -1 gid)
        # isolated = self.gid_gnode_dict[-1]
        # isolated.partition = 'train'
        # isolated.label = [1., 0.]

        if len(x_range_select) == 1:#np.array(x_range).shape == np.array([6,9]).shape:
            x_ranges = x_range_select[0]
            y_ranges = y_range_select[0]
            for gnode in self.gid_gnode_dict.values():
                #idx = self.make_arc_id(arc)
                gid = gnode.gid
                self.node_gid_to_partition[gnode.gid] = 'test'
                in_box = False
                contained = []
                for p in gnode.points:
                    if x_ranges[0] <= p[1] <= x_ranges[1] and y_ranges[0] <= p[0] <= y_ranges[1]:
                        in_box = True
                if in_box and self.node_gid_to_partition[gnode.gid] != 'val':# and gnode.z != 1:
                    gnode.box = 1
                    gnode.partition = 'train'
                    self.node_gid_to_partition[gnode.gid] = 'train'
                    if gnode.label[1] > 0:
                        self.selected_positive_arcs.add(gnode)
                        self.selected_positive_arc_ids.add(gid)
                    else:
                        self.selected_negative_arcs.add(gnode)
                        self.selected_negative_arc_ids.add(gid)
                elif self.node_gid_to_partition[gnode.gid] == '' and self.node_gid_to_partition[gnode.gid] != 'val':
                    gnode.box = 0
                    gnode.partition = 'test'
                    self.node_gid_to_partition[gnode.gid] = 'test'
                    self.selected_test_arcs.add(gnode)
                    self.selected_test_arc_ids.add(gid)
        else:


            for gnode in self.gid_gnode_dict.values():  # self.msc.arcs:
                gid = gnode.gid
                self.node_gid_to_partition[gnode.gid] = 'test'


            range_group = zip(x_range_select, y_range_select)
            empty_count = 0
            for x_rng , y_rng in range_group:

                for gnode in self.gid_gnode_dict.values():#self.msc.arcs:
                    gid = gnode.gid
                    #self.node_gid_to_partition[gnode.gid] = ''
                    p1 = gnode.points[0]
                    p2 = gnode.points[-1]
                    in_box = False
                    for p in gnode.points:#(p1, p2):
                        if x_rng[0] <= p[1] <= x_rng[1] and y_rng[0] <= p[0] <= y_rng[1]:
                            in_box = True
                    if in_box and self.node_gid_to_partition[gnode.gid] != 'val':#'#gnode.z != 1:
                        gnode.box = 1
                        gnode.partition = 'train'
                        self.node_gid_to_partition[gnode.gid] = 'train'
                        if gnode.label[1] > 0:
                            self.selected_positive_arcs.add(gnode)
                            self.selected_positive_arc_ids.add(gid)
                        else:
                            self.selected_negative_arcs.add(gnode)
                            self.selected_negative_arc_ids.add(gid)
                    elif self.node_gid_to_partition[gnode.gid] != 'train' and self.node_gid_to_partition[gnode.gid] != 'val':
                        gnode.box = 0
                        gnode.partition = 'test'
                        self.node_gid_to_partition[gnode.gid] = 'test'
                        self.selected_test_arcs.add(gnode)
                        self.selected_test_arc_ids.add(gid)
        train_length = len(self.selected_positive_arc_ids) + len(self.selected_negative_arc_ids)


        # deal with priors with incident edge and without adjacent prior( -1 gid)
        # isolated = self.gid_gnode_dict[-1]
        # isolated.partition = 'train'
        # isolated.label = [1, 0]

        num_test = len(self.selected_test_arc_ids)
        if num_test == 0:
            dummy_test_node = list(self.gid_gnode_dict.values())[1]
            dummy_test_node.partition = 'test'
            self.node_gid_to_partition[dummy_test_node.gid] = 'test'
            dn = set()
            dn_g = set()
            dn_g.add(dummy_test_node)
            dn.add(dummy_test_node.gid)
            self.selected_test_arc_ids = dn
            self.selected_test_arcs = dn_g

        self.positive_arc_ids = self.selected_positive_arc_ids
        self.negative_arc_ids = self.selected_negative_arc_ids
        self.positive_arcs = self.selected_positive_arcs
        self.negative_arcs = self.selected_negative_arcs
        self.train_classes = [self.positive_arc_ids , self.negative_arc_ids]
        self.all_test_and_val = self.selected_test_arcs


        return self.train_classes , self.all_test_and_val, (self.x_box, self.y_box)

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
    def predicted_msc(self, predictions, labels, segmentation, threshhold):
        segmentation = np.ones(self.image.shape)*0.5
        for gid in self.gid_gnode_dict.keys():
            label = self.node_gid_to_label[gid]
            pred = self.node_gid_to_prediction[gid]
            gnode = self.gid_gnode_dict[gid]
            points = get_points_from_vertices([gnode])
            for p in points:
                segmentation[p[1],p[0]] = pred[1]#1 if pred[1] > threshhold else 0
        return segmentation

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
        self.subgraph_sample_set, self.subgraph_sample_set_ids, \
        subgraph_positive_arcs, subgraph_negative_arcs = self.cvt_sample_subgraphs(sample_gnode_set=self.gid_gnode_dict.values(),
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
        train_pos_copy = train[0].copy()
        train_neg_copy = train[1].copy()
        train_copy = train_pos_copy.union(train_neg_copy)
        for vl in train_copy.intersection(val):
            train_copy.remove(vl)
        if len(train_copy) == 0:
            return False
        for vl in val:
            train_pos_copy.discard(vl)
            train_neg_copy.discard(vl)
        print("    * : size train and val after removal of overlap: ", len(train_pos_copy), len(train_pos_copy), ' val ', len(val))
        invalid = len(train_pos_copy) == 0 or len(train_neg_copy) == 0
        return not invalid


    def get_train_test_val_subgraph_split(self, validation_hops = 1, validation_samples = 1,
                                          test_hops = None, test_samples = None, multiclass= False
                                          , collect_validation=False):
        collect_validation = False
        if collect_validation:

            self.validation_set , self.validation_set_ids, _ , _ = self.cvt_sample_validation_set(hops = validation_hops,
                                                                                                  samples= validation_samples)
            for gid in self.validation_set_ids["positive"].union(self.validation_set_ids["negative"]):

                self.node_gid_to_partition[gid] = 'val'
            
            all_validation = self.validation_set_ids["positive"].union(self.validation_set_ids["negative"])
            dummy_validation = 0
        else:
            dummy_validation_pos = list(self.gid_gnode_dict.values())[0]
            dummy_validation_neg = list(self.gid_gnode_dict.values())[1]
            dummy_validation_pos.partition = 'val'
            dummy_validation_neg.partition = 'val'
            self.node_gid_to_partition[dummy_validation_pos.gid] = 'val'
            self.node_gid_to_partition[dummy_validation_neg.gid] = 'val'
            dp = set()
            dn = set()
            dp_g = set()
            dn_g = set()
            dp_g.add(dummy_validation_pos)
            dn_g.add(dummy_validation_neg)
            dp.add(dummy_validation_pos.gid)
            dn.add(dummy_validation_neg.gid)
            self.validation_set_ids = {"positive": dp, "negative": dn}
            self.validation_set = dp_g.union(dn_g)
            for gid in self.validation_set_ids["positive"].union(self.validation_set_ids["negative"]):

                self.node_gid_to_partition[gid] = 'val'
            all_validation = dp.union(dn)

        all_selected = self.selected_positive_arc_ids.union(self.selected_negative_arc_ids)

        if test_samples is not None:

            self.test_set, self.test_set_ids, _, _ =self.cvt_sample_test_set(samples=test_samples
                                            , hops=test_hops)
            all_test = self.test_set_ids["positive"].union(self.test_set_ids["negative"])
        #-- dict mapping node ids to label
        #-- dict mapping node ids to index in feature tensor
        gid_feat_idx = 0

        pout(["REMOVED FEATURE PER NODE"])

        for gnode in self.gid_gnode_dict.values():


            partition = self.node_gid_to_partition[gnode.gid]

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
            nx_gid = self.node_gid_to_graph_idx[gnode.gid]
            node = self.G.node[nx_gid]
            #
            #node["features"] =  features#gnode.features.tolist()

            node["gid"] = gnode.gid
            #getoelm = self.gid_geto_elm_dict[gnode.gid]
            #polyline = getoelm.points
            #node["geto_elm"] = polyline
            node["key"] = gnode.key
            node["box"] = gnode.box
            node["partition"] = partition
            # assign partition to node
            node["train"] = partition == 'train'
            node["sublevel_set_id"] = [0, -1]
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

                print(" FDFSDFSDFSDFSDFSDF")

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
            gid_feat_idx += 1

        self.G_dict = json_graph.node_link_data(self.G)
        # s1 = json.dumps(self.G_dict)  # input graph
        # s2 = json.dumps(self.node_gid_to_feat_idx)  # dict: nodes to ints
        # s3 = json.dumps(self.node_gid_to_label)  # dict: node_id to class

        end_time = time.time()
        return (self.G_dict, self.node_gid_to_feat_idx , self.node_gid_to_label, self.features)

    def complex_sublevel_training_set(self, level_id, num_sublevel_sets, subgraph=None, map_labels=False):
        # #for gnode in self.gid_gnode_dict.values():
        #
        #     partition = self.node_gid_to_partition[gnode.gid]
        #
        sublevel_training_set = [gnode.gid for gnode in self.gid_gnode_dict.values() if self.node_gid_to_partition[gnode.gid] == 'train' and (gnode.sublevel_set and (0 < gnode.level_id <= level_id))]

        sublevel_set = [gnode.gid for gnode in self.gid_gnode_dict.values() if gnode.sublevel_set and (0 < gnode.level_id <= level_id)]
        #     if gnode.sublevel_set:# and partition == 'train':
        #         sublevel_training_set.append(gnode.gid)
        superlevel_training_set = [gnode.gid for gnode in self.gid_gnode_dict.values() if self.node_gid_to_partition[gnode.gid] == 'train']
        #     if partition == 'train':#not gnode.sublevel_set and partition == 'train':
        #         superlevel_training_set.append(gnode.gid)

        # mark train then remainder test
        for sup_gid in sublevel_set:
            sub_gid = subgraph.sup_gid_to_sub_dict[sup_gid]
            nx_gid = subgraph.node_gid_to_graph_idx[sub_gid]
            node = subgraph.G.node[nx_gid]

            sub_gnode = subgraph.gid_gnode_dict[sub_gid]
            #
            # node["features"] =  features#gnode.features.tolist()


            node["sublevel_set_id"] = [num_sublevel_sets, level_id]

            node["gid"] = sub_gnode.gid
            # getoelm = self.gid_geto_elm_dict[gnode.gid]
            # polyline = getoelm.points
            # node["geto_elm"] = polyline
            node["key"] = sub_gnode.key
            node["box"] = sub_gnode.box
            sup_partition = self.node_gid_to_partition[sup_gid]
            node["partition"] = sup_partition
            # assign partition to node
            node["train"] = sup_partition == 'train'

            # sublevel_id = -1
            # for id, sublevel_set in enumerate(sublevel_training_sets):
            #     if gnode.gid in sublevel_set:  # and partition=='train':
            #         sublevel_id = id if sublevel_id == -1 else sublevel_id
            #
            # node["sublevel_set_id"] = [len(sublevel_training_sets), sublevel_id]
            node["test"] = sup_partition == 'test'
            node["val"] = sup_partition == 'val'
            if map_labels:
                node["label"] = [
                        int(sup_gid in self.negative_arc_ids),
                        int(sup_gid in self.positive_arc_ids)
                    ]
                subgraph.node_gid_to_label = self.node_gid_to_label
            else:
                node["label"] = subgraph.node_gid_to_label[sub_gid]

            node["prediction"] = []

            subgraph.G.node[nx_gid] = node


        return superlevel_training_set, sublevel_set, subgraph.G

    def get_complex_informed_subgraph_split(self, sublevel_training_sets,
                                                          validation_hops=1,
                                                          validation_samples=1,
                                                          test_hops=None,
                                                          test_samples=None,
                                                          multiclass=False,
                                                          collect_validation=False):



        total_sublevel_sets = len(sublevel_training_sets)
        sublevel_training_set = sublevel_training_sets[0] # change to iterate when have more complexes
        sublevel_training_set_id = 0


        pout(("NUMBER SUBLEVEL SETS in ml grpah get somplex informed", total_sublevel_sets))
        for j,i in enumerate(sublevel_training_sets):
            pout(("len "+str(j), len(i)))
        collect_validation = False
        if collect_validation:

            self.validation_set, self.validation_set_ids, _, _ = self.cvt_sample_validation_set(hops=validation_hops,
                                                                                                samples=validation_samples)
            for gid in self.validation_set_ids["positive"].union(self.validation_set_ids["negative"]):
                self.node_gid_to_partition[gid] = 'val'

            all_validation = self.validation_set_ids["positive"].union(self.validation_set_ids["negative"])
            dummy_validation = 0
        else:
            dummy_validation_pos = list(self.gid_gnode_dict.values())[0]
            dummy_validation_neg = list(self.gid_gnode_dict.values())[1]
            dummy_validation_pos.partition = 'val'
            dummy_validation_neg.partition = 'val'
            self.node_gid_to_partition[dummy_validation_pos.gid] = 'val'
            self.node_gid_to_partition[dummy_validation_neg.gid] = 'val'
            dp = set()
            dn = set()
            dp_g = set()
            dn_g = set()
            dp_g.add(dummy_validation_pos)
            dn_g.add(dummy_validation_neg)
            dp.add(dummy_validation_pos.gid)
            dn.add(dummy_validation_neg.gid)
            self.validation_set_ids = {"positive": dp, "negative": dn}
            self.validation_set = dp_g.union(dn_g)
            for gid in self.validation_set_ids["positive"].union(self.validation_set_ids["negative"]):
                self.node_gid_to_partition[gid] = 'val'
            all_validation = dp.union(dn)

        all_selected = self.selected_positive_arc_ids.union(self.selected_negative_arc_ids)

        if test_samples is not None:
            self.test_set, self.test_set_ids, _, _ = self.cvt_sample_test_set(samples=test_samples
                                                                              , hops=test_hops)
            all_test = self.test_set_ids["positive"].union(self.test_set_ids["negative"])
        # -- dict mapping node ids to label
        # -- dict mapping node ids to index in feature tensor
        gid_feat_idx = 0



        for gnode in self.gid_gnode_dict.values():

            partition = self.node_gid_to_partition[gnode.gid]

            if gnode.label is not None and gnode.gid not in all_selected:
                label = self.node_gid_to_label[gnode.gid]  # [0 , 1] if gnode.label > 0 else [1, 0]#[
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
            nx_gid = self.node_gid_to_graph_idx[gnode.gid]
            node = self.G.node[nx_gid]
            #
            # node["features"] =  features#gnode.features.tolist()

            node["gid"] = gnode.gid
            # getoelm = self.gid_geto_elm_dict[gnode.gid]
            # polyline = getoelm.points
            # node["geto_elm"] = polyline
            node["key"] = gnode.key
            node["box"] = gnode.box
            node["partition"] = partition
            # assign partition to node
            node["train"] = partition == 'train'

            sublevel_id = -1
            for id, sublevel_set in enumerate(sublevel_training_sets):
                if gnode.gid in sublevel_set:# and partition=='train':
                    sublevel_id = id  if sublevel_id == -1 else sublevel_id

            node["sublevel_set_id"] = [len(sublevel_training_sets), sublevel_id]
            node["test"] = partition == 'test'
            node["val"] = partition == 'val'
            node["label"] = label
            if self.selection_type == 'map':
                node["label_accuracy"] = gnode.label_accuracy
            node["prediction"] = []
            self.node_gid_to_prediction[gnode.gid] = []

            # labeled nodes assigned as train, test, or val
            if self.params['union_space']:
                node["label"] = label

                print(" FDFSDFSDFSDFSDFSDF")

                if self.selection_type == 'map':
                    node["label_accuracy"] = gnode.label_accuracy
                node["prediction"] = []
                modified = 0
                group = None
                if gnode.z == 1:
                    pout(["SHOULDNT BE HERE TRAIN TEST VAL "])
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
            gid_feat_idx += 1

        self.G_dict = json_graph.node_link_data(self.G)
        # s1 = json.dumps(self.G_dict)  # input graph
        # s2 = json.dumps(self.node_gid_to_feat_idx)  # dict: nodes to ints
        # s3 = json.dumps(self.node_gid_to_label)  # dict: node_id to class

        end_time = time.time()
        return (self.G_dict, self.node_gid_to_feat_idx, self.node_gid_to_label, self.features)

    def feature_importance(self, features, feature_names, labels, n_informative = 3, plot=False):
        # Build a classification task using 3 informative features
        # Build a forest and compute the impurity-based feature importances

        features = np.array(list(features.values()))
        labels = list(labels.values())
        forest = ExtraTreesClassifier(n_estimators=250,
                                      random_state=666)

        X = features
        y = np.array(labels)
        forest.fit(X, y)
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                     axis=0)
        # array of of index of decending importance feature
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("    * Feature ranking: ")
        print("importance shape",importances.shape)
        print("num_names", len(feature_names))
        print("indices shape",indices.shape)
        self.feat_importance_dict = {}
        if feature_names is not None:
            for f in range(len(feature_names)):#X.shape[1]):
                #                       fname_index   , name,                 importance
                feat_importance_info = [f + 1, feature_names[indices[f]], importances[indices[f]]]
                self.feat_importance_dict[indices[f]] = feat_importance_info

        # Plot the impurity-based feature importances of the forest
        if plot:
            plt.figure()
            plt.title("Feature importances")
            plt.bar(range(X.shape[1]), importances[indices],
                    color="r", yerr=std[indices], align="center")
            plt.xticks(range(X.shape[1]), indices)
            plt.xlim([-1, X.shape[1]])
            plt.show()
            return indices, feature_names
        return self.feat_importance_dict

    def write_feature_importance(self):
        msc_pred_file = os.path.join(self.experiment_folder,'features', "feat_importance.txt")
        print("&&&& writing feature importances in: ", msc_pred_file)
        pred_file = open(msc_pred_file, "w+")
        top_ten = 10
        for importance_rank in self.feat_importance_dict.keys():  # self.gid_gnode_dict.values():
            feat_importance_info = self.feat_importance_dict[importance_rank]
            feat_idx = feat_importance_info[0]
            feat_name = feat_importance_info[1]
            feat_importance = feat_importance_info[2]
            pred_file.write(str(importance_rank) + ' ' + str(feat_idx) + ' ' +
                            str(feat_name) + ' ' + str(feat_importance)+ "\n")
            if top_ten > 0:
                print("importance rank "+str(importance_rank) + ' feat_idx ' + str(feat_idx))
                print('     feat_name ' +str(feat_name) + ' importance ' + str(feat_importance))
                top_ten = top_ten - 1
        pred_file.close()

    def write_arc_predictions(self, dir=None, path=None, msc=None, name=''):
        if name != '':
            name = name+'.'
        if dir is not None:
            pred_session_run_path = dir
        else:
            pred_session_run_path = self.pred_session_run_path
        msc_pred_file = os.path.join(pred_session_run_path, name+"preds.txt")
        print("&&&& writing predictions in: ", msc_pred_file)
        pred_file = open(msc_pred_file,"w+")
        for gid in self.node_gid_to_prediction.keys():#self.gid_gnode_dict.values():
            pred_file.write(str(gid)+ ' '+str(self.node_gid_to_prediction[gid]) + "\n")
        pred_file.close()

    def write_graph_statistics(self, total_number_nodes, total_training_nodes, total_number_edges,
                               total_test_nodes=None, total_length_training_nodes=None,
                               total_length_positive_training_nodes=None,
                               total_length_positive_nodes=None,
                               total_length_test_nodes=None,
                               total_length_nodes=None,
                               total_nodes=None, total_foreground_nodes=None,
                               fname = 'graph_statistics'):

        msc_pred_file = os.path.join(self.pred_session_run_path, fname+'.txt')
        print("&&&& writing percentages in: ", msc_pred_file)
        pred_file = open(msc_pred_file, "w+")
        pred_file.write("num_nodes " + str(total_number_nodes)+"\n")
        pred_file.write("num_edges " + str(total_number_edges)+"\n")
        pred_file.write("total_test_nodes "+ str(total_test_nodes)+"\n")
        pred_file.write("total_training_nodes " + str(total_training_nodes )+"\n")
        pred_file.write("total_length_test_nodes " + str(total_length_test_nodes)+"\n")
        pred_file.write("total_length_train_nodes " +str(total_length_training_nodes)+"\n")
        pred_file.write("total_length_nodes " + str(total_length_nodes)+"\n")
        pred_file.write("total_nodes " + str(total_nodes) + "\n")
        pred_file.write("total_foreground_nodes " + str(total_foreground_nodes) + "\n")
        pred_file.write("total_background_nodes " + str(total_nodes - total_foreground_nodes) + "\n")
        pred_file.write("percent_graph_training " + str(100.0*(total_training_nodes/total_number_nodes)) + "\n")
        pred_file.write("Total Positive nodes length "+str(total_length_positive_nodes)+'\n')
        pred_file.write("Total Negative nodes length " + str(total_length_nodes - total_length_positive_nodes) + '\n')
        pred_file.write("percent_length_training " + str(
            100.0 * (total_length_training_nodes / total_length_nodes)) + "\n")
        pred_file.write("percent_length_positive_training " + str(
            100.0 * (total_length_positive_training_nodes / total_length_positive_nodes)) + "\n")
        pred_file.close()


    def write_training_percentages(self, dir=None, msc_segmentation=None, train_regions=None,
                                   path=None, msc=None, name='',total_positive_training_pixels=None,
                                                  total_positive_pixels=None, object='region'):
        if name != '':
            name = name+'.'
        if dir is not None:
            pred_session_run_path = dir
        else:
            pred_session_run_path = self.pred_session_run_path
        tp = ''
        if msc_segmentation is not None:
            tp = 'msc'
        if train_regions is not None:
            msc_segmentation=train_regions
            tp = 'region'
        msc_pred_file = os.path.join(pred_session_run_path, object+"_percents.txt")
        print("&&&& writing percentages in: ", msc_pred_file)
        pred_file = open(msc_pred_file,"w+")
        pred_file.write("num_pixels "+str( self.image.shape[0] * self.image.shape[1])+"\n")
        pred_file.write("num_pixels_training " + str( np.sum(msc_segmentation))+"\n")
        pred_file.write(tp+"_percent " + str( 100.0 * (np.sum(msc_segmentation) / (self.image.shape[0] * self.image.shape[1]))) + '\n')
        if total_positive_training_pixels is not None:
            pred_file.write("total_positive_training_pixels " + str(total_positive_training_pixels) + "\n")
            pred_file.write("total_positive_pixels " + str(total_positive_pixels) + "\n")
            pred_file.write("percent_positive_training " + str(
                100.0 * (total_positive_training_pixels / total_positive_pixels)) + "\n")

        pred_file.close()

    def write_training_graph_percentages(self, dir=None, msc_segmentation=None, graph_orders=None,
                                   path=None, msc=None, name=''):
        if name != '':
            name = name+'.'
        if dir is not None:
            pred_session_run_path = dir
        else:
            pred_session_run_path = self.pred_session_run_path
        tp = 'polyline'
        msc_pred_file = os.path.join(pred_session_run_path, "region_percents.txt")
        print("&&&& writing percentages in: ", msc_pred_file)
        pred_file = open(msc_pred_file,"w+")
        pred_file.write("num_lines "+str( self.image.shape[0] * self.image.shape[1])+"\n")
        pred_file.write("num_lines_training " + str( graph_orders[1])+"\n")
        pred_file.write(tp+"_percent " + str( 100.0 * (graph_orders[1]/float(graph_orders[0]))))
        pred_file.close()
    # write to file per line 'gid partition'
    # where partition = {train:0, test:1, val:2}
    def write_gnode_partitions(self, dir=None, name='', x_box=None,y_box=None):
        if name != '':
            name = name+'.'
        if dir is not None:
            pred_session_run_path = dir
        else:
            pred_session_run_path = self.pred_session_run_path
        partitions_file = os.path.join(pred_session_run_path, name+'partitions.txt')
        print("... Writing partitions file to:", partitions_file)
        partitions_file = open(partitions_file, "w+")
        for gid in self.gid_gnode_dict.keys():
            node_partition = self.node_gid_to_partition[gid]
            p = str(0) if node_partition == 'train' else str(2)
            p = str(1) if node_partition == 'test' else str(2)
            partitions_file.write(str(gid) + ' ' + str(p) + "\n")
        partitions_file.close()

    def write_selection_bounds(self,dir=None, name='',x_box=None,y_box=None, mode='w+'):
        if x_box is None:
            x_box = self.x_box
            y_box = self.y_box
        if dir is not None:
            pred_session_run_path = dir
        else:
            pred_session_run_path = self.pred_session_run_path
        print("    * : in write of bounds ")
        window_file = os.path.join(pred_session_run_path, 'window.txt')
        print("... Writing bounds file to:", window_file)

        start = not os.path.exists(window_file)

        window_file = open(window_file, mode)

        X = self.image.shape[0]
        Y = self.image.shape[1] if len(self.image.shape) == 2 else self.image.shape[2]
        ##if start:
        window_file.write(str(X) + "\n")
        window_file.write(str(Y) + "\n")
        no_new_line = len(y_box)

        for x_box, y_box in zip(x_box,y_box):
            no_new_line -= 1
            end_line = '' if 0 == no_new_line else "\n"
            window_file.write('x_box' + ' ' + str(x_box[0]) +','+str(x_box[1])+ "\n")
            window_file.write('y_box' + ' ' + str(y_box[0]) +','+str(y_box[1])+ end_line)
        window_file.close()

    def write_homophily_stats(self, homo_ratio, homo_train_ratio, sublevel_homophily_ratio, sublevel_homophily_train_ratio , dir=None ):
        if dir is not None:
            pred_session_run_path = dir
        else:
            pred_session_run_path = self.pred_session_run_path

        homo_file = os.path.join(pred_session_run_path, "homophily_ratio.txt")
        pout(("&&&& writing homophily ratio in", homo_file))
        homo_file = open(homo_file, "w+")
        homo_file.write("homophily_ratio " + str(homo_ratio) + "\n")
        homo_file.write("homophily_train_ratio " + str(homo_train_ratio) + "\n")

        homo_file.write("sublevel_homophily_ratio " + str(sublevel_homophily_ratio) + "\n")
        homo_file.write("sublevel_homophily_train_ratio " + str(sublevel_homophily_train_ratio) + "\n")
        homo_file.close()

        return 0
    def write_class_stats(self, positive , negative, pos_neg_ratio, positive_train, negative_train,
                          positive_sublevel,
                          negative_sublevel,
                          pos_neg_sublevel_ratio,
                          positive_sublevel_train,
                          negative_sublevel_train,
                          dir=None):
        if dir is not None:
            pred_session_run_path = dir
        else:
            pred_session_run_path = self.pred_session_run_path

        class_balance_file = os.path.join(pred_session_run_path, "class_balance.txt")
        pout(("&&&& writing homophily ratio in", class_balance_file))
        class_balance_file = open(class_balance_file, "w+")
        class_balance_file.write("pos_neg_ratio " + str(pos_neg_ratio) + "\n")
        class_balance_file.write("positive " + str(positive) + "\n")
        class_balance_file.write("negative " + str(negative) + "\n")
        class_balance_file.write("pos_neg_train_ratio " + str(positive_train/float(negative_train))+"\n")
        class_balance_file.write("positive_train " + str(positive_train) + "\n")
        class_balance_file.write("negative_train " + str(negative_train) + "\n")
        #
        class_balance_file.write("sublevel_pos_neg_ratio " + str(pos_neg_sublevel_ratio) + "\n")
        class_balance_file.write("sublevel_positive " + str(positive_sublevel) + "\n")
        class_balance_file.write("sublevel_negative " + str(negative_sublevel) + "\n")
        class_balance_file.write("sublevel_pos_neg_train_ratio " +str(positive_sublevel_train/float(negative_sublevel_train))+"\n")
        class_balance_file.write("sublevel_positive_train " + str(positive_sublevel_train) + "\n")
        class_balance_file.write("sublevel_negative_train " + str(negative_sublevel_train) + "\n")
        #
        class_balance_file.close()

        return 0

    def record_time(self, time, dir = None, type=''):
        if dir is not None:
            pred_session_run_path = dir
        else:
            pred_session_run_path = self.pred_session_run_path

        window_file = os.path.join(pred_session_run_path, type+'_time.txt')
        print("... Writing time file to:", window_file)

        start = not os.path.exists(window_file)

        window_file = open(window_file, 'w+')


        window_file.write(str(time))

        window_file.close()