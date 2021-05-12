import os
import numpy as np
import time
import samply
import json
from networkx.readwrite import json_graph

from ui.arcselector import ArcSelector
from getofeaturegraph import GeToFeatureGraph

class MLGraph(GeToFeatureGraph):
    def __init__(self,run_num=0, parameter_file_number = None,
                 geomsc_fname_base = None, label_file=None,
                 model_name=None, load_feature_graph_name=False,image=None, **kwargs):

        self.params = {}
        if parameter_file_number is None:
            self.params = kwargs
        else:
            for param in kwargs:
                self.params[param] = kwargs[param]


        super(MLGraph, self).__init__(parameter_file_number=parameter_file_number, run_num=run_num,
                                      name=self.params['name'],geomsc_fname_base=geomsc_fname_base,
                                      label_file=label_file,image=image,
                                      write_folder=self.params['write_folder'],
                                      model_name=model_name,load_feature_graph_name=load_feature_graph_name,
                                      params=self.params)

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
        self.train_classes = [self.positive_arc_ids , self.negative_arc_ids]
        self.all_test_and_val = self.selected_test_arcs
        return self.train_classes , self.all_test_and_val

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

    def write_arc_predictions(self, filename, path=None, msc=None, name=''):
        if name != '':
            name = name+'.'
        msc_pred_file = os.path.join(self.pred_session_run_path, name+"preds.txt")
        print("&&&& writing predictions in: ", msc_pred_file)
        pred_file = open(msc_pred_file,"w+")
        for gid in self.node_gid_to_prediction.keys():#self.gid_gnode_dict.values():
            pred_file.write(str(gid)+ ' '+str(self.node_gid_to_prediction[gid]) + "\n")
        pred_file.close()

    # write to file per line 'gid partition'
    # where partition = {train:0, test:1, val:2}
    def write_gnode_partitions(self, name):
        if name != '':
            name = name+'.'
        partitions_file = os.path.join(self.pred_session_run_path, name+'partitions.txt')
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