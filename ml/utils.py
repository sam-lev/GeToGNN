from __future__ import print_function
import tensorflow as tf
import numpy as np
import random
import json
import sys
import os

import networkx as nx
from networkx.readwrite import json_graph
version_info = list(map(int, nx.__version__.split('.')))
major = version_info[0]
minor = version_info[1]
assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"

WALK_LEN=10
N_WALKS=50

def load_data(prefix='', positive_arcs = [], negative_arcs = [], normalize=True,
              load_walks=False, train_or_test = '', scheme_required = True):
    
    print('loading json graph family')

    require_scheme = scheme_required
    label_scheme = train_or_test
    
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    
    if isinstance(G.nodes()[0], int):
        conversion = lambda n : int(n)
    else:
        conversion = lambda n : n
        
    negative_sample_count = 0
    positive_sample_count = 0
    broken_count = 0
    total_nodes = 0
    val_count = 0
    unlabeled_nodes = 0
    test_count = 0
    #if not train_or_test:
    #    sys.exit("must specify graph as train or test set")

    if positive_arcs and negative_arcs:
        cvt_indices = positive_arcs + negative_arcs

    for node in G.nodes():
        total_nodes+=1
        before = broken_count
        if not 'val' in G.node[node] or not 'test' in G.node[node] or not 'label' in G.node[node]:# or not bool(np.sum(G.node[node]["label"])):
            G.remove_node(node)
            broken_count += 1
            
        if before == broken_count and "label" in G.node[node]:
            if G.node[node]["label"][0] == 1 and G.node[node]['train']:
                negative_sample_count+=1
            if G.node[node]["label"][1] == 1 and G.node[node]['train']:
                positive_sample_count+=1
            if G.node[node]['val']:
                val_count+=1
            if G.node[node]['test']:
                test_count+=1
 
        if before == broken_count  and train_or_test == 'test' and require_scheme and not bool(np.sum(G.node[node]["label"])):
            G.remove_node(node)
            unlabeled_nodes+=1

    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    
    else:
        print("No features present.. Only identity features will be used.")
        feats = None

    
    print('feature shape: ', feats.shape)
    
    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {conversion(k):int(v) for k,v in id_map.items()}
    walks = []
    class_map = json.load(open(prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n : n
    else:
        lab_conversion = lambda n : int(n)

    class_map = {conversion(k):lab_conversion(v) for k,v in class_map.items()}# if bool(np.sum(v))==require_scheme}


    print("Removed {:d} nodes that lacked proper annotations".format(broken_count))
    print("&..Total Nodes: ",total_nodes)
    print("&..Negative Samples: ", negative_sample_count)
    print("&..Positive Samples: ", positive_sample_count)
    print("&..Validation Samples: ", val_count)
    print("&..Unlabeled Samples: ", unlabeled_nodes )

    print("..test nodes: ", test_count)
    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
            G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[n] for n in G.nodes() if G.node[n]['train']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)
    
    if load_walks:
        with open('./data/random_walks/'+load_walks + "-walks.txt") as fp:
            for line in fp:
                walks.append(map(conversion, line.split()))

    return G, feats, id_map, walks, class_map, negative_sample_count, positive_sample_count

def format_data(dual=None, features=None, node_id=None, id_map=None
                , node_classes=None, train_or_test = '', scheme_required = True
                , load_walks=False, normalize=True, test_graph=False):

    #if not train_or_test:
    #sys.exit("must specify graph as train or test set")
    typeG = nx.Graph()
    if dual:
        if type(dual) != type(typeG):
            G = json_graph.node_link_graph(dual)
        else:
            G = dual
        if isinstance(G.nodes()[0], int):
            conversion = lambda n : int(n)
        else:
            conversion = lambda n : n

    feats = features

    if node_id:
        id_map = node_id
        id_map = {conversion(k):int(v) for k,v in id_map.items()}

    class_map = node_classes
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n : n
    else:
        lab_conversion = lambda n : int(n)
    class_map = {conversion(k):lab_conversion(v) for k,v in class_map.items()}


    require_scheme = scheme_required
    if isinstance(G.nodes()[0], int):
        conversion = lambda n : int(n)
    else:
        conversion = lambda n : n
        
    negative_sample_count = 0
    positive_sample_count = 0
    broken_count = 0
    total_nodes = 0
    unlabeled_nodes = 0
    val_count = 0
    test_count = 0
    for node in G.nodes():
        total_nodes+=1
        before = broken_count
        
        if not 'val' in G.node[node] or not 'test' in G.node[node] or not 'train' in G.node[node]:
            G.remove_node(node)
            broken_count += 1
        if before == broken_count and "label" in G.node[node]:
            if G.node[node]["label"][0] > 0 and G.node[node]['train']:
                negative_sample_count+=1
            if G.node[node]["label"][1] > 0 and G.node[node]['train']:
                positive_sample_count+=1
            if G.node[node]['val']:
                val_count+=1
            if G.node[node]['test']:
                test_count+=1
        #if before == broken_count and train_or_test and not G.node[node]['val']:
        #G.node[node]["train"] = label_scheme == 'train'
        #G.node[node]["test"] = label_scheme == 'test'
        if before==broken_count and require_scheme and not bool(np.sum(G.node[node]["label"])):
            G.remove_node(node)
            unlabeled_nodes+=1


    walks = []
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n : n
    else:
        lab_conversion = lambda n : int(n)


    print("Removed {:d} nodes that lacked proper annotations".format(broken_count))
    print("&..total nodes: ",total_nodes)
    print("&..Positive Samples: ",positive_sample_count)
    print("&..Negative Samples: ", negative_sample_count)
    print("&..Validation Samples: ", val_count)
    print("&..Unlabeled Samples: ", unlabeled_nodes )
    print("&..test nodes: ", test_count)
    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
            G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    if normalize and feats is not None and not test_graph:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[n] for n in G.nodes() if G.node[n]['train']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)

    if normalize and feats is not None and test_graph:
        from sklearn.preprocessing import StandardScaler
        test_ids = np.array([id_map[n] for n in G.nodes() if G.node[n]['test']])
        pout(("ID MAP IN UTILS", id_map, "features in utils", feats))
        test_feats = feats[test_ids]
        scaler = StandardScaler()
        scaler.fit(test_feats)
        feats = scaler.transform(feats)
    
    if load_walks:
        print('loading walks...')
        with open(load_walks + "-walks.txt") as fp:
            for line in fp:
                walks.append(map(conversion, line.split()))

    return G, feats, id_map, walks, class_map, negative_sample_count, positive_sample_count

def format_gnn_dicts(dual=None,
                     features=None,
                     # node_id=nx_idx_to_feat_idx,
                     id_map=None,
                     node_classes=None,
                     train_or_test='',
                     scheme_required=True,
                     load_walks=False):
    #(node_gid_to_graph_idx, node_gid_to_feat_idx, node_gid_to_label, G, features):
    # nx_idx_to_feat_idx = {node_gid_to_graph_idx[gid]: feat for gid, feat
    #                       in node_gid_to_feat_idx.items()}
    # nx_idx_to_label_idx = {node_gid_to_graph_idx[gid]: label for gid, label
    #                        in node_gid_to_label.items()}
    # nx_idx_to_getoelm_idx = {self.node_gid_to_graph_idx[gid]: getoelm_idx for gid, getoelm_idx
    #                          in self.gid_to_getoelm_idx.items()} if self.getoelms is not None else None

    G, features, nx_idx_to_feat_idx, _, nx_idx_to_label_idx, _, _ \
        = format_data(dual=dual,
                      features=features,
                      # node_id=nx_idx_to_feat_idx,
                      id_map=id_map,
                      node_classes=node_classes,
                      train_or_test='',
                      scheme_required=True,
                      load_walks=False)
    return G, features, nx_idx_to_feat_idx, nx_idx_to_label_idx

def run_random_walks(G, nodes, num_walks=N_WALKS, walk_len = WALK_LEN):
    print("... Doing walks")
    pairs = []
    for count, node in enumerate(nodes):
        if G.degree(node) == 0:
            continue
        for i in range(num_walks):
            curr_node = node
            for j in range(walk_len):
                next_node = random.choice(G.neighbors(curr_node))
                # self co-occurrences are useless
                if curr_node != node:
                    pairs.append((node,curr_node))
                curr_node = next_node
        if count % 1000 == 0:
            print("Done walks for", count, "nodes")
    return pairs

def random_walk_embedding(G, walk_length, number_walks, out_file, load_graph=False):
    WALK_LEN = walk_length
    N_WALKS = number_walks
    #G_data = json.load(open(graph_file))
    #G = json_graph.node_link_graph(G_data)
    if load_graph:
        G = json_graph.node_link_graph(G)
    nodes = [n for n in G.nodes() if not G.node[n]["val"] and not G.node[n]["test"]]#or 'val' in G.node[n] and (G.node[n]['train'] or G.node[n]['val'])]
    #print(len(nodes))
    G = G.subgraph(nodes)
    pairs = run_random_walks(G, nodes, num_walks=number_walks, walk_len=walk_length)
    dir, fname = os.path.split(os.path.abspath(out_file))
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(out_file+'-walks.txt', "w") as fp:
        fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs]))
    
def get_train_test_val_partitions(node_gid_to_partition, gid_to_node, test_all=False):
    training_set = []
    test_set = []
    val_set = []
    for gid in node_gid_to_partition.keys():
        partition = node_gid_to_partition[gid]
        if partition == 'train':
            training_set.append(gid_to_node[gid])
        if partition == 'val':
            val_set.append(gid_to_node[gid])
        elif not test_all:
            test_set.append(gid_to_node[gid])
        if test_all:
            test_set.append(gid_to_node[gid])
    return training_set, test_set, val_set

#
# Create prediction and label array
# where idx prediction is idx of label for same node
def make_binary_prediction_label_pairs(predictions, labels, threshold=0.5):
    predictions_b = predictions
    labels_b = []
    for l in labels:
        labels_b.append(l)
    predictions_b = np.array(predictions_b)
    labels_b = np.array(labels_b)
    predictions_b[predictions_b >= threshold] = 1.
    predictions_b[predictions_b < threshold] = 0.
    return predictions_b, labels_b

def get_partition_feature_label_pairs(node_gid_to_partition, node_gid_to_feature,
                                      node_gid_to_label, test_all=False):
    training_gid_to_labels = {}
    test_gid_to_labels = {}
    val_gid_to_labels = {}
    training_gid_to_feats = {}
    test_gid_to_feats = {}
    val_gid_to_feats = {}

    all_gid_to_feats = {}
    all_gid_to_labels = {}

    for gid in node_gid_to_partition.keys():
        partition = node_gid_to_partition[gid]
        if partition == 'train':
            training_gid_to_labels[gid] = node_gid_to_label[gid]
            training_gid_to_feats[gid] = node_gid_to_feature[gid]
        if partition == 'val':
            val_gid_to_labels[gid] = node_gid_to_label[gid]
            val_gid_to_feats[gid] = node_gid_to_feature[gid]
        elif not test_all:
            test_gid_to_labels[gid] = node_gid_to_label[gid]
            test_gid_to_feats[gid] = node_gid_to_feature[gid]
        if test_all:
            test_gid_to_labels[gid] = node_gid_to_label[gid]
            test_gid_to_feats[gid] = node_gid_to_feature[gid]
        all_gid_to_feats[gid] = node_gid_to_feature[gid]
        all_gid_to_labels[gid] = node_gid_to_label[gid]

    partition_label_dict = {'train': training_gid_to_labels,
            'test': test_gid_to_labels,
            'val': val_gid_to_labels,
            'all': all_gid_to_labels}
    partition_feature_dict = {'train': training_gid_to_feats,
            'test': test_gid_to_feats,
            'val': val_gid_to_feats,
            'all':all_gid_to_feats}

    return partition_label_dict, partition_feature_dict


def get_merged_features(model):
    feat_idx = 0
    features = []
    node_gid_to_feature = {}
    node_gid_to_feat_idx = {}
    for gid, gnode in model.gid_gnode_dict.items():
        feats = list(model.node_gid_to_standard_feature[gid])
        geomfeats = list(model.node_gid_to_geom_feature[gid])
        combined_feats = feats + geomfeats
        node_gid_to_feature[gid] = np.array(combined_feats)
        feat_idx += 1
        features.append(combined_feats)
        node_gid_to_feat_idx[gid] = feat_idx
    return node_gid_to_feature, node_gid_to_feat_idx, features


def pout(show=None):
    if isinstance(show, list) or isinstance(show, tuple):
        print("    *")
        for elm in show:
            if isinstance(elm, str):
                print("    * ",elm)
            else:
                print("    * ", str(elm))
        print("    *")
def pouts(**show):
    print("    *")
    for item_name, item in show.items():
        print("    *", str(item))
    print("    *")

def get_subgraph_attr(sublevel_samples_dict, subadj_idx, labels=False):
    #for subadj_idx in sublevel_ids:
    sb_name = 'sub_batch' + str(subadj_idx)
    sb_sz_name = sb_name + '_size'
    sb_lb_name = sb_name + '_labels'
    subsamples_i = sublevel_samples_dict[sb_name]
    # self.subbatch_dict[sb_sz_name] = placeholders[sb_sz_name]
    if not labels:
        return subsamples_i
    else:
        subsamples_labels_i = sublevel_samples_dict[sb_lb_name]
        return subsamples_i, subsamples_labels_i

def append_subgraph_attr(sublevel_samples_dict, x, subadj_idx):#, labels=False):
    #for subadj_idx in sublevel_ids:
    sb_name = 'sub_batch' + str(subadj_idx)
    sb_sz_name = sb_name + '_size'
    sb_lb_name = sb_name + '_labels'
    subsamples_i = sublevel_samples_dict[sb_name]
    # self.subbatch_dict[sb_sz_name] = placeholders[sb_sz_name]
    subsamples_i.append(x)
    sublevel_samples_dict[sb_name] = subsamples_i
    return sublevel_samples_dict

def map_fn_subgraph_attr(sublevel_samples_dict, fn, subadj_idx):#, labels=False):
    #for subadj_idx in sublevel_ids:
    sb_name = 'sub_batch' + str(subadj_idx)
    sb_sz_name = sb_name + '_size'
    sb_lb_name = sb_name + '_labels'
    subsamples_i = sublevel_samples_dict[sb_name]
    # self.subbatch_dict[sb_sz_name] = placeholders[sb_sz_name]
    subsamples_i_out = fn(subsamples_i)
    sublevel_samples_dict[sb_name] = subsamples_i_out
    return sublevel_samples_dict

def gather_neighbors(adj, indices, num_samples):
    adj_lists_T = tf.transpose(adj)
    adj_lists_T_nbr = tf.gather(adj_lists_T, indices)
    adj_lists_nbr = tf.transpose(adj_lists_T_nbr)

    adj_lists_all = tf.slice(adj_lists_nbr, [0, 0], [-1, num_samples])


# Dataset class for the retina dataset
# each item of the dataset is a tuple with three items:
# - the first element is the input image to be segmented
# - the second element is the segmentation ground truth image
# - the third element is a mask to know what parts of the input image should be used (for training and for scoring)
class dataset():
    def transpose_first_index(self, x, with_hand_seg=False, with_range=True):
        if with_range:
            x2 =(x[0], x[1], x[2])#(np.transpose(x[0], [1, 0]), x[1])
            #, np.transpose(x[2], [2, 0, 1]))
        else:
            x2 =(x[0], x[1])#(np.transpose(x[0], [1, 0]), x[1])#(np.transpose(x[0], [2, 0, 1]), np.transpose(x[1], [2, 0, 1]), np.transpose(x[2], [2, 0, 1]),
            #      np.transpose(x[3], [2, 0, 1]))
        return x2

    def __init__(self, data_array, split='train', do_transform=False,
                 with_hand_seg=False, with_range=False):

        self.with_hand_seg = with_hand_seg
        self.with_range = with_range

        indexes_this_split = np.arange(len(data_array))#get_split(np.arange(len(retina_array), dtype=np.int), split)
        self.data_array = [self.transpose_first_index(data_array[i],
                                                      self.with_hand_seg,
                                                      with_range=self.with_range) for i in
                             indexes_this_split]



        self.split = split
        self.do_transform = do_transform

    def __getitem__(self, index):
        if self.with_range:
            sample = [self.data_array[index][0],
                      self.data_array[index][1],
                      self.data_array[index][2]]
        else:
            sample = [self.data_array[index][0],
                      self.data_array[index][1]]


        return sample

    def __len__(self):
        return len(self.data_array)

    def get_images(self):
        return np.array([np.array(samp[0],dtype=np.float32) for samp in self.data_array ], dtype=np.float32)

    def get_segmentations(self):
        return np.array([np.array(samp[1],dtype=np.uint8) for samp in self.data_array], dtype=np.uint8)

if __name__ == "__main__":
    """ Run random walks """
    #example run: python3 topoml/graphsage/utils.py ./data/json_graphs/test_ridge_arcs-G.json ./data/random_walks/full_msc_n-1_k-40
    graph_file = sys.argv[1]
    out_file = sys.argv[2]
    G_data = json.load(open(graph_file))
    G = json_graph.node_link_graph(G_data)
    nodes = [n for n in G.nodes() if not G.node[n]["val"] and not G.node[n]["test"]]# [n for n in G.nodes() if 'train' in G.node[n] or 'val' in G.node[n] and (G.node[n]['train'] or G.node[n]['val'])]

    G = G.subgraph(nodes)
    pairs = run_random_walks(G, nodes)
    with open(out_file+'-walks.txt', "w") as fp:
        fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs]))
