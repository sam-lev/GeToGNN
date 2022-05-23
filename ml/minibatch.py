from __future__ import division
from __future__ import print_function

import numpy as np

np.random.seed(123)

from ml.utils import pout

class EdgeMinibatchIterator(object):
    
    """ This minibatch iterator iterates over batches of sampled edges or
    random pairs of co-occuring edges.

    G -- networkx graph
    id2idx
    placeholders -- tensorflow placeholders object
    context_pairs -- if not none, then a list of co-occuring node pairs (from random walks)
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    n2v_retrain -- signals that the iterator is being used to add new embeddings to a n2v model
    fixed_n2v -- signals that the iterator is being used to retrain n2v with only existing nodes as context
    """
    def __init__(self, G, id2idx, 
            placeholders, nx_idx_to_getoelm_idx=None,
                 context_pairs=None, batch_size=100, max_degree=25,
            n2v_retrain=False, fixed_n2v=False,
            **kwargs):

        self.G = G
        self.nodes = G.nodes()
        self.id2idx = id2idx
        self.id2getoidx = nx_idx_to_getoelm_idx
        self.use_geto = self.id2getoidx is not None
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0

        self.nodes = np.random.permutation(G.nodes())
        self.adj, self.deg, self.geto_adj = self.construct_adj()
        self.test_adj, self.test_geto_adj = self.construct_test_adj()
        if context_pairs is None:
            edges = G.edges()
        else:
            edges = context_pairs
        self.train_edges = self.edges = np.random.permutation(edges)
        if not n2v_retrain:
            self.train_edges = self._remove_isolated(self.train_edges)
            self.val_edges = [e for e in G.edges() if G[e[0]][e[1]]['train_removed']]
        else:
            if fixed_n2v:
                self.train_edges = self.val_edges = self._n2v_prune(self.edges)
            else:
                self.train_edges = self.val_edges = self.edges

        print(len([n for n in G.nodes() if not G.node[n]['test'] and not G.node[n]['val']]), 'train nodes')
        print(len([n for n in G.nodes() if G.node[n]['test'] or G.node[n]['val']]), 'test or val nodes')
        self.val_set_size = len(self.val_edges)

    def _n2v_prune(self, edges):
        is_val = lambda n : self.G.node[n]["val"] or self.G.node[n]["test"]
        return [e for e in edges if not is_val(e[1])]

    def _remove_isolated(self, edge_list):
        new_edge_list = []
        missing = 0
        for n1, n2 in edge_list:
            if not n1 in self.G.node or not n2 in self.G.node:
                missing += 1
                continue
            if (self.deg[self.id2idx[n1]] == 0 or self.deg[self.id2idx[n2]] == 0) \
                    and (not self.G.node[n1]['test'] or self.G.node[n1]['val']) \
                    and (not self.G.node[n2]['test'] or self.G.node[n2]['val']):
                continue
            else:
                new_edge_list.append((n1,n2))
        print("unexpected missing node connections: ", missing)
        return new_edge_list

    def construct_adj(self):

        #degree_dict = {}
        #for i in range(1000):
        #    degree_dict[i] = 0
        def row_major_index(row, col , num_col):
            lin_index = row*num_col + col
            return lin_index

        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        if self.use_geto:
            geto_adj = np.ones((len(self.id2getoidx)+1, self.max_degree))
            print("geto adjacency length:",len(geto_adj))
            print("geto id map length:", len(self.id2getoidx))
        else:
            geto_adj = None
        deg = np.zeros((len(self.id2idx),))
        for nodeid in self.G.nodes():
            if self.G.node[nodeid]['test'] or self.G.node[nodeid]['val']:
                continue
            neighbor_ids = np.array([neighbor
                      for neighbor in self.G.neighbors(nodeid)
                      if (not self.G[nodeid][neighbor]['train_removed'])])
            neighbors_feats = np.array([self.id2idx[neighbor]
                for neighbor in neighbor_ids])
            if self.use_geto:
                neighbors_getoelms = np.array([self.id2getoidx[(nodeid,neighbor)]
                                               if (nodeid,neighbor) in self.id2getoidx.keys() else self.id2getoidx[(neighbor,nodeid)]
                                               for neighbor in neighbor_ids])
            deg[self.id2idx[nodeid]] = len(neighbors_feats)
            if len(neighbors_feats) == 0:
                continue
            if len(neighbors_feats) > self.max_degree:
                pruned_neighors = np.random.choice(range(len(neighbors_feats)),
                                                   self.max_degree, replace=False)
                neighbors_feats = np.array([neighbors_feats[nbr] for nbr in pruned_neighors])
                if self.use_geto:
                    neighbors_getoelms = np.array([neighbors_getoelms[nbr] for nbr in pruned_neighors])
                #np.random.choice(neighbors_feats, self.max_degree, replace=False)
            elif len(neighbors_feats) < self.max_degree:
                resampled_neighors = np.random.choice(range(len(neighbors_feats)),
                                                      self.max_degree, replace=True)
                #np.random.choice(neighbors_feats, self.max_degree, replace=True)
                neighbors_feats = np.array([neighbors_feats[nbr] for nbr in resampled_neighors])
                if self.use_geto:
                    neighbors_getoelms = np.array([neighbors_getoelms[nbr] for nbr in resampled_neighors])
            #occurance_degree  = degree_dict[len(neighbors)] + 1
            #degree_dict[len(neighbors)] += 1# occurance_degree

            adj[self.id2idx[nodeid], :] = neighbors_feats
            if self.use_geto:
                geto_adj[self.id2getoidx[(nodeid,nodeid)], :] = neighbors_getoelms
        #print("    observed degree counts")
        #print(degree_dict)
        #print(">>>>")
        return adj, deg, geto_adj

    def construct_test_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        if self.use_geto:
            geto_adj = np.ones((len(self.id2getoidx)+1, self.max_degree))
        else:
            geto_adj = None
        for nodeid in self.G.nodes():
            #neighbors = np.array([self.id2idx[neighbor]
            #    for neighbor in self.G.neighbors(nodeid)])
            neighbor_ids = np.array([neighbor
                                     for neighbor in self.G.neighbors(nodeid)])
            neighbors_feats = np.array([self.id2idx[neighbor]
                                        for neighbor in neighbor_ids])
            if self.use_geto:
                neighbors_getoelms = np.array([self.id2getoidx[(nodeid, neighbor)]
                                               if (nodeid, neighbor) in self.id2getoidx.keys() else self.id2getoidx[(neighbor, nodeid)]
                                               for neighbor in neighbor_ids])
            if len(neighbors_feats) == 0:
                continue
            if len(neighbors_feats) > self.max_degree:
                pruned_neighors = np.random.choice(range(len(neighbors_feats)),
                                                   self.max_degree, replace=False)
                neighbors_feats = np.array([neighbors_feats[nbr] for nbr in pruned_neighors])
                if self.use_geto:
                    neighbors_getoelms = np.array([neighbors_getoelms[nbr] for nbr in pruned_neighors])
                #neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors_feats) < self.max_degree:
                resampled_neighors = np.random.choice(range(len(neighbors_feats)),
                                                      self.max_degree, replace=True)
                neighbors_feats = np.array([neighbors_feats[nbr] for nbr in resampled_neighors])
                if self.use_geto:
                    neighbors_getoelms = np.array([neighbors_getoelms[nbr] for nbr in resampled_neighors])
                #neighbors = np.random.choice(neighbors, self.max_degree, replace=True)

            adj[self.id2idx[nodeid], :] = neighbors_feats
            if self.use_geto:
                geto_adj[self.id2getoidx[(nodeid,nodeid)], :] = neighbors_getoelms
        return adj, geto_adj

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_edges)

    def batch_feed_dict(self, batch_edges):
        batch1 = []
        batch2 = []
        self.current_batch=[]
        for node1, node2 in batch_edges:
            self.current_batch.append(node1)
            self.current_batch.append(node2)
            batch1.append(self.id2idx[node1])
            batch2.append(self.id2idx[node2])

        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size'] : len(batch_edges)})
        feed_dict.update({self.placeholders['batch1']: batch1})
        feed_dict.update({self.placeholders['batch2']: batch2})

        return feed_dict

    #def update_batch_prediction(self, preds):
    #    for i, nodeid in enumerate(self.current_batch):#, preds):
    #        self.G.node[nodeid]['prediction'] = preds[-1][i,:]
    def update_batch_prediction(self, preds):
        batch1id = self.current_batch
        a = b = c = 0
        for i, n in enumerate(batch1id):
            self.G.node[n]['prediction'] = preds[i]

    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_edges))
        batch_edges = self.train_edges[start_idx : end_idx]
        return self.batch_feed_dict(batch_edges)

    def num_training_batches(self):
        return len(self.train_edges) // self.batch_size + 1

    def val_feed_dict(self, size=None):
        edge_list = self.val_edges
        if size is None:
            return self.batch_feed_dict(edge_list)
        else:
            ind = np.random.permutation(len(edge_list))
            val_edges = [edge_list[i] for i in ind[:min(size, len(ind))]]
            return self.batch_feed_dict(val_edges)

    def incremental_val_feed_dict(self, size, iter_num):
        edge_list = self.val_edges
        val_edges = edge_list[iter_num*size:min((iter_num+1)*size, 
            len(edge_list))]
        return self.batch_feed_dict(val_edges), (iter_num+1)*size >= len(self.val_edges), val_edges

    def incremental_embed_feed_dict(self, size, iter_num):
        node_list = self.nodes
        val_nodes = node_list[iter_num*size:min((iter_num+1)*size, 
            len(node_list))]
        val_edges = [(n,n) for n in val_nodes]
        return self.batch_feed_dict(val_edges), (iter_num+1)*size >= len(node_list), val_edges

    def label_val(self):
        train_edges = []
        val_edges = []
        for n1, n2 in self.G.edges():
            if (self.G.node[n1]['val'] or self.G.node[n1]['test'] 
                    or self.G.node[n2]['val'] or self.G.node[n2]['test']):
                val_edges.append((n1,n2))
            else:
                train_edges.append((n1,n2))
        return train_edges, val_edges

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_edges = np.random.permutation(self.train_edges)
        self.nodes = np.random.permutation(self.nodes)
        self.batch_num = 0

class NodeMinibatchIterator(object):
    
    """ 
    This minibatch iterator iterates over nodes for supervised learning.

    G -- networkx graph
    id2idx -- dict mapping node ids to integer values indexing feature tensor
    placeholders -- standard tensorflow placeholders object for feeding
    label_map -- map from node ids to class values (integer or list)
    num_classes -- number of output classes
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    """
    def __init__(self, G, id2idx, 
            placeholders, label_map, num_classes,
                 nx_idx_to_getoelm_idx=None,
            batch_size=100, max_degree=25, train=True,
            **kwargs):

        self.G = G
        self.nodes = G.nodes()
        self.id2idx = id2idx
        self.id2getoidx = nx_idx_to_getoelm_idx
        self.use_geto = nx_idx_to_getoelm_idx is not None
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0


        self.label_map = label_map
        self.num_classes = num_classes

        self.adj, self.deg, self.geto_adj = self.construct_adj()
        self.test_adj, self.test_geto_adj = self.construct_test_adj()

        self.val_nodes = [n for n in self.G.nodes() if self.G.node[n]['val']]
        #if train:
        self.test_nodes = [n for n in self.G.nodes() if self.G.node[n]['test'] ]
        self.all_nodes = [n for n in self.G.nodes() ]
        #else:
        #    print("performing inference")
        #    self.test_nodes = [n for n in self.G.nodes()]
        self.no_train_nodes_set = set(self.val_nodes + self.test_nodes)
        self.train_nodes = set(G.nodes()).difference(self.no_train_nodes_set)
        # don't train on nodes that only have edges to test set
        self.train_nodes = [n for n in self.train_nodes if self.deg[id2idx[n]] > 0]

        self.sublevel_sets = [n for n in self.G.nodes() if self.G.node[n]['sublevel_set_id'][1] != -1]
        self.level_set_nodes = self.sublevel_sets

        pout(["len sublevelnodes",len(self.sublevel_sets)])
        if len(self.sublevel_sets) != 0:
            self.sublevel_set_id = 0
            self.observed_sublevel_set_ids = []
            self.total_sublevel_sets = self.G.node[self.sublevel_sets[0]]['sublevel_set_id'][0]
            pout(["total sublevel sets", self.total_sublevel_sets])
            self.levelset_batch_nums = np.zeros(self.total_sublevel_sets, dtype=np.int32)
            self.levelset_node_lists = []
            self.subadj_list = []#len(self.id2idx) * np.ones((self.total_sublevel_sets,len(self.id2idx) + 1, self.max_degree))
            for sub_id in range(1,self.total_sublevel_sets+1):
                sublevel_i= [n for n in self.sublevel_sets if self.G.node[n]['sublevel_set_id'][1] == sub_id]
                sublevel_i = [n for n in sublevel_i if self.deg[self.id2idx[n]] > 0]
                self.levelset_node_lists.append(sublevel_i)
                sub_adj, sub_deg, sub_geto_adj = self.construct_subgraph_adj(sublevel_i)
                self.subadj_list.append(sub_adj)#[sub_id-1,:,:] = sub_adj
            self.subadj_list.append(self.adj)#[-1,:,:]= self.adj
            self.subadj_list =  np.array(self.subadj_list).astype(dtype=np.int32)
            self.update_sublevel_training_set()


    def update_sublevel_training_set(self):

        self.sublevel_set_id += 1
        self.observed_sublevel_set_ids.append(self.sublevel_set_id)
        if self.sublevel_set_id == 1:
            #self.train_nodes = []
            self.observed_sublevel_sets = []
        if self.sublevel_set_id <= self.total_sublevel_sets:
            self.hier_superlevel_nodes = []
            for sub_i in range(self.sublevel_set_id):
                self.hier_superlevel_nodes += self.levelset_node_lists[sub_i]#[n for n in self.sublevel_sets if self.G.node[n]['sublevel_set_id'][1] in self.observed_sublevel_set_ids ]
            # don't train on nodes that only have edges to test set
            #self.hier_superlevel_nodes = [n for n in self.hier_superlevel_nodes if self.deg[self.id2idx[n]] > 0]
            pout(["adding sublevel nodes ",len(self.hier_superlevel_nodes)])
            #self.train_nodes = self.hier_superlevel_nodes
            self.observed_sublevel_set1 = self.hier_superlevel_nodes
            self.observed_sublevel_sets.append(self.hier_superlevel_nodes)
        else:
            self.train_nodes = set(self.G.nodes()).difference(self.no_train_nodes_set)
            # don't train on nodes that only have edges to test set
            self.train_nodes = [n for n in self.train_nodes if self.deg[self.id2idx[n]] > 0]
        pout(['updating training subgraph', 'length_training',len(self.train_nodes),'length observed sublevel',
              len(self.observed_sublevel_set1)])

    def _make_label_vec(self, node):
        label = self.label_map[node]
        if isinstance(label, list):
            label_vec = np.array(label)
        else:
            label_vec = np.zeros((self.num_classes))
            class_ind = self.label_map[node]
            label_vec[class_ind] = 1
        return label_vec

    def _make_pred_vec(self,pred):
        if isinstance(pred, list):
            pred_vec = pred
        else:
            pred_vec = [1. - pred , pred]
        return pred_vec

    def construct_adj(self):

        # degree_dict = {}
        # for i in range(1000):
        #    degree_dict[i] = 0
        def row_major_index(row, col, num_col):
            lin_index = row * num_col + col
            return lin_index

        adj = len(self.id2idx) * np.ones((len(self.id2idx) + 1, self.max_degree))

        if self.use_geto:
            geto_adj = np.ones((len(self.id2getoidx) + 1, self.max_degree)) if self.use_geto else None
        else:
            geto_adj = None
        # id2idx graph_id to feat_idx
        deg = np.zeros((len(self.id2idx),))
        for nodeid in self.G.nodes():
            if self.G.node[nodeid]['test'] or self.G.node[nodeid]['val']:
                continue
            neighbor_ids = np.array([neighbor
                                     for neighbor in self.G.neighbors(nodeid)
                                     if (not self.G[nodeid][neighbor]['train_removed'])])
            neighbors_feats = np.array([self.id2idx[neighbor]
                                        for neighbor in neighbor_ids])
            if self.use_geto:
                neighbors_getoelms = np.array([self.id2getoidx[neighbor]#(nodeid, neighbor)]
                #                               if (nodeid, neighbor) in self.id2getoidx.keys() else self.id2getoidx[
                #                                (neighbor, nodeid)]
                                                for neighbor in neighbor_ids]) if self.use_geto else None
            deg[self.id2idx[nodeid]] = len(neighbors_feats)
            if len(neighbors_feats) == 0:
                continue
            if len(neighbors_feats) > self.max_degree:
                pruned_neighors = np.random.choice(range(len(neighbors_feats)),
                                                   self.max_degree, replace=False)
                neighbors_feats = np.array([neighbors_feats[nbr] for nbr in pruned_neighors])
                if self.use_geto:
                    neighbors_getoelms = np.array([neighbors_getoelms[nbr] for nbr in pruned_neighors]) if self.use_geto else None
                # np.random.choice(neighbors_feats, self.max_degree, replace=False)
            elif len(neighbors_feats) < self.max_degree:
                resampled_neighors = np.random.choice(range(len(neighbors_feats)),
                                                      self.max_degree, replace=True)
                # np.random.choice(neighbors_feats, self.max_degree, replace=True)
                neighbors_feats = np.array([neighbors_feats[nbr] for nbr in resampled_neighors])
                if self.use_geto:
                    neighbors_getoelms = np.array([neighbors_getoelms[nbr] for nbr in resampled_neighors]) if self.use_geto else None
            # occurance_degree  = degree_dict[len(neighbors)] + 1
            # degree_dict[len(neighbors)] += 1# occurance_degree

            adj[self.id2idx[nodeid], :] = neighbors_feats
            if self.use_geto:
                geto_adj[self.id2getoidx[nodeid], :] = neighbors_getoelms if self.use_geto else None
                #geto_adj[self.id2getoidx[(nodeid, nodeid)], :] = neighbors_getoelms if self.use_geto else None
        # print("    observed degree counts")
        # print(degree_dict)
        # print(">>>>")
        return adj, deg, geto_adj

    def construct_subgraph_adj(self, subgraph_ids):

        # degree_dict = {}
        # for i in range(1000):
        #    degree_dict[i] = 0
        def row_major_index(row, col, num_col):
            lin_index = row * num_col + col
            return lin_index

        adj = len(self.id2idx) * np.ones((len(self.id2idx) + 1, self.max_degree))

        if self.use_geto:
            geto_adj = np.ones((len(self.id2getoidx) + 1, self.max_degree)) if self.use_geto else None
        else:
            geto_adj = None
        # id2idx graph_id to feat_idx
        deg = np.zeros((len(self.id2idx),))
        for nodeid in self.G.nodes():
            if self.G.node[nodeid]['test'] or self.G.node[nodeid]['val']:
                continue
            neighbor_ids = np.array([neighbor
                                     for neighbor in self.G.neighbors(nodeid)
                                     if ((not self.G[nodeid][neighbor]['train_removed']) and (neighbor in subgraph_ids)  )])
            neighbors_feats = np.array([self.id2idx[neighbor]
                                        for neighbor in neighbor_ids])
            if self.use_geto:
                neighbors_getoelms = np.array([self.id2getoidx[neighbor]#(nodeid, neighbor)]
                #                               if (nodeid, neighbor) in self.id2getoidx.keys() else self.id2getoidx[
                #                                (neighbor, nodeid)]
                                                for neighbor in neighbor_ids]) if self.use_geto else None
            deg[self.id2idx[nodeid]] = len(neighbors_feats)
            if len(neighbors_feats) == 0:
                continue
            if len(neighbors_feats) > self.max_degree:
                pruned_neighors = np.random.choice(range(len(neighbors_feats)),
                                                   self.max_degree, replace=False)
                neighbors_feats = np.array([neighbors_feats[nbr] for nbr in pruned_neighors])
                if self.use_geto:
                    neighbors_getoelms = np.array([neighbors_getoelms[nbr] for nbr in pruned_neighors]) if self.use_geto else None
                # np.random.choice(neighbors_feats, self.max_degree, replace=False)
            elif len(neighbors_feats) < self.max_degree:
                resampled_neighors = np.random.choice(range(len(neighbors_feats)),
                                                      self.max_degree, replace=True)
                # np.random.choice(neighbors_feats, self.max_degree, replace=True)
                neighbors_feats = np.array([neighbors_feats[nbr] for nbr in resampled_neighors])
                if self.use_geto:
                    neighbors_getoelms = np.array([neighbors_getoelms[nbr] for nbr in resampled_neighors]) if self.use_geto else None
            # occurance_degree  = degree_dict[len(neighbors)] + 1
            # degree_dict[len(neighbors)] += 1# occurance_degree

            adj[self.id2idx[nodeid], :] = neighbors_feats
            if self.use_geto:
                geto_adj[self.id2getoidx[nodeid], :] = neighbors_getoelms if self.use_geto else None
                #geto_adj[self.id2getoidx[(nodeid, nodeid)], :] = neighbors_getoelms if self.use_geto else None
        # print("    observed degree counts")
        # print(degree_dict)
        # print(">>>>")
        return adj, deg, geto_adj
    def construct_test_adj(self):
        adj = len(self.id2idx) * np.ones((len(self.id2idx) + 1, self.max_degree))
        geto_adj = np.ones((len(self.id2getoidx) + 1, self.max_degree)) if self.use_geto else None
        for nodeid in self.G.nodes():
            # neighbors = np.array([self.id2idx[neighbor]
            #    for neighbor in self.G.neighbors(nodeid)])
            neighbor_ids = np.array([neighbor
                                     for neighbor in self.G.neighbors(nodeid)])
            neighbors_feats = np.array([self.id2idx[neighbor]
                                        for neighbor in neighbor_ids])
            if self.use_geto:
                neighbors_getoelms = np.array([self.id2getoidx[neighbor]#(nodeid, neighbor)]
                #                               if (nodeid, neighbor) in self.id2getoidx.keys() else self.id2getoidx[
                #                                (neighbor, nodeid)]
                                                for neighbor in neighbor_ids])
            if len(neighbors_feats) == 0:
                continue
            if len(neighbors_feats) > self.max_degree:
                pruned_neighors = np.random.choice(range(len(neighbors_feats)),
                                                   self.max_degree, replace=False)
                neighbors_feats = np.array([neighbors_feats[nbr] for nbr in pruned_neighors])
                if self.use_geto:
                    neighbors_getoelms = np.array([neighbors_getoelms[nbr] for nbr in pruned_neighors])
                # neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors_feats) < self.max_degree:
                resampled_neighors = np.random.choice(range(len(neighbors_feats)),
                                                      self.max_degree, replace=True)
                neighbors_feats = np.array([neighbors_feats[nbr] for nbr in resampled_neighors])
                if self.use_geto:
                    neighbors_getoelms = np.array([neighbors_getoelms[nbr] for nbr in resampled_neighors])

            adj[self.id2idx[nodeid], :] = neighbors_feats
            if self.use_geto:
                geto_adj[self.id2getoidx[nodeid], :] = neighbors_getoelms
                #geto_adj[self.id2getoidx[(nodeid, nodeid)], :] = neighbors_getoelms
        return adj, geto_adj

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_nodes)

    def batch_feed_dict(self, batch_nodes, sup_only=False):


        feed_dict = dict()
        sub_feed_dict = {}

        batch1id = batch_nodes

        if sup_only:

            superlevel_batch = [n for n in batch1id]
            level_id = 0
            sublevel_batch1 = superlevel_batch
            self.current_batch = superlevel_batch
            full_labels = np.vstack([self._make_label_vec(node) for node in superlevel_batch])
            batch1 = [self.id2idx[n] for n in superlevel_batch]
            super_level_bs = int(len(superlevel_batch))

            sub_level_bs = super_level_bs#int(0)
            sub_batch0 = batch1
            levelset_labels = full_labels
            sub_feed_dict.update({self.placeholders['sub_batch0_size']: sub_level_bs})
            sub_feed_dict.update({self.placeholders['sub_batch0']: sub_batch0})
            sub_feed_dict.update({self.placeholders['sub_batch0_labels']: levelset_labels})
        else:

            superlevel_batch = [n for n in batch1id if n not in self.level_set_nodes]
            # level_id = 0
            # sublevel_batch1   = [n for n in batch1id if n in self.levelset_node_lists[level_id]]
            # order added matters, see call to sample / build
            # for sample, to order level set batches to preserve nbr ordering and labels
            # for build for label ordering
            full_batch = superlevel_batch

            for level_id, level_set in enumerate(self.observed_sublevel_sets):
                level_set_batch = [n for n in batch1id if n in level_set]
                level_set_name = "sub_batch" + str(level_id)

                level_set_size_name = level_set_name+'_size'

                new_levelset_samples, new_levelset_labels = self.next_levelset_minibatch(level_id=level_id,
                                                                                         batch_size=len(
                                                                                             superlevel_batch))
                level_set_batch_i = new_levelset_samples + level_set_batch
                full_batch += level_set_batch
                sub_batch_i = [self.id2idx[n] for n in level_set_batch_i]
                levelset_labels_i = np.vstack([self._make_label_vec(node) for node in level_set_batch_i])
                level_set_batch_size = len(level_set_batch_i)
                level_set_label_name = level_set_name+'_labels'

                sub_feed_dict.update({self.placeholders[level_set_name]: sub_batch_i})
                sub_feed_dict.update({self.placeholders[level_set_size_name]: level_set_batch_size})
                sub_feed_dict.update({self.placeholders[level_set_label_name]: levelset_labels_i})

            # order added matters, see call to sample / build

            super_level_bs = int(len(full_batch)) if len(full_batch) > 0 else int(0)
            batch1 = [self.id2idx[n] for n in full_batch]  # superlevel_batch]
            self.current_batch = full_batch
            full_labels = np.vstack([self._make_label_vec(node) for node in full_batch])
            # levelset_labels_init = [self._make_label_vec(node) for node in sublevel_batch1]
            # # pout(['SUB BATCH SIZE', len(sublevel_batch1), 'sublevellabels0', len(levelset_labels)])
            # new_levelset_samples, new_levelset_labels = self.next_levelset_minibatch(level_id=level_id,
            #                                                                          batch_size=len(superlevel_batch))
            # pout(['levelsetsamps',len(new_levelset_samples)])
            # sublevel_batch_sampled0 = new_levelset_samples + sublevel_batch1
            # levelset_labels = [self._make_label_vec(node) for node in sublevel_batch_sampled0]#new_levelset_labels + levelset_labels


            #levelset_labels = np.vstack(levelset_labels)


            # sub_batch0 = [self.id2idx[n] for n in sublevel_batch_sampled0]

            #sub_level_bs = int(len(sub_batch0)) if len(sub_batch0) > 0 else int(0)
        feed_dict.update(sub_feed_dict)


        # sub_batch2 = [self.id2idx[n] for n in sublevel_batch2]
        getobatch1 = [self.id2getoidx[n] for n in batch1id ] if self.use_geto else [0]#if n not in self.observed_sublevel_sets]
        sub_getobatch1 = [self.id2getoidx[n] for n in batch1id if n in self.observed_sublevel_set1] if self.use_geto else [0]



        feed_dict.update({self.placeholders['batch_size'] : super_level_bs})
        #feed_dict.update({self.placeholders['sub_batch0_size']: sub_level_bs})
        feed_dict.update({self.placeholders['batch']: batch1})
        #feed_dict.update({self.placeholders['sub_batch0']:sub_batch0})
        # feed_dict.update({self.placeholders['sub_batch2']: sub_batch2})
        feed_dict.update({self.placeholders['getobatch']: getobatch1})
        feed_dict.update({self.placeholders['sub_getobatch']: sub_getobatch1})
        feed_dict.update({self.placeholders['labels']: full_labels})
        # feed_dict.update({self.placeholders['sub_batch0_id']: self.sublevel_set_id-1})
        #feed_dict.update({self.placeholders['sub_batch0_labels']: levelset_labels})
        feed_dict.update({self.placeholders['subcomplex_weight']: 1.0})
        return feed_dict, full_labels

    def next_levelset_minibatch(self, level_id, batch_size):
        start_idx = self.levelset_batch_nums[level_id]

        self.levelset_batch_nums[level_id] += batch_size
        level_set = self.levelset_node_lists[level_id]
        if self.levelset_batch_nums[level_id] >= len(level_set):
            self.levelset_batch_nums[level_id] = 0


        end_idx = min(start_idx + batch_size, len(level_set))
        level_set_batch = level_set[start_idx: end_idx]#[n for n in batch1id if n in level_set]
        if end_idx != start_idx + batch_size:
            end_idx = (start_idx+batch_size) - len(level_set)
            overflow_batch = level_set[0:end_idx]
            self.levelset_batch_nums[level_id] += end_idx
            level_set_batch = level_set_batch + overflow_batch
        levelset_labels =[self._make_label_vec(node) for node in level_set_batch]
        # level_set_label_name = level_set_name+'_labels'
        #full_batch += level_set_batch
        #if not inference:
        return level_set_batch, levelset_labels



    def update_batch_prediction(self, preds):
        batch1id = self.current_batch
        a = b = c = 0
        for i, n in enumerate(batch1id):
            self.G.node[n]['prediction'] = preds[i]
        #labels = np.vstack([self._make_label_vec(node) for node in batch1id])
        #feed_dict = dict()
        #feed_dict.update({self.placeholders['batch_size']: len(batch1)})
        #feed_dict.update({self.placeholders['batch']: batch1})
        #feed_dict.update({self.placeholders['labels']: labels})

        #return feed_dict, labels

    def node_val_feed_dict(self, size=None, test=False, sup_only=True):
        if test:
            val_nodes = self.test_nodes
            # self.current_batch = self.test_nodes
        else:
            val_nodes = self.val_nodes
            #self.current_batch = self.val_nodes
        if not size is None:
            val_nodes = np.random.choice(val_nodes, size, replace=True)

        self.current_batch=val_nodes
        # add a dummy neighbor
        ret_val = self.batch_feed_dict(val_nodes, sup_only=sup_only)
        ret_val[0].update({self.placeholders['subcomplex_weight']: -1})
        return ret_val[0], ret_val[1]

    def incremental_node_val_feed_dict(self, size, iter_num, test=False, sup_only=True):
        if test:
            val_nodes = self.all_nodes#test_nodes
        else:
            val_nodes = self.val_nodes
        val_node_subset = val_nodes[iter_num*size:min((iter_num+1)*size, 
            len(val_nodes))]

        # add a dummy neighbor
        ret_val = self.batch_feed_dict(val_node_subset, sup_only=sup_only)
        ret_val[0].update({self.placeholders['subcomplex_weight']: -1})
        return ret_val[0], ret_val[1], (iter_num+1)*size >= len(val_nodes), val_node_subset

    def num_training_batches(self):
        return len(self.train_nodes) // self.batch_size + 1

    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_nodes))
        batch_nodes = self.train_nodes[start_idx : end_idx]
        return self.batch_feed_dict(batch_nodes)

    def incremental_embed_feed_dict(self, size, iter_num):
        node_list = self.nodes
        val_nodes = node_list[iter_num*size:min((iter_num+1)*size, 
            len(node_list))]
        return self.batch_feed_dict(val_nodes), (iter_num+1)*size >= len(node_list), val_nodes

    def get_graph(self):
        return self.G

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_nodes = np.random.permutation(self.train_nodes)
        self.batch_num = 0
