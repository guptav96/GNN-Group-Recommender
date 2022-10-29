from __future__ import print_function
import numpy as np
import random
from tqdm import tqdm
import os, sys, pdb, math, time
from copy import deepcopy
import multiprocessing as mp
import networkx as nx
import argparse
import scipy.io as sio
import scipy.sparse as ssp
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
import warnings
warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import pdb
# pdb.set_trace()

class SparseRowIndexer:
    def __init__(self, csr_matrix):
        data = []
        indices = []
        indptr = []

        for row_start, row_end in zip(csr_matrix.indptr[:-1], csr_matrix.indptr[1:]):
            data.append(csr_matrix.data[row_start:row_end])
            indices.append(csr_matrix.indices[row_start:row_end])
            indptr.append(row_end - row_start)  # nnz of the row

        self.data = np.array(data, dtype=object)
        self.indices = np.array(indices, dtype=object)
        self.indptr = np.array(indptr, dtype=object)
        self.shape = csr_matrix.shape

    def __getitem__(self, row_selector):
        indices = np.concatenate(self.indices[row_selector])
        data = np.concatenate(self.data[row_selector])
        indptr = np.append(0, np.cumsum(self.indptr[row_selector]))
        shape = [indptr.shape[0] - 1, self.shape[1]]
        return ssp.csr_matrix((data, indices, indptr), shape=shape)


class SparseColIndexer:
    def __init__(self, csc_matrix):
        data = []
        indices = []
        indptr = []

        for col_start, col_end in zip(csc_matrix.indptr[:-1], csc_matrix.indptr[1:]):
            data.append(csc_matrix.data[col_start:col_end])
            indices.append(csc_matrix.indices[col_start:col_end])
            indptr.append(col_end - col_start)

        self.data = np.array(data, dtype=object)
        self.indices = np.array(indices, dtype=object)
        self.indptr = np.array(indptr, dtype=object)
        self.shape = csc_matrix.shape

    def __getitem__(self, col_selector):
        indices = np.concatenate(self.indices[col_selector])
        data = np.concatenate(self.data[col_selector])
        indptr = np.append(0, np.cumsum(self.indptr[col_selector]))

        shape = [self.shape[0], indptr.shape[0] - 1]
        return ssp.csc_matrix((data, indices, indptr), shape=shape)


class MyDataset(InMemoryDataset):
    def __init__(self, root, A_user, A_train, A_group_user, links, labels, h, sample_ratio, max_nodes_per_hop,
                 u_features, v_features, class_values, max_num=None, parallel=True):
        self.A_group_item_row = SparseRowIndexer(A_train)
        self.A_group_item_col = SparseColIndexer(A_train.tocsc())
        self.A_user_item_row = SparseRowIndexer(A_user)
        self.A_user_item_col = SparseColIndexer(A_user.tocsc())
        self.A_group_user_row = SparseRowIndexer(A_group_user)
        self.A_group_user_col = SparseColIndexer(A_group_user.tocsc())
        self.links = links
        self.labels = labels
        self.h = h
        self.sample_ratio = sample_ratio
        self.max_nodes_per_hop = max_nodes_per_hop
        self.u_features = u_features
        self.v_features = v_features
        self.class_values = class_values
        self.parallel = parallel
        self.max_num = max_num
        if max_num is not None:
            np.random.seed(123)
            num_links = len(links[0])
            perm = np.random.permutation(num_links)
            perm = perm[:max_num]
            self.links = (links[0][perm], links[1][perm])
            self.labels = labels[perm]
        super(MyDataset, self).__init__(root)
        print("reading from:", self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        name = 'data.pt'
        if self.max_num is not None:
            name = 'data_{}.pt'.format(self.max_num)
        return [name]

    def process(self):
        # Extract enclosing subgraphs and save to disk
        data_list = links2subgraphs(self.A_group_item_row, self.A_group_item_col, 
                                    self.A_user_item_row, self.A_user_item_col,
                                    self.A_group_user_row, self.A_group_user_col, 
                                    self.links, self.labels, self.h,
                                    self.sample_ratio, self.max_nodes_per_hop,
                                    self.u_features, self.v_features,
                                    self.class_values, self.parallel)

        data, slices = self.collate(data_list)
        print("saving to:", self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[0])
        del data_list


class MyDynamicDataset(Dataset):
    def __init__(self, root, A, links, labels, h, sample_ratio, max_nodes_per_hop,
                 u_features, v_features, class_values, max_num=None):
        super(MyDynamicDataset, self).__init__(root)
        self.Arow = SparseRowIndexer(A)
        self.Acol = SparseColIndexer(A.tocsc())
        self.links = links
        self.labels = labels
        self.h = h
        self.sample_ratio = sample_ratio
        self.max_nodes_per_hop = max_nodes_per_hop
        self.u_features = u_features
        self.v_features = v_features
        self.class_values = class_values
        if max_num is not None:
            np.random.seed(123)
            num_links = len(links[0])
            perm = np.random.permutation(num_links)
            perm = perm[:max_num]
            self.links = (links[0][perm], links[1][perm])
            self.labels = labels[perm]

    def len(self):
        return self.__len__()

    def __len__(self):
        return len(self.links[0])

    def get(self, idx):
        i, j = self.links[0][idx], self.links[1][idx]
        g_label = self.labels[idx]
        tmp = subgraph_extraction_labeling(
            (i, j), self.A, self.h, self.sample_ratio, self.max_nodes_per_hop,
            self.u_features, self.v_features, self.class_values, g_label
        )
        return construct_pyg_graph(*tmp)


def links2subgraphs(A_group_item_row,
                    A_group_item_col,
                    A_user_item_row,
                    A_user_item_col,
                    A_group_user_row,
                    A_group_user_col,
                    links,
                    labels,
                    h=1,
                    sample_ratio=1.0,
                    max_nodes_per_hop=None,
                    u_features=None,
                    v_features=None,
                    class_values=None,
                    parallel=True):
    # extract enclosing subgraphs
    print('Enclosing subgraph extraction begins...')
    g_list = []
    if not parallel:
        with tqdm(total=len(links[0])) as pbar:
            for i, j, g_label in zip(links[0], links[1], labels):
                tmp = subgraph_extraction_labeling(
                    (i, j), A_group_item_row, A_group_item_col, A_user_item_row, A_user_item_col, A_group_user_row, A_group_user_col, h, sample_ratio, max_nodes_per_hop, u_features,
                    v_features, class_values, g_label
                )
                data = construct_pyg_graph(*tmp)
                g_list.append(data)
                pbar.update(1)
    else:
        start = time.time()
        pool = mp.Pool(mp.cpu_count())
        results = pool.starmap_async(
            subgraph_extraction_labeling,
            [
                ((i, j), A_group_item_row, A_group_item_col, A_user_item_row, A_user_item_col, A_group_user_row, A_group_user_col, h, sample_ratio, max_nodes_per_hop, u_features,
                v_features, class_values, g_label)
                for i, j, g_label in zip(links[0], links[1], labels)
            ]
        )
        remaining = results._number_left
        pbar = tqdm(total=remaining)
        while True:
            pbar.update(remaining - results._number_left)
            if results.ready(): break
            remaining = results._number_left
            time.sleep(1)
        results = results.get()
        pool.close()
        pbar.close()
        end = time.time()
        print("Time elapsed for subgraph extraction: {}s".format(end-start))
        print("Transforming to pytorch_geometric graphs...")
        g_list = []
        pbar = tqdm(total=len(results))
        while results:
            tmp = results.pop()
            g_list.append(construct_pyg_graph(*tmp))
            pbar.update(1)
        pbar.close()
        end2 = time.time()
        print("Time elapsed for transforming to pytorch_geometric graphs: {}s".format(end2-end))
    return g_list


def subgraph_extraction_labeling(ind, A_group_item_row, A_group_item_col, A_user_item_row, A_user_item_col, A_group_user_row, A_group_user_col, h=1, sample_ratio=1.0, max_nodes_per_hop=None,
                                 u_features=None, v_features=None, class_values=None,
                                 y=1):
    # extract the h-hop enclosing subgraph around link 'ind'
    group_nodes, item_nodes = [ind[0]], [ind[1]]
    group_dist, item_dist = [0], [0]
    group_visited, item_visited = set([ind[0]]), set([ind[1]])
    group_fringe, item_fringe = set([ind[0]]), set([ind[1]])
    user_nodes = []
    user_dist = []
    user_visited = set([])
    user_fringe = set([])
    for dist in range(1, 2):
        item_fringe, group_fringe = neighbors(group_fringe, A_group_item_row), neighbors(item_fringe, A_group_item_col)
        group_fringe = group_fringe - group_visited
        item_fringe = item_fringe - item_visited
        group_visited = group_visited.union(group_fringe)
        item_visited = item_visited.union(item_fringe)
        user_fringe_1, user_fringe_2 = neighbors(group_visited, A_group_user_row), neighbors(item_visited, A_user_item_col)
        # print("len user fringe 1, group visited", len(user_fringe_1), len(group_visited))
        # print("len user fringe 2, item  visited", len(user_fringe_2), len(item_visited))
        user_fringe_1 = user_fringe_1 - user_fringe
        user_fringe_2 = user_fringe_2 - user_fringe
        user_fringe = user_fringe_1
        user_visited = user_visited.union(user_fringe)
        # print("len user fringe_1, fringe_2, fringe, visited: ", len(user_fringe_1), len(user_fringe_2), len(user_fringe), len(user_visited))
        if sample_ratio < 1.0: # not required
            u_fringe = random.sample(u_fringe, int(sample_ratio*len(u_fringe)))
            v_fringe = random.sample(v_fringe, int(sample_ratio*len(v_fringe)))
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(group_fringe):
                group_fringe = random.sample(group_fringe, max_nodes_per_hop)
            if max_nodes_per_hop < len(item_fringe):
                item_fringe = random.sample(item_fringe, max_nodes_per_hop)
            if max_nodes_per_hop < len(user_fringe):
                user_fringe = random.sample(user_fringe, max_nodes_per_hop)
        if len(group_fringe) == 0 and len(item_fringe) == 0:
            break
        group_nodes = group_nodes + list(group_fringe)
        item_nodes = item_nodes + list(item_fringe)
        user_nodes = user_nodes + list(user_fringe)
        group_dist = group_dist + [dist] * len(group_fringe)
        item_dist = item_dist + [dist] * len(item_fringe)
        user_dist = user_dist + [dist] * len(user_fringe)
    group_subgraph = A_group_item_row[group_nodes][:, item_nodes]
    user_subgraph = A_user_item_row[user_nodes][:, item_nodes]
    group_user_subraph = A_group_user_row[group_nodes][:, user_nodes]
    # remove link between target nodes
    group_subgraph[0, 0] = 0

    # prepare pyg graph constructor input
    group, item_group, rating_group = ssp.find(group_subgraph)  # r is 1, 2... (rating labels + 1)
    user, item_user, rating_user = ssp.find(user_subgraph)
    group_user, user_group, label_group_user = ssp.find(group_user_subraph)
    # print(len(item_group), len(item_user), len(group), len(user))
    # assert(len(item_group) == len(item_user))
    item_group += len(group_nodes)
    item_user += len(group_nodes)
    user += len(group_nodes) + len(item_nodes)
    user_group += len(group_nodes) + len(item_nodes)
    rating_group = rating_group - 1  # transform r back to rating label
    rating_user = rating_user - 1
    rating_max = max(np.max(rating_user), np.max(rating_group))
    num_nodes = len(group_nodes) + len(item_nodes) + len(user_nodes)
    node_labels = [x*3 for x in group_dist] + [x*3+1 for x in item_dist] + [x*3+2 for x in user_dist]
    max_node_label = 3*h + 2
    # print("num nodes: ", num_nodes)
    y = class_values[y]

    # get node features
    if u_features is not None:
        u_features = u_features[u_nodes]
    if v_features is not None:
        v_features = v_features[v_nodes]
    node_features = None
    if False:
        # directly use padded node features
        if u_features is not None and v_features is not None:
            u_extended = np.concatenate(
                [u_features, np.zeros([u_features.shape[0], v_features.shape[1]])], 1
            )
            v_extended = np.concatenate(
                [np.zeros([v_features.shape[0], u_features.shape[1]]), v_features], 1
            )
            node_features = np.concatenate([u_extended, v_extended], 0)
    if False:
        # use identity features (one-hot encodings of node idxes)
        u_ids = one_hot(u_nodes, Arow.shape[0] + Arow.shape[1])
        v_ids = one_hot([x+Arow.shape[0] for x in v_nodes], Arow.shape[0] + Arow.shape[1])
        node_ids = np.concatenate([u_ids, v_ids], 0)
        #node_features = np.concatenate([node_features, node_ids], 1)
        node_features = node_ids
    if True:
        # only output node features for the target user and item
        if u_features is not None and v_features is not None:
            node_features = [u_features[0], v_features[0]]

    return group, item_group, user, item_user, group_user, user_group, rating_group, rating_user, rating_max, label_group_user, node_labels, max_node_label, y, node_features


def construct_pyg_graph(group, item_group, user, item_user, group_user, user_group, rating_group, rating_user, rating_max, label_group_user, node_labels, max_node_label, y, node_features):
    group, item_group, user, item_user, group_user, user_group = torch.LongTensor(group), torch.LongTensor(item_group), torch.LongTensor(user), torch.LongTensor(item_user), torch.LongTensor(group_user), torch.LongTensor(user_group)
    # Normalising edge label for ratings to 0-1
    rating_group = torch.FloatTensor(rating_group)/rating_max
    rating_user = torch.FloatTensor(rating_user)/rating_max
    label_group_user = torch.LongTensor(label_group_user)
    edge_index = torch.stack([torch.cat([group, item_group, user, item_user, group_user, user_group]), torch.cat([item_group, group, item_user, user, user_group, group_user])], 0)
    edge_type = torch.cat([rating_group, rating_group, rating_user, rating_user, label_group_user, label_group_user])
    x = torch.FloatTensor(one_hot(node_labels, max_node_label+1))
    y = torch.FloatTensor([y])
    data = Data(x, edge_index, edge_type=edge_type, y=y)

    if node_features is not None:
        if type(node_features) == list:  # a list of u_feature and v_feature
            u_feature, v_feature = node_features
            data.u_feature = torch.FloatTensor(u_feature).unsqueeze(0)
            data.v_feature = torch.FloatTensor(v_feature).unsqueeze(0)
        else:
            x2 = torch.FloatTensor(node_features)
            data.x = torch.cat([data.x, x2], 1)
    return data


def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    if not fringe:
        return set([])
    return set(A[list(fringe)].indices)


def one_hot(idx, length):
    idx = np.array(idx)
    x = np.zeros([len(idx), length])
    x[np.arange(len(idx)), idx] = 1.0
    return x


def PyGGraph_to_nx(data):
    edges = list(zip(data.edge_index[0, :].tolist(), data.edge_index[1, :].tolist()))
    g = nx.from_edgelist(edges)
    g.add_nodes_from(range(len(data.x)))  # in case some nodes are isolated
    # transform r back to rating label
    edge_types = {(u, v): data.edge_type[i].item() for i, (u, v) in enumerate(edges)}
    nx.set_edge_attributes(g, name='type', values=edge_types)
    node_types = dict(zip(range(data.num_nodes), torch.argmax(data.x, 1).tolist()))
    nx.set_node_attributes(g, name='type', values=node_types)
    g.graph['rating'] = data.y.item()
    return g
