import torch
import numpy as np
import scipy.sparse as sp
import pandas as pd
import random


def get_map(adj, pathlist):
    adj = adj[adj['0'].isin(pathlist)]
    adj = adj[adj['1'].isin(pathlist)]
    idx = np.array(pathlist, dtype=np.str)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.array(adj.values.tolist())
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.str).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(idx.shape[0], idx.shape[0]), dtype=np.float32)
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = torch.FloatTensor(np.array(adj.todense()))
    return adj


def normalize(adj):
    degree = np.array(adj.sum(1))
    d_inv_sq = np.power(degree, -0.5).flatten()
    d_inv_sq[np.isinf(d_inv_sq)] = 0.
    d_mat_inv_sq = sp.diags(d_inv_sq)
    # tocoo/tocsr/tocsc
    mx = adj.dot(d_mat_inv_sq).transpose().dot(d_mat_inv_sq)
    return mx


def preprocess(data):
    data.dropna(axis=1, how='all')
    data = data.groupby(data.index).mean()
    data.fillna(0, inplace=True)
    # print(np.any(data.isnull()))
    return data