import torch
import numpy as np
import scipy.sparse as sp
import pandas as pd
import random


def get_map(adj, pathlist):
    adj = adj[adj['0'].isin(pathlist)]
    adj = adj[adj['1'].isin(pathlist)]
    ##
    # PPI-based approach
    # adj = PPI_based_edge(pathlist)
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


def transposition():
    or_mutation = pd.read_csv("E:/Data/GraphPath/PNET/metastatic/or_mutation.csv")
    or_mutation.fillna(0, inplace=True)
    or_mutation.to_csv("E:/Data/GraphPath/PNET/metastatic/tf_mutation.csv", sep=',', index=False, header=True)

    or_cna = pd.read_csv("E:/Data/GraphPath/PNET/metastatic/or_cna.csv")
    tf_cna = pd.DataFrame(or_cna.values.T, index=or_cna.columns, columns=or_cna.index)
    tf_cna = tf_cna.loc[:, (tf_cna != 0).any(axis=0)]
    tf_cna.fillna(0, inplace=True)
    tf_cna.to_csv("E:/Data/GraphPath/PNET/metastatic/tf_cna.csv", sep=',', index=True, header=False)
    return


def PPI_based_edge(pathlist):
    path_gene = pd.read_csv("../../data/GraphPath/kegg_gene.csv")
    protein_links = pd.read_csv("../../data/GraphPath/9606.protein.links.v11.5.txt", sep=' ')
    PPI = protein_links[protein_links['combined_score'] >= 700]
    protein_info = pd.read_csv("../../data/GraphPath/9606.protein.info.v11.5.txt", sep='\t')
    adj = []
    for i in range(len(pathlist)):
        print(i, end='\r')
        p1 = pathlist[i]
        p1_genelist = path_gene[path_gene['ko'] == p1]['gene'].tolist()
        cut = i+1
        for j in range(cut, len(pathlist)):
            p2 = pathlist[j]
            p2_genelist = path_gene[path_gene['ko'] == p2]['gene'].tolist()
            interSet = list(set(p1_genelist) & set(p2_genelist))
            p1_compSet = list(set(p1_genelist) - set(p2_genelist))
            p2_compSet = list(set(p2_genelist) - set(p1_genelist))
            if not interSet:
                continue
            else:
                inter_proteinID = []
                for g_inter in interSet:
                    g_inter_proteinID = protein_info.loc[protein_info['preferred_name'] == g_inter]
                    if not g_inter_proteinID.empty:
                        g_inter_proteinID = g_inter_proteinID.iloc[0, 0]
                        inter_proteinID.append(g_inter_proteinID)
                judge1 = False
                for g1 in p1_compSet:
                    g1_proteinID = protein_info.loc[protein_info['preferred_name'] == g1]
                    if not g1_proteinID.empty:
                        g1_proteinID = g1_proteinID.iloc[0, 0]
                    else:
                        continue
                    g1_PPI = PPI[PPI['protein1'] == g1_proteinID].iloc[:, 1].tolist()
                    if list(set(g1_PPI) & set(inter_proteinID)):
                        judge1 = True
                        break
                judge2 = False
                for g2 in p2_compSet:
                    g2_proteinID = protein_info.loc[protein_info['preferred_name'] == g2]
                    if not g2_proteinID.empty:
                        g2_proteinID = g2_proteinID.iloc[0, 0]
                    else:
                        continue
                    g2_PPI = PPI[PPI['protein1'] == g2_proteinID].iloc[:, 1].tolist()
                    if list(set(g2_PPI) & set(inter_proteinID)):
                        judge2 = True
                        break
                if judge1 and judge2:
                    adj.append([p1, p2])
    adj = pd.DataFrame(adj)
    print("PPI-based edges: " + str(adj.shape[0]))
    return adj
