import torch
import numpy as np
import pandas as pd
import utils

minsize = 0
maxsize = 1926


def omix_data(path, omics_files):
    for omics in omics_files:
        print(omics + '.', end='')
    labels = pd.read_csv(path + "response.csv")
    patientID = labels.loc[:, 'id'].tolist()

    omicslist = []
    for i in range(len(omics_files)):
        temp = pd.read_csv(path + omics_files[i] + ".csv", index_col=0)
        tempID = temp.index.tolist()
        patientID = list(set(patientID) & set(tempID))
    for i in range(len(omics_files)):
        temp = pd.read_csv(path + omics_files[i] + ".csv", index_col=0)
        temp = temp.loc[patientID, :]
        temp = utils.preprocess(temp)
        temp.sort_index(inplace=True)
        omicslist.append(temp)
    labels = labels[labels['id'].isin(patientID)]
    labels.set_index(['id'], inplace=True)
    labels.sort_index(inplace=True)

    return omicslist, labels


def gene_info(path, omics_files):
    omics_genelist = []
    for i in range(len(omics_files)):
        temp_gene = pd.read_csv(path + omics_files[i] + "_genelist.csv")
        omics_genelist.append(temp_gene)
    path_gene = pd.read_csv(path + "/kegg_gene.csv")
    pathsize = path_gene.groupby(['ko']).size()
    pathlist = pathsize[((pathsize >= minsize) & (pathsize <= maxsize))].index.tolist()
    return omics_genelist, path_gene, pathlist


def omix_info(path):
    edges = pd.read_csv(path + "/edges.csv")
    return edges


def feat_extract(path, omics_files):
    omicslist, labels = omix_data(path, omics_files)
    omics_genelist, path_gene, pathlist = gene_info(path, omics_files)

    undup_path_genes = set(path_gene['gene'].values.tolist())
    undup_omic_genes = []
    for genelist in omics_genelist:
        temp_genelist = genelist['genes'].values.tolist()
        temp_set = set(temp_genelist)
        temp_list = list(undup_path_genes.intersection(temp_set))
        undup_omic_genes.append(temp_list)
    for omic_genes in undup_omic_genes:
        print(str(len(omic_genes)) + '.', end='')
    print()
    total_genes = []
    for omic_genes in undup_omic_genes:
        total_genes.extend(omic_genes)
    total_genes = set(total_genes)

    input = []
    i = 1
    for ko in pathlist:
        print(str(i) + ': ' + ko, end='\r')
        i = i + 1
        ko_genelist = path_gene[path_gene['ko'] == ko]['gene'].tolist()
        ko_genelist = list(set(ko_genelist))
        path_feat = data_extract(omicslist, undup_omic_genes, ko_genelist)
        path_feat = path_feat.to(torch.float32)
        input.append(path_feat)
    features = torch.cat(input, dim=1)

    print("Samples: " + str(features.shape[0]) + ", Pathways: " + str(features.shape[1]) + ", Genes: " + str(len(total_genes)))
    return features, pathlist, labels


def data_extract(omicslist, undup_omic_genes, ko_genelist):
    features_omics = []
    for i in range(len(omicslist)):
        feature = np.zeros((omicslist[i].shape[0], len(undup_omic_genes[i])))
        feature = pd.DataFrame(feature, columns=undup_omic_genes[i])
        for g in ko_genelist:
            if g in undup_omic_genes[i]:
                if g in omicslist[i].columns:
                    value = omicslist[i].loc[:, g]
                    feature.loc[:, g] = np.array(value)
        feature = torch.tensor(np.array(feature))
        features_omics.append(feature)

    feature = torch.cat(features_omics, dim=1)
    f = feature.unsqueeze(1)
    return f
