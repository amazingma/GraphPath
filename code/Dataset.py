import random
import pandas as pd
import torch.utils.data as data
from sklearn.model_selection import train_test_split, StratifiedKFold
import DataLoader as myLoader


def getLoader(x, y, batch):
    dataset = data.TensorDataset(x, y)
    loader = data.DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=0)
    return loader


def get_split(x, y, r):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=r)
    x_train, x_vali, y_train, y_vali = train_test_split(x_train, y_train, test_size=len(y_test), stratify=y_train, random_state=r)
    return x_train, y_train, x_vali, y_vali, x_test, y_test


def get_skf(k):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=666)
    return skf


def get_stratify(x, y, l):
    x_train, x_vali, y_train, y_vali = train_test_split(x, y, test_size=l, stratify=y, shuffle=True, random_state=6666)
    return x_train, x_vali, y_train, y_vali


def get_balance(x, y):
    df = pd.DataFrame(y)
    index_pos = df[(df.iloc[:, 0] == 1)].index.tolist()
    index_neg = df[(df.iloc[:, 0] == 0)].index.tolist()
    index_sub = random.sample(index_neg, 333)
    index_pos.extend(index_sub)
    x_train = x[index_pos]
    y_train = y[index_pos]

    # primary/metastatic
    omics_files = ['cna', 'mutation']
    path = "E:/Data/GraphPath/PNET/primary/"
    x_n_test, _, y_n_test = myLoader.feat_extract(path, omics_files)
    path = "E:/Data/GraphPath/PNET/metastatic/"
    x_p_test, _, y_p_test = myLoader.feat_extract(path, omics_files)

    return x_train, y_train, x_n_test, y_n_test, x_p_test, y_p_test
