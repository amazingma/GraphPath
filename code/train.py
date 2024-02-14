from __future__ import division
from __future__ import print_function
import DataLoader as myLoader
from utils import get_map
from Dataset import getLoader, get_split
from GAT import GAT
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import auc, f1_score, roc_auc_score, recall_score, precision_score, accuracy_score
from sklearn.metrics import roc_curve, precision_recall_curve
import numpy as np
import time
import argparse
import random
import copy
# import os
import warnings
warnings.filterwarnings('ignore')


path = "../../data/GraphPath/"
omics_files = ['cna', 'mutation']
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--hidden', type=int, default=20, help='Number of hidden units.')
parser.add_argument('--heads', type=int, default=4, help='Multi-head Attention.')
parser.add_argument('--out', type=int, default=1, help='Number of output units.')
parser.add_argument('--alpha', type=float, default=0.182, help='Greater than zero.')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate(1 - keep probability).')
parser.add_argument('--lr', type=float, default=0.04, help='Initial learning rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum.')
parser.add_argument('--weight_decay', type=float, default=0.04, help='L2 loss on parameters.')
parser.add_argument('--replications', type=int, default=30, help='Times of repetitions of cross-validation.')
parser.add_argument('--epoch', type=int, default=1000, help='Max value of epochs to train.')
parser.add_argument('--batch', type=int, default=162, help='batch size.')

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)
rint = np.random.randint(1, 1000, args.replications)

# features, pathlist, labels = myLoader.feat_extract(path, omics_files)
# edges = myLoader.omix_info(path)
# adj = get_map(edges, pathlist)
# np.save('log/features.npy', features)
# np.save('log/labels.npy', labels)
# np.save('log/adj.npy', adj)

features = np.load('log/features.npy')
labels = np.load('log/labels.npy')
adj = np.load('log/adj.npy')
features = torch.tensor(np.array(features), dtype=torch.float32)
labels = torch.tensor(np.array(labels), dtype=torch.float32)
adj = torch.tensor(np.array(adj), dtype=torch.float32)
if args.cuda:
    adj = adj.cuda()


AUPR_list = []
AUC_list = []
F1_list = []
Recall_list = []
Precision_list = []
Accuracy_list = []
def k_fold(k, x, y, epo):
    for f in rint:
        torch.cuda.empty_cache()
        model = GAT(n_feat=features.shape[2], n_hid=args.hidden, n_class=args.out, n_path=features.shape[1], alpha=args.alpha, dropout=args.dropout, n_heads=args.heads)
        # model = nn.DataParallel(model, device_ids=[0])
        criterion = nn.BCELoss()
        opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        x_train, y_train, x_valid, y_valid, x_test, y_test = get_split(x, y, f)
        loader = getLoader(x_train, y_train, batch=args.batch)

        loss_min = 1.0
        model_best = None
        counter = 0
        for e in range(epo):
            if (e - counter) > 25 and e > 100:
                # PATH = './model/model_' + str(r+1) + '_' + str(f+1) + '.pt'
                # torch.save(model_best, PATH)
                # model_test = torch.load(PATH)
                break
            for bat, (x_bat, y_bat) in enumerate(loader):
                loss_vali = train(e, x_bat, y_bat, x_valid, y_valid, model, criterion, opt)
                if loss_vali < loss_min:
                    counter = e
                    loss_min = loss_vali
                    model_best = copy.deepcopy(model)
        loss_test, AUPR_test, F1_test, AUC_test, Recall_test, Precision_test, Accuracy_test = test(model_best, x_test, y_test, criterion, save=True)
        print('\033[1;31mTest Loss=' + format(loss_test.item(), '.4f') + ', Metrics=[' + format(AUPR_test, '.3f') + ', ' + format(AUC_test, '.3f') + ', '
              + format(F1_test, '.3f') + ', ' + format(Recall_test, '.3f') + ', ' + format(Precision_test, '.3f') + ', ' + format(Accuracy_test, '.3f') + ']\033[0m')

        AUPR_list.append(AUPR_test)
        AUC_list.append(AUC_test)
        F1_list.append(F1_test)
        Recall_list.append(Recall_test)
        Precision_list.append(Precision_test)
        Accuracy_list.append(Accuracy_test)
    return


def train(epo, x_train, y_train, x_valid, y_valid, model, criterion, opt):
    torch.cuda.empty_cache()
    if args.cuda:
        x_train = x_train.cuda()
        y_train = y_train.cuda()
    model.train()
    opt.zero_grad()
    pred_train = model(x_train, adj)
    # loss_train = criterion(pred_train, y_train)
    weight = torch.zeros_like(y_train)
    weight = torch.fill_(weight, 0.85)
    weight[y_train > 0] = 1.5
    loss_train = nn.BCELoss(weight=weight, size_average=True)(pred_train, y_train)
    loss_train.backward()
    opt.step()

    y_train = y_train.type(torch.IntTensor).numpy()
    Y_train = []
    for i in y_train:
        Y_train.append(i[0])
    Pred_train = np.round(pred_train.data.numpy()).astype(int)
    P_train = []
    for i in Pred_train:
        P_train.append(i[0])
    pred_train = pred_train.data.numpy()
    p_train = []
    for i in pred_train:
        p_train.append(i[0])

    F1_train = f1_score(Y_train, P_train)
    Recall_train = recall_score(Y_train, P_train)
    Precision_train = precision_score(Y_train, P_train)
    Accuracy_train = accuracy_score(Y_train, P_train)
    AUC_train = roc_auc_score(Y_train, p_train)
    pre, rec, thresholds = precision_recall_curve(Y_train, p_train)
    AUPR_train = auc(rec, pre)

    if not args.fastmode:
        loss_vali, AUPR_vali, F1_vali, AUC_vali, Recall_vali, Precision_vali, Accuracy_vali = test(model, x_valid, y_valid, criterion, save=False)
        # print('Epo' + str(epo) + '\033[1;32m Train\033[0m Loss=' + format(loss_train.item(), '.4f') + ', [' + format(AUPR_train, '.3f') + ', ' + format(AUC_train, '.3f')
        #       + ', ' + format(F1_train, '.3f') + ', ' + format(Recall_train, '.3f') + ', ' + format(Precision_train, '.3f') + ', ' + format(Accuracy_train, '.3f') + ']'
        #       + ';\033[1;35m Validation\033[0m Loss=' + format(loss_vali.item(), '.4f') + ', [' + format(AUPR_vali, '.3f') + ', ' + format(AUC_vali, '.3f')
        #       + ', ' + format(F1_vali, '.3f') + ', ' + format(Recall_vali, '.3f') + ', ' + format(Precision_vali, '.3f') + ', ' + format(Accuracy_vali, '.3f') + ']')
        return loss_vali


def test(model_t, x_test, y_test, criterion, save):
    torch.cuda.empty_cache()
    if args.cuda:
        x_test = x_test.cuda()
        y_test = y_test.cuda()
    model_t.eval()
    with torch.no_grad():
        pred_test = model_t(x_test, adj)
        loss_test = criterion(pred_test, y_test)

    y_test = y_test.type(torch.IntTensor).numpy()
    Y_test = []
    for i in y_test:
        Y_test.append(i[0])
    Pred_test = np.round(pred_test.data.numpy()).astype(int)
    P_test = []
    for i in Pred_test:
        P_test.append(i[0])
    pred_test = pred_test.data.numpy()
    p_test = []
    for i in pred_test:
        p_test.append(i[0])

    F1_test = f1_score(Y_test, P_test)
    Recall_test = recall_score(Y_test, P_test)
    Precision_test = precision_score(Y_test, P_test)
    Accuracy_test = accuracy_score(Y_test, P_test)
    fpr, tpr, thres_roc = roc_curve(Y_test, p_test)
    AUC_test = auc(fpr, tpr)
    pre, rec, thres_pr = precision_recall_curve(Y_test, p_test)
    AUPR_test = auc(rec, pre)
    # if save:
    #     np.save('log/' + str(AUPR_test) + '_Y_test.npy', Y_test)
    #     np.save('log/' + str(AUPR_test) + '_p_test.npy', p_test)

    return loss_test, AUPR_test, F1_test, AUC_test, Recall_test, Precision_test, Accuracy_test


k_fold(args.replications, features, labels, args.epoch)
print('The Final Metrics AUPR:' + format(np.mean(AUPR_list), '.6f') + '+-' + format(np.std(AUPR_list), '.6f') + ', AUC:' + format(np.mean(AUC_list), '.6f') + '+-' + format(np.std(AUC_list), '.6f')
      + ', F1:' + format(np.mean(F1_list), '.6f') + '+-' + format(np.std(F1_list), '.6f') + ', Recall:' + format(np.mean(Recall_list), '.6f') + '+-' + format(np.std(Recall_list), '.6f')
      + ', Precision:' + format(np.mean(Precision_list), '.6f') + '+-' + format(np.std(Precision_list), '.6f') + ', Accuracy:' + format(np.mean(Accuracy_list), '.6f') + '+-' + format(np.std(Accuracy_list), '.6f'))
