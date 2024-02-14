import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import time
from GraphAttentionLayer import GraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, n_path, alpha, dropout, n_heads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.activ = nn.Tanh()

        self.AttLayers = [GraphAttentionLayer(n_feat, n_hid, alpha, dropout, concat=True) for _ in range(n_heads)]
        for i, attention in enumerate(self.AttLayers):
            self.add_module('attention_{}'.format(i), attention)
        self.lin1 = nn.Linear(n_heads * n_hid, 1)
        self.lin2 = nn.Linear(n_path, n_class)

    def forward(self, x, adj, out=False):
        x = torch.cat([attention(x, adj) for attention in self.AttLayers], dim=2)
        x = f.dropout(x, self.dropout, training=self.training)
        x = torch.squeeze(self.activ(self.lin1(x)))
        if out:
            t = round(time.time())
            np.save('log/node_' + str(t) + '.npy', x.detach().numpy())
        x = self.lin2(x)

        return torch.sigmoid(x)
