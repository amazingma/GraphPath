import torch
import torch.nn as nn
import torch.nn.functional as f


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, alpha, dropout, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.dropout = dropout
        self.concat = concat

        torch.manual_seed(66)
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.activ = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.matmul(input, self.W)

        N = h.size()[1]
        a_input = torch.cat([h.repeat(1, 1, N).view(-1, N * N, self.out_features), h.repeat(1, N, 1)], dim=2).view(-1, N, N, 2 * self.out_features)
        # [B, N, N]
        e = self.activ(torch.matmul(a_input, self.a).squeeze(3))
        zero_vec = -9e15 * torch.ones_like(e)
        attn = torch.where(adj > 0, e, zero_vec)
        attn = f.softmax(attn, dim=1)
        attn = f.dropout(attn, self.dropout, training=self.training)

        h_prime = torch.matmul(attn, h)
        if self.concat:
            return f.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
