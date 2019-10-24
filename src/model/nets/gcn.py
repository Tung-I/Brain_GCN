import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
#import scipy.sparse as sp

from src.model.nets.base_net import BaseNet
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GCN(BaseNet):
    def __init__(self, n_feat, n_hide, n_class, dropout_rate):
        super().__init__()

        self.n_feat = n_feat
        self.n_hide = n_hide
        self.n_class = n_class
        self.dropout_rate = dropout_rate

        self.gc1 = GraphConvolution(n_feat, n_hide)
        self.gc2 = GraphConvolution(n_hide, n_class)
        self.dropout_rate = dropout_rate

    def forward(self, x, a):
        #print(a.shape)
        i = torch.eye(a.size(0)).cuda()
        a_hat = a + i
        d = torch.diag(a_hat.sum(1))
        adj_arr = torch.mm(torch.inverse(d), a_hat)
        ###
        #to_save = adj_arr.clone().detach()
        #to_save = np.asarray(to_save.cpu())
        #np.save('/home/tony/Documents/adj_arr.npy', to_save)
        ###
        #print(d.shape):
        x = F.relu(self.gc1(x, adj_arr))
        x = F.dropout(x, self.dropout_rate)
        x = self.gc2(x, adj_arr)
        x = F.softmax(x, dim=1)
        return x


class GraphConvolution(nn.Sequential):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + 'Hi'

