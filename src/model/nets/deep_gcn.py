import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
#import scipy.sparse as sp

from src.model.nets.base_net import BaseNet
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class DeepGCN(BaseNet):
    def __init__(self, n_feat, n_hide, n_class, dropout_rate):
        super().__init__()

        self.n_feat = n_feat
        self.n_hide = n_hide
        self.n_class = n_class
        self.dropout_rate = dropout_rate

        n_hide = [256, 512, 1024, 512, 256]
        self.gc1 = GraphConvolution(n_feat, n_hide[0])
        self.gc2 = GraphConvolution(n_hide[0], n_hide[1])
        self.gc3 = GraphConvolution(n_hide[1], n_hide[2])
        self.gc4 = GraphConvolution(n_hide[2], n_hide[3])
        self.gc5 = GraphConvolution(n_hide[3], n_hide[4])
        self.gc6 = GraphConvolution(n_hide[4], n_class)
        
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
        x1 = F.relu(self.gc1(x, adj_arr))
        x1 = F.dropout(x1, self.dropout_rate)
        x2 = F.relu(self.gc2(x1, adj_arr))
        x2 = F.dropout(x2, self.dropout_rate)
        x3 = F.relu(self.gc3(x2, adj_arr))
        x3 = F.dropout(x3, self.dropout_rate)
        x4 = F.relu(self.gc4(x3, adj_arr))
        x4 = F.dropout(x4, self.dropout_rate)
        x5 = F.relu(self.gc5(x4+x2, adj_arr))
        x5 = F.dropout(x5, self.dropout_rate)
        x6 = self.gc6(x5+x1, adj_arr)
        x6 = F.softmax(x6, dim=1)
        return x6


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

