#
# ported from https://github.com/IllinoisGraphBenchmark/IGB-Datasets/blob/main/igb/models.py
# add activation
#

import dgl.nn.pytorch as dglnn
import torch.nn as nn
from contextlib import contextmanager

class GAT(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_heads, n_layers, activation, dropout):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_heads = n_heads

        self.layers.append(dglnn.GATConv(in_feats, n_hidden, n_heads, allow_zero_in_degree=True))
        for _ in range(1, n_layers-1):
            self.layers.append(dglnn.GATConv(n_hidden * n_heads, n_hidden, n_heads, allow_zero_in_degree=True))
        self.layers.append(dglnn.GATConv(n_hidden * n_heads, n_classes, n_heads, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation # should be torch.nn.functional.relu

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_dst = h[:block.num_dst_nodes()]
            if l < len(self.layers) - 1:
                h = layer(block, (h, h_dst)).flatten(1)
                h = self.activation(h)
                h = self.dropout(h)
            else:
                h = layer(block, (h, h_dst)).mean(1)  
        return h

    def inference(self, g, x, batch_size, device):
        pass

    @contextmanager
    def join(self):
        """dummy join for standalone"""
        yield
