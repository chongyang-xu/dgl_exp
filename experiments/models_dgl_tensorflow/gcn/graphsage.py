import tensorflow as tf
from tensorflow.keras import layers

from dgl.nn.tensorflow import SAGEConv


class GraphSAGE(tf.keras.Model):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, activation, dropout
    ):
        super(GraphSAGE, self).__init__()
        self.layer_list = []
        # input layer
        self.layer_list.append(
            SAGEConv(in_feats, n_hidden, aggregator_type='mean',
                     activation=activation)
        )
        # hidden layers
        for i in range(n_layers - 1):  # TODO: take care layer num
            self.layer_list.append(
                SAGEConv(n_hidden, n_hidden, aggregator_type='mean',
                         activation=activation)
            )
        # output layer
        self.layer_list.append(
            SAGEConv(n_hidden, n_classes, aggregator_type='mean'))
        self.dropout = layers.Dropout(dropout)

    def call(self, g, features):
        h = features
        for i, layer in enumerate(self.layer_list):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h
