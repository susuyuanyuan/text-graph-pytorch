import torch
import torch.nn as nn


class GraphLayer(nn.Module):
    def __init__( self, input_dim, \
                        output_dim, \
                        support, \
                        activision_func = None, \
                        featureless = False, \
                        dropout_rate = 0., \
                        bias=False):
        super(GraphLayer, self).__init__()
        self.support = support
        self.featureless = featureless
        self.activision_func = activision_func
        self.dropout = nn.Dropout(dropout_rate)

        # initialize weights and bias
        for i in range(len(self.support)):
            setattr(self, 'W{}'.format(i),
                    nn.Parameter(torch.randn(input_dim, output_dim)))

        if bias:
            self.b = nn.Parameter(torch.zeros(1, output_dim))

    def forward(self, x):
        x = self.dropout(x)

        for i in range(len(self.support)):
            if self.featureless:
                pre_sup = getattr(self, 'W{}'.format(i))
            else:
                pre_sup = x.mm(getattr(self, 'W{}'.format(i)))

            if i == 0:
                out = self.support[i].mm(pre_sup)
            else:
                out += self.support[i].mm(pre_sup)

        if self.activision_func is not None:
            out = self.activision_func(out)

        self.embedding = out
        return out


class GCN(nn.Module):
    def __init__(self,
                 input_dim,
                 support,
                 output_dim=200,
                 dropout_rate=0.0,
                 num_classes=10,
                 featureless=True):
        super(GCN, self).__init__()

        self.layer1 = GraphLayer(input_dim,
                                 output_dim,
                                 support,
                                 activision_func=nn.ReLU(),
                                 featureless=featureless,
                                 dropout_rate=dropout_rate)
        self.layer2 = GraphLayer(output_dim,
                                 num_classes,
                                 support,
                                 dropout_rate=dropout_rate)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out
