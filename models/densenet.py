u"""
DenseNet architecture
"""

from __future__ import division
from __future__ import absolute_import
from math import sqrt
import torch
import torch.nn as nn
from .dense_modules import *
from .transitions import Transition, Transition2
import sys
from itertools import izip


class DenseBlock(nn.Sequential):
    def __init__(self, num_layers,
                 num_input_features,
                 kernels,
                 params):
        super(DenseBlock, self).__init__()
        layer_type = params.get(u'layer_type', 1)
        growth_rate = params.get(u'growth_rate', 32)
        if layer_type == u"regular":
            LayerModule = DenseLayer
        elif layer_type == u"mid-dropout":  # Works fine, basically another dropout
            LayerModule = DenseLayer_midDP
        elif layer_type == u"nobn":  # W/o BN works fine if weights initialized "correctly"
            LayerModule = DenseLayer_noBN
        elif layer_type == u"asym":
            LayerModule = DenseLayer_Asym
        elif layer_type == u"dilated": # 3 conv in each layer, the 3rd being dilated
            LayerModule = DenseLayer_Dil
        else:
            raise ValueError(u'Unknown type: %d' % layer_type)
        print u'Dense channels:', num_input_features,; sys.stdout.write(u'')
        for i in xrange(num_layers):
            print u">", num_input_features + (i + 1) * growth_rate,; sys.stdout.write(u'')
            layer = LayerModule(
                num_input_features + i * growth_rate,
                kernels[i],
                params,
                first=i==0,
                )
            self.add_module(u'denselayer%d' % (i + 1), layer)
        
    def update(self, x):
        for layer in list(self.children()):
            x = layer.update(x)
        return x

    def reset_buffers(self):
        for layer in list(self.children()):
            layer.reset_buffers()

    def track(self, x):
        activations = []
        for layer in list(self.children()):
            # layer is a DenseLayer
            x, newf = layer.track(x)
            activations.append(newf.data.cpu().numpy())
            x = torch.cat([x, newf], 1)
        return x, activations


class DenseNet(nn.Module):
    def __init__(self, num_init_features, params):
        super(DenseNet, self).__init__()
        block_layers = params.get(u'num_layers', (24))
        block_kernels = params[u'kernels']
        growth_rate = params.get(u'growth_rate', 32)
        divide_channels = params.get(u'divide_channels', 2)
        init_weights = params.get(u'init_weights', 0)
        normalize_channels = params.get(u'normalize_channels', 0)
        transition_type = params.get(u'transition_type', 1)
        skip_last_trans = params.get(u'skip_last_trans', 0)

        if transition_type == 1:
            TransitionLayer = Transition
        elif transition_type == 2:
            TransitionLayer = Transition2

        self.features = nn.Sequential()
        num_features = num_init_features
        # start by normalizing the input channels #FIXME
        if normalize_channels:
            self.features.add_module(u'initial_norm',
                                     nn.GroupNorm(1, num_features))

        # start by reducing the input channels
        if divide_channels > 1:
            # In net2: trans = TransitionLayer
            trans = nn.Conv2d(num_features, num_features // divide_channels, 1)
            if init_weights == u"manual":
                std = sqrt(1 / num_features)
                trans.weight.data.normal_(0, std)
            self.features.add_module(u'initial_transition', trans)
            num_features = num_features // divide_channels
        # Each denseblock
        for i, (num_layers, kernels) in enumerate(izip(block_layers,
                                                      block_kernels)):
            block = DenseBlock(num_layers, num_features,
                                kernels, params)
            self.features.add_module(u'denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            # In net2: Only between blocks
            if not i == len(block_layers) - 1 or not skip_last_trans:
                trans = TransitionLayer(
                    num_input_features=num_features,
                    num_output_features=num_features // 2,
                    init_weights=init_weights)
                self.features.add_module(u'transition%d' % (i + 1), trans)
                num_features = num_features // 2
                print u"> (trans) ", num_features,; sys.stdout.write(u'')
        print
        self.output_channels = num_features
        # Final batch norm
        self.features.add_module(u'norm_last', nn.BatchNorm2d(num_features))
        self.features.add_module(u'relu_last', nn.ReLU(inplace=True))

    def forward(self, x):
        return self.features(x.contiguous())

    def update(self, x):
        x = x.contiguous()
        for layer in list(self.features.children()):
            if isinstance(layer, DenseBlock):
                x = layer.update(x)
            else:
                x = layer(x)
        return x

    def reset_buffers(self):
        for layer in list(self.features.children()):
            if isinstance(layer, DenseBlock):
                layer.reset_buffers()

    def track(self, x):
        activations = []
        x = x.contiguous()
        for layer in list(self.features.children()):
            if isinstance(layer, DenseBlock):
                x, actv = layer.track(x)
                activations.append(actv)
            else:
                x = layer(x)
        return x, activations
