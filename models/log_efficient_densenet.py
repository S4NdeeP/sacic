# Adapted from:
# https://github.com/gpleiss/efficient_densenet_pytorch/blob/master/models/densenet.py

from __future__ import division
from __future__ import absolute_import
from math import sqrt, log, floor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from .dense_modules import _setup_conv
from .transitions import Transition
import sys
from itertools import izip


def is_power2(num):
    u""" True iff integer is a power of 2"""
    return ((num & (num - 1)) == 0) and num != 0


def _bn_function_factory(norm, relu, conv, index, mode=1):
    # for index i, select {i-[2^k] for k in 0,.. log(i)}
    if mode == 1:
        connexions = [index - 2**k for k in xrange(1+floor(log(index,2)))]
    elif mode == 2:
        # Make sure the first input is always in
        connexions = [index - 2**k for k in xrange(1+floor(log(index,2)))]
        if 0 not in connexions:
            connexions.append(0)

    # print('index:', index, ">> connected to:", connexions)
    def bn_function(*inputs):
        concatenated_features = torch.cat([inputs[c] for c in connexions], 1)
        # print('concat feature:', concatenated_features.size())
        bottleneck_output = conv(relu(norm(concatenated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Module):
    def __init__(self,
                 num_input_features,
                 kernel_size,
                 params,
                 index):
        super(_DenseLayer, self).__init__()
        self.kernel_size = kernel_size
        self.bn_size = params.get(u'bn_size', 4)
        self.growth_rate = params.get(u'growth_rate', 32)
        self.drop_rate = params.get(u'conv_dropout', 0.)
        self.index = index
        self.mode = params.get(u'log_mode', 1)
        conv1, conv2 = _setup_conv(num_input_features, kernel_size, params)
        self.add_module(u'norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module(u'relu1', nn.ReLU(inplace=True)),
        self.add_module(u'conv1', conv1)
        self.add_module(u'norm2', nn.BatchNorm2d(self.bn_size * self.growth_rate)),
        self.add_module(u'relu2', nn.ReLU(inplace=True)),
        self.add_module(u'conv2', conv2)
        
    def forward(self, *prev_features):
        bn_function = _bn_function_factory(
            self.norm1,
            self.relu1,
            self.conv1,
            self.index,
            self.mode
            )
        if any(prev_feature.requires_grad for prev_feature in prev_features):
            # Does not compute intermediate values
            # but recompute them in the backward pass
            # tradeoff btw memory & computation
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        # new_features has g channels
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features,
                 kernels, params):
        super(_DenseBlock, self).__init__()
        growth_rate = params.get(u'growth_rate', 32)
        log_mode = params.get(u'log_mode', 1)
        print u'Dense channels:', num_input_features,; sys.stdout.write(u'')
        for i in xrange(num_layers):
            index = i + 1
            numc = floor(log(index,2)) + 1
            if log_mode == 1:
                if is_power2(index):
                    numf = num_input_features + (numc - 1) * growth_rate
                else:
                    numf = numc * growth_rate
            elif log_mode == 2:
                if is_power2(index):
                    numf = num_input_features + (numc - 1) * growth_rate
                else:
                    numf = numc * growth_rate + num_input_features

            print u"> (%d)" % numc, numf,; sys.stdout.write(u'')
            layer = _DenseLayer(
                numf,
                kernels[i],
                params,
                i+1,
            )
            self.add_module(u'denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for layer in self.children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class Log_Efficient_DenseNet(nn.Module):
    u""" 
    set to True to use checkpointing. Much more memory efficient, but slower.
    log connections inside a block:
    x_i = f_i(concat({x_{i-[2^k]}, i < [log(i)]}))
    Implementation of Log-Densenet V1 described in:
    ``LOG-DENSENET: HOW TO SPARSIFY A DENSENET``
    arxiv: https://arxiv.org/pdf/1711.00002.pdf
    """
    def __init__(self, num_init_features, params):
        super(Log_Efficient_DenseNet, self).__init__()
        growth_rate = params.get(u'growth_rate', 32)
        block_layers = params.get(u'num_layers', (6, 12, 24, 16))
        block_kernels = params[u'kernels']
        init_weights = params.get(u'init_weights', 0)
        divide_channels = params.get(u'divide_channels', 2)
        skip_last_trans = params.get(u'skip_last_trans', 0)
        self.features = nn.Sequential()
        num_features = num_init_features
        # start by reducig the input channels
        if divide_channels > 1:
            trans = nn.Conv2d(num_features, num_features // divide_channels, 1)
            if init_weights == u"manual":
                std = sqrt(1 / num_features)
                trans.weight.data.normal_(0, std)
            self.features.add_module(u'initial_transition', trans)
            num_features = num_features // divide_channels

        # Each denseblock
        for i, (num_layers, kernels) in enumerate(izip(block_layers,
                                                      block_kernels)):
            block = _DenseBlock(num_layers,
                                num_features,
                                kernels,
                                params
                               )
            self.features.add_module(u'denseblock%d' % (i + 1), block)
            num_features += num_layers * growth_rate
            if not i == len(block_layers) - 1 or not skip_last_trans:
                trans = Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2,
                    init_weights=init_weights)
                self.features.add_module(u'transition%d' % (i + 1), trans)
                num_features = num_features // 2
                print u"> (trans) ", num_features,; sys.stdout.write(u'')
        print
        self.output_channels = num_features
        # Final batch norm
        self.features.add_module(u'norm_final', nn.BatchNorm2d(num_features))
        self.features.add_module(u'relu_last', nn.ReLU(inplace=True))

    def forward(self, x):
        return  self.features(x.contiguous())
