from __future__ import division
from __future__ import absolute_import
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [u'truncated_max', u"truncated_mean",
           u'average_code', u'max_code',
           u"MaxAttention",
          ]


def truncated_max(tensor, src_lengths, track=False, *args):
    u"""
    Max-pooling up to effective legth
    """
    # input size: N, d, Tt, Ts
    # src_lengths : N,
    Pool = []
    Attention = []
    for n in xrange(tensor.size(0)):
        X = tensor[n]
        xpool, attn = X[:, :, :src_lengths[n]].max(dim=2)
        if track:
            targets = torch.arange(src_lengths[n])
            align = targets.apply_(lambda k: sum(attn[:, -1] == k))
            align = align/align.sum()
            Attention.append(align.unsqueeze(0))
        Pool.append(xpool.unsqueeze(0))
    result = torch.cat(Pool, dim=0)
    if track:
        return result, torch.cat(Attention, dim=0).cuda()
    return result


def truncated_mean(tensor, src_lengths, *args):
    u"""
    Average-pooling up to effective legth
    """
    # input size: N, d, Tt, Ts
    # src_lengths : N,
    # n=1
    # print('tensor:', tensor.size(), 'lengths:', src_lengths)
    Pool = []
    Attention = []
    for n in xrange(tensor.size(0)):
        X = tensor[n]
        xpool = X[:, :, :src_lengths[n]].mean(dim=2)
        xpool *=  math.sqrt(src_lengths[n])
        Pool.append(xpool.unsqueeze(0))
    result = torch.cat(Pool, dim=0)
    return result


def average_code(tensor, *args):
    return tensor.mean(dim=3)


def max_code(tensor, src_lengths=None, track=False, track_all = False):
    # input size: N, d, Tt, Ts
    # src_lengths : N, 1
    if track and not track_all:
        batch_size, nchannels, _, max_len = tensor.size()
        xpool, attn = tensor.max(dim=3)
        targets = torch.arange(max_len).type_as(attn)
        align = []
        activ_distrib = []
        activ = []
        for n in xrange(batch_size):
            # distribution of the argmax indices
            align.append(np.array([
                torch.sum(attn[n, :, -1] == k, dim=-1).data.item() / nchannels
                for k in targets
            ]))
            # weighted distribution of the argmax indices
            activ_distrib.append(np.array([
                torch.sum((attn[n, :, -1] == k).float() * xpool[n, :, -1], dim=-1).data.item()
                for k in targets
            ]))
            # return the sparse tensor (0 if not pooled, value otherwise)
            activ.append(np.array([
                ((attn[n, :, -1] == k).float() * xpool[n, :, -1]).data.cpu().numpy()
                for k in targets
            ]))

        align = np.array(align)
        activ = np.array(activ)
        activ_distrib = np.array(activ_distrib)
        return xpool, (None, align, activ_distrib, activ)
    if track and track_all:
        batch_size, nchannels, source_len, max_len = tensor.size()
        xpool, attn = tensor.max(dim=3)
        targets = torch.arange(max_len).type_as(attn)
        align = []
        activ_distrib = []
        activ = []
        #Implemented only for batch_size=1
        assert (batch_size == 1)
        for n in xrange(batch_size):
            for i in xrange(source_len):
                # distribution of the argmax indices
                align.append(np.array([
                    torch.sum(attn[n, :, i] == k, dim=-1).data.item() / nchannels
                    for k in targets
                ]))
                # weighted distribution of the argmax indices
                activ_distrib.append(np.array([
                    torch.sum((attn[n, :, i] == k).float() * xpool[n, :, i], dim=-1).data.item()
                    for k in targets
                ]))
                # return the sparse tensor (0 if not pooled, value otherwise)
                activ.append(np.array([
                    ((attn[n, :, i] == k).float() * xpool[n, :, i]).data.cpu().numpy()
                    for k in targets
                ]))

        align = np.array(align)
        activ = np.array(activ)
        activ_distrib = np.array(activ_distrib)
        return xpool, (None, align, activ_distrib, activ)
    else:
        return tensor.max(dim=3)[0]


class MaxAttention(nn.Module):
    def __init__(self, params, in_channels):
        super(MaxAttention, self).__init__()
        self.in_channels = in_channels
        self.attend = nn.Linear(in_channels, 1)
        self.dropout = params[u'attention_dropout']
        self.scale_ctx = params.get(u'scale_ctx', 1)
        if params[u'nonlin'] == u"tanh":
            self.nonlin = F.tanh
        elif params[u'nonlin'] == u"relu":
            self.nonlin = F.relu
        else:
            self.nonlin = lambda x: x
        if params[u'first_aggregator'] == u"max":
            self.max = max_code
        elif params[u'first_aggregator'] == u"truncated-max":
            self.max = truncated_max
        elif params[u'first_aggregator'] == u"skip":
            self.max = None
        else:
            raise ValueError(u'Unknown mode for first aggregator ', params[u'first_aggregator'])

    def forward(self, X, src_lengths, track=False, *args):
        if track:
            N, d, Tt, Ts = X.size()
            Xatt = X.permute(0, 2, 3, 1)
            alphas = self.nonlin(self.attend(Xatt))
            alphas = F.softmax(alphas, dim=2)
            # print('alpha:', alphas.size(), alphas)
            # alphas : N, Tt, Ts , 1
            context = alphas.expand_as(Xatt) * Xatt
            # Mean over Ts >>> N, Tt, d
            context = context.mean(dim=2).permute(0, 2, 1)
            if self.scale_ctx:
                context = math.sqrt(Ts) * context
            # Projection N, Tt, d
            if self.max is not None:
                Xpool, tracking = self.max(X,
                                           src_lengths,
                                           track=True)
                feat = torch.cat((Xpool, context), dim=1)
                #return feat, (alphas[0, -1, :, 0].data.cpu().numpy(), *tracking[1:]) # Check this
                return feat, (alphas[0, -1, :, 0].data.cpu().numpy(), tracking[1:])  # Check this
            else:
                return context
        else:
            N, d, Tt, Ts = X.size()
            Xatt = X.permute(0, 2, 3, 1)
            alphas = self.nonlin(self.attend(Xatt))
            alphas = F.softmax(alphas, dim=2)
            # alphas : N, Tt, Ts , 1
            context = alphas.expand_as(Xatt) * Xatt
            # Mean over Ts >>> N, Tt, d
            context = context.mean(dim=2).permute(0, 2, 1)
            if self.scale_ctx:
                context = math.sqrt(Ts) * context
            # Projection N, Tt, d
            if self.max is not None:
                Xpool = self.max(X, src_lengths)
                return torch.cat((Xpool, context), dim=1)
            else:
                return context

