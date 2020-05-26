u"""
Pervasive attention
"""
from __future__ import absolute_import
import math
import itertools
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .densenet import DenseNet
from .efficient_densenet import Efficient_DenseNet
from .log_efficient_densenet import Log_Efficient_DenseNet

from .aggregator import Aggregator
from .embedding import Embedding, ConvEmbedding
from .beam_search import Beam
from .AttModel import sort_pack_padded_sequence, pad_unsort_packed_sequence, pack_wrapper
from .CaptionModel import CaptionModel

att_res = 7
def _expand(tensor, dim, reps):
    # Expand 4D tensor in the source or the target dimension
    if dim == 1:
        return tensor.repeat(1, reps, 1, 1)
        # return tensor.expand(-1, reps, -1, -1)
    if dim == 2:
        return tensor.repeat(1, 1, reps, 1)
        # return tensor.expand(-1, -1, reps, -1)
    else:
        raise NotImplementedError


class Pervasive(CaptionModel):
    def __init__(self, jobname, params, src_vocab_size, trg_vocab_size,
                 special_tokens):
        nn.Module.__init__(self)
        self.use_obj_mcl_loss = params[u'use_obj_mcl_loss']
        self.num_heads = params[u'num_heads']
        self.use_obj_att = params[u'use_obj_att']
        self.use_seg_feat = params[u'use_seg_feat']
        self.use_img_feat = params[u'use_img_feat']
        self.logger = logging.getLogger(jobname)
        self.version = u'conv'
        self.params = params
        self.merge_mode  = params[u'network'].get(u'merge_mode', u'concat')
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.padding_idx = special_tokens[u'PAD']
        self.mask_version = params.get(u'mask_version', -1)
        # assert self.padding_idx == 0, "Padding token should be 0"
        self.bos_token = special_tokens[u'BOS']
        self.eos_token = special_tokens[u'EOS']
        self.kernel_size = max(list(itertools.chain.from_iterable(
            params[u'network'][u'kernels']
            )))
        self.seg_feat_size = 310

        self.trg_embedding = Embedding(
            params[u'decoder'],
            trg_vocab_size,
            padding_idx=self.padding_idx,
            pad_left=True
            )

        self.ss_prob = 0.0  # Schedule sampling probability
        self.drop_prob_lm = params['drop_prob_lm']
        self.use_bn = getattr(params, 'use_bn', 0)
        self.att_feat_size = params['att_feat_size']
        self.input_encoding_size = params['input_encoding_size']
        self.input_obj_encoding_size = params['input_obj_encoding_size']

        class FeedforwardNeuralNetModel(nn.Module):
          def __init__(self, input_dim, hidden_dim, output_dim):
            super(FeedforwardNeuralNetModel, self).__init__()
            # Linear function
            self.fc1 = nn.Linear(input_dim, hidden_dim)

            # Non-linearity
            self.relu = nn.ReLU()

            # Linear function (readout)
            self.fc2 = nn.Linear(hidden_dim, output_dim)

          def forward(self, x):
            # Linear function  # LINEAR
            out = self.fc1(x)

            # Non-linearity  # NON-LINEAR
            out = self.relu(out)

            # Linear function (readout)  # LINEAR
            out = self.fc2(out)
            return out

        if self.use_img_feat:
            self.att_embed = nn.Sequential(*(
              ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
              (nn.Linear(self.att_feat_size, self.input_encoding_size),
               nn.ReLU(),
               nn.Dropout(self.drop_prob_lm)) +
              ((nn.BatchNorm1d(self.input_encoding_size),) if self.use_bn == 2 else ())))

        if self.use_obj_att:
            self.obj_att_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
                (nn.Linear(self.att_feat_size, self.input_obj_encoding_size),
                 nn.ReLU(),
                 nn.Dropout(self.drop_prob_lm)) +
                ((nn.BatchNorm1d(self.input_obj_encoding_size),) if self.use_bn == 2 else ())))

        if self.use_seg_feat:
            self.seg_att_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(self.seg_feat_size),) if self.use_bn else ()) +
                (nn.Linear(self.seg_feat_size, self.input_obj_encoding_size),
                 nn.ReLU(),
                 nn.Dropout(self.drop_prob_lm)) +
                ((nn.BatchNorm1d(self.input_obj_encoding_size),) if self.use_bn == 2 else ())))

        ''''
        self.att_embed = nn.Sequential(*(
          ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
          (FeedforwardNeuralNetModel(self.att_feat_size, 512, self.input_encoding_size),
           nn.ReLU(),
           nn.Dropout(self.drop_prob_lm)) +
          ((nn.BatchNorm1d(self.input_encoding_size),) if self.use_bn == 2 else ())))
        '''
        #self.att_embed = nn.Sequential()
        #self.input_channels_initial = 256

        if not self.use_seg_feat:
            self.input_channels_initial = self.input_encoding_size
        else:
            if self.use_img_feat:
                self.input_channels_initial = self.input_encoding_size + self.input_obj_encoding_size #self.seg_feat_size
            else:
                self.input_channels_initial = self.input_obj_encoding_size

        self.src_encoding_size = 128
        if self.merge_mode == u'concat':
            self.input_channels = self.src_encoding_size + \
                                  self.trg_embedding.dimension
        elif self.merge_mode == u"product":
            self.input_channels = self.input_encoding_size
        elif self.merge_mode == u"bilinear":
            bilinear_dim = params[u'network'].get(u'bilinear_dimension', 128)
            self.input_channels = bilinear_dim
            std = params[u'encoder'].get(u'init_std', 0.01)
            self.bw = nn.Parameter(std * torch.randn(bilinear_dim))
        elif self.merge_mode == u"multi-sim":
            self.sim_dim = params[u'network'].get(u'similarity_dimension', 128)
            self.input_channels = self.sim_dim
            std = params[u'encoder'].get(u'init_std', 0.01)
            self.bw = nn.Parameter(std * torch.randn(self.sim_dim,
                                                     self.trg_embedding.dimension,
                                                     self.input_encoding_size))

        elif self.merge_mode == u"multi-sim2":
            self.sim_dim = params[u'network'].get(u'similarity_dimension', 128)
            self.input_channels = self.sim_dim
            std = params[u'encoder'].get(u'init_std', 0.01)
            self.bw = nn.Parameter(torch.empty(self.sim_dim,
                                               self.trg_embedding.dimension,
                                               self.input_encoding_size))

            nn.init.orthogonal_(self.bw)
        else:
            raise ValueError(u'Unknown merging mode')


        self.logger.info(u'Model input channels: %d', self.input_channels)
        self.logger.info(u"Selected network: %s", params[u'network'][u'type'])


        if params[u'network'][u'divide_channels'] > 1:
            self.logger.warning(u'Reducing the input channels by %d',
                                params[u'network'][u'divide_channels'])

        if params[u"network"][u'type'] == u"densenet":
            self.net = DenseNet(self.input_channels, params[u'network'])
            self.network_output_channels = self.net.output_channels

        elif params[u"network"][u'type'] == u"efficient-densenet":

            self.net_initial = Efficient_DenseNet(self.input_channels_initial, params[u'init_network'])
            self.network_output_channels_initial = self.net_initial.output_channels
            if(self.use_obj_att):
                self.input_channels_obj_att = 256 * 2
                self.net_obj_att = Efficient_DenseNet(self.input_channels_obj_att, params[u'obj_att_network'])
                self.network_output_channels_obj_att = self.net_obj_att.output_channels
            self.net = Efficient_DenseNet(self.input_channels, params[u'network'])
            self.network_output_channels = self.net.output_channels

        elif params[u"network"][u'type'] == u"log-densenet":
            self.net = Log_Efficient_DenseNet(self.input_channels, params[u'network'])
            self.network_output_channels = self.net.output_channels
        else:
            raise ValueError(
                u'Unknown architecture %s' % params[u'network'][u'type'])

        self.tie_target_weights = params[u'decoder'][u'tie_target_weights']
        self.copy_source_weights = params[u'decoder'][u'copy_source_weights']

        if self.tie_target_weights:
            self.logger.warning(u'Tying the decoder weights')
            last_dim = params[u'decoder'][u'input_dim']
        else:
            last_dim = None

        if self.use_obj_att:
            self.aggregator_initial = Aggregator(self.network_output_channels_initial,
                                                 self.input_channels_obj_att/2,
                                                 params[u'aggregator'])
            self.aggregator_obj_att = Aggregator(self.network_output_channels_obj_att,
                                                 last_dim,
                                                 params[u'aggregator'])
        else:
            self.aggregator_initial = Aggregator(self.network_output_channels_initial,
                                                 last_dim,
                                                 params[u'aggregator'])
        if True:
            aggregator_params = params[u'aggregator'].copy()
            aggregator_params[u'mode'] = u'max'

            if(self.use_obj_att):
                self.aggregator_initial_mean = Aggregator(self.network_output_channels_initial,
                                                self.input_channels_obj_att/2,
                                                aggregator_params)
            else:
                self.aggregator_initial_mean = Aggregator(self.network_output_channels_initial,
                                                          last_dim,
                                                          aggregator_params)


        self.aggregator = Aggregator(self.network_output_channels,
                                     last_dim,
                                     params[u'aggregator'])
        self.final_output_channels = self.aggregator.output_channels  # d_h

        self.prediction_dropout = nn.Dropout(
            params[u'decoder'][u'prediction_dropout'])
        self.logger.info(u'Output channels: %d', self.final_output_channels)
        self.prediction = nn.Linear(self.final_output_channels,
                                    self.trg_vocab_size)
        if self.tie_target_weights:
            self.prediction.weight = self.trg_embedding.label_embedding.weight

        num_classes = 512
        if self.use_obj_att:
            num_features = self.input_channels_obj_att
        else:
            num_features = 128

        if self.use_obj_mcl_loss:
            self.multi_label_classifier = nn.Linear(num_features, num_classes)

    def init_weights(self):
        u"""
        Called after setup.buil_model to intialize the weights
        """
        if self.params[u'network'][u'init_weights'] == u"kaiming":
            nn.init.kaiming_normal_(self.prediction.weight)

        #self.src_embedding.init_weights()
        self.trg_embedding.init_weights()
        self.prediction.bias.data.fill_(0)
    
    def merge(self, src_emb, trg_emb):
        u"""
        Merge source and target embeddings
        *_emb : N, T_t, T_s, d
        """
        N, Tt, Ts, _ = src_emb.size()
        if self.merge_mode == u'concat':
            # 2d grid:
            return torch.cat((src_emb, trg_emb), dim=3)
        elif self.merge_mode == u'product':
            return src_emb * trg_emb
        elif self.merge_mode == u'bilinear':
            # self.bw : d
            # for every target position
            X = []
            for t in xrange(Tt):
                # trg_emb[:, t, :] (N, 1, d_t)
                e = trg_emb[:, t:t+1, 0, :]
                w = self.bw.expand(N, -1).unsqueeze(-1)
                # print('e:', e.size())
                # print('bw:', w.size())
                x = torch.bmm(w, e).transpose(1, 2)
                # print('x:', x.size())
                # x  (N, d_t, d) & src_emb (N, T_s, d_s = d_t) => (N, 1, T_s, d)
                x = torch.bmm(src_emb[:,0], x).unsqueeze(1)
                # print('appending:', x.size())
                X.append(x)
            return torch.cat(X, dim=1)

        elif self.merge_mode == u"multi-sim":
            # self.bw d, d_t, ds
            X = []
            for k in xrange(self.sim_dim):
                w = self.bw[k].expand(N, -1, -1)
                # print('w:', w.size())
                # print(trg_emb[:,:,0].size())
                # print(src_emb[:,0].size())
                x = torch.bmm(torch.bmm(trg_emb[:,:,0], w), src_emb[:,0].transpose(1,2)).unsqueeze(-1)
                # print('x:', x.size())
                X.append(x)
            return torch.cat(X, dim=-1)

        elif self.merge_mode == u"multi-sim2":
            # self.bw d, d_t, ds
            X = []
            for n in xrange(N):
                x = torch.bmm(torch.bmm(trg_emb[n:n+1,:,0].expand(self.sim_dim, -1, -1), self.bw),
                              src_emb[n:n+1,0].expand(self.sim_dim, -1, -1).transpose(1,2)
                             ).unsqueeze(0)
                X.append(x)
            return torch.cat(X, dim=0).permute(0, 2, 3, 1)

        else:
            raise ValueError(u'Unknown merging mode')


    def subsequent_mask(self, size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            seq = seq[:,:-1]
            seq_mask = (seq.data > 0)
            seq_mask[:,0] += 1

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & self.subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    def _prepare_feature_obj(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        att_feats = pack_wrapper(self.obj_att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            seq = seq[:,:-1]
            seq_mask = (seq.data > 0)
            seq_mask[:,0] += 1

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & self.subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    def _prepare_feature_seg(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        att_feats = pack_wrapper(self.seg_att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            seq = seq[:,:-1]
            seq_mask = (seq.data > 0)
            seq_mask[:,0] += 1

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & self.subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    def _forward(self, fc_feats, att_feats, seq, att_masks = None):

        if self.use_obj_att and self.use_seg_feat:
            obj_att_feats = att_feats[1]
            obj_att_masks = att_masks[1]
            seg_feat_feats = att_feats[2]
            seg_feat_masks = att_masks[2]
            att_feats = att_feats[0]
            att_masks = att_masks[0]
            if self.use_img_feat:
                att_feats, seq, att_masks, seq_mask = self._prepare_feature(att_feats, att_masks, seq)
            seg_feat_feats, seq, seg_feat_masks, seq_mask = self._prepare_feature_seg(seg_feat_feats, seg_feat_masks, seq)
            obj_att_feats, seq, obj_att_masks, seq_mask = self._prepare_feature_obj(obj_att_feats, obj_att_masks, seq)
        elif not self.use_obj_att and self.use_seg_feat:
            seg_feat_feats = att_feats[1]
            seg_feat_masks = att_masks[1]
            att_feats = att_feats[0]
            att_masks = att_masks[0]
            if self.use_img_feat:
                att_feats, seq, att_masks, seq_mask = self._prepare_feature(att_feats, att_masks, seq)
            seg_feat_feats, seq, seg_feat_masks, seq_mask = self._prepare_feature_seg(seg_feat_feats, seg_feat_masks, seq)
        elif self.use_obj_att and not self.use_seg_feat:
            obj_att_feats = att_feats[1]
            obj_att_masks = att_masks[1]
            att_feats = att_feats[0]
            att_masks = att_masks[0]
            att_feats, seq, att_masks, seq_mask = self._prepare_feature(att_feats, att_masks, seq)
            obj_att_feats, seq, obj_att_masks, seq_mask = self._prepare_feature_obj(obj_att_feats, obj_att_masks, seq)
        elif not self.use_obj_att and not self.use_seg_feat:
            att_feats, seq, att_masks, seq_mask = self._prepare_feature(att_feats, att_masks, seq)

        X = torch.reshape(att_feats, [att_feats.size(0), att_res, att_res, att_feats.size(2)])
        if(att_res == 14):
            X = X.permute(0, 3, 1, 2)
            X = F.adaptive_avg_pool2d(X, [att_res, att_res])
            X = X.permute(0, 2, 3, 1)

        if self.use_seg_feat:
            X_seg = torch.reshape(seg_feat_feats, [seg_feat_feats.size(0), att_res, att_res, seg_feat_feats.size(2)])
            if self.use_img_feat:
                X = torch.cat((X, X_seg), 3)
            else:
                X = X_seg

        src_masks = att_feats.new_ones(X.shape[:2], dtype=torch.long)
        src_masks = src_masks.unsqueeze(-2)

        src_lengths = torch.squeeze(torch.sum(src_masks, dim=2))

        if(self.use_obj_mcl_loss == 0):
            X, X_obj = self._forward_initial_multi(X, src_lengths)
            src_emb = torch.cat((X, X_obj), dim=2)
            dim = src_emb.size()
            src_emb = torch.reshape(src_emb, (dim[0], dim[1] * 2, dim[2] / 2))
        elif(self.use_obj_mcl_loss == 1):
            X, X_obj = self._forward_initial_multi(X, src_lengths)
            src_emb = torch.cat((X, X_obj), dim=2)
            dim = src_emb.size()
            src_emb = torch.reshape(src_emb, (dim[0], dim[1]*2, dim[2]/2))
            if self.use_obj_att:
                out = F.adaptive_avg_pool2d(src_emb, (1, self.input_channels_obj_att)).view(X_obj.size(0), -1)
            else:
                out = F.adaptive_avg_pool2d(src_emb, (1, 128)).view(X_obj.size(0), -1)
            out = self.multi_label_classifier(out)

        #src_emb = self.src_embedding(att_feats)

        if (self.use_obj_att):
            trg_emb = src_emb
            src_emb = obj_att_feats
            src_masks = obj_att_masks
            Ts = src_emb.size(1)  # source sequence length
            Tt = trg_emb.size(1)  # target sequence length

            # 2d grid:
            src_emb = _expand(src_emb.unsqueeze(1), 1, Tt)
            trg_emb = _expand(trg_emb.unsqueeze(2), 2, Ts)

            X = self.merge(src_emb, trg_emb)
            src_lengths = torch.squeeze(torch.sum(src_masks, dim=2))

            X = self._forward_obj_attn(X, src_lengths)
            src_emb = X

        trg_emb = self.trg_embedding(seq)

        Ts = src_emb.size(1)  # source sequence length
        Tt = trg_emb.size(1)  # target sequence length
        # 2d grid:
        src_emb = _expand(src_emb.unsqueeze(1), 1, Tt)
        trg_emb = _expand(trg_emb.unsqueeze(2), 2, Ts)
        X = self.merge(src_emb, trg_emb)
        # del src_emb, trg_emb
        #src_lengths = torch.squeeze(torch.sum(src_masks,dim = 2))
        #X = self._forward_(X, data_src[u'lengths'])

        X = self._forward_(X, src_lengths)
        logits = F.log_softmax(
            self.prediction(self.prediction_dropout(X)), dim=2)

        if self.use_obj_mcl_loss == 1 and self.training:
            return logits, out
        return logits

    # @profile
    def _forward_obj_attn(self, X, src_lengths=None, track=False):
        X = X.permute(0, 3, 1, 2)
        X = self.net_obj_att(X)
        if track:
            X, attn = self.aggregator_obj_att(X, src_lengths, track=True, track_all = True)
            return X, attn
        X = self.aggregator_obj_att(X, src_lengths, track=track)
        return X

    # @profile
    def _forward_initial(self, X, src_lengths=None, track=False):
        X = X.permute(0, 3, 1, 2)
        X = self.net_initial(X)
        if track:
            X, attn = self.aggregator_initial(X, src_lengths, track=True, track_all = True)
            return X, attn
        X = self.aggregator_initial(X, src_lengths, track=track)
        return X

    # @profile
    def _forward_initial_multi(self, X, src_lengths=None, track=False):
        X = X.permute(0, 3, 1, 2)
        X = self.net_initial(X)
        if track:
            X_max, attn_max = self.aggregator_initial(X, src_lengths, track=True, track_all = True)
            set_1 = X_max, attn_max
        X_max = self.aggregator_initial(X, src_lengths, track=track)

        X = X.permute(0,1,3,2)
        if track:
            X_mean, attn_mean = self.aggregator_initial_mean(X, src_lengths, track=True, track_all = True)
            set_2 = X_mean, attn_mean
        X_mean = self.aggregator_initial_mean(X, src_lengths, track=track)
        #X_mean = X_max
        if track:
            return set_1, set_2
        return X_max, X_mean

    # @profile
    def _forward_(self, X, src_lengths=None, track=False):
        X = X.permute(0, 3, 1, 2)
        X = self.net(X)
        if track:
            X, attn = self.aggregator(X, src_lengths, track=True)
            return X, attn
        X = self.aggregator(X, src_lengths, track=track)
        return X

    def update(self, X, src_lengths=None, track=False):
        X = X.permute(0, 3, 1, 2)
        X = self.net.update(X)
        attn = None
        if track:
            X, attn = self.aggregator(X, src_lengths, track=track)
        else:
            X = self.aggregator(X, src_lengths, track=track)
        return X, attn

    def track_update(self, data_src, kwargs={}):
        u"""
        Sample and return tracked activations
        Using update where past activations are discarded
        """
        batch_size = data_src[u'labels'].size(0)
        src_emb = self.src_embedding(data_src)
        Ts = src_emb.size(1)  # source sequence length
        max_length = int(
            kwargs.get(u'max_length_a', 0) * Ts +
            kwargs.get(u'max_length_b', 50)
            )

        trg_labels = torch.LongTensor(
            [[self.bos_token] for i in xrange(batch_size)]
            ).cuda()
        trg_emb = self.trg_embedding.single_token(trg_labels, 0)
        # 2d grid:
        src_emb = src_emb.unsqueeze(1)  # Tt=1
        src_emb_ = src_emb
        seq = []
        alphas = []
        aligns = []
        activ_aligns = []
        activs = []
        trg_emb = _expand(trg_emb.unsqueeze(2), 2, Ts)
        for t in xrange(max_length):
            X = self.merge(src_emb, trg_emb)
            Y, attn = self.update(X, data_src[u"lengths"], track=True)
            # align, activ_distrib, activ = attn
            if attn[0] is not None:
                alphas.append(attn[0])
            aligns.append(attn[1])
            activ_aligns.append(attn[2])
            activs.append(attn[3][0])
            proj = self.prediction_dropout(Y[:, -1, :])
            logits = F.log_softmax(self.prediction(proj), dim=1)
            if self.padding_idx:
                logits[:, self.padding_idx] = -math.inf
                npargmax = logits.data.cpu().numpy().argmax(axis=-1)
            else:
                logits = logits[:, 1:]  # remove pad
                npargmax = 1 + logits.data.cpu().numpy().argmax(axis=-1)
            next_preds = torch.from_numpy(npargmax).view(-1, 1).cuda()
            seq.append(next_preds)
            trg_emb_t = self.trg_embedding.single_token(next_preds,
                                                        t).unsqueeze(2)
            trg_emb_t = _expand(trg_emb_t, 2, Ts)
            max_h = self.kernel_size // 2 + 1  
            # keep only what's needed
            if trg_emb.size(1) > max_h:
                trg_emb = trg_emb[:, -max_h:, :, :]
            trg_emb = torch.cat((trg_emb, trg_emb_t), dim=1)
            src_emb = _expand(src_emb_, 1, trg_emb.size(1))
            if t >= 1:
                # stop when all finished
                unfinished = torch.add(
                    torch.mul((next_preds == self.eos_token).type_as(logits),
                              -1), 1)
                if unfinished.sum().data.item() == 0:
                    break
        seq = torch.cat(seq, 1)
        self.net.reset_buffers()
        self.trg_embedding.reset_buffers()
        return seq, alphas, aligns, activ_aligns, activs

    def _track(self, fc_feats, att_feats, att_masks = None, opt = {}):
        u"""
        Sample and return tracked activations
        """
        if self.use_obj_att and self.use_seg_feat:
            obj_att_feats = att_feats[1]
            obj_att_masks = att_masks[1]
            seg_feat_feats = att_feats[2]
            seg_feat_masks = att_masks[2]
            att_feats = att_feats[0]
            att_masks = att_masks[0]
            att_feats, seq, att_masks, seq_mask = self._prepare_feature(att_feats, att_masks)
            obj_att_feats, seq, obj_att_masks, seq_mask = self._prepare_feature_obj(obj_att_feats, obj_att_masks, seq)
        elif not self.use_obj_att and self.use_seg_feat:
            seg_feat_feats = att_feats[1]
            seg_feat_masks = att_masks[1]
            att_feats = att_feats[0]
            att_masks = att_masks[0]
            att_feats, seq, att_masks, seq_mask = self._prepare_feature(att_feats, att_masks)
        elif self.use_obj_att and self.use_seg_feat:
            obj_att_feats = att_feats[1]
            obj_att_masks = att_masks[1]
            att_feats = att_feats[0]
            att_masks = att_masks[0]
            att_feats, seq, att_masks, seq_mask = self._prepare_feature(att_feats, att_masks)
            obj_att_feats, seq, obj_att_masks, seq_mask = self._prepare_feature_obj(obj_att_feats, obj_att_masks, seq)
        elif not self.use_obj_att and not self.use_seg_feat:
            att_feats, seq, att_masks, seq_mask = self._prepare_feature(att_feats, att_masks)

        X = torch.reshape(att_feats, [att_feats.size(0), att_res, att_res, att_feats.size(2)])

        if self.use_seg_feat:
            X_seg = torch.reshape(seg_feat_feats, [seg_feat_feats.size(0), att_res, att_res, seg_feat_feats.size(2)])
            if self.use_img_feat:
                X = torch.cat((X, X_seg), 3)
            else:
                X = X_seg

        src_masks = att_feats.new_ones(X.shape[:2], dtype=torch.long)
        src_masks = src_masks.unsqueeze(-2)

        src_lengths = torch.squeeze(torch.sum(src_masks, dim=2))

        if (self.use_obj_mcl_loss == 0):
            X, init_attn = self._forward_initial(X, src_lengths, track=True)
            src_emb = X
        elif (self.use_obj_mcl_loss == 1):
            set_1, set_2 = self._forward_initial_multi(X, src_lengths, track=True)
            X, init_attn = set_1
            X_obj, init_attn2 = set_2
            #out = F.adaptive_avg_pool2d(X_obj, (1, 128)).view(X_obj.size(0), -1)
            #out = self.multi_label_classifier(out)
            X = torch.cat((X, X_obj), dim=2)
            src_emb = X

            if self.use_obj_att:
                out = F.adaptive_avg_pool2d(src_emb, (1, self.input_channels_obj_att)).view(X_obj.size(0), -1)
            else:
                out = F.adaptive_avg_pool2d(src_emb, (1, 128)).view(X_obj.size(0), -1)
            out = self.multi_label_classifier(out)

            dim = src_emb.size()
            src_emb = torch.reshape(src_emb, (dim[0], dim[1] * 2, dim[2] / 2))

        if (self.use_obj_att):
            trg_emb = src_emb
            src_emb = obj_att_feats
            src_masks = obj_att_masks
            Ts = src_emb.size(1)  # source sequence length
            Tt = trg_emb.size(1)  # target sequence length

            # 2d grid:
            src_emb = _expand(src_emb.unsqueeze(1), 1, Tt)
            trg_emb = _expand(trg_emb.unsqueeze(2), 2, Ts)

            X = self.merge(src_emb, trg_emb)
            src_lengths = torch.squeeze(torch.sum(src_masks, dim=2))

            X = self._forward_obj_attn(X, src_lengths)
            src_emb = X

        batch_size = X.size(0)

        Ts = src_emb.size(1)  # source sequence length
        max_length = int(
            opt[u'max_length_a'] * Ts +
            opt[u'max_length_b']
        )
        trg_labels = torch.LongTensor(
            [[self.bos_token] for i in xrange(batch_size)]
        ).cuda()
        trg_emb = self.trg_embedding.single_token(trg_labels, 0)
        # 2d grid:
        src_emb = src_emb.unsqueeze(1)  # Tt=1
        src_emb_ = src_emb
        seq = []

        alphas = []
        aligns = []
        activ_aligns = []
        activs = []
        trg_emb = _expand(trg_emb.unsqueeze(2), 2, Ts)
        for t in xrange(max_length):
            X = self.merge(src_emb, trg_emb)
            src_lengths = torch.squeeze(torch.sum(att_masks, dim=2))
            Y, attn = self._forward_(X, src_lengths, track=True)
            if attn[0] is not None:
                alphas.append(attn[0])
            aligns.append(attn[1])
            activ_aligns.append(attn[2])
            activs.append(attn[3][0])
            proj = self.prediction_dropout(Y[:, -1, :])
            logits = F.log_softmax(self.prediction(proj), dim=1)
            if self.padding_idx:
                logits[:, self.padding_idx] = -math.inf
                npargmax = logits.data.cpu().numpy().argmax(axis=-1)
            else:
                logits = logits[:, 1:]  # remove pad
                npargmax = 1 + logits.data.cpu().numpy().argmax(axis=-1)
            next_preds = torch.from_numpy(npargmax).view(-1, 1).cuda()
            seq.append(next_preds)
            trg_emb_t = self.trg_embedding.single_token(next_preds,
                                                        t).unsqueeze(2)
            trg_emb_t = _expand(trg_emb_t, 2, Ts)
            trg_emb = torch.cat((trg_emb, trg_emb_t), dim=1)
            src_emb = _expand(src_emb_, 1, trg_emb.size(1))
            if t >= 1:
                # stop when all finished
                unfinished = torch.add(
                    torch.mul((next_preds == self.eos_token).type_as(logits),
                              -1), 1)
                if unfinished.sum().data.item() == 0:
                    break
        seq = torch.cat(seq, 1)
        self.trg_embedding.reset_buffers()
        final_attn = alphas, aligns, activ_aligns, activs

        if (self.use_obj_mcl_loss == 0):
            return seq, init_attn, final_attn
        if (self.use_obj_mcl_loss == 1):
            return seq, init_attn, init_attn2, final_attn

    def sample_update(self, data_src, scorer, kwargs={}):
        u"""
        Sample in evaluation mode
        Using update where past activations are discarded
        """
        beam_size = kwargs.get(u'beam_size', 1)
        if beam_size > 1:
            # Without update
            return self.sample_beam(data_src, kwargs)
        batch_size = data_src[u'labels'].size(0)
        src_emb = self.src_embedding(data_src)
        Ts = src_emb.size(1)  # source sequence length
        max_length = int(
            kwargs.get(u'max_length_a', 0) * Ts +
            kwargs.get(u'max_length_b', 50)
            )
        trg_labels = torch.LongTensor(
            [[self.bos_token] for i in xrange(batch_size)]
            ).cuda()
        trg_emb = self.trg_embedding.single_token(trg_labels, 0)
        # 2d grid:
        src_emb = src_emb.unsqueeze(1)  # Tt=1
        src_emb_ = src_emb
        seq = []
        trg_emb = _expand(trg_emb.unsqueeze(2), 2, Ts)
        for t in xrange(max_length):
            X = self.merge(src_emb, trg_emb)
            Y, _ = self.update(X, data_src[u"lengths"])
            proj = self.prediction_dropout(Y[:, -1, :])
            logits = F.log_softmax(self.prediction(proj), dim=1)
            if self.padding_idx:
                logits[:, self.padding_idx] = -math.inf
                npargmax = logits.data.cpu().numpy().argmax(axis=-1)
            else:
                logits = logits[:, 1:]  # remove pad
                npargmax = 1 + logits.data.cpu().numpy().argmax(axis=-1)
            next_preds = torch.from_numpy(npargmax).view(-1, 1).cuda()
            seq.append(next_preds)
            trg_emb_t = self.trg_embedding.single_token(next_preds,
                                                        t).unsqueeze(2)
            trg_emb_t = _expand(trg_emb_t, 2, Ts)
            max_h = self.kernel_size // 2 + 1 
            # keep only what's needed
            if trg_emb.size(1) > max_h:
                trg_emb = trg_emb[:, -max_h:, :, :]
            trg_emb = torch.cat((trg_emb, trg_emb_t), dim=1)
            src_emb = _expand(src_emb_, 1, trg_emb.size(1))
            if t >= 1:
                # stop when all finished
                unfinished = torch.add(
                    torch.mul((next_preds == self.eos_token).type_as(logits),
                              -1), 1)
                if unfinished.sum().data.item() == 0:
                    break
        seq = torch.cat(seq, 1)
        self.net.reset_buffers()
        self.trg_embedding.reset_buffers()
        return seq, None

    def _sample(self, fc_feats, att_feats, att_masks = None, opt = {}):
        u"""
        Sample in evaluation mode
        """
        beam_size = opt['beam_size']

        # Need to be cleaned
        # if beam_size > 1:
        #    return self.sample_beam(fc_feats, att_feats, att_masks, opt)

        if self.use_obj_att and self.use_seg_feat:
            obj_att_feats = att_feats[1]
            obj_att_masks = att_masks[1]
            seg_feat_feats = att_feats[2]
            seg_feat_masks = att_masks[2]
            att_feats = att_feats[0]
            att_masks = att_masks[0]
            if self.use_img_feat:
                att_feats, seq, att_masks, seq_mask = self._prepare_feature(att_feats, att_masks)
            seg_feat_feats, seq, seg_feat_masks, seq_mask = self._prepare_feature_seg(seg_feat_feats, seg_feat_masks)
            obj_att_feats, seq, obj_att_masks, seq_mask = self._prepare_feature_obj(obj_att_feats, obj_att_masks)
            att_masks = seg_feat_masks
        elif not self.use_obj_att and self.use_seg_feat:
            seg_feat_feats = att_feats[1]
            seg_feat_masks = att_masks[1]
            att_feats = att_feats[0]
            att_masks = att_masks[0]
            if self.use_img_feat:
                att_feats, seq, att_masks, seq_mask = self._prepare_feature(att_feats, att_masks)
            seg_feat_feats, seq, seg_feat_masks, seq_mask = self._prepare_feature_seg(seg_feat_feats, seg_feat_masks)
            att_masks = seg_feat_masks
        elif self.use_obj_att and self.use_seg_feat:
            obj_att_feats = att_feats[1]
            obj_att_masks = att_masks[1]
            att_feats = att_feats[0]
            att_masks = att_masks[0]
            att_feats, seq, att_masks, seq_mask = self._prepare_feature(att_feats, att_masks)
            obj_att_feats, seq, obj_att_masks, seq_mask = self._prepare_feature_obj(obj_att_feats, obj_att_masks)
        elif not self.use_obj_att and not self.use_seg_feat:
            att_feats, seq, att_masks, seq_mask = self._prepare_feature(att_feats, att_masks)

        X = torch.reshape(att_feats, [att_feats.size(0), att_res, att_res, att_feats.size(2)])

        if (att_res == 14):
            X = X.permute(0, 3, 1, 2)
            X = F.adaptive_avg_pool2d(X, [att_res, att_res])
            X = X.permute(0, 2, 3, 1)

        if self.use_seg_feat:
            X_seg = torch.reshape(seg_feat_feats, [seg_feat_feats.size(0), att_res, att_res, seg_feat_feats.size(2)])
            if self.use_img_feat:
                X = torch.cat((X, X_seg), 3)
            else:
                X = X_seg

        src_masks = att_feats.new_ones(X.shape[:2], dtype=torch.long)
        src_masks = src_masks.unsqueeze(-2)

        src_lengths = torch.squeeze(torch.sum(src_masks, dim=2))

        if (self.use_obj_mcl_loss == 0):
            X = self._forward_initial(X, src_lengths)
            src_emb = X
        elif (self.use_obj_mcl_loss == 1):
            X, X_obj = self._forward_initial_multi(X, src_lengths)
            X = torch.cat((X, X_obj), dim=2)
            src_emb = X

            if self.use_obj_att:
                out = F.adaptive_avg_pool2d(src_emb, (1, self.input_channels_obj_att)).view(X_obj.size(0), -1)
            else:
                out = F.adaptive_avg_pool2d(src_emb, (1, 128)).view(X_obj.size(0), -1)
            out = self.multi_label_classifier(out)
            dim = src_emb.size()
            src_emb = torch.reshape(src_emb, (dim[0], dim[1] * 2, dim[2] / 2))

        if (self.use_obj_att):
            trg_emb = src_emb
            src_emb = obj_att_feats
            src_masks = obj_att_masks
            Ts = src_emb.size(1)  # source sequence length
            Tt = trg_emb.size(1)  # target sequence length

            # 2d grid:
            src_emb = _expand(src_emb.unsqueeze(1), 1, Tt)
            trg_emb = _expand(trg_emb.unsqueeze(2), 2, Ts)

            X = self.merge(src_emb, trg_emb)
            src_lengths = torch.squeeze(torch.sum(src_masks, dim=2))

            X = self._forward_obj_attn(X, src_lengths)
            src_emb = X

        batch_size = X.size(0)

        Ts = src_emb.size(1)  # source sequence length
        max_length = int(
            opt[u'max_length_a'] * Ts +
            opt[u'max_length_b']
            )
        trg_labels = torch.LongTensor(
            [[self.bos_token] for i in xrange(batch_size)]
            ).cuda()
        trg_emb = self.trg_embedding.single_token(trg_labels, 0)
        # 2d grid:
        src_emb = src_emb.unsqueeze(1)  # Tt=1
        src_emb_ = src_emb
        seq = []
        trg_emb = _expand(trg_emb.unsqueeze(2), 2, Ts)
        for t in xrange(max_length):
            X = self.merge(src_emb, trg_emb)
            src_lengths = torch.squeeze(torch.sum(att_masks, dim=2))
            Y = self._forward_(X, src_lengths)
            proj = self.prediction_dropout(Y[:, -1, :])
            logits = F.log_softmax(self.prediction(proj), dim=1)
            if self.padding_idx:
                logits[:, self.padding_idx] = -math.inf
                npargmax = logits.data.cpu().numpy().argmax(axis=-1)
            else:
                logits = logits[:, 1:]  # remove pad
                npargmax = 1 + logits.data.cpu().numpy().argmax(axis=-1)
            next_preds = torch.from_numpy(npargmax).view(-1, 1).cuda()
            seq.append(next_preds)
            trg_emb_t = self.trg_embedding.single_token(next_preds,
                                                        t).unsqueeze(2)
            trg_emb_t = _expand(trg_emb_t, 2, Ts)
            trg_emb = torch.cat((trg_emb, trg_emb_t), dim=1)
            src_emb = _expand(src_emb_, 1, trg_emb.size(1))
            if t >= 1:
                # stop when all finished
                unfinished = torch.add(
                    torch.mul((next_preds == self.eos_token).type_as(logits),
                              -1), 1)
                if unfinished.sum().data.item() == 0:
                    break
        seq = torch.cat(seq, 1)
        return seq, None

    def sample_beam(self, fc_feats, att_feats, att_masks = None, kwargs={}):
        beam_size = kwargs[u'beam_size']
        if self.use_obj_att and self.use_seg_feat:
            obj_att_feats = att_feats[1]
            obj_att_masks = att_masks[1]
            seg_feat_feats = att_feats[2]
            seg_feat_masks = att_masks[2]
            att_feats = att_feats[0]
            att_masks = att_masks[0]
            att_feats, seq, att_masks, seq_mask = self._prepare_feature(att_feats, att_masks)
            obj_att_feats, seq, obj_att_masks, seq_mask = self._prepare_feature_obj(obj_att_feats, obj_att_masks, seq)
        elif not self.use_obj_att and self.use_seg_feat:
            seg_feat_feats = att_feats[1]
            seg_feat_masks = att_masks[1]
            att_feats = att_feats[0]
            att_masks = att_masks[0]
            att_feats, seq, att_masks, seq_mask = self._prepare_feature(att_feats, att_masks)
        elif self.use_obj_att and self.use_seg_feat:
            obj_att_feats = att_feats[1]
            obj_att_masks = att_masks[1]
            att_feats = att_feats[0]
            att_masks = att_masks[0]
            att_feats, seq, att_masks, seq_mask = self._prepare_feature(att_feats, att_masks)
            obj_att_feats, seq, obj_att_masks, seq_mask = self._prepare_feature_obj(obj_att_feats, obj_att_masks, seq)
        elif not self.use_obj_att and not self.use_seg_feat:
            att_feats, seq, att_masks, seq_mask = self._prepare_feature(att_feats, att_masks)

        X = torch.reshape(att_feats, [att_feats.size(0), att_res, att_res, att_feats.size(2)])

        if (att_res == 14):
            X = X.permute(0, 3, 1, 2)
            X = F.adaptive_avg_pool2d(X, [att_res, att_res])
            X = X.permute(0, 2, 3, 1)

        if self.use_seg_feat:
            X_seg = torch.reshape(seg_feat_feats, [seg_feat_feats.size(0), att_res, att_res, seg_feat_feats.size(2)])
            X = torch.cat((X, X_seg), 3)

        src_masks = att_feats.new_ones(X.shape[:2], dtype=torch.long)
        src_masks = src_masks.unsqueeze(-2)

        src_lengths = torch.squeeze(torch.sum(src_masks, dim=2))


        if (self.use_obj_mcl_loss == 0):
            X = self._forward_initial(X, src_lengths)
            src_emb_initial = X
        elif (self.use_obj_mcl_loss == 1):
            X, X_obj = self._forward_initial_multi(X, src_lengths)
            X = torch.cat((X, X_obj), dim=2)
            src_emb_initial = X

            if self.use_obj_att:
                out = F.adaptive_avg_pool2d(src_emb_initial, (1, self.input_channels_obj_att)).view(X_obj.size(0), -1)
            else:
                out = F.adaptive_avg_pool2d(src_emb_initial, (1, 128)).view(X_obj.size(0), -1)
            out = self.multi_label_classifier(out)
            dim = src_emb_initial.size()
            src_emb_initial = torch.reshape(src_emb_initial, (dim[0], dim[1] * 2, dim[2] / 2))

        if (self.use_obj_att):
            trg_emb = src_emb_initial
            src_emb_initial = obj_att_feats
            src_masks = obj_att_masks
            Ts = src_emb_initial.size(1)  # source sequence length
            Tt = trg_emb.size(1)  # target sequence length

            # 2d grid:
            src_emb_initial = _expand(src_emb_initial.unsqueeze(1), 1, Tt)
            trg_emb = _expand(trg_emb.unsqueeze(2), 2, Ts)

            X = self.merge(src_emb_initial, trg_emb)
            src_lengths = torch.squeeze(torch.sum(src_masks, dim=2))

            X = self._forward_obj_attn(X, src_lengths)
            src_emb_initial = X


        batch_size = src_emb_initial.size(0)
        beam = [Beam(beam_size, kwargs) for k in xrange(batch_size)]
        batch_idx = range(batch_size)
        remaining_sents = batch_size
        Ts = src_emb_initial.size(1)  # source sequence length
        max_length = int(
            kwargs.get(u'max_length_a', 0) * Ts +
            kwargs.get(u'max_length_b', 50)
            )
        src_emb_initial = src_emb_initial.repeat(beam_size, 1, 1)
        src_lengths = src_lengths.repeat(beam_size, 1)

        for t in xrange(max_length):
            # Source:
            src_emb = src_emb_initial.unsqueeze(1).repeat(1, t + 1, 1, 1)
            trg_labels_t = torch.stack([
                b.get_current_state() for b in beam if not b.done
            ]).t().contiguous().view(-1, 1)
            if t:
                # append to the previous tokens
                trg_labels = torch.cat((trg_labels, trg_labels_t), dim=1)
            else:
                trg_labels = trg_labels_t

            trg_emb = self.trg_embedding(trg_labels).unsqueeze(2).repeat(1, 1, Ts, 1)
            # X: N, Tt, Ts, Ds+Dt
            X = self.merge(src_emb, trg_emb)
            Y = self._forward_(X, src_lengths)
            proj = self.prediction_dropout(Y[:, -1, :])
            logits = F.log_softmax(self.prediction(proj), dim=1)
            word_lk = logits.view(beam_size,
                                  remaining_sents,
                                  -1).transpose(0, 1).contiguous()
            active = []
            for b in xrange(batch_size):
                if beam[b].done:
                    continue
                idx = batch_idx[b]
                if not beam[b].advance(word_lk.data[idx], t):
                    active += [b]
                trg_labels_prev = trg_labels.view(beam_size,
                                                  remaining_sents,
                                                  t + 1)
                trg_labels = trg_labels_prev[
                    beam[b].get_current_origin()].view(-1, t + 1)
            if not active:
                break
            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            active_idx = torch.cuda.LongTensor([batch_idx[k] for k in active])
            batch_idx = dict((beam, idx) for idx, beam in enumerate(active))

            def update_active(t):
                # select only the remaining active sentences
                view = t.data.contiguous().view(beam_size,
                                                remaining_sents,
                                                *t.size()[1:])
                new_size = list(view.size())
                new_size[1] = new_size[1] * len(active_idx) \
                    // remaining_sents
                result = view.index_select(1, active_idx).view(*new_size)
                return result.view(-1, result.size(-1))

            def update_active_emb(t):
                # select only the remaining active sentences
                view = t.data.contiguous().view(beam_size,
                                                remaining_sents,
                                                *t.size()[1:])
                new_size = list(view.size())
                new_size[1] = new_size[1] * len(active_idx) \
                    // remaining_sents
                result = view.index_select(1, active_idx).view(*new_size)
                return result.view(-1, result.size(2),result.size(-1))

            src_emb_initial = update_active_emb(src_emb_initial)
            src_lengths = update_active(src_lengths)
            trg_labels = update_active(trg_labels)
            remaining_sents = len(active)

        # Wrap up
        allHyp, allScores = [], []
        n_best = 1
        for b in xrange(batch_size):
            scores, ks = beam[b].sort_best()
            allScores += [scores[:n_best]]
            hyps = beam[b].get_hyp(ks[0])
            allHyp += [hyps]
        return allHyp, allScores