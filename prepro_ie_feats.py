from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
from six.moves import cPickle
import numpy as np
import torch
import torchvision.models as models
import skimage.io
import torchtext.vocab as vocab  # use this to load glove vector
import math
import torch.nn.functional as F
from matplotlib import pyplot as plt

from torchvision import transforms as trn

def main(params):
  panoptic_train = json.load(open(params['input_json_panoptic_train'], 'r'))
  panoptic_val = json.load(open(params['input_json_panoptic_val'], 'r'))
  att_size = args.att_size
  annotations = panoptic_val[u'annotations']
  N = len(annotations)
  cname2cid = {}
  cid2cname = {}
  categories = panoptic_train[u'categories']

  overall_max_inst = 0
  for i, category in enumerate(categories):
    cname2cid[category[u'name']] = category[u'id']
    cid2cname[category[u'id']] = category[u'name']

  seed(123) # make reproducible
  np.random.seed(123)
  dir_ie = params['output_dir']+'_ie_feat'

  if not os.path.isdir(dir_ie):
    os.mkdir(dir_ie)

  cuda = torch.device('cuda')
  with torch.cuda.device(0) and torch.no_grad():
    max_len = 100
    d_model = 10

    ie = torch.eye(d_model, device=cuda)
    ie_rest = torch.zeros(d_model, device=cuda)
    ie_rest[d_model-1] = 1
    ie_rest = ie_rest.repeat(max_len-d_model,1)
    ie = torch.cat([ie,ie_rest], dim=0)

    for i, ann in enumerate(annotations):
      if (os.path.isfile(os.path.join(dir_ie, str(ann['image_id']))+'.npz')):
        print('Skiping ' + str(ann['image_id']))
        continue
      # load the image
      filename = ann[u'file_name']
      I = skimage.io.imread(os.path.join(params['seg_root'], filename))

      seg_info = ann[u'segments_info']


      sid2cid = {}
      sid2cid[0] = 0
      countc = {}
      countc[0] = 0

      for seg in seg_info:
        sid2cid[seg[u'id']] = seg[u'category_id']
        countc[seg[u'category_id']] = 0

      sid2instnum = {}
      for s in sid2cid.keys():
        countc[sid2cid[s]] += 1
        instnum = countc[sid2cid[s]]
        sid2instnum[s] = instnum

      I = torch.from_numpy(I).cuda()
      colour_to_id_transform = torch.tensor([[1], [256], [256 * 256]], device=cuda)
      I = torch.matmul(I.float(), colour_to_id_transform.float())

      I = I.squeeze().long()


      # 300 for cid and sid conflic avoid
      I_inst = I + 300

      for s in sid2cid.keys():
        I_inst = torch.where(I_inst - 300 == s, torch.tensor(sid2instnum[s]).cuda(), I_inst)

      dim_x, dim_y = I.size()

      I_pe = torch.zeros(dim_x, dim_y, d_model, device=cuda).double()
      I_inst = I_inst.cuda()
      ie = ie.cuda()
      max_inst = max(sid2instnum.values())
      if max_inst > overall_max_inst:
        overall_max_inst = max_inst
      I_pe_zeros = torch.zeros(dim_x, dim_y, d_model, device=cuda).double()
      I_inst_extended = I_inst.unsqueeze(-1).expand(dim_x, dim_y, d_model)
      for inst in range(max_inst):
        pe_i = inst
        pe_inst = ie[pe_i].double()
        I_pe_inst = pe_inst.repeat(dim_x, dim_y, 1)
        I_pe_inst = torch.where(I_inst_extended == inst, I_pe_inst, I_pe_zeros)
        I_pe = I_pe + I_pe_inst
      del I_pe_zeros
      del I_inst_extended

      I_seg = I_pe

      I_seg = I_seg.permute(2, 0, 1)
      I_seg = I_seg.unsqueeze(0)
      seg_feat = F.adaptive_avg_pool2d(I_seg, [att_size, att_size]).squeeze().permute(1, 2, 0)

      np.savez_compressed(os.path.join(dir_ie, str(ann['image_id'])), seg_feat.data.cpu().float().numpy())
      torch.cuda.empty_cache()

      del I_pe
      del I_seg
      del seg_feat
      if i % 1000 == 0:
        print(datetime.datetime.now())
        print('processing %d/%d (%.2f%% done)' % (i, N, i*100.0/N))
        print('Current max inst range:' + str(overall_max_inst))
    print('wrote ', params['output_dir'])

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # input json
  parser.add_argument('--input_json', required=True, help='input json file to process into hdf5')
  parser.add_argument('--input_json_panoptic_train', required=True, help='input json file to process into hdf5')
  parser.add_argument('--input_json_panoptic_val', required=True, help='input json file to process into hdf5')
  parser.add_argument('--output_dir', default='data', help='output h5 file')

  # options
  parser.add_argument('--seg_root', default='', help='root location in which images are stored, to be prepended to file_path in input json')
  parser.add_argument('--att_size', default=7, type=int, help='14x14 or 7x7')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print('parsed input parameters:')
  print(json.dumps(params, indent=2))
  main(params)
