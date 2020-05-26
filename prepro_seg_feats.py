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
preprocess = trn.Compose([
        #trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def main(params):
  #imgs = json.load(open(params['input_json'], 'r'))
  panoptic_train = json.load(open(params['input_json_panoptic_train'], 'r'))
  panoptic_val = json.load(open(params['input_json_panoptic_val'], 'r'))
  #panoptic = panoptic_train
  #imgs = imgs['images']
  att_size = args.att_size
  annotations = panoptic_train[u'annotations']
  N = len(annotations)
  cname2cid = {}
  cid2cname = {}
  cid2glovei = {}
  categories = panoptic_train[u'categories']

  for i, category in enumerate(categories):
    cname2cid[category[u'name']] = category[u'id']
    cid2cname[category[u'id']] = category[u'name']

  seed(123) # make reproducible
  np.random.seed(123)
  dir_seg = params['output_dir']+'_seg_feat'

  if not os.path.isdir(dir_seg):
    os.mkdir(dir_seg)

  glove = vocab.GloVe(name='6B', dim=300)
  # get the glove vector for the fg detections.

  glove_c_all = np.zeros((len(cname2cid) + 1, 300))

  unknown_vector = 2 * np.random.rand(300) - 1
  glove_c_all[0] = unknown_vector
  cid2glovei[0] = 0

  for i, word in enumerate(cname2cid.keys()):
    vector = np.zeros((300))
    count = 0
    if word == 'playingfield':
      word = 'playing field'
    for w in word.replace(' ', '-').split('-'):
      count += 1
      if w in glove.stoi:
        glove_vector = glove.vectors[glove.stoi[w]]
        vector += glove_vector.numpy()
      else:  # use a random vector instead
        random_vector = 2 * np.random.rand(300) - 1
        vector += random_vector
    glove_c_all[i + 1] = vector / count
    if word == 'playing field':
      word = 'playingfield'
    cid2glovei[cname2cid[word]] = i + 1

  cuda = torch.device('cuda')
  with torch.cuda.device(0) and torch.no_grad():
    glove_c_all_tensor = torch.from_numpy(glove_c_all).cuda()
    max_len = 100
    d_model = 300

    pe = torch.zeros(max_len, d_model, device=cuda)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                         -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    for i, ann in enumerate(annotations):
      if (os.path.isfile(os.path.join(dir_seg, str(ann['image_id']))+'.npz')):
        print('Skiping ' + str(ann['image_id']))
        continue
      # load the image
      filename = ann[u'file_name']
      I = skimage.io.imread(os.path.join(params['seg_root'], filename))
      #skimage.io.imshow(I)
      #plt.show()

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


      I_cat = I + 300 # 300 for cid and sid conflic avoidance
      I_inst = I + 300

      for s in sid2cid.keys():
        I_cat = torch.where(I_cat - 300 == s, torch.tensor(sid2cid[s]).cuda(), I_cat)

      for s in sid2cid.keys():
        I_inst = torch.where(I_inst - 300 == s, torch.tensor(sid2instnum[s]).cuda(), I_inst)

      dim_x, dim_y = I.size()

      I_glove = torch.zeros(dim_x, dim_y, 300).double().cuda()
      I_glove_c_zeros = torch.zeros(dim_x, dim_y, 300, device=cuda).double()
      I_cat_extended = I_cat.unsqueeze(-1).expand(dim_x, dim_y, 300)
      for c in countc.keys():
        glove_i = cid2glovei[c]
        glove_c = glove_c_all_tensor[glove_i]
        I_glove_c = glove_c.repeat(dim_x, dim_y, 1)
        I_glove_c = torch.where(I_cat_extended == c, I_glove_c, I_glove_c_zeros)
        I_glove = I_glove + I_glove_c
      del I_glove_c_zeros
      del I_cat_extended

      if params['use_pe'] == 1:
        I_pe = torch.zeros(dim_x, dim_y, 300, device=cuda).double()
        I_inst = I_inst.cuda()
        pe = pe.cuda()
        max_inst = max(sid2instnum.values())
        I_pe_zeros = torch.zeros(dim_x, dim_y, 300, device=cuda).double()
        I_inst_extended = I_inst.unsqueeze(-1).expand(dim_x, dim_y, 300)
        for inst in range(max_inst):
          pe_i = inst
          pe_inst = pe[pe_i].double()
          I_pe_inst = pe_inst.repeat(dim_x, dim_y, 1)
          I_pe_inst = torch.where(I_inst_extended == inst, I_pe_inst, I_pe_zeros)
          I_pe = I_pe + I_pe_inst
        del I_pe_zeros
        del I_inst_extended

      I_glove = I_glove
      if params['use_pe'] == 1:
        I_pe = I_pe
        I_seg = I_glove + I_pe
      else:
        I_seg = I_glove


      I_seg = I_seg.permute(2, 0, 1)
      I_seg = I_seg.unsqueeze(0)
      seg_feat = F.adaptive_avg_pool2d(I_seg, [att_size, att_size]).squeeze().permute(1, 2, 0)
      np.savez_compressed(os.path.join(dir_seg, str(ann['image_id'])), seg_feat.data.cpu().float().numpy())
      torch.cuda.empty_cache()
      del I_glove
      if params['use_pe'] == 1:
        del I_pe
      del I_seg
      del seg_feat
      if i % 1000 == 0:
        print(datetime.datetime.now())
        print('processing %d/%d (%.2f%% done)' % (i, N, i*100.0/N))
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
  parser.add_argument('--use_pe', default=1, type=int)

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print('parsed input parameters:')
  print(json.dumps(params, indent = 2))
  main(params)
