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
  annotations = panoptic_val[u'annotations']
  N = len(annotations)

  seg_dir = params['output_seg_dir']

  if not os.path.isdir(seg_dir):
    os.mkdir(seg_dir)

  cuda = torch.device('cuda')
  #with torch.cuda.device(0) and torch.no_grad():
  for i, ann in enumerate(annotations):
    if (os.path.isfile(os.path.join(seg_dir, str(ann['image_id']))+'.npz')):
      print('Skiping ' + str(ann['image_id']))
      continue
    # load the image
    glove_feat = np.load(os.path.join(params['glove_dir'], str(ann['image_id'])) + '.npz')['arr_0']
    ie_feat = np.load(os.path.join(params['ie_dir'], str(ann['image_id'])) + '.npz')['arr_0']

    seg_feat = np.concatenate([glove_feat,ie_feat], axis=2)
    np.savez_compressed(os.path.join(seg_dir, str(ann['image_id'])), seg_feat)
    #torch.cuda.empty_cache()

    if i % 1000 == 0:
      print(datetime.datetime.now())
      print('processing %d/%d (%.2f%% done)' % (i, N, i*100.0/N))
  print('wrote ', seg_dir)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # input dirs
  parser.add_argument('--glove_dir', default='data', help='input glove dir')
  parser.add_argument('--ie_dir', default='data', help='input ie dir')

  # options
  parser.add_argument('--seg_root', default='', help='root location in which images are stored, to be prepended to file_path in input json')
  parser.add_argument('--att_size', default=7, type=int, help='14x14 or 7x7')

  parser.add_argument('--input_json_panoptic_train', required=True, help='input json file to process into hdf5')
  parser.add_argument('--input_json_panoptic_val', required=True, help='input json file to process into hdf5')

  parser.add_argument('--output_seg_dir', default='data', help='output h5 file')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print('parsed input parameters:')
  print(json.dumps(params, indent=2))
  main(params)
