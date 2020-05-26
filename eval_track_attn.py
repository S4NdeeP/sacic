from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
from dataloaderraw import *
import eval_utils
import argparse
import misc.utils as utils
import torch

# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--model_path', type=str, default='',
                help='path to model to evaluate')
parser.add_argument('--cnn_model', type=str,  default='resnet101',
                help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str, default='',
                help='path to infos to evaluate')
# Basic options
parser.add_argument('--batch_size', type=int, default=0,
                help='if > 0 then overrule, otherwise load from checkpoint.')
parser.add_argument('--num_images', type=int, default=-1,
                help='how many images to use when periodically evaluating the loss? (-1 = all)')
parser.add_argument('--language_eval', type=int, default=0,
                help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
parser.add_argument('--dump_images', type=int, default=1,
                help='Dump images into vis/imgs folder for vis? (1=yes,0=no)')
parser.add_argument('--dump_json', type=int, default=1,
                help='Dump json with predictions into vis folder? (1=yes,0=no)')
parser.add_argument('--dump_path', type=int, default=0,
                help='Write image paths along with predictions into vis json? (1=yes,0=no)')

# Sampling options
parser.add_argument('--sample_max', type=int, default=1,
                help='1 = sample argmax words. 0 = sample from distributions.')
parser.add_argument('--max_ppl', type=int, default=0,
                help='beam search by max perplexity or max probability.')
parser.add_argument('--beam_size', type=int, default=2,
                help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
parser.add_argument('--group_size', type=int, default=1,
                help='used for diverse beam search. if group_size is 1, then it\'s normal beam search')
parser.add_argument('--diversity_lambda', type=float, default=0.5,
                help='used for diverse beam search. Usually from 0.2 to 0.8. Higher value of lambda produces a more diverse list')
parser.add_argument('--temperature', type=float, default=1.0,
                help='temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
parser.add_argument('--decoding_constraint', type=int, default=0,
                help='If 1, not allowing same word in a row')
# For evaluation on a folder of images:
parser.add_argument('--image_folder', type=str, default='', 
                help='If this is nonempty then will predict on the images in this folder path')
parser.add_argument('--image_root', type=str, default='', 
                help='In case the image paths have to be preprended with a root path to an image folder')
# For evaluation on MSCOCO images from some split:
parser.add_argument('--input_fc_dir', type=str, default='',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_att_dir', type=str, default='',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_obj_att_dir', type=str, default='data/cocotalk_att',
                        help='path to the directory containing the preprocessed object att feats')
parser.add_argument('--input_seg_feat_dir', type=str, default='data/seg_feat',
                        help='path to the directory containing the preprocessed seg feats')
parser.add_argument('--input_box_dir', type=str, default='',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_label_h5', type=str, default='',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_json', type=str, default='', 
                help='path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
parser.add_argument('--split', type=str, default='test', 
                help='if running on MSCOCO images, which split to use: val|test|train')
parser.add_argument('--coco_json', type=str, default='', 
                help='if nonempty then use this file in DataLoaderRaw (see docs there). Used only in MSCOCO test evaluation, where we have a specific json file of only test set images.')
# misc
parser.add_argument('--id', type=str, default='', 
                help='an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')
parser.add_argument('--verbose_beam', type=int, default=1, 
                help='if we need to print out all beam search beams.')
parser.add_argument('--verbose_loss', type=int, default=0, 
                help='if we need to calculate loss.')

# Pervasive
"""
Parse parametres from config file for training
"""
parser.add_argument("-c", "--config",
                    help="Specify config file", metavar="FILE")
parser.add_argument("-v", "--verbose", type=int,
                    default=1, help="code verbosity")
parser.add_argument("-g", "--gpu_id", type=str,
                    default='0', help="gpu id")

# Pervasive Eval Parameters
parser.add_argument("-o", "--offset", type=int,
                    default=0, help="starting index used to visualize a specific batch")
parser.add_argument("--read_length", type=int,
                    default=0, help="max length for loading")
parser.add_argument("--max_length_a", type=float,
                    default=0, help="Decode up to a*source_lenght + b : (a)")
parser.add_argument("--max_length_b", type=int,
                    default=25, help="Decode up to a*source_lenght + b : (b)")
parser.add_argument("-l", "--last", action="store_true",
                    help="evaluate with the last checkpoint instead of the best one")
parser.add_argument("--norm", action="store_true", help="Normalize scores by length")
parser.add_argument("-m", "--max_samples", type=int,
                    default=100, help="Decode up to max_samples sequences")
parser.add_argument("--block_ngram_repeat", type=int,
                    default=0, help="GNMT parameter")
parser.add_argument("--length_penalty", "-p", type=float,
                    default=0, help="length penalty for GNMT")
parser.add_argument("--length_penalty_mode", type=str,
                    default="wu", help="length penalty mode, either wu or avg for GNMTscorer")

parser.add_argument("--stepwise_penalty", action="store_true")

parser.add_argument('--model', type=str, default="show_tell",
                    help='show_tell, show_attend_tell, all_img, fc, att2in, att2in2, att2all2, adaatt, adaattmo, topdown, stackatt, denseatt, transformer, pervasive')
parser.add_argument('--num_heads', type=int, default=1,
                        help='number of heads')

# Parameters for attribute, stem decomposition
parser.add_argument('--use_obj_mcl_loss', type=int, default=0,
                    help='If 1, then use obj multi-class classification loss, if 0 dont use the loss for training')
parser.add_argument('--use_obj_att', type=int, default=0,
                        help='If 1, then do use obj attn also')
parser.add_argument('--use_seg_feat', type=int, default=0,
                        help='If 1, then do use seg feat also')
parser.add_argument('--use_img_feat', type=int, default=1,
                        help='If 1, then do use image feat also')
opt = parser.parse_args()

# Load infos
with open(opt.infos_path) as f:
    infos = cPickle.load(f)

# override and collect parameters
if len(opt.input_fc_dir) == 0:
    opt.input_fc_dir = infos['opt'].input_fc_dir
    opt.input_att_dir = infos['opt'].input_att_dir
    opt.input_box_dir = infos['opt'].input_box_dir
    opt.input_label_h5 = infos['opt'].input_label_h5
if len(opt.input_json) == 0:
    opt.input_json = infos['opt'].input_json
if opt.batch_size == 0:
    opt.batch_size = infos['opt'].batch_size
if len(opt.id) == 0:
    opt.id = infos['opt'].id
ignore = ["id", "batch_size", "beam_size", "start_from", "language_eval"]
for k in vars(infos['opt']).keys():
    if k not in ignore:
        if k in vars(opt):
            assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
        else:
            vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

vocab = infos['vocab'] # ix -> word mapping

# Setup the model
model = models.setup(opt)
model.load_state_dict(torch.load(opt.model_path))
model.cuda()
model.eval()
crit = utils.LanguageModelCriterion()

# Create the Data Loader instance
if len(opt.image_folder) == 0:
  loader = DataLoader(opt)
else:
  loader = DataLoaderRaw({'folder_path': opt.image_folder, 
                            'coco_json': opt.coco_json,
                            'batch_size': opt.batch_size,
                            'cnn_model': opt.cnn_model})
# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
loader.ix_to_word = infos['vocab']


# Set sample options
loss, split_predictions, lang_stats = eval_utils.eval_split_track_attn(model, crit, loader,
    vars(opt))

print('loss: ', loss)
if lang_stats:
  print(lang_stats)

if opt.dump_json == 1:
    # dump the json
    json.dump(split_predictions, open('vis/vis.json', 'w'))
