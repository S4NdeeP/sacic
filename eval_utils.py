from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.misc import imresize
import skimage.transform
from PIL import Image

def language_eval(dataset, preds, model_id, split):
    import sys
    sys.path.append("coco-caption")
    annFile = 'coco-caption/annotations/captions_val2014.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    # encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', model_id + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out

def eval_split(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)

    use_obj_att = eval_kwargs.get('use_obj_att', 0)
    use_seg_feat = eval_kwargs.get('use_seg_feat', 0)
    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        if data.get('labels', None) is not None and verbose_loss:
            # forward the model to get loss
            if use_obj_att and use_seg_feat:
                tmp = [data['fc_feats'], data['att_feats'], data['obj_att_feats'], data['seg_feat_feats'], data['labels'], data['masks'],
                       data['att_masks'], data['obj_att_masks'], data['seg_feat_masks']]
                tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
                fc_feats, att_feats, obj_att_feats, seg_feat_feats, labels, masks, att_masks, obj_att_masks, seg_feat_masks = tmp
            elif not use_obj_att and use_seg_feat:
                tmp = [data['fc_feats'], data['att_feats'], data['seg_feat_feats'], data['labels'], data['masks'],
                       data['att_masks'], data['seg_feat_masks']]
                tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
                fc_feats, att_feats, seg_feat_feats, labels, masks, att_masks, seg_feat_masks = tmp
            elif not use_obj_att and not use_seg_feat:
                tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
                tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
                fc_feats, att_feats, labels, masks, att_masks = tmp
            elif use_obj_att and not use_seg_feat:
                tmp = [data['fc_feats'], data['att_feats'], data['obj_att_feats'], data['labels'], data['masks'],
                       data['att_masks'], data['obj_att_masks']]
                tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
                fc_feats, att_feats, obj_att_feats, labels, masks, att_masks, obj_att_masks = tmp


            with torch.no_grad():
                if use_obj_att and use_seg_feat:
                    loss = crit(model(fc_feats, [att_feats, obj_att_feats, seg_feat_feats], labels, [att_masks, obj_att_masks, seg_feat_masks]),
                            labels[:, 1:], masks[:, 1:]).item()
                elif not use_obj_att and use_seg_feat:
                    loss = crit(model(fc_feats, [att_feats, seg_feat_feats], labels, [att_masks, seg_feat_masks]),
                            labels[:, 1:], masks[:, 1:]).item()
                elif not use_obj_att and not use_seg_feat:
                    loss = crit(model(fc_feats, att_feats, labels, att_masks), labels[:,1:], masks[:,1:]).item()
                elif use_obj_att and not use_seg_feat:
                    loss = crit(model(fc_feats, [att_feats, obj_att_feats], labels, [att_masks, obj_att_masks]), labels[:,1:], masks[:,1:]).item()
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        if use_obj_att and use_seg_feat:
            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['obj_att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['seg_feat_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data['att_masks'] is not None else None,
                   data['obj_att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data[
                                                                                               'obj_att_masks'] is not None else None,
                   data['seg_feat_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data[
                                                                                                   'seg_feat_masks'] is not None else None
                   ]
            tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
            fc_feats, att_feats, obj_att_feats, seg_feat_feats, att_masks, obj_att_masks, seg_feat_masks = tmp
        elif not use_obj_att and use_seg_feat:
            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['seg_feat_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data['att_masks'] is not None else None,
                   data['seg_feat_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data[
                                                                                               'seg_feat_masks'] is not None else None]
            tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
            fc_feats, att_feats, seg_feat_feats, att_masks, seg_feat_masks = tmp
        elif not use_obj_att:
            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data['att_masks'] is not None else None]
            tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
            fc_feats, att_feats, att_masks = tmp
        elif use_obj_att:
            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['obj_att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data['att_masks'] is not None else None,
                   data['obj_att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data[
                                                                                               'obj_att_masks'] is not None else None]
            tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
            fc_feats, att_feats, obj_att_feats, att_masks, obj_att_masks = tmp
        # forward the model to also get generated samples for each image
        with torch.no_grad():
            if use_obj_att and use_seg_feat:
                seq = model(fc_feats, [att_feats, obj_att_feats, seg_feat_feats], [att_masks, obj_att_masks, seg_feat_masks], opt=eval_kwargs, mode='sample')[0]
            elif not use_obj_att and use_seg_feat:
                seq = model(fc_feats, [att_feats, seg_feat_feats], [att_masks, seg_feat_masks], opt=eval_kwargs, mode='sample')[0]
            if not use_obj_att and not use_seg_feat:
                seq = model(fc_feats, att_feats, att_masks, opt=eval_kwargs, mode='sample')[0]
            elif use_obj_att and not use_seg_feat:
                seq = model(fc_feats, [att_feats, obj_att_feats], [att_masks, obj_att_masks], opt=eval_kwargs, mode='sample')[0]

            if isinstance(seq, list):
                #seq = np.array(seq).reshape(1, -1)
                seq = torch.stack(seq[0]).unsqueeze(0)
            else:
                seq = seq.data
        
        # Print beam search
        '''
        if beam_size > 1 and verbose_beam:
            for i in range(loader.batch_size):
                print('\n'.join([utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                print('--' * 10)
        '''
        sents = utils.decode_sequence(loader.get_vocab(), seq)

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                print(cmd)
                os.system(cmd)

            if verbose:
                print('image %s: %s' %(entry['image_id'], entry['caption']))

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, eval_kwargs['id'], split)

    # Switch back to training mode
    model.train()
    return loss_sum/loss_evals, predictions, lang_stats

def crop_center(img,cropx,cropy):
    y,x, channels = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx, :]

def replicate(img, shape):
    new_img = np.zeros(shape)
    step_x = shape[0]/7
    step_y = shape[1]/7
    for i in range(shape[0]):
        for j in range(shape[1]):
            new_img[i,j] = img[int(i/step_x),int(j/step_y)]
    return new_img

def visualize(caption, init_attn, final_attn, image_path):
    # Plot original image
    img = ndimage.imread(image_path)

    '''
    if(image_path.find(u'000000101985') == -1):
        return
    
    if not 'frisbee' in caption:
        return
    '''

    reshape = False
    if reshape:
        img_size = img.shape
        if img_size[0] < img_size[1]:
            shape_0 = 576
            shape_1 = int(img_size[1] * 576/img_size[0])
            img = skimage.transform.resize(img,[shape_0,shape_1])
            img = crop_center(img, 512,512)
        if img_size[0] > img_size[1]:
            shape_1 = 576
            shape_0 = int(img_size[0] * 576/img_size[1])
            img = skimage.transform.resize(img,[shape_0,shape_1])
            img = crop_center(img, 512, 512)
    plt.subplot(4, 5, 1)
    plt.imshow(img)
    plt.axis('off')
    if type(init_attn) is tuple:
        init_attn1 = init_attn[0][2]
        init_attn1 = np.asarray(init_attn1)
        '''
        row_sums = init_attn1.sum(axis=1)
        init_attn1 = init_attn1 / row_sums[:, np.newaxis]
        '''

        init_attn2 = init_attn[1][2]
        init_attn2 = np.asarray(init_attn2)

        '''
        col_sums = init_attn2.sum(axis=0)
        init_attn2 = init_attn2 / col_sums[np.newaxis, :]
        '''
    final_attn = final_attn[2]

    if type(init_attn) is not tuple:
        init_attn = init_attn[2]
        init_attn = np.asarray(init_attn)

    '''
    for i in range(0,7):
        init_attn[i] = 1/7
    '''

    # Plot images with attention weights
    words = caption.split(" ")
    for t in range(len(words)):
        if t > 18:
            break
        plt.subplot(4, 5, t + 2)
        plt.text(0, 1, '%s' % words[t], color='black', backgroundcolor='white', fontsize=8)
        plt.imshow(img)

        #alp_curr = init_attn.reshape(7, 7)
        if type(init_attn) is not tuple:
            alp_curr = np.multiply(np.asarray(final_attn[t]).reshape(7, 1), init_attn)
        if type(init_attn) is tuple:
            #init_attn = init_attn1 + init_attn2.transpose()
            alp_curr = init_attn
            if True:
                final_attn[t] = np.asarray(final_attn[t]).reshape(7, 2)
            alp_curr1 = np.multiply(np.asarray(final_attn[t][:, 0]).reshape(7, 1), init_attn1)
            alp_curr2 = np.multiply(np.asarray(final_attn[t][:, 1]).reshape(7, 1), init_attn2)
            alp_curr = alp_curr1 + alp_curr2.transpose()
            #alp_curr = alp_curr + np.multiply(np.asarray(final_attn[t]).reshape(1, 7), init_attn2)

        #alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=16, sigma=20)
        alp_img = alp_curr
        #alp_img = skimage.transform.resize(alp_img, img.shape[0:2], preserve_range=True, anti_aliasing=True)
        alp_img = replicate(alp_img, img.shape[0:2])
        #alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=16, sigma=20)
        #alp_img = skimage.transform.pyramid_expand(alp_img)

        #alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=16, sigma=20) # added

        #plt.imshow(alp_img, alpha=0.85, cmap='gray')
        plt.imshow(alp_img, alpha=0.85)
        plt.savefig("att_result.png", format='png', dpi=1000)
        plt.axis('off')
    plt.show()


def eval_split_track_attn(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)

    use_obj_att = eval_kwargs.get('use_obj_att', 0)
    use_seg_feat = eval_kwargs.get('use_seg_feat', 0)

    use_obj_mcl_loss = eval_kwargs.get('use_obj_mcl_loss', 0)
    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        if data.get('labels', None) is not None and verbose_loss:
            # forward the model to get loss
            if use_obj_att and use_seg_feat:
                tmp = [data['fc_feats'], data['att_feats'], data['obj_att_feats'], data['seg_feat_feats'], data['labels'], data['masks'],
                       data['att_masks'], data['obj_att_masks'], data['seg_feat_masks']]
                tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
                fc_feats, att_feats, obj_att_feats, seg_feat_feats, labels, masks, att_masks, obj_att_masks, seg_feat_masks = tmp
            elif not use_obj_att and use_seg_feat:
                tmp = [data['fc_feats'], data['att_feats'], data['seg_feat_feats'], data['labels'], data['masks'],
                       data['att_masks'], data['seg_feat_masks']]
                tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
                fc_feats, att_feats, seg_feat_feats, labels, masks, att_masks, seg_feat_masks = tmp
            elif not use_obj_att and not use_seg_feat:
                tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
                tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
                fc_feats, att_feats, labels, masks, att_masks = tmp
            elif use_obj_att and not use_seg_feat:
                tmp = [data['fc_feats'], data['att_feats'], data['obj_att_feats'], data['labels'], data['masks'],
                       data['att_masks'], data['obj_att_masks']]
                tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
                fc_feats, att_feats, obj_att_feats, labels, masks, att_masks, obj_att_masks = tmp

            with torch.no_grad():
                if use_obj_att and use_seg_feat:
                    loss = crit(model(fc_feats, [att_feats, obj_att_feats, seg_feat_feats], labels, [att_masks, obj_att_masks, seg_feat_masks]),
                            labels[:, 1:], masks[:, 1:]).item()
                elif not use_obj_att and use_seg_feat:
                    loss = crit(model(fc_feats, [att_feats, seg_feat_feats], labels, [att_masks, seg_feat_masks]),
                            labels[:, 1:], masks[:, 1:]).item()
                elif not use_obj_att and not use_seg_feat:
                    loss = crit(model(fc_feats, att_feats, labels, att_masks), labels[:,1:], masks[:,1:]).item()
                elif use_obj_att and not use_seg_feat:
                    loss = crit(model(fc_feats, [att_feats, obj_att_feats], labels, [att_masks, obj_att_masks]), labels[:,1:], masks[:,1:]).item()

            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        if use_obj_att and use_seg_feat:
            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['obj_att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['seg_feat_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data['att_masks'] is not None else None,
                   data['obj_att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data[
                                                                                               'obj_att_masks'] is not None else None,
                   data['seg_feat_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data[
                                                                                                   'seg_feat_masks'] is not None else None
                   ]
            tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
            fc_feats, att_feats, obj_att_feats, seg_feat_feats, att_masks, obj_att_masks, seg_feat_masks = tmp
        elif not use_obj_att and use_seg_feat:
            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['seg_feat_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data['att_masks'] is not None else None,
                   data['seg_feat_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data[
                                                                                               'seg_feat_masks'] is not None else None]
            tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
            fc_feats, att_feats, seg_feat_feats, att_masks, seg_feat_masks = tmp
        elif not use_obj_att:
            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data['att_masks'] is not None else None]
            tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
            fc_feats, att_feats, att_masks = tmp
        elif use_obj_att:
            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['obj_att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data['att_masks'] is not None else None,
                   data['obj_att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data[
                                                                                               'obj_att_masks'] is not None else None]
            tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
            fc_feats, att_feats, obj_att_feats, att_masks, obj_att_masks = tmp

        # forward the model to also get generated samples for each image
        with torch.no_grad():
            if (use_obj_mcl_loss == 0):
                if use_obj_att and use_seg_feat:
                    seq, init_attn, final_attn = model(fc_feats, [att_feats, obj_att_feats, seg_feat_feats], [att_masks, obj_att_masks, seg_feat_masks], opt=eval_kwargs, mode='track')
                elif not use_obj_att and use_seg_feat:
                    seq, init_attn, final_attn = model(fc_feats, [att_feats, seg_feat_feats], [att_masks, seg_feat_masks], opt=eval_kwargs, mode='track')
                if not use_obj_att and not use_seg_feat:
                    seq, init_attn, final_attn = model(fc_feats, att_feats, att_masks, opt=eval_kwargs, mode='track')
                elif use_obj_att and not use_seg_feat:
                    seq, init_attn, final_attn = model(fc_feats, [att_feats, obj_att_feats], [att_masks, obj_att_masks], opt=eval_kwargs, mode='track')
            elif (use_obj_mcl_loss == 1):
                if use_obj_att and use_seg_feat:
                    seq, init_attn, init_attn2, final_attn = model(fc_feats, [att_feats, obj_att_feats, seg_feat_feats], [att_masks, obj_att_masks, seg_feat_masks], opt=eval_kwargs, mode='track')
                elif not use_obj_att and use_seg_feat:
                    seq, init_attn, init_attn2, final_attn = model(fc_feats, [att_feats, seg_feat_feats], [att_masks, seg_feat_masks], opt=eval_kwargs, mode='track')
                if not use_obj_att and not use_seg_feat:
                    seq, init_attn, init_attn2, final_attn = model(fc_feats, att_feats, att_masks, opt=eval_kwargs, mode='track')
                elif use_obj_att and not use_seg_feat:
                    seq, init_attn, init_attn2, final_attn = model(fc_feats, [att_feats, obj_att_feats], [att_masks, obj_att_masks], opt=eval_kwargs, mode='track')
                init_attn = init_attn, init_attn2

            if isinstance(seq, list):
                # seq = np.array(seq).reshape(1, -1)
                seq = torch.stack(seq[0]).unsqueeze(0)
            else:
                seq = seq.data

        # Print beam search
        '''
        if beam_size > 1 and verbose_beam:
            for i in range(loader.batch_size):
                print('\n'.join([utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                print('--' * 10)
        '''
        sents = utils.decode_sequence(loader.get_vocab(), seq)

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)

            image_path = os.path.join(eval_kwargs['image_root'],
                                            data['infos'][k]['file_path'])
            visualize(sent, init_attn, final_attn, image_path)

            if verbose:
                print('image %s: %s' % (entry['image_id'], entry['caption']))

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' % (ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, eval_kwargs['id'], split)

    # Switch back to training mode
    model.train()
    return loss_sum / loss_evals, predictions, lang_stats
