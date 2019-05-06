"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

# general packages
import os
import argparse
import numpy as np
from collections import defaultdict
import json
import subprocess
import csv
import yaml

# torch
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

# misc
from data.anet_test_dataset import ANetTestDataset, anet_test_collate_fn
from data.anet_dataset import get_vocab_and_sentences
from model.action_prop_dense_cap import ActionPropDenseCap
from tools.eval_proposal_anet import ANETproposal
from data.utils import update_values

parser = argparse.ArgumentParser()

# Data input settings
parser.add_argument('--cfgs_file', default='cfgs/anet.yml', type=str, help='dataset specific settings. anet | yc2')
parser.add_argument('--dataset', default='', type=str, help='which dataset to use. two options: anet | yc2')
parser.add_argument('--dataset_file', default='', type=str)
parser.add_argument('--feature_root', default='', type=str, help='the feature root')
parser.add_argument('--dur_file', default='', type=str)
parser.add_argument('--val_data_folder', default='validation', help='validation data folder')
parser.add_argument('--densecap_eval_file', default='/z/subsystem/densevid_eval/evaluate.py')
parser.add_argument('--densecap_references', default='', type=str)
parser.add_argument('--start_from', default='', help='path to a model checkpoint to initialize model weights from. Empty = dont')
parser.add_argument('--max_sentence_len', default=20, type=int)
parser.add_argument('--num_workers', default=2, type=int)

# Model settings: General
parser.add_argument('--d_model', default=1024, type=int, help='size of the rnn in number of hidden nodes in each layer')
parser.add_argument('--d_hidden', default=2048, type=int)
parser.add_argument('--n_heads', default=8, type=int)
parser.add_argument('--in_emb_dropout', default=0.1, type=float)
parser.add_argument('--attn_dropout', default=0.2, type=float)
parser.add_argument('--vis_emb_dropout', default=0.1, type=float)
parser.add_argument('--cap_dropout', default=0.2, type=float)
parser.add_argument('--image_feat_size', default=3072, type=int, help='the encoding size of the image feature')
parser.add_argument('--n_layers', default=2, type=int, help='number of layers in the sequence model')

# Model settings: Proposal and mask
parser.add_argument('--slide_window_size', default=480, type=int, help='the (temporal) size of the sliding window')
parser.add_argument('--slide_window_stride', default=20, type=int, help='the step size of the sliding window')
parser.add_argument('--sampling_sec', default=0.5, help='sample frame (RGB and optical flow) with which time interval')
parser.add_argument('--kernel_list', default=[1, 2, 3, 4, 5, 7, 9, 11, 15, 21, 29, 41, 57, 71, 111, 161, 211, 251],
                    type=int, nargs='+')
parser.add_argument('--max_prop_num', default=500, type=int, help='the maximum number of proposals per video')
parser.add_argument('--min_prop_num', default=50, type=int, help='the minimum number of proposals per video')
parser.add_argument('--min_prop_before_nms', default=200, type=int, help='the minimum number of proposals per video')
parser.add_argument('--pos_thresh', default=0.7, type=float)
parser.add_argument('--stride_factor', default=50, type=int, help='the proposal temporal conv kernel stride is determined by math.ceil(kernel_len/stride_factor)')

parser.add_argument('--gated_mask', action='store_true', dest='gated_mask')
parser.add_argument('--learn_mask', action='store_true', dest='learn_mask')

# Optimization: General
parser.add_argument('--batch_size', default=1, type=int, help='what is the batch size in number of images per batch? (there will be x seq_per_img sentences)')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='use gpu')
parser.add_argument('--id', default='', help='an id identifying this run/job. used in cross-val and appended when writing progress files')


parser.set_defaults(cuda=False, learn_mask=False, gated_mask=False)

args = parser.parse_args()

with open(args.cfgs_file, 'r') as handle:
    options_yaml = yaml.load(handle)
update_values(options_yaml, vars(args))
print(args)

# arguments inspection
assert args.batch_size == 1, "Batch size has to be 1!"
if args.slide_window_size < args.slide_window_stride:
    raise Exception("arguments insepection failed!")


def get_dataset(args):
    # process text
    text_proc, raw_data = get_vocab_and_sentences(args.dataset_file, args.max_sentence_len)

    # Create the dataset and data loader instance
    test_dataset = ANetTestDataset(args.feature_root,
                                   args.slide_window_size,
                                   text_proc, raw_data, args.val_data_folder,
                                   learn_mask=args.learn_mask)

    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             collate_fn=anet_test_collate_fn)

    return test_loader, text_proc


def get_model(text_proc, args):
    sent_vocab = text_proc.vocab
    model = ActionPropDenseCap(d_model=args.d_model,
                               d_hidden=args.d_hidden,
                               n_layers=args.n_layers,
                               n_heads=args.n_heads,
                               vocab=sent_vocab,
                               in_emb_dropout=args.in_emb_dropout,
                               attn_dropout=args.attn_dropout,
                               vis_emb_dropout=args.vis_emb_dropout,
                               cap_dropout=args.cap_dropout,
                               nsamples=0,
                               kernel_list=args.kernel_list,
                               stride_factor=args.stride_factor,
                               learn_mask=args.learn_mask)

    # Initialize the networks and the criterion
    if len(args.start_from) > 0:
        print("Initializing weights from {}".format(args.start_from))
        model.load_state_dict(torch.load(args.start_from,
                                              map_location=lambda storage, location: storage))

    # Ship the model to GPU, maybe
    if args.cuda:
        model.cuda()

    return model


### Validation ##
def validate(model, loader, args):
    model.eval()
    densecap_result = defaultdict(list)
    prop_result = defaultdict(list)

    avg_prop_num = 0

    frame_to_second = {}
    with open(args.dur_file) as f:
        if args.dataset == 'anet':
            for line in f:
                vid_name, vid_dur, vid_frame = [l.strip() for l in line.split(',')]
                frame_to_second[vid_name] = float(vid_dur)*int(float(vid_frame)*1./int(float(vid_dur))*args.sampling_sec)*1./float(vid_frame)
            frame_to_second['_0CqozZun3U'] = args.sampling_sec # a missing video in anet
        elif args.dataset == 'yc2':
            import math
            for line in f:
                vid_name, vid_dur, vid_frame = [l.strip() for l in line.split(',')]
                frame_to_second[vid_name] = float(vid_dur)*math.ceil(float(vid_frame)*1./float(vid_dur)*args.sampling_sec)*1./float(vid_frame) # for yc2
        else:
            raise NotImplementedError

    for data in loader:
        image_feat, original_num_frame, video_prefix = data
        with torch.no_grad():
            image_feat = Variable(image_feat)
            # ship data to gpu
            if args.cuda:
                image_feat = image_feat.cuda()

            dtype = image_feat.data.type()
            if video_prefix[0].split('/')[-1] not in frame_to_second:
                frame_to_second[video_prefix[0].split('/')[-1]] = args.sampling_sec
                print("cannot find frame_to_second for video {}".format(video_prefix[0].split('/')[-1]))
            sampling_sec = frame_to_second[video_prefix[0].split('/')[-1]] # batch_size has to be 1
            all_proposal_results = model.inference(image_feat,
                                                   original_num_frame,
                                                   sampling_sec,
                                                   args.min_prop_num,
                                                   args.max_prop_num,
                                                   args.min_prop_before_nms,
                                                   args.pos_thresh,
                                                   args.stride_factor,
                                                   gated_mask=args.gated_mask)

            for b in range(len(video_prefix)):
                vid = video_prefix[b].split('/')[-1]
                print('Write results for video: {}'.format(vid))
                for pred_start, pred_end, pred_s, sent in all_proposal_results[b]:
                    densecap_result['v_'+vid].append(
                        {'sentence':sent,
                         'timestamp':[pred_start * sampling_sec,
                                      pred_end * sampling_sec]})

                    prop_result[vid].append(
                        {'segment':[pred_start * sampling_sec,
                                    pred_end * sampling_sec],
                         'score':pred_s})

                avg_prop_num += len(all_proposal_results[b])

    print("average proposal number: {}".format(avg_prop_num/len(loader.dataset)))

    return eval_results(densecap_result, prop_result, args)


def eval_results(densecap_result, prop_result, args):

    # write captions to json file for evaluation (densecap)
    dense_cap_all = {'version':'VERSION 1.0', 'results':densecap_result,
                     'external_data':{'used':'true',
                      'details':'global_pool layer from BN-Inception pretrained from ActivityNet \
                                 and ImageNet (https://github.com/yjxiong/anet2016-cuhk)'}}
    with open(os.path.join('./results/', 'densecap_'+args.val_data_folder+'_'+args.id+ '.json'), 'w') as f:
        json.dump(dense_cap_all, f)

    subprocess.Popen(["python2", args.densecap_eval_file, "-s", \
                      os.path.join('./results/', 'densecap_'+args.val_data_folder+'_' + args.id + '.json'), \
                      "-v", "-r"] + \
                      args.densecap_references \
                      )

    # write proposals to json file for evaluation (proposal)
    prop_all = {'version':'VERSION 1.0', 'results':prop_result,
                'external_data':{'used':'true',
                'details':'global_pool layer from BN-Inception pretrained from ActivityNet \
                           and ImageNet (https://github.com/yjxiong/anet2016-cuhk)'}}
    with open(os.path.join('./results/', 'prop_'+args.val_data_folder+'_'+args.id+ '.json'), 'w') as f:
        json.dump(prop_all, f)

    anet_proposal = ANETproposal(args.dataset_file,
                                 os.path.join('./results/', 'prop_'+args.val_data_folder+'_' + args.id + '.json'),
                                 tiou_thresholds=np.linspace(0.5, 0.95, 10),
                                 max_avg_nr_proposals=100,
                                 subset=args.val_data_folder, verbose=True, check_status=True)

    anet_proposal.evaluate()

    return anet_proposal.area


def main():

    print('loading dataset')
    test_loader, text_proc = get_dataset(args)

    print('building model')
    model = get_model(text_proc, args)

    recall_area = validate(model, test_loader, args)

    print('proposal recall area: {:.6f}'.format(recall_area))


if __name__ == "__main__":
    main()
