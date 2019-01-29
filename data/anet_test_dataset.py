"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset


class ANetTestDataset(Dataset):
    def __init__(self, image_path, slide_window_size,
                 text_proc, raw_data, split, learn_mask=False):
        super(ANetTestDataset, self).__init__()

        self.split = split
        split_path = os.path.join(image_path, self.split)
        self.slide_window_size = slide_window_size
        self.learn_mask = learn_mask

        self.sample_list = []  # list of list for data samples

        test_sentences = []
        for vid, val in raw_data.items():
            annotations = val['annotations']
            if val['subset'] == self.split and os.path.isfile(os.path.join(split_path, vid+'_bn.npy')):
                video_prefix = os.path.join(split_path, vid)
                self.sample_list.append(video_prefix)
                for ind, ann in enumerate(annotations):
                    ann['sentence'] = ann['sentence'].strip()
                    test_sentences.append(ann['sentence'])

        test_sentences = list(map(text_proc.preprocess, test_sentences))
        sentence_idx = text_proc.numericalize(text_proc.pad(test_sentences),
                                                   device=-1)  # put in memory

        if sentence_idx.nelement() != 0 and len(test_sentences) != 0:
            if sentence_idx.size(0) != len(test_sentences):
                raise Exception("Error in numericalize sentences")

        idx = 0
        for vid, val in raw_data.items():
            if val['subset'] == self.split and os.path.isfile(os.path.join(split_path, vid+'_bn.npy')):
                for ann in val['annotations']:
                    ann['sentence_idx'] = sentence_idx[idx]
                    idx += 1

        print('total number of samples (unique videos): {}'.format(
            len(self.sample_list)))
        print('total number of sentences: {}'.format(len(test_sentences)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        video_prefix = self.sample_list[index]

        resnet_feat = torch.from_numpy(np.load(video_prefix + '_resnet.npy')).float()
        bn_feat = torch.from_numpy(np.load(video_prefix + '_bn.npy')).float()

        if self.learn_mask:
            img_feat = torch.FloatTensor(np.zeros((self.slide_window_size,
                                                   resnet_feat.size(1)+bn_feat.size(1))))
            torch.cat((resnet_feat, bn_feat), dim=1,
                      out=img_feat[:min(bn_feat.size(0), self.slide_window_size)])
        else:
            img_feat = torch.cat((resnet_feat, bn_feat), 1)

        return img_feat, bn_feat.size(0), video_prefix


def anet_test_collate_fn(batch_lst):
    img_feat, _, _ = batch_lst[0]

    batch_size = len(batch_lst)

    img_batch = torch.FloatTensor(batch_size,
                                  img_feat.size(0),
                                  img_feat.size(1)).zero_()

    frame_length = torch.IntTensor(batch_size).zero_()

    video_prefix = []

    for batch_idx in range(batch_size):
        img_feat, T, vid = batch_lst[batch_idx]

        img_batch[batch_idx,:] = img_feat
        frame_length[batch_idx] = T
        video_prefix.append(vid)

    return img_batch, frame_length, video_prefix
