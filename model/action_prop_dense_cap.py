"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


from torch import nn
from torch.autograd import Variable
from .transformer import Transformer, RealTransformer
import torch
import numpy as np
import torch.nn.functional as F
import math
from data.utils import segment_iou
import time

def positional_encodings(x, D):
    # input x a vector of positions
    encodings = torch.zeros(x.size(0), D)
    if x.is_cuda:
        encodings = encodings.cuda(x.get_device())
    encodings = Variable(encodings)

    for channel in range(D):
        if channel % 2 == 0:
            encodings[:,channel] = torch.sin(
                x / 10000 ** (channel / D))
        else:
            encodings[:,channel] = torch.cos(
                x / 10000 ** ((channel - 1) / D))
    return encodings

class DropoutTime1D(nn.Module):
    '''
        assumes the first dimension is batch, 
        input in shape B x T x H
        '''
    def __init__(self, p_drop):
        super(DropoutTime1D, self).__init__()
        self.p_drop = p_drop

    def forward(self, x):
        if self.training:
            mask = x.data.new(x.data.size(0),x.data.size(1), 1).uniform_()
            mask = Variable((mask > self.p_drop).float())
            return x * mask
        else:
            return x * (1-self.p_drop)

    def init_params(self):
        pass

    def __repr__(self):
        repstr = self.__class__.__name__ + ' (\n'
        repstr += "{:.2f}".format(self.p_drop)
        repstr += ')'
        return repstr


class ActionPropDenseCap(nn.Module):
    def __init__(self, d_model, d_hidden, n_layers, n_heads, vocab,
                 in_emb_dropout, attn_dropout, vis_emb_dropout,
                 cap_dropout, nsamples, kernel_list, stride_factor,
                 learn_mask=False, window_length=480):
        super(ActionPropDenseCap, self).__init__()

        self.kernel_list = kernel_list
        self.nsamples = nsamples
        self.learn_mask = learn_mask
        self.d_model = d_model

        self.mask_model = nn.Sequential(
            nn.Linear(d_model+window_length, d_model, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Linear(d_model, window_length),
        )

        self.rgb_emb = nn.Linear(2048, d_model // 2)
        self.flow_emb = nn.Linear(1024, d_model // 2)
        self.emb_out = nn.Sequential(
            # nn.BatchNorm1d(h_dim),
            DropoutTime1D(in_emb_dropout),
            nn.ReLU()
        )

        self.vis_emb = Transformer(d_model, 0, 0,
                                   d_hidden=d_hidden,
                                   n_layers=n_layers,
                                   n_heads=n_heads,
                                   drop_ratio=attn_dropout)

        self.vis_dropout = DropoutTime1D(vis_emb_dropout)

        self.prop_out = nn.ModuleList(
            [nn.Sequential(
                nn.BatchNorm1d(d_model),
                nn.Conv1d(d_model, d_model,
                          self.kernel_list[i],
                          stride=math.ceil(kernel_list[i]/stride_factor),
                          groups=d_model,
                          bias=False),
                nn.BatchNorm1d(d_model),
                nn.Conv1d(d_model, d_model,
                          1, bias=False),
                nn.BatchNorm1d(d_model),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(d_model),
                nn.Conv1d(d_model, 4,
                          1)
                )
                for i in range(len(self.kernel_list))
            ])

        self.cap_model = RealTransformer(d_model,
                                         self.vis_emb.encoder, #share the encoder
                                         vocab,
                                         d_hidden=d_hidden,
                                         n_layers=n_layers,
                                         n_heads=n_heads,
                                         drop_ratio=cap_dropout)

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.reg_loss = nn.SmoothL1Loss()
        self.l2_loss = nn.MSELoss()

    def forward(self, x, s_pos, s_neg, sentence,
                sample_prob=0, stride_factor=10, scst=False,
                gated_mask=False):
        B, T, _ = x.size()
        dtype = x.data.type()

        x_rgb, x_flow = torch.split(x, 2048, 2)
        x_rgb = self.rgb_emb(x_rgb.contiguous())
        x_flow = self.flow_emb(x_flow.contiguous())

        x = torch.cat((x_rgb, x_flow), 2)

        x = self.emb_out(x)

        vis_feat, all_emb = self.vis_emb(x)
        # vis_feat = self.vis_dropout(vis_feat)

        # B x T x H -> B x H x T
        # for 1d conv
        vis_feat = vis_feat.transpose(1,2).contiguous()

        prop_lst = []
        for i, kernel in enumerate(self.prop_out):

            kernel_size = self.kernel_list[i]
            if kernel_size <= vis_feat.size(-1):
                pred_o = kernel(vis_feat)
                anchor_c = Variable(torch.FloatTensor(np.arange(
                    float(kernel_size)/2.0,
                    float(T+1-kernel_size/2.0),
                    math.ceil(kernel_size/stride_factor)
                )).type(dtype))
                if anchor_c.size(0) != pred_o.size(-1):
                    raise Exception("size mismatch!")

                anchor_c = anchor_c.expand(B, 1, anchor_c.size(0))
                anchor_l = Variable(torch.FloatTensor(anchor_c.size()).fill_(kernel_size).type(dtype))

                pred_final = torch.cat((pred_o, anchor_l, anchor_c), 1)
                prop_lst.append(pred_final)
            else:
                print('skipping kernel sizes greater than {}'.format(
                    self.kernel_list[i]))
                break

        # Important! In prop_all, for the first dimension, the four values are proposal score, overlapping score (DEPRECATED!), length offset, and center offset, respectively
        prop_all = torch.cat(prop_lst, 2)

        if B != s_pos.size(0) or B != s_neg.size(0):
            raise Exception('feature and ground-truth segment do not match!')

        sample_each = self.nsamples // 2
        pred_score = Variable(torch.FloatTensor(np.zeros((sample_each*B, 2))).type(dtype))
        gt_score = Variable(torch.FloatTensor(np.zeros((sample_each*B, 2))).type(dtype))
        pred_offsets = Variable(torch.FloatTensor(np.zeros((sample_each*B,2))).type(dtype))
        gt_offsets = Variable(torch.FloatTensor(np.zeros((sample_each*B,2))).type(dtype))

        # B x T x H
        batch_mask = Variable(torch.FloatTensor(np.zeros((B,T,1))).type(dtype))

        # store positional encodings, size of B x 4,
        # the first B values are predicted starts,
        # second B values are predicted ends,
        # third B values are anchor starts,
        # last B values are anchor ends
        pe_locs = Variable(torch.zeros(B*4).type(dtype))
        anchor_window_mask = Variable(torch.zeros(B, T).type(dtype),
                                      requires_grad=False)
        pred_bin_window_mask = Variable(torch.zeros(B, T).type(dtype),
                                      requires_grad=False)
        gate_scores = Variable(torch.zeros(B,1,1).type(dtype))

        mask_loss = None

        pred_len = prop_all[:, 4, :] * torch.exp(prop_all[:, 2, :])
        pred_cen = prop_all[:, 5, :] + prop_all[:, 4, :] * prop_all[:, 3, :]

        for b in range(B):
            pos_anchor = s_pos[b]
            neg_anchor = s_neg[b]

            if pos_anchor.size(0) != sample_each or neg_anchor.size(0) != sample_each:
                raise Exception("# of positive or negative samples does not match")

            # randomly choose one of the positive samples to caption
            pred_index = np.random.randint(sample_each)

            # random sample anchors from different length
            for i in range(sample_each):
                # sample pos anchors
                pos_sam = pos_anchor[i].data
                pos_sam_ind = int(pos_sam[0])

                pred_score[b*sample_each+i, 0] = prop_all[b, 0, pos_sam_ind]
                gt_score[b*sample_each+i, 0] = 1

                pred_offsets[b*sample_each+i] = prop_all[b, 2:4, pos_sam_ind]
                gt_offsets[b*sample_each+i] = pos_sam[2:]

                # sample neg anchors
                neg_sam = neg_anchor[i].data
                neg_sam_ind = int(neg_sam[0])
                pred_score[b*sample_each+i, 1] = prop_all[b, 0, neg_sam_ind]
                gt_score[b*sample_each+i, 1] = 0

                # caption the segment
                if i == pred_index: # only need once, since one sample corresponds to one sentence only
                           # TODO Is that true? Why cannot caption all?
                    # anchor length, 4, 5 are the indices for anchor length and center
                    anc_len = prop_all[b, 4, pos_sam_ind].data.item()
                    anc_cen = prop_all[b, 5, pos_sam_ind].data.item()

                    # grount truth length, 2 and 3 are indices for ground truth length and center
                    # see line 260 and 268 in anet_dataset.py
                    # dont need to use index since now i is 0 and everything is matching
                    # length is after taking log
                    gt_len = np.exp(pos_sam[2].item()) * anc_len
                    gt_cen = pos_sam[3].item() * anc_len + anc_cen
                    gt_window_mask = torch.zeros(T, 1).type(dtype)

                    gt_window_mask[
                    max(0, math.floor(gt_cen - gt_len / 2.)):
                    min(T, math.ceil(gt_cen + gt_len / 2.)), :] = 1.
                    gt_window_mask = Variable(gt_window_mask,
                                              requires_grad=False)

                    batch_mask[b] = gt_window_mask
                    # batch_mask[b] = anchor_window_mask

                    crt_pred_cen = pred_cen[b, pos_sam_ind]
                    crt_pred_len = pred_len[b, pos_sam_ind]
                    pred_start_w = crt_pred_cen - crt_pred_len / 2.0
                    pred_end_w = crt_pred_cen + crt_pred_len/2.0

                    pred_bin_window_mask[b,
                    math.floor(max(0, min(T-1, pred_start_w.data.item()))):
                    math.ceil(max(1, min(T, pred_end_w.data.item())))] = 1.

                    if self.learn_mask:
                        anchor_window_mask[b,
                        max(0, math.floor(anc_cen - anc_len / 2.)):
                        min(T, math.ceil(anc_cen + anc_len / 2.))] = 1.

                        pe_locs[b] = pred_start_w
                        pe_locs[B + b] = pred_end_w
                        pe_locs[B*2 + b] = Variable(torch.Tensor([max(0, math.floor(anc_cen - anc_len / 2.))]).type(dtype))
                        pe_locs[B*3 + b] = Variable(torch.Tensor([min(T, math.ceil(anc_cen + anc_len / 2.))]).type(dtype))

                        # gate_scores[b] = pred_score[b*sample_each+i, 0].detach()
                        gate_scores[b] = pred_score[b*sample_each+i, 0]

        if self.learn_mask:
            pos_encs = positional_encodings(pe_locs, self.d_model//4)
            in_pred_mask = torch.cat((pos_encs[:B], pos_encs[B:B*2],
                                      pos_encs[B*2:B*3], pos_encs[B*3:B*4],
                                      anchor_window_mask), 1)
            pred_mask = self.mask_model(in_pred_mask).view(B, T, 1)

            if gated_mask:
                gate_scores = F.sigmoid(gate_scores)
                window_mask = (gate_scores * pred_bin_window_mask.view(B, T, 1)
                # window_mask = (gate_scores * batch_mask
                                +  (1-gate_scores) * pred_mask)
            else:
                window_mask = pred_mask

            mask_loss = F.binary_cross_entropy_with_logits(pred_mask, pred_bin_window_mask.view(B, T, 1))
            # mask_loss = F.binary_cross_entropy_with_logits(window_mask, batch_mask)
        else:
            window_mask = pred_bin_window_mask.view(B, T, 1)
            # window_mask = batch_mask

        pred_sentence, gt_cent= self.cap_model(x, sentence, window_mask,
                                               sample_prob=sample_prob)

        scst_loss = None
        if scst:
            scst_loss = self.cap_model.scst(x, batch_mask, sentence)

        return (pred_score, gt_score,
                pred_offsets, gt_offsets,
                pred_sentence, gt_cent,
                scst_loss, mask_loss)


    def inference(self, x, actual_frame_length, sampling_sec,
                  min_prop_num, max_prop_num,
                  min_prop_num_before_nms, pos_thresh, stride_factor,
                  gated_mask=False):
        B, T, _ = x.size()
        dtype = x.data.type()

        x_rgb, x_flow = torch.split(x, 2048, 2)
        x_rgb = self.rgb_emb(x_rgb.contiguous())
        x_flow = self.flow_emb(x_flow.contiguous())

        x = torch.cat((x_rgb, x_flow), 2)

        x = self.emb_out(x)

        vis_feat, all_emb = self.vis_emb(x)
        # vis_feat = self.vis_dropout(vis_feat)

        # B x T x H -> B x H x T
        # for 1d conv
        vis_feat = vis_feat.transpose(1,2).contiguous()

        prop_lst = []
        for i, kernel in enumerate(self.prop_out):

            kernel_size = self.kernel_list[i]
            if kernel_size <= actual_frame_length[0]: # no need to use larger kernel size in this case, batch size is only 1
                pred_o = kernel(vis_feat)
                anchor_c = Variable(torch.FloatTensor(np.arange(
                    float(kernel_size)/2.0,
                    float(T+1-kernel_size/2.0),
                    math.ceil(kernel_size/stride_factor)
                )).type(dtype))
                if anchor_c.size(0) != pred_o.size(-1):
                    raise Exception("size mismatch!")

                anchor_c = anchor_c.expand(B, 1, anchor_c.size(0))
                anchor_l = Variable(torch.FloatTensor(anchor_c.size()).fill_(kernel_size).type(dtype))

                pred_final = torch.cat((pred_o, anchor_l, anchor_c), 1)
                prop_lst.append(pred_final)
            else:
                print('skipping kernel sizes greater than {}'.format(
                    self.kernel_list[i]))
                break

        prop_all = torch.cat(prop_lst, 2)

        # assume 1st and 2nd are action prediction and overlap, respectively
        prop_all[:,:2,:] = F.sigmoid(prop_all[:,:2,:])


        pred_len = prop_all[:, 4, :] * torch.exp(prop_all[:, 2, :])
        pred_cen = prop_all[:, 5, :] + prop_all[:, 4, :] * prop_all[:, 3, :]

        nms_thresh_set = np.arange(0.9, 0.95, 0.05).tolist()
        all_proposal_results = []

        # store positional encodings, size of B x 4,
        # the first B values are predicted starts,
        # second B values are predicted ends,
        # third B values are anchor starts,
        # last B values are anchor ends
        pred_start_lst = [] #torch.zeros(B * 4).type(dtype)
        pred_end_lst = []
        anchor_start_lst = []
        anchor_end_lst = []
        anchor_window_mask = [] #Variable(torch.zeros(B, T).type(dtype))
        gate_scores = [] #Variable(torch.zeros(B, 1).type(dtype))

        for b in range(B):
            crt_pred = prop_all.data[b]
            crt_pred_cen = pred_cen.data[b]
            crt_pred_len = pred_len.data[b]
            pred_masks = []
            batch_result = []
            crt_nproposal = 0
            nproposal = torch.sum(torch.gt(prop_all.data[b, 0, :], pos_thresh))
            nproposal = min(max(nproposal, min_prop_num_before_nms),
                            prop_all.size(-1))
            pred_results = np.empty((nproposal, 3))
            _, sel_idx = torch.topk(crt_pred[0], nproposal)
 
            start_t = time.time()
            for nms_thresh in nms_thresh_set:
                for prop_idx in range(nproposal):
                    original_frame_len = actual_frame_length[b].item() + sampling_sec*2 # might be truncated at the end, hence + frame_to_second*2
                    pred_start_w = crt_pred_cen[sel_idx[prop_idx]] - crt_pred_len[sel_idx[prop_idx]] / 2.0
                    pred_end_w = crt_pred_cen[sel_idx[prop_idx]] + crt_pred_len[sel_idx[prop_idx]] / 2.0
                    pred_start = pred_start_w
                    pred_end = pred_end_w
                    if pred_start >= pred_end:
                        continue
                    if pred_end >= original_frame_len or pred_start < 0:
                        continue

                    hasoverlap = False
                    if crt_nproposal > 0:
                        if np.max(segment_iou(np.array([pred_start, pred_end]), pred_results[:crt_nproposal])) > nms_thresh:
                            hasoverlap = True

                    if not hasoverlap:
                        pred_bin_window_mask = torch.zeros(1, T, 1).type(dtype)
                        win_start = math.floor(max(min(pred_start, min(original_frame_len, T)-1), 0))
                        win_end = math.ceil(max(min(pred_end, min(original_frame_len, T)), 1))
                        # if win_start >= win_end:
                        #     print('length: {}, mask window start: {} >= window end: {}, skipping'.format(
                        #         original_frame_len, win_start, win_end,
                        #     ))
                        #     continue

                        pred_bin_window_mask[:, win_start:win_end] = 1
                        pred_masks.append(pred_bin_window_mask)

                        if self.learn_mask:
                            # 4, 5 are the indices for anchor length and center
                            anc_len = crt_pred[4, sel_idx[prop_idx]]
                            anc_cen = crt_pred[5, sel_idx[prop_idx]]
                            # only use the pos sample to train, could potentially use more sample for training mask, but this is easier to do
                            amask = torch.zeros(1,T).type(dtype)
                            amask[0,
                            max(0, math.floor(anc_cen - anc_len / 2.)):
                            min(T, math.ceil(anc_cen + anc_len / 2.))] = 1.
                            anchor_window_mask.append(amask)

                            pred_start_lst.append(torch.Tensor([pred_start_w]).type(dtype))
                            pred_end_lst.append(torch.Tensor([pred_end_w]).type(dtype))
                            anchor_start_lst.append(torch.Tensor([max(0,
                                                                 math.floor(
                                                                 anc_cen - anc_len / 2.))]).type(
                                                                 dtype))
                            anchor_end_lst.append(torch.Tensor([min(T,
                                                               math.ceil(
                                                               anc_cen + anc_len / 2.))]).type(
                                                               dtype))

                            gate_scores.append(torch.Tensor([crt_pred[0, sel_idx[prop_idx]]]).type(dtype))

                        pred_results[crt_nproposal] = np.array([win_start,
                                                                win_end,
                                                                crt_pred[0, sel_idx[prop_idx]]])
                        crt_nproposal += 1

                    if crt_nproposal >= max_prop_num:
                        break

                if crt_nproposal >= min_prop_num:
                    break

            mid1_t = time.time()

            if len(pred_masks) == 0: # append all-one window if no window is proposed
                pred_masks.append(torch.ones(1, T, 1).type(dtype))
                pred_results[0] = np.array([0, min(original_frame_len, T), pos_thresh])
                crt_nproposal = 1

            pred_masks = Variable(torch.cat(pred_masks, 0))
            batch_x = x[b].unsqueeze(0).expand(pred_masks.size(0), x.size(1), x.size(2))

            if self.learn_mask:
                pe_pred_start = torch.cat(pred_start_lst, 0)
                pe_pred_end = torch.cat(pred_end_lst, 0)
                pe_anchor_start = torch.cat(anchor_start_lst, 0)
                pe_anchor_end = torch.cat(anchor_end_lst, 0)

                pe_locs = torch.cat((pe_pred_start, pe_pred_end, pe_anchor_start, pe_anchor_end), 0)
                pos_encs = positional_encodings(pe_locs, self.d_model // 4)
                npos = pos_encs.size(0)
                anchor_window_mask = Variable(torch.cat(anchor_window_mask, 0))
                in_pred_mask = torch.cat((pos_encs[:npos//4], pos_encs[npos//4:npos//4*2],
                                          pos_encs[npos//4 * 2:npos//4 * 3],
                                          pos_encs[npos//4 * 3:npos//4 * 4],
                                          anchor_window_mask), 1)
                pred_cont_masks  = self.mask_model(in_pred_mask).unsqueeze(2)

                if gated_mask:
                    gate_scores = Variable(torch.cat(gate_scores, 0).view(-1,1,1))
                    window_mask = (gate_scores * pred_masks
                                   + (1 - gate_scores) * pred_cont_masks)

                else:
                    window_mask = pred_cont_masks
            else:
                window_mask = pred_masks

            mid2_t = time.time()

            pred_sentence = []
            # use cap_batch as caption batch size
            cap_batch = math.ceil(480*256/T)
            for sent_i in range(math.ceil(window_mask.size(0)/cap_batch)):
                batch_start = sent_i*cap_batch
                batch_end = min((sent_i+1)*cap_batch, window_mask.size(0))
                pred_sentence += self.cap_model.greedy(batch_x[batch_start:batch_end],
                                                       window_mask[batch_start:batch_end], 20)

            pred_results = pred_results[:crt_nproposal]
            assert len(pred_sentence) == crt_nproposal, (
                "number of predicted sentence and proposal does not match"
            )

            for idx in range(len(pred_results)):
                batch_result.append((pred_results[idx][0],
                                     pred_results[idx][1],
                                     pred_results[idx][2],
                                     pred_sentence[idx]))
            all_proposal_results.append(tuple(batch_result))

            end_t = time.time()
            print('Processing time for tIoU: {:.2f}, mask: {:.2f}, caption: {:.2f}'.format(mid1_t-start_t, mid2_t-mid1_t, end_t-mid2_t))

        return all_proposal_results
