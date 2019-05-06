"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

# general packages
import os
import errno
import argparse
import numpy as np
import random
import time
import yaml

# torch
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
import torch.utils.data.distributed

# misc
from data.anet_dataset import ANetDataset, anet_collate_fn, get_vocab_and_sentences
from model.action_prop_dense_cap import ActionPropDenseCap
from data.utils import update_values

parser = argparse.ArgumentParser()

# Data input settings
parser.add_argument('--cfgs_file', default='cfgs/anet.yml', type=str, help='dataset specific settings. anet | yc2')
parser.add_argument('--dataset', default='', type=str, help='which dataset to use. two options: anet | yc2')
parser.add_argument('--dataset_file', default='', type=str)
parser.add_argument('--feature_root', default='', type=str, help='the feature root')
parser.add_argument('--dur_file', default='', type=str)
parser.add_argument('--train_data_folder', default=['training'], type=str, nargs='+', help='training data folder')
parser.add_argument('--val_data_folder', default=['validation'], help='validation data folder')
parser.add_argument('--save_train_samplelist', action='store_true')
parser.add_argument('--load_train_samplelist', action='store_true')
parser.add_argument('--train_samplelist_path', type=str, default='/z/home/luozhou/subsystem/densecap_vid/train_samplelist.pkl')
parser.add_argument('--save_valid_samplelist', action='store_true')
parser.add_argument('--load_valid_samplelist', action='store_true')
parser.add_argument('--valid_samplelist_path', type=str, default='/z/home/luozhou/subsystem/densecap_vid/valid_samplelist.pkl')
parser.add_argument('--start_from', default='', help='path to a model checkpoint to initialize model weights from. Empty = dont')
parser.add_argument('--max_sentence_len', default=20, type=int)
parser.add_argument('--num_workers', default=1, type=int)

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
parser.add_argument('--train_sample', default=20, type=int, help='total number of positive+negative training samples (2*U)')
parser.add_argument('--sample_prob', default=0, type=float, help='probability for use model samples during training')

# Model settings: Proposal and mask
parser.add_argument('--slide_window_size', default=480, type=int, help='the (temporal) size of the sliding window')
parser.add_argument('--slide_window_stride', default=20, type=int, help='the step size of the sliding window')
parser.add_argument('--sampling_sec', default=0.5, help='sample frame (RGB and optical flow) with which time interval')
parser.add_argument('--kernel_list', default=[1, 2, 3, 4, 5, 7, 9, 11, 15, 21, 29, 41, 57, 71, 111, 161, 211, 251],
                    type=int, nargs='+')
parser.add_argument('--pos_thresh', default=0.7, type=float)
parser.add_argument('--neg_thresh', default=0.3, type=float)
parser.add_argument('--stride_factor', default=50, type=int, help='the proposal temporal conv kernel stride is determined by math.ceil(kernel_len/stride_factor)')

# Optimization: General
parser.add_argument('--max_epochs', default=20, type=int, help='max number of epochs to run for')
parser.add_argument('--batch_size', default=32, type=int, help='what is the batch size in number of images per batch? (there will be x seq_per_img sentences)')
parser.add_argument('--valid_batch_size', default=64, type=int)
parser.add_argument('--cls_weight', default=1.0, type=float)
parser.add_argument('--reg_weight', default=10, type=float)
parser.add_argument('--sent_weight', default=0.25, type=float)
parser.add_argument('--scst_weight', default=0.0, type=float)
parser.add_argument('--mask_weight', default=0.0, type=float)
parser.add_argument('--gated_mask', action='store_true', dest='gated_mask')

# Optimization
parser.add_argument('--optim',default='sgd', help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
parser.add_argument('--learning_rate', default=0.1, type=float, help='learning rate')
parser.add_argument('--alpha', default=0.95, type=float, help='alpha for adagrad/rmsprop/momentum/adam')
parser.add_argument('--beta', default=0.999, type=float, help='beta used for adam')
parser.add_argument('--epsilon', default=1e-8, help='epsilon that goes into denominator for smoothing')
parser.add_argument('--loss_alpha_r', default=2, type=int, help='The weight for regression loss')
parser.add_argument('--patience_epoch', default=1, type=int, help='Epoch to wait to determine a pateau')
parser.add_argument('--reduce_factor', default=0.5, type=float, help='Factor of learning rate reduction')
parser.add_argument('--grad_norm', default=1, type=float, help='Gradient clipping norm')

# Data parallel
parser.add_argument('--dist_url', default='file:///home/luozhou/nonexistent_file', type=str, help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')

# Evaluation/Checkpointing
parser.add_argument('--save_checkpoint_every', default=1, type=int, help='how many epochs to save a model checkpoint?')
parser.add_argument('--checkpoint_path', default='./checkpoint', help='folder to save checkpoints into (empty = this folder)')
parser.add_argument('--losses_log_every', default=1, type=int, help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
parser.add_argument('--seed', default=213, type=int, help='random number generator seed to use')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='use gpu')
parser.add_argument('--enable_visdom', action='store_true', dest='enable_visdom')


parser.set_defaults(cuda=False, save_train_samplelist=False,
                    load_train_samplelist=False,
                    save_valid_samplelist=False,
                    load_valid_samplelist=False,
                    gated_mask=False,
                    enable_visdom=False)

args = parser.parse_args()

with open(args.cfgs_file, 'r') as handle:
    options_yaml = yaml.load(handle)
update_values(options_yaml, vars(args))
print(args)

# arguments inspection
assert(args.slide_window_size >= args.slide_window_stride)
assert(args.sampling_sec == 0.5) # attention! sampling_sec is hard coded as 0.5

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed_all(args.seed)


def get_dataset(args):
    # process text
    text_proc, raw_data = get_vocab_and_sentences(args.dataset_file, args.max_sentence_len)

    # Create the dataset and data loader instance
    train_dataset = ANetDataset(args.feature_root,
                                args.train_data_folder,
                                args.slide_window_size,
                                args.dur_file,
                                args.kernel_list,
                                text_proc, raw_data,
                                args.pos_thresh, args.neg_thresh,
                                args.stride_factor,
                                args.dataset,
                                save_samplelist=args.save_train_samplelist,
                                load_samplelist=args.load_train_samplelist,
                                sample_listpath=args.train_samplelist_path,
                                )

    # dist parallel, optional
    args.distributed = args.world_size > 1
    if args.distributed and args.cuda:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=args.num_workers,
                              collate_fn=anet_collate_fn)

    valid_dataset = ANetDataset(args.feature_root,
                                args.val_data_folder,
                                args.slide_window_size,
                                args.dur_file,
                                args.kernel_list,
                                text_proc, raw_data,
                                args.pos_thresh, args.neg_thresh,
                                args.stride_factor,
                                args.dataset,
                                save_samplelist=args.save_valid_samplelist,
                                load_samplelist=args.load_valid_samplelist,
                                sample_listpath=args.valid_samplelist_path
                                )

    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.valid_batch_size,
                              shuffle=False,
                              num_workers=args.num_workers,
                              collate_fn=anet_collate_fn)

    return train_loader, valid_loader, text_proc, train_sampler


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
                               nsamples=args.train_sample,
                               kernel_list=args.kernel_list,
                               stride_factor=args.stride_factor,
                               learn_mask=args.mask_weight>0)

    # Initialize the networks and the criterion
    if len(args.start_from) > 0:
        print("Initializing weights from {}".format(args.start_from))
        model.load_state_dict(torch.load(args.start_from,
                                              map_location=lambda storage, location: storage))

    # Ship the model to GPU, maybe
    if args.cuda:
        if args.distributed:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
        else:
            model = torch.nn.DataParallel(model).cuda()
        # elif torch.cuda.device_count() > 1:
        #     model = torch.nn.DataParallel(model).cuda()
        # else:
        #     model.cuda()
    return model


def main(args):
    try:
        os.makedirs(args.checkpoint_path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory already exists.')
        else:
            raise

    print('loading dataset')
    train_loader, valid_loader, text_proc, train_sampler = get_dataset(args)

    print('building model')
    model = get_model(text_proc, args)

    # filter params that don't require gradient (credit: PyTorch Forum issue 679)
    # smaller learning rate for the decoder
    if args.optim == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            args.learning_rate, betas=(args.alpha, args.beta), eps=args.epsilon)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            args.learning_rate,
            weight_decay=1e-5,
            momentum=args.alpha,
            nesterov=True
        )
    else:
        raise NotImplementedError

    # learning rate decay every 1 epoch
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.reduce_factor,
                                               patience=args.patience_epoch,
                                               verbose=True)
    # scheduler = lr_scheduler.ExponentialLR(optimizer, 0.6)

    # Number of parameter blocks in the network
    print("# of param blocks: {}".format(str(len(list(model.parameters())))))

    best_loss = float('inf')

    if args.enable_visdom:
        import visdom
        vis = visdom.Visdom()
        vis_window={'iter': None,
                    'loss': None}
    else:
        vis, vis_window = None, None

    all_eval_losses = []
    all_cls_losses = []
    all_reg_losses = []
    all_sent_losses = []
    all_mask_losses = []
    all_training_losses = []
    for train_epoch in range(args.max_epochs):
        t_epoch_start = time.time()
        print('Epoch: {}'.format(train_epoch))

        if args.distributed:
            train_sampler.set_epoch(train_epoch)

        epoch_loss = train(train_epoch, model, optimizer, train_loader,
                           vis, vis_window, args)
        all_training_losses.append(epoch_loss)

        (valid_loss, val_cls_loss,
         val_reg_loss, val_sent_loss, val_mask_loss) = valid(model, valid_loader)

        all_eval_losses.append(valid_loss)
        all_cls_losses.append(val_cls_loss)
        all_reg_losses.append(val_reg_loss)
        all_sent_losses.append(val_sent_loss)
        all_mask_losses.append(val_mask_loss)

        if args.enable_visdom:
            if vis_window['loss'] is None:
                if not args.distributed or (args.distributed and dist.get_rank() == 0):
                    vis_window['loss'] = vis.line(
                    X=np.tile(np.arange(len(all_eval_losses)),
                              (6,1)).T,
                    Y=np.column_stack((np.asarray(all_training_losses),
                                       np.asarray(all_eval_losses),
                                       np.asarray(all_cls_losses),
                                       np.asarray(all_reg_losses),
                                       np.asarray(all_sent_losses),
                                       np.asarray(all_mask_losses))),
                    opts=dict(title='Loss',
                              xlabel='Validation Iter',
                              ylabel='Loss',
                              legend=['train',
                                      'dev',
                                      'dev_cls',
                                      'dev_reg',
                                      'dev_sentence',
                                      'dev_mask']))
            else:
                if not args.distributed or (
                    args.distributed and dist.get_rank() == 0):
                    vis.line(
                    X=np.tile(np.arange(len(all_eval_losses)),
                              (6, 1)).T,
                    Y=np.column_stack((np.asarray(all_training_losses),
                                       np.asarray(all_eval_losses),
                                       np.asarray(all_cls_losses),
                                       np.asarray(all_reg_losses),
                                       np.asarray(all_sent_losses),
                                       np.asarray(all_mask_losses))),
                    win=vis_window['loss'],
                    opts=dict(title='Loss',
                              xlabel='Validation Iter',
                              ylabel='Loss',
                              legend=['train',
                                      'dev',
                                      'dev_cls',
                                      'dev_reg',
                                      'dev_sentence',
                                      'dev_mask']))

        if valid_loss < best_loss:
            best_loss = valid_loss
            if (args.distributed and dist.get_rank() == 0) or not args.distributed:
                torch.save(model.module.state_dict(), os.path.join(args.checkpoint_path, 'best_model.t7'))
            print('*'*5)
            print('Better validation loss {:.4f} found, save model'.format(valid_loss))

        # save eval and train losses
        if (args.distributed and dist.get_rank() == 0) or not args.distributed:
            torch.save({'train_loss':all_training_losses,
                        'eval_loss':all_eval_losses,
                        'eval_cls_loss':all_cls_losses,
                        'eval_reg_loss':all_reg_losses,
                        'eval_sent_loss':all_sent_losses,
                        'eval_mask_loss':all_mask_losses,
                        }, os.path.join(args.checkpoint_path, 'model_losses.t7'))

        # learning rate decay
        scheduler.step(valid_loss)

        # validation/save checkpoint every a few epochs
        if train_epoch%args.save_checkpoint_every == 0 or train_epoch == args.max_epochs:
            if (args.distributed and dist.get_rank() == 0) or not args.distributed:
                torch.save(model.module.state_dict(),
                       os.path.join(args.checkpoint_path, 'model_epoch_{}.t7'.format(train_epoch)))

        # all other process wait for the 1st process to finish
        # if args.distributed:
        #     dist.barrier()

        print('-'*80)
        print('Epoch {} summary'.format(train_epoch))
        print('Train loss: {:.4f}, val loss: {:.4f}, Time: {:.4f}s'.format(
            epoch_loss, valid_loss, time.time()-t_epoch_start
        ))
        print('val_cls: {:.4f}, '
              'val_reg: {:.4f}, val_sentence: {:.4f}, '
              'val mask: {:.4f}'.format(
            val_cls_loss, val_reg_loss, val_sent_loss, val_mask_loss
        ))
        print('-'*80)


### Training the network ###
def train(epoch, model, optimizer, train_loader, vis, vis_window, args):
    model.train() # training mode
    train_loss = []
    nbatches = len(train_loader)
    t_iter_start = time.time()

    sample_prob = min(args.sample_prob, int(epoch/5)*0.05)
    for train_iter, data in enumerate(train_loader):
        (img_batch, tempo_seg_pos, tempo_seg_neg, sentence_batch) = data
        img_batch = Variable(img_batch)
        tempo_seg_pos = Variable(tempo_seg_pos)
        tempo_seg_neg = Variable(tempo_seg_neg)
        sentence_batch = Variable(sentence_batch)

        if args.cuda:
            img_batch = img_batch.cuda()
            tempo_seg_neg = tempo_seg_neg.cuda()
            tempo_seg_pos = tempo_seg_pos.cuda()
            sentence_batch = sentence_batch.cuda()

        t_model_start = time.time()
        (pred_score, gt_score,
        pred_offsets, gt_offsets,
        pred_sentence, gt_sent,
         scst_loss, mask_loss) = model(img_batch, tempo_seg_pos,
                                       tempo_seg_neg, sentence_batch,
                                       sample_prob, args.stride_factor,
                                       scst=args.scst_weight > 0,
                                       gated_mask=args.gated_mask)

        cls_loss = model.module.bce_loss(pred_score, gt_score) * args.cls_weight
        reg_loss = model.module.reg_loss(pred_offsets, gt_offsets) * args.reg_weight
        sent_loss = F.cross_entropy(pred_sentence, gt_sent) * args.sent_weight

        total_loss = cls_loss + reg_loss + sent_loss

        if scst_loss is not None:
            scst_loss *= args.scst_weight
            total_loss += scst_loss

        if mask_loss is not None:
            mask_loss = args.mask_weight * mask_loss
            total_loss += mask_loss
        else:
            mask_loss = cls_loss.new(1).fill_(0)

        optimizer.zero_grad()
        total_loss.backward()

        # enable the clipping for zero mask loss training
        total_grad_norm = clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
                                         args.grad_norm)

        optimizer.step()

        train_loss.append(total_loss.data.item())

        if args.enable_visdom:
            if vis_window['iter'] is None:
                if not args.distributed or (
                    args.distributed and dist.get_rank() == 0):
                    vis_window['iter'] = vis.line(
                        X=np.arange(epoch*nbatches+train_iter, epoch*nbatches+train_iter+1),
                        Y=np.asarray(train_loss),
                        opts=dict(title='Training Loss',
                                  xlabel='Training Iteration',
                                  ylabel='Loss')
                    )
            else:
                if not args.distributed or (
                    args.distributed and dist.get_rank() == 0):
                    vis.line(
                        X=np.arange(epoch*nbatches+train_iter, epoch*nbatches+train_iter+1),
                        Y=np.asarray([np.mean(train_loss)]),
                        win=vis_window['iter'],
                        opts=dict(title='Training Loss',
                                  xlabel='Training Iteration',
                                  ylabel='Loss'),
                        update='append'
                    )

        t_model_end = time.time()
        print('iter: [{}/{}], training loss: {:.4f}, '
              'class: {:.4f}, '
              'reg: {:.4f}, sentence: {:.4f}, '
              'mask: {:.4f}, '
              'grad norm: {:.4f} '
              'data time: {:.4f}s, total time: {:.4f}s'.format(
            train_iter, nbatches, total_loss.data.item(), cls_loss.data.item(),
            reg_loss.data.item(), sent_loss.data.item(), mask_loss.data.item(),
            total_grad_norm,
            t_model_start - t_iter_start,
            t_model_end - t_iter_start
        ), end='\r')

        t_iter_start = time.time()

    return np.mean(train_loss)


### Validation ##
def valid(model, loader):
    model.eval()
    valid_loss = []
    val_cls_loss = []
    val_reg_loss = []
    val_sent_loss = []
    val_mask_loss = []
    for iter, data in enumerate(loader):
        (img_batch, tempo_seg_pos, tempo_seg_neg, sentence_batch) = data
        with torch.no_grad():
            img_batch = Variable(img_batch)
            tempo_seg_pos = Variable(tempo_seg_pos)
            tempo_seg_neg = Variable(tempo_seg_neg)
            sentence_batch = Variable(sentence_batch)

            if args.cuda:
                img_batch = img_batch.cuda()
                tempo_seg_neg = tempo_seg_neg.cuda()
                tempo_seg_pos = tempo_seg_pos.cuda()
                sentence_batch = sentence_batch.cuda()

            (pred_score, gt_score,
             pred_offsets, gt_offsets,
             pred_sentence, gt_sent,
             _, mask_loss) = model(img_batch, tempo_seg_pos,
                                    tempo_seg_neg, sentence_batch,
                                    stride_factor=args.stride_factor,
                                    gated_mask=args.gated_mask)

            cls_loss = model.module.bce_loss(pred_score, gt_score) * args.cls_weight
            reg_loss = model.module.reg_loss(pred_offsets, gt_offsets) * args.reg_weight
            sent_loss = F.cross_entropy(pred_sentence, gt_sent) * args.sent_weight

            total_loss = cls_loss + reg_loss + sent_loss

            if mask_loss is not None:
                mask_loss = args.mask_weight * mask_loss
                total_loss += mask_loss
            else:
                mask_loss = cls_loss.new(1).fill_(0)

            valid_loss.append(total_loss.data.item())
            val_cls_loss.append(cls_loss.data.item())
            val_reg_loss.append(reg_loss.data.item())
            val_sent_loss.append(sent_loss.data.item())
            val_mask_loss.append(mask_loss.data.item())

    return (np.mean(valid_loss), np.mean(val_cls_loss),
            np.mean(val_reg_loss), np.mean(val_sent_loss), np.mean(val_mask_loss))


if __name__ == "__main__":
    main(args)
