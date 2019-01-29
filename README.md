# End-to-End Dense Video Captioning with Masked Transformer

This is the source code for our paper [End-to-End Dense Video Captioning with Masked Transformer](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0037.pdf)

## Requirements (Recommended)
1) [Miniconda3](https://conda.io/miniconda.html) for Python 3

2) CUDA 9.2 and CUDNN v7.1

3) [PyTorch 0.4.0](https://pytorch.org/get-started/locally/). Follow the instructions to install pytorch and torchvision.

4) Install other required modules (e.g., torchtext)

   `pip install -r requirements.txt`

Optional: If you would like to use visdom to track training do `pip install visdom`

Optional: If you would like to use spacy tokenizer do `pip install spacy`



## Data Preparation
### Annotation and feature
For ActivityNet, download the re-formatted annotation files from [here](http://youcook2.eecs.umich.edu/static/dat/anet_densecap/anet.tar.gz), decompress and place under directory `data`. The frame-wise appearance (with suffix `_resnet.npy`) and motion (with suffix `_bn.npy`) feature files for each spilt are available [[train (27.7GB)](http://youcook2.eecs.umich.edu/static/dat/anet_densecap/training_feat_anet.tar.gz), [val (13.7GB)](http://youcook2.eecs.umich.edu/static/dat/anet_densecap/validation_feat_anet.tar.gz), [test (13.6GB)](http://youcook2.eecs.umich.edu/static/dat/anet_densecap/testing_feat_anet.tar.gz)] and should be decompressed and placed under your dataset directory (refer to as `feature_root` in the configuration files).

Similarly for YouCook2, the annotation files are available [here](http://youcook2.eecs.umich.edu/static/dat/yc2_densecap/yc2.tar.gz) and should be placed under `data`. The feature files are [[train (9.6GB)](http://youcook2.eecs.umich.edu/static/dat/yc2_densecap/training_feat_yc2.tar.gz), [val (3.2GB)](http://youcook2.eecs.umich.edu/static/dat/yc2_densecap/validation_feat_yc2.tar.gz), [test (1.5GB)](http://youcook2.eecs.umich.edu/static/dat/yc2_densecap/testing_feat_yc2.tar.gz)].

### Evaluate scripts
Download the dense video captioning evaluation [scripts](https://github.com/LuoweiZhou/densevid_eval) and place it under the `tools` directory. Make sure you recursively clone the repo. Our code is equavalent to the official evaluation code from ActivityNet 2017 Challenge, but faster. Note that the current evaluation [scripts](https://github.com/ranjaykrishna/densevid_eval) had a few major bugs fixed towards ActivityNet 2018 Challenge.

The evaluate script for event proposal can be found under `tools`.


## Training and Validation
Configuration files for ActivityNet and YouCook2 are under `cfgs`. Create new directories `log` and `results` under the root directory to save log and result files.

The example command on running a 4-GPU distributed data parallel job (for ActivityNet):

For Masked Transformer:
```
CUDA_VISIBLE_DEVICES=0 python3 scripts/train.py --dist_url $dist_url --cfgs_file $cfgs_file --checkpoint_path ./checkpoint/$id \
    --d_model 1024 --d_hidden 2048 --in_emb_dropout 0.1 --attn_dropout $enc_drop --vis_emb_dropout 0.1 --cap_dropout $dec_drop \
    --n_layers $nlayer --optim sgd --learning_rate $lr --alpha 0.95 --batch_size $batch_size --patience_epoch 1 --world_size 4 \
    --stride_factor $stride --cuda --seed $seed --sent_weight $sent_weight | tee log/$id-0 &
CUDA_VISIBLE_DEVICES=1 python3 scripts/train.py --dist_url $dist_url --cfgs_file $cfgs_file --checkpoint_path ./checkpoint/$id \
    --d_model 1024 --d_hidden 2048 --in_emb_dropout 0.1 --attn_dropout $enc_drop --vis_emb_dropout 0.1 --cap_dropout $dec_drop \
    --n_layers $nlayer --optim sgd --learning_rate $lr --alpha 0.95 --batch_size $batch_size --patience_epoch 1 --world_size 4 \
    --stride_factor $stride --cuda --seed $seed --sent_weight $sent_weight | tee log/$id-1 &
CUDA_VISIBLE_DEVICES=2 python3 scripts/train.py --dist_url $dist_url --cfgs_file $cfgs_file --checkpoint_path ./checkpoint/$id \
    --d_model 1024 --d_hidden 2048 --in_emb_dropout 0.1 --attn_dropout $enc_drop --vis_emb_dropout 0.1 --cap_dropout $dec_drop \
    --n_layers $nlayer --optim sgd --learning_rate $lr --alpha 0.95 --batch_size $batch_size --patience_epoch 1 --world_size 4 \
    --stride_factor $stride --cuda --seed $seed --sent_weight $sent_weight | tee log/$id-2 &
CUDA_VISIBLE_DEVICES=3 python3 scripts/train.py --dist_url $dist_url --cfgs_file $cfgs_file --checkpoint_path ./checkpoint/$id \
    --d_model 1024 --d_hidden 2048 --in_emb_dropout 0.1 --attn_dropout $enc_drop --vis_emb_dropout 0.1 --cap_dropout $dec_drop \
    --n_layers $nlayer --optim sgd --learning_rate $lr --alpha 0.95 --batch_size $batch_size --patience_epoch 1 --world_size 4 \
    --stride_factor $stride --cuda --seed $seed --sent_weight $sent_weight | tee log/$id-3
```
For End-to-End Masked Transformer:
```
CUDA_VISIBLE_DEVICES=0 python3 scripts/train.py --dist_url $dist_url --cfgs_file $cfgs_file --checkpoint_path ./checkpoint/$id \
    --d_model 1024 --d_hidden 2048 --in_emb_dropout 0.1 --attn_dropout $enc_drop --vis_emb_dropout 0.1 --cap_dropout $dec_drop \
    --n_layers $nlayer --optim sgd --learning_rate $lr --alpha 0.95 --batch_size $batch_size --patience_epoch 1 --world_size 4 \
    --stride_factor $stride --cuda --seed $seed --sent_weight $sent_weight --mask_weight $mask_weight --gated_mask | tee log/$id-0 &
CUDA_VISIBLE_DEVICES=1 python3 scripts/train.py --dist_url $dist_url --cfgs_file $cfgs_file --checkpoint_path ./checkpoint/$id \
    --d_model 1024 --d_hidden 2048 --in_emb_dropout 0.1 --attn_dropout $enc_drop --vis_emb_dropout 0.1 --cap_dropout $dec_drop \
    --n_layers $nlayer --optim sgd --learning_rate $lr --alpha 0.95 --batch_size $batch_size --patience_epoch 1 --world_size 4 \
    --stride_factor $stride --cuda --seed $seed --sent_weight $sent_weight --mask_weight $mask_weight --gated_mask | tee log/$id-1 &
CUDA_VISIBLE_DEVICES=2 python3 scripts/train.py --dist_url $dist_url --cfgs_file $cfgs_file --checkpoint_path ./checkpoint/$id \
    --d_model 1024 --d_hidden 2048 --in_emb_dropout 0.1 --attn_dropout $enc_drop --vis_emb_dropout 0.1 --cap_dropout $dec_drop \
    --n_layers $nlayer --optim sgd --learning_rate $lr --alpha 0.95 --batch_size $batch_size --patience_epoch 1 --world_size 4 \
    --stride_factor $stride --cuda --seed $seed --sent_weight $sent_weight --mask_weight $mask_weight --gated_mask | tee log/$id-2 &
CUDA_VISIBLE_DEVICES=3 python3 scripts/train.py --dist_url $dist_url --cfgs_file $cfgs_file --checkpoint_path ./checkpoint/$id \
    --d_model 1024 --d_hidden 2048 --in_emb_dropout 0.1 --attn_dropout $enc_drop --vis_emb_dropout 0.1 --cap_dropout $dec_drop \
    --n_layers $nlayer --optim sgd --learning_rate $lr --alpha 0.95 --batch_size $batch_size --patience_epoch 1 --world_size 4 \
    --stride_factor $stride --cuda --seed $seed --sent_weight $sent_weight --mask_weight $mask_weight --gated_mask | tee log/$id-3
```

Arguments: `nlayer=2`, `batch_size=14`, `stride=50`, `split='training'`, `enc_drop=0.2`, `dec_drop=0.2`, `mask_weight=1.0`, `patience_epoch=1`, `sent_weight=0.25`, `momentum=0.95`, `optim='sgd'`, `lr=0.1`, `seed=213`, `cfgs_file='cfgs/anet.yml'`, `dist_url='file:///private/home/luoweizhou/nonexistent_file'`, `id` indicates the model name.

For YouCook2 dataset, you can simply replace `cfgs/anet.yml` with `cfgs/yc2.yml`. To monitor the training (e.g., training & validation losses), start the visdom server with `python -m visdom.server` or `visdom` in the background (e.g., tmux). Then, add `--enable_visdom` as a command argument.

You need at least 15 GB of free RAM for the training.

### Pre-trained Models
The pre-trained models can be downloaded from [here (1GB)](http://youcook2.eecs.umich.edu/static/dat/densecap_checkpoints/pre-trained-models.tar.gz). Make sure you uncompress the file under the `checkpoint` directory (create one under the root directory if not exists).


## Testing
For Masked Transformer (`id=anet-2L-gt-mask`):
```
python3 scripts/test.py --cfgs_file $cfgs_file --densecap_eval_file ./tools/densevid_eval/evaluate.py --batch_size 1 \
    --start_from ./checkpoint/$id/model_epoch_$epoch.t7 --n_layers $nlayer --d_model 1024 --d_hidden 2048 --id $id-$epoch \
    --stride_factor $stride --in_emb_dropout 0.1 --attn_dropout $enc_drop --vis_emb_dropout 0.1 --cap_dropout $dec_drop \
    --val_data_folder $split --cuda | tee log/eval-$id-epoch$epoch
```

For End-to-End Masked Transformer (`id=anet-2L-e2e-mask`):
```
python3 scripts/test.py --cfgs_file $cfgs_file --densecap_eval_file ./tools/densevid_eval/evaluate.py --batch_size 1 \
    --start_from ./checkpoint/$id/model_epoch_$epoch.t7 --n_layers $nlayer --d_model 1024 --d_hidden 2048 --id $id-$epoch \
    --stride_factor $stride --in_emb_dropout 0.1 --attn_dropout $enc_drop --vis_emb_dropout 0.1 --cap_dropout $dec_drop \
    --val_data_folder $split --learn_mask --gated_mask --cuda | tee log/eval-$id-epoch$epoch
```

Arguments: `epoch=19`, `stride=50`, `split='validation'`, `enc_drop=0.2`, `dec_drop=0.2`, `nlayer=2`, `cfgs_file='cfgs/anet.yml'`

You need at least 8GB of free GPU memory for the evaluation. The current evaluation script only supports `batch_size=1` and is slow (1hr for yc2 and 4hr for anet). We actively welcome pull requests.

### Results
On ActivityNet Captions:

| Method                        | Bleu@4 | METEOR | CIDEr |
|-------------------------------|--------|--------|-------|
| Masked Transformer            | 2.39   | 10.12  | 19.94 |
| End-to-End Masked Transformer | 2.62   | 10.17  | 23.59 |


On YouCook2:

| Method                        | Bleu@4 | METEOR | CIDEr |
|-------------------------------|--------|--------|-------|
| Masked Transformer            | 0.83   | 8.08   | 18.14 |
| End-to-End Masked Transformer | 1.16   | 8.73   | 22.67 |

## Notes
We use a different code base for captioning-only models (dense captioning w/ GT segments). Please contact <luozhou@umich.edu> for details. Note that it can potentially work with this code base if you feed in GT segments into the captioning module rather than the generated segments. However, there is no guarantee on reproducing the results from the paper.


## Citation
```
@inproceedings{zhou2018end,
  title={End-to-End Dense Video Captioning with Masked Transformer},
  author={Zhou, Luowei and Zhou, Yingbo and Corso, Jason J and Socher, Richard and Xiong, Caiming},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={8739--8748},
  year={2018}
}
```
