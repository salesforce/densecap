# End-to-End Dense Video Captioning with Masked Transformer

This is the source code for our paper [End-to-End Dense Video Captioning with Masked Transformer](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0037.pdf). It mainly supports dense video captioning on **generated** segments. To generate captions on **GT** segments, please refer to our new GVD [repo](https://github.com/facebookresearch/grounded-video-description) and also our [notes](#endnotes).


## Requirements (Recommended)
1) [Miniconda3](https://conda.io/miniconda.html) for Python 3.6

2) CUDA 9.2 and CUDNN v7.1

3) [PyTorch 0.4.0](https://pytorch.org/get-started/locally/). Follow the instructions to install pytorch and torchvision.

4) Install other required modules (e.g., torchtext)

`pip install -r requirements.txt`

Optional: If you would like to use visdom to track training do `pip install visdom`

Optional: If you would like to use spacy tokenizer do `pip install spacy`

Note: The code has been tested on a variety of GPUs, including 1080 Ti, Titan Xp, P100, V100 etc. However, for the latest RTX GPUs (e.g., 2080 Ti), CUDA 10.0 and hence PyTorch 1.0 are required. The code needs to be upgraded to PyTorch 1.0.


## Data Preparation
### Annotation and feature
For ActivityNet, download the re-formatted annotation files from [here](http://youcook2.eecs.umich.edu/static/dat/anet_densecap/anet.tar.gz), decompress and place under directory `data`. The frame-wise appearance (with suffix `_resnet.npy`) and motion (with suffix `_bn.npy`) feature files for each spilt are available [[train (27.7GB)](http://youcook2.eecs.umich.edu/static/dat/anet_densecap/training_feat_anet.tar.gz), [val (13.7GB)](http://youcook2.eecs.umich.edu/static/dat/anet_densecap/validation_feat_anet.tar.gz), [test (13.6GB)](http://youcook2.eecs.umich.edu/static/dat/anet_densecap/testing_feat_anet.tar.gz)] and should be decompressed and placed under your dataset directory (refer to as `feature_root` in the configuration files).

Similarly for YouCook2, the annotation files are available [here](http://youcook2.eecs.umich.edu/static/dat/yc2_densecap/yc2.tar.gz) and should be placed under `data`. The feature files are [[train (9.6GB)](http://youcook2.eecs.umich.edu/static/dat/yc2_densecap/training_feat_yc2.tar.gz), [val (3.2GB)](http://youcook2.eecs.umich.edu/static/dat/yc2_densecap/validation_feat_yc2.tar.gz), [test (1.5GB)](http://youcook2.eecs.umich.edu/static/dat/yc2_densecap/testing_feat_yc2.tar.gz)].

You could also extract the feature on your own with this [code](https://github.com/LuoweiZhou/anet2016-cuhk-feature). Note that ActivityNet is processed with an older version of the repo while YouCook2 is processed with the latest code which had a minor change regarding the [sampling](https://github.com/LuoweiZhou/anet2016-cuhk-feature/blob/master/pyActionRec/video_proc.py#L85) approach. This accounts for the difference in the formulation of [frame_to_second](https://github.com/salesforce/densecap/blob/master/data/anet_dataset.py#L131) conversion.


### Evaluate scripts
Download the dense video captioning evaluation [scripts](https://github.com/LuoweiZhou/densevid_eval) and place it under the `tools` directory. Make sure you recursively clone the repo. Our code is equavalent to the official evaluation code from ActivityNet 2017 Challenge, but faster. Note that the current evaluation [scripts](https://github.com/ranjaykrishna/densevid_eval) had a few major bugs fixed towards ActivityNet 2018 Challenge.

The evaluate script for event proposal can be found under `tools`.


## Training and Validation
First, set the paths in configuration files (under `cfgs`) to your own data and feature directories. Create new directories `log` and `results` under the root directory to save log and result files.

The example command on running a 4-GPU distributed data parallel job (for ActivityNet):

For Masked Transformer:
```
CUDA_VISIBLE_DEVICES=0 python3 scripts/train.py --dist_url $dist_url --cfgs_file $cfgs_file \
    --checkpoint_path ./checkpoint/$id --batch_size $batch_size --world_size 4 \
    --cuda --sent_weight $sent_weight | tee log/$id-0 &
CUDA_VISIBLE_DEVICES=1 python3 scripts/train.py --dist_url $dist_url --cfgs_file $cfgs_file \
    --checkpoint_path ./checkpoint/$id --batch_size $batch_size --world_size 4 \
    --cuda --sent_weight $sent_weight | tee log/$id-1 &
CUDA_VISIBLE_DEVICES=2 python3 scripts/train.py --dist_url $dist_url --cfgs_file $cfgs_file \
    --checkpoint_path ./checkpoint/$id --batch_size $batch_size --world_size 4 \
    --cuda --sent_weight $sent_weight | tee log/$id-2 &
CUDA_VISIBLE_DEVICES=3 python3 scripts/train.py --dist_url $dist_url --cfgs_file $cfgs_file \
    --checkpoint_path ./checkpoint/$id --batch_size $batch_size --world_size 4 \
    --cuda --sent_weight $sent_weight | tee log/$id-3
```
For End-to-End Masked Transformer:
```
CUDA_VISIBLE_DEVICES=0 python3 scripts/train.py --dist_url $dist_url --cfgs_file $cfgs_file \
    --checkpoint_path ./checkpoint/$id --batch_size $batch_size --world_size 4 \
    --cuda --sent_weight $sent_weight --mask_weight $mask_weight --gated_mask | tee log/$id-0 &
CUDA_VISIBLE_DEVICES=1 python3 scripts/train.py --dist_url $dist_url --cfgs_file $cfgs_file \
    --checkpoint_path ./checkpoint/$id --batch_size $batch_size --world_size 4 \
    --cuda --sent_weight $sent_weight --mask_weight $mask_weight --gated_mask | tee log/$id-1 &
CUDA_VISIBLE_DEVICES=2 python3 scripts/train.py --dist_url $dist_url --cfgs_file $cfgs_file \
    --checkpoint_path ./checkpoint/$id --batch_size $batch_size --world_size 4 \
    --cuda --sent_weight $sent_weight --mask_weight $mask_weight --gated_mask | tee log/$id-2 &
CUDA_VISIBLE_DEVICES=3 python3 scripts/train.py --dist_url $dist_url --cfgs_file $cfgs_file \
    --checkpoint_path ./checkpoint/$id --batch_size $batch_size --world_size 4 \
    --cuda --sent_weight $sent_weight --mask_weight $mask_weight --gated_mask | tee log/$id-3
```

Arguments: `batch_size=14`, `mask_weight=1.0`, `sent_weight=0.25`, `cfgs_file='cfgs/anet.yml'`, `dist_url='file:///home/luozhou/nonexistent_file'` (replace with your directory), `id` indicates the model name.

For YouCook2 dataset, you can simply replace `cfgs/anet.yml` with `cfgs/yc2.yml`. To monitor the training (e.g., training & validation losses), start the visdom server with `visdom` in the background (e.g., tmux). Then, add `--enable_visdom` as a command argument.

Note that at least 15 GB of free RAM is required for the training. The `nonexistent_file` will normally be cleaned up automatically, but might need a manual delete if otherwise. More about distributed data parallel see [here (0.4.0)](https://pytorch.org/docs/0.4.0/distributed.html). You can also run the code with a single GPU by setting `world_size=1`.

Due to legacy reasons, we store the feature files as individual `.npy` files, which causes latency in data loading and hence instability during distributed model training. By default, we set the value of `num_workers` to 1. It could be set up to 6 for a faster data loading. However, if encouter any data parallel issue, try setting it to 0.


### Pre-trained Models
The pre-trained models can be downloaded from [here (1GB)](http://youcook2.eecs.umich.edu/static/dat/densecap_checkpoints/pre-trained-models.tar.gz). Make sure you uncompress the file under the `checkpoint` directory (create one under the root directory if not exists).


## Testing
For Masked Transformer (`id=anet-2L-gt-mask`):
```
python3 scripts/test.py --cfgs_file $cfgs_file --densecap_eval_file ./tools/densevid_eval/evaluate.py \
    --batch_size 1 --start_from ./checkpoint/$id/model_epoch_$epoch.t7 --id $id-$epoch \
    --val_data_folder $split --cuda | tee log/eval-$id-epoch$epoch
```

For End-to-End Masked Transformer (`id=anet-2L-e2e-mask`):
```
python3 scripts/test.py --cfgs_file $cfgs_file --densecap_eval_file ./tools/densevid_eval/evaluate.py \
    --batch_size 1 --start_from ./checkpoint/$id/model_epoch_$epoch.t7 --id $id-$epoch \
    --val_data_folder $split --learn_mask --gated_mask --cuda | tee log/eval-$id-epoch$epoch
```

Arguments: `epoch=19`, `split='validation'`, `cfgs_file='cfgs/anet.yml'`

This gives you the language evaluation results on the validation set. You need at least 8GB of free GPU memory for the evaluation. The current evaluation script only supports `batch_size=1` and is slow (1hr for yc2 and 4hr for anet). We actively welcome pull requests.

### Leaderboard (for the test set)
The official evaluation servers are available under [ActivityNet](http://activity-net.org/challenges/2018/evaluation.html) and [YouCook2](http://youcook2.eecs.umich.edu/leaderboard). Note that the NEW evaluation [scripts](https://github.com/ranjaykrishna/densevid_eval) from ActivityNet 2018 Challenge are used in both cases.


## <a name="endnotes"></a>Notes
We use a different code base for captioning-only models (dense captioning on GT segments). Please contact <luozhou@umich.edu> for details. Note that it can potentially work with this code base if you feed in GT segments into the captioning module rather than the generated segments. However, there is no guarantee on reproducing the results from the paper. You can also refer to this [implementation](https://github.com/facebookresearch/grounded-video-description) where you need to config `--att_model` to 'transformer'.


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
