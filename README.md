# End-to-End Dense Video Captioning with Masked Transformer

This is the source code for our paper [End-to-End Dense Video Captioning with Masked Transformer](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0037.pdf)

## Requirements (Recommended)
1) [Miniconda3](https://conda.io/miniconda.html) for Python 3

2) CUDA 9.2 and CUDNN v7.1

3) [PyTorch 0.4.0](https://pytorch.org/get-started/locally/). Follow the instructions to install pytorch and torchvision.

4) Install other required modules (e.g., torchtext)

`pip install -r requirements.txt`

## Data Preparation
### Annotation and feature
For activityNet, download the re-formatted annotation files from [here](http://youcook2.eecs.umich.edu/static/dat/anet_densecap/anet.tar.gz), decompress and place under directory `data`. The frame-wise appearance (with suffix `_resnet.npy`) and motion (with suffix `_bn.npy`) feature files for each spilt are available [[train(27.7 GB)](http://youcook2.eecs.umich.edu/static/dat/anet_densecap/training_feat_anet.tar.gz), [val(13.7 GB)](http://youcook2.eecs.umich.edu/static/dat/anet_densecap/validation_feat_anet.tar.gz), [test(13.6 GB)](http://youcook2.eecs.umich.edu/static/dat/anet_densecap/testing_feat_anet.tar.gz)]  and should be decompressed placed under your dataset directory (which we refer to as `feature_root`).

Similarly for YouCook2, the annotation files are available [here](http://youcook2.eecs.umich.edu/static/dat/yc2_densecap/yc2.tar.gz) and should be placed under `data`. The feature files are [[train(9.6 GB)](http://youcook2.eecs.umich.edu/static/dat/yc2_densecap/training_feat_yc2.tar.gz), [val(3.2 GB)](http://youcook2.eecs.umich.edu/static/dat/yc2_densecap/validation_feat_yc2.tar.gz), [test(1.5 GB)](http://youcook2.eecs.umich.edu/static/dat/yc2_densecap/testing_feat_yc2.tar.gz)].

### Evaluate scripts
Download the dense video captioning evaluation [scripts](https://github.com/LuoweiZhou/densevid_eval) and place it under the `tools` directory. Make sure you recursively clone the coco-caption repo. Our code is equavalent to the official evaluation code from ActivityNet 2017 Challenge, but faster. Note that the current evaluation [scripts](https://github.com/ranjaykrishna/densevid_eval) had a few major bugs fixed towards ActivityNet 2018 Challenge.

The evaluate script for event proposal can be found under `tools`.

## Training and Validation
Configuration files for ActivityNet and YouCook2 are under `cfgs`.

The example command on running a 4-GPU distributed data parallel job (for ActivityNet):
```python scripts/train.py```

The example command on running a single-GPU job (for ActivityNet):
```python scripts/train.py```

For YouCook2 dataset, you can simply replace `cfgs/anet.yml` with `cfgs/yc2.yml`. To monitor the training (e.g., training & validation losses), start the visdom server with `python -m visdom.server` or `visdom` in the background (e.g., tmux). Then, add `--enable_visdom` as a command argument.

You need at least 15 GB of free RAM for the training.

## Testing
```python scripts/train.py```

### Results
ActivityNet

YouCook2

## Notes
We use a different code base for captioning-only models (dense captioning w/ GT segments). Please contact luozhou@umich.edu for details. Note that it can potentially work with this code base if you feed in GT segments into the captioning module rather than the generated segments. However, there is no guarantee on reproducing the results from the paper.


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