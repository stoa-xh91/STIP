# Semantic-aware Transfer with Instance-adaptive Parsing for Crowded Scenes Pose Estimation
This is an official implementation of:

**Xuanhan Wang, Lianli Gao, Yan Dai, Yixuan Zhou and Jingkuan Song**. ***Semantic-aware Transfer with Instance-adaptive Parsing for Crowded Scenes Pose Estimation***

This repo is built on [deep-high-resolution-net](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch).

## Main Results
### Results on CrowdPose test set
| Method             | Backbone   | Input size  |  AP   | Ap .5 | AP .75 | AP (Easy) | AP (Medium) | AP (Hard) |
|--------------------|------------|------------ |-------|-------|--------|-----------|-------------|-----------|
| HRNet              | HRNet-w32  | 256 x 192   | 71.7  | 89.8  |  76.9  |  79.6     |    72.7     |    61.5   | 
| HRNet + STIP       | HRNet-w32  | 256 x 192   | 74.1  | 90.0  |  79.9  |  81.6     |    75.1     |    64.3   |
| HRNet              | HRNet-w48  | 256 x 192   | 73.3  | 90.0  |  78.7  |  81.0     |    74.4     |    63.4   | 
| HRNet + STIP       | HRNet-w48  | 256 x 192   | 74.8  | 90.8  |  80.1  |  82.0     |    75.7     |    65.0   |

## Installation

The environment can be referred to [README.md](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/blob/master/README.md).

The details about dataset can be referred to [README.md](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation/blob/master/README.md).

Downlaod pretrained weights from [BaidunYun](https://pan.baidu.com/s/186ktb9KvF0Vz425mZuIPrA)(Password: wp30) to [./models](!./models).

#### Testing on CrowdPose dataset using pretrained weights

Testing HRNet
```
python tools/script_test.py \
    --cfg experiments/crowdpose/hrnet/w32_256x192_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pytorch/crowdpose/hrnet_w32_256x192.pth
```

Testing STIP net
```
python tools/script_test.py \
    --cfg experiments/crowdpose/partnet/stipnet_w32_256x192_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pytorch/crowdpose/stipnet_w32_256x192.pth
```

#### Training on CrowdPose dataset

```
python tools/script_train.py \
    --cfg experiments/crowdpose/partnet/stipnet_w32_256x192_adam_lr1e-3.yaml 
```