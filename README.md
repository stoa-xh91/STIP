# Semantic-aware Transfer with Instance-adaptive Parsing for Crowded Scenes Pose Estimation
This is an official implementation of Semantic-aware Transfer with Instance-adaptive Parsing for Crowded Scenes Pose Estimation

This repo is built on [deep-high-resolution-net](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch).

![](/figures/framework.png)
![](/figures/vis_model.png)
## Main Results
### Results on CrowdPose test set
| Method             | Backbone   | Input size  |  AP   | Ap@50 | AP@75 | AP (Easy) | AP (Medium) | AP (Hard) |
|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|
| HRNet              | HRNet-w32  | 256 x 192   | 71.7  | 89.8  |  76.9  |  79.6     |    72.7     |    61.5   | 
| HRNet + STIP       | HRNet-w32  | 256 x 192   | 74.1  | 90.0  |  79.9  |  81.6     |    75.1     |    64.3   |
| HRNet              | HRNet-w48  | 256 x 192   | 73.3  | 90.0  |  78.7  |  81.0     |    74.4     |    63.4   | 
| HRNet + STIP       | HRNet-w48  | 256 x 192   | 74.8  | 90.8  |  80.1  |  82.0     |    75.7     |    65.0   |

### Results on COCO val set
| Method             | Backbone   | Input size  |  AP   | Ap@50 | AP@75 | AP (Small) | AP (Medium) | AP (Large) |
|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|
| HRNet              | HRNet-w32  | 256 x 192   | 74.4  | 90.5  |  81.9  |  70.8     |    81.0     |    79.8   | 
| HRNet + STIP       | HRNet-w32  | 256 x 192   | 75.8  | 90.3  |  82.4  |  72.1     |    82.4     |    80.8   |
| HRNet              | HRNet-w48  | 256 x 192   | 75.1  | 90.6  |  82.2  |  71.5     |    81.8     |    80.4   | 
| HRNet + STIP       | HRNet-w48  | 256 x 192   | 76.0  | 90.4  |  82.2  |  72.2     |    82.9     |    81.1   |

## Installation

The environment can be referred to [README.md](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/blob/master/README.md).

The details about dataset can be referred to [README.md](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation/blob/master/README.md).

Downlaod pretrained weights from [BaidunYun](https://pan.baidu.com/s/1BavoWmV6aOuGiJzbSsSaMw?pwd=cr1x)(Password: cr1x) to [./models](!./models).

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

## Citation
If you find this work or code is helpful in your research, please cite:
```
@inproceedings{pose:stip,
	author={Xuanhan Wang and Lianli Gao and Yan Dai and Yixuan Zhou and Jingkuan Song},
	title={Semantic-aware Transfer with Instance-adaptive Parsing for Crowded Scenes Pose Estimation},
	pages={686--694},
	booktitle = {ACM MM},
	year={2021}
}
```