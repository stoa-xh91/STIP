
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn

from torch.nn import functional as F
from .registry import Registry

from .kernel_generator import SingleHead
POSE_RESNET_REGISTRY = Registry("POSE_NET")
BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(
                in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False
            ),
            # nn.GroupNorm(32, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            # nn.GroupNorm(32, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels):
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            # nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class GlobalAveragePooling2D(nn.Module):
    def __init__(self, output_size):
        super(GlobalAveragePooling2D, self).__init__()
        self.pooler = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        b, c, h, w = x.size()
        x = self.pooler(x)
        x = x.reshape(b, c)
        return x

class FPN(nn.Module):

    def __init__(
        self, in_channels, out_channels
    ):
        super(FPN, self).__init__()
        lateral_convs = []
        lateral_norms = []
        output_convs = []
        output_norms = []

        for idx, in_channels in enumerate(in_channels):

            lateral_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1
            )
            output_conv = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
            lateral_convs.append(lateral_conv)
            lateral_norms.append(nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM))
            output_convs.append(output_conv)
            output_norms.append(nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM))

        self.lateral_convs = nn.ModuleList(lateral_convs)
        self.output_convs = nn.ModuleList(output_convs)
        self.lateral_norms = nn.ModuleList(lateral_norms)
        self.output_norms = nn.ModuleList(output_norms)

    def forward(self, x):

        results = []
        prev_features = F.relu(self.lateral_norms[0](self.lateral_convs[0](x[0])))
        results.append(F.relu(self.output_norms[0](self.output_convs[0](prev_features))))
        for features, lateral_conv, lateral_norm, output_conv, output_norm in zip(
            x[1:], self.lateral_convs[1:], self.lateral_norms[1:], self.output_convs[1:], self.output_norms[1:]
        ):
            top_down_features = F.interpolate(prev_features, scale_factor=2, mode="nearest")
            lateral_features = lateral_norm(lateral_conv(features))
            prev_features = F.relu(lateral_features + top_down_features)
            results.insert(0, output_conv(prev_features))

        return results


@POSE_RESNET_REGISTRY.register()
class PoseResNet(nn.Module):

    def __init__(self, block, layers, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pre_stage_channels = []
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.pre_stage_channels.append(self.inplanes)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.pre_stage_channels.append(self.inplanes)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.pre_stage_channels.append(self.inplanes)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.pre_stage_channels.append(self.inplanes)
        self.pre_stage_channels = self.pre_stage_channels[::-1]
        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)
        x = self.final_layer(x)
        net_outputs = {}
        net_outputs['keypoint'] = [x]
        net_outputs['kernel'] = torch.mean(x ** 2, dim=(1, 2, 3))
        net_outputs['semantics'], _ = torch.max(x.reshape(x.size(0),x.size(1),-1), dim=2)
        return net_outputs

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.normal_(m.weight, std=0.001)
                    # nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)


@POSE_RESNET_REGISTRY.register()
class CPResNet(PoseResNet):

    def __init__(self, block, layers, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        super().__init__(block, layers, cfg, **kwargs)
        pre_stage_channels = self.pre_stage_channels
        print(pre_stage_channels)
        self.FPN = FPN(pre_stage_channels, 128)
        pre_stage_channels = [128, 128, 128, 128]
        self.inplanes = 128
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )
        self.scale_factor = cfg.MODEL.UP_SCALE
        self.ASPP_stage4 = ASPP(pre_stage_channels[0], [1, 1, 2], pre_stage_channels[0])
        self.latent_channels = pre_stage_channels[0]
        self.weight_size = pre_stage_channels[0] * pre_stage_channels[0]

        self.num_outputs = cfg.MODEL.NUM_JOINTS
        coord = cfg.MODEL.ENCODER_NET.COORD
        num_convs = cfg.MODEL.ENCODER_NET.NUM_CONVS

        norm = cfg.MODEL.ENCODER_NET.NORM
        self.num_convs = num_convs
        self.context_head = nn.Sequential(
            *[SingleHead(self.latent_channels, self.latent_channels, 1, coord=False, norm=True),
              GlobalAveragePooling2D(1),
              nn.Linear(self.latent_channels, self.latent_channels, bias=False),
              nn.BatchNorm1d(self.latent_channels, momentum=BN_MOMENTUM),
              nn.ReLU(),
              nn.Linear(self.latent_channels, self.num_outputs),
              nn.Sigmoid()])
        self.kernel_head = ASPP(pre_stage_channels[0], [1, 1, 2], pre_stage_channels[0])
        self.kernel_generator = nn.Sequential(
            *[nn.Conv2d(pre_stage_channels[0], pre_stage_channels[0] * cfg.MODEL.NUM_JOINTS, 1)])
        self.projector = nn.Sequential(nn.Conv2d(2 * self.latent_channels, self.latent_channels, 1, bias=False),
                                       nn.BatchNorm2d(self.latent_channels, momentum=BN_MOMENTUM),
                                       nn.ReLU())

        self.spatial_encoder_head = SingleHead(self.latent_channels + 2 if coord else self.latent_channels,
                                               self.latent_channels,
                                               num_convs,
                                               coord=coord,
                                               norm=norm)
        self.kpt_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )
        self.parts_predictor = nn.Conv2d(
            in_channels=pre_stage_channels[-2],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.mask_predictor = nn.Conv2d(
            in_channels=pre_stage_channels[-1],
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )
    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                print('init linear:',m)
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        features = []
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        features = features[::-1]

        features = self.FPN(features)
        stage2_outputs = features[0]
        stage3_outputs = F.interpolate(features[1], size=stage2_outputs.shape[-2:], mode="bilinear",
                                            align_corners=False)

        context_scores = self.context_head(features[-1])
        init_features = self.deconv_layers(features[-1])
        # predictions

        if self.scale_factor > 1:
            stage2_outputs = F.interpolate(stage2_outputs, scale_factor=self.scale_factor, mode="bilinear",
                                           align_corners=False)
            stage3_outputs = F.interpolate(stage3_outputs, scale_factor=self.scale_factor, mode="bilinear",
                                           align_corners=False)
            kpt_features = F.interpolate(init_features, scale_factor=self.scale_factor, mode="bilinear",
                                         align_corners=False)

        pose_score = self.kpt_layer(kpt_features)
        parts_score = self.parts_predictor(stage3_outputs)
        mask_score = F.sigmoid(self.mask_predictor(stage2_outputs))
        net_outputs = {'keypoint': [pose_score], 'sp_keypoint': parts_score, 'mask': [mask_score],
                       'semantics': [context_scores]}
        ######
        num_inst = pose_score.size(0)
        num_kpts = pose_score.size(1)

        instance_aware_params = self.context_head[5].weight.squeeze() * 1.
        pose_score_reshaped = F.interpolate(pose_score, size=init_features.shape[-2:], mode="bilinear",
                                            align_corners=False)
        pose_score_reshaped = pose_score_reshaped.reshape(num_inst, num_kpts, -1).permute(0, 2, 1).contiguous()
        spatial_instance_embs = torch.matmul(pose_score_reshaped, instance_aware_params).permute(0, 2, 1).contiguous().reshape(init_features.shape)
        #
        instance_kernel_embs = self.kernel_head(spatial_instance_embs)
        instance_aware_embs = self.spatial_encoder_head(spatial_instance_embs)
        instance_aware_embs = torch.cat([instance_aware_embs, init_features], dim=1)
        instance_aware_embs = self.projector(instance_aware_embs)
        instance_aware_embs = self.ASPP_stage4(instance_aware_embs)
        if self.scale_factor > 1:
            instance_aware_embs = F.interpolate(instance_aware_embs, scale_factor=self.scale_factor, mode="bilinear",
                                                align_corners=False)
            instance_kernel_embs = F.interpolate(instance_kernel_embs, scale_factor=self.scale_factor, mode="bilinear",
                                                 align_corners=False)
        num_inst, num_feats, h, w = instance_aware_embs.size()

        kernels = self.kernel_generator(instance_kernel_embs)
        # kernel_loss = 0
        kernels = kernels.reshape(num_inst, num_kpts, num_feats, h, w)

        feat_reshaped = instance_aware_embs.reshape(num_inst, 1, num_feats, h, w)

        instance_scores = torch.mean(kernels * feat_reshaped, dim=2)

        instance_scores = instance_scores.reshape(num_inst, num_kpts, h, w)

        net_outputs['keypoint'] = [instance_scores, pose_score]
        net_outputs['kernel'] = torch.mean(kernels ** 2, dim=(1, 2, 3, 4))
        return net_outputs


@POSE_RESNET_REGISTRY.register()
class IADyResNet(PoseResNet):

    def __init__(self, block, layers, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        super().__init__(block, layers, cfg, **kwargs)
        pre_stage_channels = self.pre_stage_channels
        self.FPN = FPN(pre_stage_channels, 128)
        pre_stage_channels = [128, 128, 128, 128]
        self.inplanes = 128
        # self.deconv_layers = self._make_deconv_layer(
        #     extra.NUM_DECONV_LAYERS,
        #     extra.NUM_DECONV_FILTERS,
        #     extra.NUM_DECONV_KERNELS,
        # )
        self.deconv_layers = None
        self.final_layer = None
        self.scale_factor = cfg.MODEL.UP_SCALE
        self.ASPP_stage4 = SingleHead(pre_stage_channels[0], pre_stage_channels[0], 1, coord=False, norm=True)
        self.latent_channels = pre_stage_channels[0]
        self.weight_size = pre_stage_channels[0] * pre_stage_channels[0]

        self.num_outputs = cfg.MODEL.NUM_JOINTS
        coord = cfg.MODEL.ENCODER_NET.COORD
        num_convs = cfg.MODEL.ENCODER_NET.NUM_CONVS

        norm = cfg.MODEL.ENCODER_NET.NORM
        self.num_convs = num_convs
        self.context_head = nn.Sequential(
            *[GlobalAveragePooling2D(1),
              nn.Linear(self.latent_channels, self.latent_channels, bias=False),
              nn.BatchNorm1d(self.latent_channels, momentum=BN_MOMENTUM),
              nn.ReLU(),
              nn.Linear(self.latent_channels, self.num_outputs),
              nn.Sigmoid()])
        self.kernel_head = SingleHead(pre_stage_channels[0], pre_stage_channels[0], 1, coord=False, norm=True)
        self.kernel_generator = nn.Sequential(
            *[nn.Conv2d(pre_stage_channels[0], pre_stage_channels[0] * cfg.MODEL.NUM_JOINTS, 1)])
        self.projector = nn.Sequential(nn.Conv2d(2 * self.latent_channels, self.latent_channels, 1, bias=False),
                                       nn.BatchNorm2d(self.latent_channels, momentum=BN_MOMENTUM),
                                       nn.ReLU())

        self.spatial_encoder_head = SingleHead(self.latent_channels + 2 if coord else self.latent_channels,
                                               self.latent_channels,
                                               num_convs,
                                               coord=coord,
                                               norm=norm)
        self.kpt_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )
        self.parts_predictor = nn.Conv2d(
            in_channels=pre_stage_channels[-2],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.mask_predictor = nn.Conv2d(
            in_channels=pre_stage_channels[-1],
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0
        )
    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                print('init linear:',m)
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        features = []
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        features = features[::-1]

        features = self.FPN(features)

        stage3_outputs = F.interpolate(features[1], size=features[0].shape[-2:], mode="bilinear",
                                            align_corners=False)

        context_scores = self.context_head(features[-1])
        init_features = features[0]
        # predictions

        if self.scale_factor > 1:
            stage3_outputs = F.interpolate(stage3_outputs, scale_factor=self.scale_factor, mode="bilinear",
                                           align_corners=False)
            kpt_features = F.interpolate(init_features, scale_factor=self.scale_factor, mode="bilinear",
                                         align_corners=False)

        pose_score = self.kpt_layer(kpt_features)
        parts_score = self.parts_predictor(stage3_outputs)
        net_outputs = {'keypoint': [pose_score], 'sp_keypoint': parts_score, 'semantics': [context_scores]}
        ######
        num_inst = pose_score.size(0)
        num_kpts = pose_score.size(1)

        instance_aware_params = self.context_head[4].weight.squeeze() * 1.
        pose_score_reshaped = F.interpolate(pose_score, size=init_features.shape[-2:], mode="bilinear",
                                            align_corners=False)
        pose_score_reshaped = pose_score_reshaped.reshape(num_inst, num_kpts, -1).permute(0, 2, 1).contiguous()
        spatial_instance_embs = torch.matmul(pose_score_reshaped, instance_aware_params).permute(0, 2, 1).contiguous().reshape(init_features.shape)
        #
        instance_kernel_embs = self.kernel_head(spatial_instance_embs)
        instance_aware_embs = self.spatial_encoder_head(spatial_instance_embs)
        instance_aware_embs = torch.cat([instance_aware_embs, init_features], dim=1)
        instance_aware_embs = self.projector(instance_aware_embs)
        instance_aware_embs = self.ASPP_stage4(instance_aware_embs)
        if self.scale_factor > 1:
            instance_aware_embs = F.interpolate(instance_aware_embs, scale_factor=self.scale_factor, mode="bilinear",
                                                align_corners=False)
            instance_kernel_embs = F.interpolate(instance_kernel_embs, scale_factor=self.scale_factor, mode="bilinear",
                                                 align_corners=False)
        num_inst, num_feats, h, w = instance_aware_embs.size()

        kernels = self.kernel_generator(instance_kernel_embs)
        # kernel_loss = 0
        kernels = kernels.reshape(num_inst, num_kpts, num_feats, h, w)

        feat_reshaped = instance_aware_embs.reshape(num_inst, 1, num_feats, h, w)

        instance_scores = torch.sum(kernels * feat_reshaped, dim=2)

        instance_scores = instance_scores.reshape(num_inst, num_kpts, h, w)

        net_outputs['keypoint'] = [instance_scores, pose_score]
        net_outputs['kernel'] = torch.mean(kernels ** 2, dim=(1, 2, 3, 4))
        return net_outputs

@POSE_RESNET_REGISTRY.register()
class GPResNet(PoseResNet):

    def __init__(self, block, layers, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        super().__init__(block, layers, cfg, **kwargs)
        pre_stage_channels = self.pre_stage_channels
        print(pre_stage_channels)
        self.FPN = FPN(pre_stage_channels, 128)
        pre_stage_channels = [128, 128, 128, 128]
        self.inplanes = 128
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )
        self.scale_factor = cfg.MODEL.UP_SCALE

        self.latent_channels = pre_stage_channels[0]
        self.weight_size = pre_stage_channels[0] * pre_stage_channels[0]

        self.num_outputs = cfg.MODEL.NUM_JOINTS
        coord = cfg.MODEL.ENCODER_NET.COORD
        num_convs = cfg.MODEL.ENCODER_NET.NUM_CONVS

        norm = cfg.MODEL.ENCODER_NET.NORM
        self.num_convs = num_convs
        self.context_head = nn.Sequential(
            *[SingleHead(self.latent_channels, self.latent_channels, 1, coord=False, norm=True),
              GlobalAveragePooling2D(1),
              nn.Linear(self.latent_channels, self.latent_channels, bias=False),
              nn.BatchNorm1d(self.latent_channels, momentum=BN_MOMENTUM),
              nn.ReLU(),
              nn.Linear(self.latent_channels, self.num_outputs),
              nn.Sigmoid()])
        self.kernel_head = SingleHead(pre_stage_channels[0], pre_stage_channels[0], 2, coord=True, norm=True) #ASPP(pre_stage_channels[0], [1, 1, 2], pre_stage_channels[0])
        self.kernel_generator = nn.Sequential(
            *[nn.Conv2d(pre_stage_channels[0], pre_stage_channels[0] * cfg.MODEL.NUM_JOINTS, 1)])
        self.projector = nn.Sequential(nn.Conv2d(2 * self.latent_channels, self.latent_channels, 1, bias=False),
                                       nn.BatchNorm2d(self.latent_channels, momentum=BN_MOMENTUM),
                                       nn.ReLU())

        self.spatial_encoder_head = SingleHead(self.latent_channels + 2 if coord else self.latent_channels,
                                               self.latent_channels,
                                               num_convs,
                                               coord=coord,
                                               norm=norm)
        self.refine_head = SingleHead(self.latent_channels + 2 if coord else self.latent_channels,
                                      self.latent_channels,
                                      num_convs,
                                      coord=coord,
                                      norm=norm)
        self.kpt_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )
        self.parts_predictor = nn.Conv2d(
            in_channels=pre_stage_channels[-2],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.mask_predictor = nn.Conv2d(
            in_channels=pre_stage_channels[-1],
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                print('init linear:',m)
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        features = []
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        features = features[::-1]

        features = self.FPN(features)
        stage2_outputs = features[0]
        stage3_outputs = F.interpolate(features[1], size=stage2_outputs.shape[-2:], mode="bilinear",
                                            align_corners=False)

        context_scores = self.context_head(features[-1])
        init_features = self.deconv_layers(features[-1])
        # predictions

        if self.scale_factor > 1:
            stage2_outputs = F.interpolate(stage2_outputs, scale_factor=self.scale_factor, mode="bilinear",
                                           align_corners=False)
            stage3_outputs = F.interpolate(stage3_outputs, scale_factor=self.scale_factor, mode="bilinear",
                                           align_corners=False)
            kpt_features = F.interpolate(init_features, scale_factor=self.scale_factor, mode="bilinear",
                                         align_corners=False)

        pose_score = self.kpt_layer(kpt_features)
        parts_score = self.parts_predictor(stage3_outputs)
        mask_score = F.sigmoid(self.mask_predictor(stage2_outputs))
        net_outputs = {'keypoint': [pose_score], 'sp_keypoint': parts_score, 'mask': [mask_score],
                       'semantics': [context_scores]}
        ######
        num_inst = pose_score.size(0)
        num_kpts = pose_score.size(1)

        instance_aware_params = self.context_head[5].weight.squeeze() * 1.
        pose_score_reshaped = F.interpolate(pose_score, size=init_features.shape[-2:], mode="bilinear",
                                            align_corners=False)
        pose_score_reshaped = pose_score_reshaped.reshape(num_inst, num_kpts, -1).permute(0, 2, 1).contiguous()
        spatial_instance_embs = torch.matmul(pose_score_reshaped, instance_aware_params).permute(0, 2, 1).contiguous().reshape(init_features.shape)
        #
        instance_kernel_embs = self.kernel_head(spatial_instance_embs)
        instance_aware_embs = self.spatial_encoder_head(spatial_instance_embs)
        instance_aware_embs = torch.cat([instance_aware_embs, init_features], dim=1)
        instance_aware_embs = self.projector(instance_aware_embs)
        instance_aware_embs = self.ASPP_stage4(instance_aware_embs)
        if self.scale_factor > 1:
            instance_aware_embs = F.interpolate(instance_aware_embs, scale_factor=self.scale_factor, mode="bilinear",
                                                align_corners=False)
            instance_kernel_embs = F.interpolate(instance_kernel_embs, scale_factor=self.scale_factor, mode="bilinear",
                                                 align_corners=False)
        num_inst, num_feats, h, w = instance_aware_embs.size()

        kernels = self.kernel_generator(instance_kernel_embs)
        # kernel_loss = 0
        kernels = kernels.reshape(num_inst, num_kpts, num_feats, h, w)

        instance_aware_embs = self.refine_head(instance_aware_embs)
        feat_reshaped = instance_aware_embs.reshape(num_inst, 1, num_feats, h, w)

        instance_scores = torch.mean(kernels * feat_reshaped, dim=2)

        instance_scores = instance_scores.reshape(num_inst, num_kpts, h, w)

        net_outputs['keypoint'] = [instance_scores, pose_score]
        net_outputs['kernel'] = torch.mean(kernels ** 2, dim=(1, 2, 3, 4))
        return net_outputs

resnet_spec = {
    18: (BasicBlock, [2, 2, 2, 2]),
    34: (BasicBlock, [3, 4, 6, 3]),
    50: (Bottleneck, [3, 4, 6, 3]),
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3])
}


def get_pose_net(cfg, is_train, **kwargs):
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS

    block_class, layers = resnet_spec[num_layers]

    # model = PoseResNet(block_class, layers, cfg, **kwargs)
    POSE_NAME = cfg.MODEL.POSE_NAME
    logger.info('=> construct pose model {}'.format(POSE_NAME))
    model = POSE_RESNET_REGISTRY.get(POSE_NAME)(block_class, layers, cfg, **kwargs)
    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model
