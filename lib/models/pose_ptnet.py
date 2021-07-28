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
BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)
POSE_NET_REGISTRY = Registry("POSE_NET")

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

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

class GlobalAveragePooling2D(nn.Module):
    def __init__(self, output_size):
        super(GlobalAveragePooling2D, self).__init__()
        self.pooler = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        b, c, h, w = x.size()
        x = self.pooler(x)
        x = x.reshape(b, c)
        return x

class GlobalMaxPooling2D(nn.Module):
    def __init__(self, output_size):
        super(GlobalMaxPooling2D, self).__init__()
        self.pooler = nn.AdaptiveMaxPool2d(output_size)

    def forward(self, x):
        b,c = x.size()[:2]
        x = self.pooler(x)
        x = x.view(b, c)
        return x

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):

            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )
            # if i == 0:
            #     branches.append(
            #         self._make_one_branch(i, block, num_blocks, num_channels)
            #     )
            # else:
            #     branches.append(
            #         self._make_one_branch(i, BasicBlock, num_blocks, num_channels)
            #     )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck,
}

modules_dict = {
    'BASIC': HighResolutionModule,
}

@POSE_NET_REGISTRY.register()
class HRNet(nn.Module):
    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        super(HRNet, self).__init__()
        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = cfg['MODEL']['EXTRA']['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = cfg['MODEL']['EXTRA']['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = cfg['MODEL']['EXTRA']['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=False)
        self.pre_stage_channels = pre_stage_channels
        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )
        self.pretrained_layers = cfg['MODEL']['EXTRA']['PRETRAINED_LAYERS']

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1, bias=False
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False
                            ),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels
    def _forward_backbone(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        stage2_y_list = y_list
        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        stage3_y_list = y_list
        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        return y_list

    def forward(self, x):
        y_list = self._forward_backbone(x)
        x = self.final_layer(y_list[0])
        net_outputs = {}
        net_outputs['keypoint'] = [x]
        net_outputs['kernel'] = torch.mean(x ** 2, dim=(1, 2, 3))
        net_outputs['semantics'], _ = torch.max(x.reshape(x.size(0), x.size(1), -1), dim=2)
        return net_outputs

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
            elif isinstance(m, nn.GroupNorm):
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

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] is '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))

@POSE_NET_REGISTRY.register()
class GPEmbNet(HRNet):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        super().__init__(cfg, **kwargs)
        pre_stage_channels = self.pre_stage_channels
        self.scale_factor = cfg.MODEL.UP_SCALE
        self.ASPP_stage4 = ASPP(pre_stage_channels[0], [1, 1, 2], pre_stage_channels[0])
        self.latent_channels = pre_stage_channels[0]
        self.weight_size = pre_stage_channels[0]*pre_stage_channels[0]

        self.num_outputs = cfg.MODEL.NUM_JOINTS
        coord = cfg.MODEL.ENCODER_NET.COORD
        num_convs = cfg.MODEL.ENCODER_NET.NUM_CONVS

        norm = cfg.MODEL.ENCODER_NET.NORM
        self.num_convs = num_convs
        self.context_head = nn.Sequential(*[SingleHead(self.latent_channels, self.latent_channels, 1, coord=False, norm=True),
                                        GlobalAveragePooling2D(1),
                                        nn.Linear(self.latent_channels, self.latent_channels, bias=False),
                                        nn.BatchNorm1d(self.latent_channels, momentum=BN_MOMENTUM),
                                        nn.ReLU()])
        self.context_predictor = nn.Sequential(*[nn.Linear(self.latent_channels, self.num_outputs),
                                                nn.Sigmoid()])

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
            in_channels=pre_stage_channels[0],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.mask_predictor = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.pretrained_layers = cfg['MODEL']['EXTRA']['PRETRAINED_LAYERS']
    def _forward_relation(self, global_context, local_context):
        num_inst, num_feat, feat_h, feat_w = local_context.size()
        local_context_reshaped = local_context.view(num_inst, num_feat, feat_h*feat_w)
        global_context = global_context.view(num_inst, 1, num_feat)
        relation_scores = torch.bmm(global_context, local_context_reshaped)
        relation_scores = relation_scores.view(num_inst, 1, feat_h, feat_w)
        relation_scores = F.sigmoid(relation_scores)
        return relation_scores

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        stage2_outputs = y_list[0]*1.
        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        stage3_outputs = y_list[0]*1.
        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        # predictions
        context_embeddings = self.context_head(y_list[0])
        context_scores = self.context_predictor(context_embeddings)
        init_features = y_list[0] * 1.
        rel_scores = self._forward_relation(context_embeddings, init_features)

        if self.scale_factor > 1:
            stage2_outputs = F.interpolate(stage2_outputs, scale_factor=self.scale_factor, mode="bilinear",
                                            align_corners=False)
            stage3_outputs = F.interpolate(stage3_outputs, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
            kpt_features = F.interpolate(init_features, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        else:
            kpt_features = init_features

        pose_score = self.kpt_layer(kpt_features)
        parts_score = self.parts_predictor(stage3_outputs)
        mask_score = F.sigmoid(self.mask_predictor(stage2_outputs))
        net_outputs = {'keypoint': [pose_score], 'sp_keypoint': parts_score, 'mask': [mask_score], 'semantics':[context_scores]}
        ######
        num_inst = pose_score.size(0)
        num_kpts = pose_score.size(1)

        instance_aware_params = self.context_predictor[0].weight.squeeze()*1.
        pose_score_reshaped = F.interpolate(pose_score, size=init_features.shape[-2:], mode="bilinear",
                                            align_corners=False)
        pose_score_reshaped = pose_score_reshaped * rel_scores
        pose_score_reshaped = pose_score_reshaped.reshape(num_inst, num_kpts, -1).permute(0, 2, 1).contiguous()
        spatial_instance_embs = torch.matmul(pose_score_reshaped, instance_aware_params).permute(0, 2, 1).contiguous().reshape(init_features.shape)
        #
        instance_aware_embs = self.spatial_encoder_head(spatial_instance_embs)
        instance_aware_embs = torch.cat([instance_aware_embs, init_features], dim=1)
        instance_aware_embs = self.projector(instance_aware_embs)
        instance_aware_embs = self.ASPP_stage4(instance_aware_embs)
        if self.scale_factor > 1:
            instance_aware_embs = F.interpolate(instance_aware_embs, scale_factor=self.scale_factor, mode="bilinear",
                                     align_corners=False)
        num_inst, num_feats, h, w = instance_aware_embs.size()

        # kernel_loss = 0
        instance_aware_embs = self.refine_head(instance_aware_embs)

        instance_scores = self.final_layer(instance_aware_embs)

        net_outputs['keypoint'] = [instance_scores, pose_score]
        net_outputs['kernel'] = torch.mean(instance_scores**2, dim=(1, 2, 3))
        return net_outputs

@POSE_NET_REGISTRY.register()
class STIPNet(HRNet):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        super().__init__(cfg, **kwargs)
        pre_stage_channels = self.pre_stage_channels
        self.scale_factor = cfg.MODEL.UP_SCALE
        self.ASPP_stage4 = ASPP(pre_stage_channels[0], [1, 1, 2], pre_stage_channels[0])
        self.latent_channels = pre_stage_channels[0]
        self.weight_size = pre_stage_channels[0]*pre_stage_channels[0]

        self.num_outputs = cfg.MODEL.NUM_JOINTS
        coord = cfg.MODEL.ENCODER_NET.COORD
        num_convs = cfg.MODEL.ENCODER_NET.NUM_CONVS

        norm = cfg.MODEL.ENCODER_NET.NORM
        self.num_convs = num_convs
        self.context_head = nn.Sequential(*[SingleHead(self.latent_channels, self.latent_channels, 1, coord=False, norm=True),
                                        GlobalAveragePooling2D(1),
                                        nn.Linear(self.latent_channels, self.latent_channels, bias=False),
                                        nn.BatchNorm1d(self.latent_channels, momentum=BN_MOMENTUM),
                                        nn.ReLU()]
                                        )
        self.context_predictor = nn.Sequential(*[nn.Linear(self.latent_channels, self.num_outputs),
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
            in_channels=pre_stage_channels[0],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.mask_predictor = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.pretrained_layers = cfg['MODEL']['EXTRA']['PRETRAINED_LAYERS']
    def _forward_relation(self, global_context, local_context):
        num_inst, num_feat, feat_h, feat_w = local_context.size()
        local_context_reshaped = local_context.view(num_inst, num_feat, feat_h*feat_w)
        global_context = global_context.view(num_inst, 1, num_feat)
        relation_scores = torch.bmm(global_context, local_context_reshaped)
        relation_scores = relation_scores.view(num_inst, 1, feat_h, feat_w)
        relation_scores = F.sigmoid(relation_scores)
        return relation_scores

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        stage2_outputs = y_list[0]*1.
        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        stage3_outputs = y_list[0]*1.
        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        # predictions
        context_embeddings = self.context_head(y_list[0])
        context_scores = self.context_predictor(context_embeddings)
        init_features = y_list[0] * 1.
        rel_scores = self._forward_relation(context_embeddings, init_features)

        if self.scale_factor > 1:
            stage2_outputs = F.interpolate(stage2_outputs, scale_factor=self.scale_factor, mode="bilinear",
                                            align_corners=False)
            stage3_outputs = F.interpolate(stage3_outputs, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
            kpt_features = F.interpolate(init_features, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
            up_rel_scores = F.interpolate(rel_scores, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)

        pose_score = self.kpt_layer(kpt_features)
        parts_score = self.parts_predictor(stage3_outputs)
        mask_score = F.sigmoid(self.mask_predictor(stage2_outputs))
        net_outputs = {'keypoint': [pose_score], 'sp_keypoint': parts_score, 'mask': [mask_score,up_rel_scores], 'semantics':[context_scores]}
        ######
        num_inst = pose_score.size(0)
        num_kpts = pose_score.size(1)

        instance_aware_params = self.context_predictor[0].weight.squeeze()*1.
        pose_score_reshaped = F.interpolate(pose_score, size=init_features.shape[-2:], mode="bilinear",
                                            align_corners=False)
        pose_score_reshaped = pose_score_reshaped * rel_scores
        transfer_matrix = F.interpolate(pose_score_reshaped, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        net_outputs['mask'] += [transfer_matrix]
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
        kernels = kernels.reshape(num_inst, num_kpts, num_feats, h, w)
        instance_aware_embs = self.refine_head(instance_aware_embs)
        feat_reshaped = instance_aware_embs.reshape(num_inst, 1, num_feats, h, w)

        instance_scores = torch.mean(kernels * feat_reshaped, dim=2)

        instance_scores = instance_scores.reshape(num_inst, num_kpts, h, w)
        kernel_maps = kernels.mean(dim=2)
        net_outputs['keypoint'] = [instance_scores, pose_score, kernel_maps]
        net_outputs['kernel'] = torch.mean(kernels**2, dim=(1, 2, 3, 4))
        return net_outputs


def get_pose_net(cfg, is_train, **kwargs):
    POSE_NAME = cfg.MODEL.POSE_NAME
    logger.info('=> construct pose model {}'.format(POSE_NAME))
    model = POSE_NET_REGISTRY.get(POSE_NAME)(cfg, **kwargs)

    if is_train or cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model