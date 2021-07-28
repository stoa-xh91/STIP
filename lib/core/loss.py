# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import logging
from torch.nn import functional as F
logger = logging.getLogger(__name__)
class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints

class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)


def make_input(t, requires_grad=False, need_cuda=True):
    inp = torch.autograd.Variable(t, requires_grad=requires_grad)
    inp = inp.sum()
    if need_cuda:
        inp = inp.cuda()
    return inp


class LimbsAwareLosses(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.limbs_criterion = nn.BCELoss(reduction='mean')
        self.limbs_aware_criterion = nn.BCELoss(size_average=True)
        self.limb_loss_factor = cfg.LOSS.LIMBS_LOSS_WEIGHT
        print('limb loss weight:',self.limb_loss_factor)

    def forward(self, predictions, ground_truth):

        limbs_predictions = predictions['limbs']
        limbs_gt = ground_truth['limbs']
        limbs_aware_gt = limbs_gt[0].view(-1)
        limbs_aware_predictions = limbs_predictions[0].view(-1)
        limbs_aware_loss = self.limbs_aware_criterion(limbs_aware_predictions, limbs_aware_gt.float()) * \
                           self.limb_loss_factor[0]
        # print(limbs_predictions[1].size(), limbs_gt[1].size())
        limb_loss = self.limbs_criterion(limbs_predictions[1], limbs_gt[1]) * self.limb_loss_factor[1]

        return {'limb_loss': limb_loss, 'limb_aware_loss': limbs_aware_loss}

class SemsAwareLosses(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.sem_aware_criterion = nn.BCELoss(reduction='mean')
        self.sem_loss_factor = cfg.LOSS.SEM_LOSS_WEIGHT
        print('sem loss weight:',self.sem_loss_factor)

    def forward(self, predictions, ground_truth):
        sems_predictions = predictions['semantics']
        sems_gt = ground_truth['semantics']
        sems_aware_gt = sems_gt[0].view(-1).float()
        all_sems_losses = []

        if isinstance(sems_predictions, list):
            sems_aware_predictions = sems_predictions[0].view(-1)
            sems_aware_loss = self.sem_aware_criterion(sems_aware_predictions, sems_aware_gt) * \
                              self.sem_loss_factor[0]
            all_sems_losses.append(sems_aware_loss)
            for output in sems_predictions[1:]:
                sems_aware_predictions = output.view(-1)
                sems_aware_loss = self.sem_aware_criterion(sems_aware_predictions, sems_aware_gt) * \
                              self.sem_loss_factor[0]
                all_sems_losses.append(sems_aware_loss)
        else:
            sems_aware_predictions = sems_predictions.view(-1)
            sems_aware_loss = self.sem_aware_criterion(sems_aware_predictions, sems_aware_gt) * \
                              self.sem_loss_factor[0]
            all_sems_losses = sems_aware_loss
        return all_sems_losses

class MultiTaskLosses(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.POSE_ON = cfg.MODEL.POSE_ON
        self.SPATIAL_POSE_ON = cfg.MODEL.SPATIAL_POSE_ON
        self.MASK_ON = cfg.MODEL.MASK_ON
        self.SEM_ON = cfg.MODEL.SEM_ON
        self.LIMBS_ON = cfg.MODEL.LIMBS_ON
        num_tasks = 0
        if self.POSE_ON:
            self.pose_criterion = JointsMSELoss(cfg.LOSS.USE_TARGET_WEIGHT)
            self.pose_loss_factor = cfg.LOSS.POSE_LOSS_WEIGHT
            print('heatmap loss weight:', self.pose_loss_factor)

            if self.SPATIAL_POSE_ON:
                self.sp_pose_criterion = JointsMSELoss(cfg.LOSS.USE_TARGET_WEIGHT)

            num_tasks += 1

        if self.MASK_ON:
            self.mask_criterion = nn.BCELoss(size_average=True)
            self.mask_loss_factor = cfg.LOSS.MASK_LOSS_WEIGHT
            num_tasks += 1
            print('mask loss weight:', self.mask_loss_factor)
        if self.SEM_ON:
            self.sem_criterion = SemsAwareLosses(cfg)
            self.sem_loss_factor = cfg.LOSS.SEM_LOSS_WEIGHT
            num_tasks += 1

        if self.LIMBS_ON:
            self.limbs_criterion = nn.BCELoss(reduction='mean')
            self.limb_loss_factor = cfg.LOSS.LIMBS_LOSS_WEIGHT
            num_tasks += 1

        if num_tasks == 0:
            logger.error('At least enable one loss!')
        logger.info("build {} learning tasks".format(num_tasks))

    def forward(self, predictions, ground_truth):

        loss = {}
        total_loss = 0
        if self.POSE_ON:
            pose_predictions = predictions['pose']
            pose_gt = ground_truth['pose']
            all_pose_losses = []
            if isinstance(pose_predictions, list):
                # print(pose_predictions[0].size(), pose_gt[0].size(),pose_gt[1].size())
                pose_loss = self.pose_criterion(pose_predictions[0], pose_gt[0], pose_gt[1]) * self.pose_loss_factor
                all_pose_losses.append(pose_loss)
                total_loss += pose_loss
                for output in pose_predictions[1:]:
                    pose_loss = self.pose_criterion(output, pose_gt[0], pose_gt[1]) * self.pose_loss_factor
                    all_pose_losses.append(pose_loss)
                    total_loss += pose_loss
            else:
                pose_loss = self.pose_criterion(pose_predictions, pose_gt[0], pose_gt[1]) * self.pose_loss_factor
                all_pose_losses = pose_loss

            loss['pose'] = all_pose_losses

            if self.SPATIAL_POSE_ON:
                sp_pose_predictions = predictions['spatial_pose']
                pose_gt = ground_truth['spatial_pose']
                sp_poss_loss = self.sp_pose_criterion(sp_pose_predictions, pose_gt[0], pose_gt[1])
                total_loss += sp_poss_loss
                loss['sp_pose'] = sp_poss_loss

        if self.MASK_ON:
            mask_predictions = predictions['mask']
            mask_gt = ground_truth['mask']
            if isinstance(mask_predictions, list):
                mask_prediction = mask_predictions[0].view(-1)
                mask_gt = mask_gt.view(-1)
                mask_pred = mask_prediction.clone().detach().cpu().numpy()
                m_gt = mask_gt.clone().detach().cpu().numpy()
                if mask_pred.max()>1 or mask_pred.min()<0:
                    print(100*'*')
                    print('prediction not in range')
                if m_gt.max()>1 or m_gt.min()<0:
                    print('ground truth not in range')
                mask_loss = self.mask_criterion(mask_prediction, mask_gt.float())
                mask_loss = mask_loss * self.mask_loss_factor
                for mask_prediction in mask_predictions[1:]:
                    mask_prediction = mask_prediction.view(-1)
                    mask_loss += (self.mask_criterion(mask_prediction, mask_gt.float()) * self.mask_loss_factor)
            else:
                mask_prediction = mask_predictions.view(-1)
                mask_gt = mask_gt.view(-1)
                mask_loss = self.mask_criterion(mask_prediction, mask_gt.float())
                mask_loss = mask_loss * self.mask_loss_factor
            total_loss += mask_loss
            loss['mask'] = mask_loss

        if self.SEM_ON:
            sems_losses = self.sem_criterion(predictions, ground_truth)
            if isinstance(sems_losses, list):
                for s_l in sems_losses:
                    total_loss += s_l
            else:
                total_loss += sems_losses
            loss['semantics'] = sems_losses

        if self.LIMBS_ON:
            limbs_predictions = predictions['limbs']
            limbs_gt = ground_truth['limbs'][1]
            if isinstance(limbs_predictions, list):
                limb_prediction = limbs_predictions[0].view(-1)
                limbs_gt = limbs_gt.view(-1)
                limb_loss = self.limbs_criterion(limb_prediction, limbs_gt.float())
                limb_loss = limb_loss * self.limb_loss_factor[1]
                for limb_prediction in limbs_predictions[1:]:
                    limb_prediction = limb_prediction.view(-1)
                    limb_loss += (self.limbs_criterion(limb_prediction, limbs_gt.float()) * self.limb_loss_factor[1] * 0.5)
            else:
                limb_loss = self.limbs_criterion(limbs_predictions[0], limbs_gt[1]) * self.limb_loss_factor[1]

            loss['limbs'] = limb_loss
            total_loss += limb_loss
        loss['total loss'] = total_loss
        return loss