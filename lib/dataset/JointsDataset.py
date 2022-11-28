# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import math
from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints
from pycocotools import mask as mask_util

logger = logging.getLogger(__name__)


class JointsDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.target_type = cfg.MODEL.TARGET_TYPE
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        self.joints_weight = 1

        self.transform = transform
        self.db = []

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.is_train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                )

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1

        # trans = self.get_warpmatrix(r, c * 2.0, self.image_size - 1.0, s)
        trans = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)
        # cv2.imwrite('img.jpg',input)
        if self.transform:
            input = self.transform(input)

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        target, target_weight = self.generate_target(joints, joints_vis)
        # cv2.imwrite('heatmap.jpg',np.max(target,axis=0)*255)
        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score,
            'pose': target,
            'pose_weight': target_weight
        }

        return input, target, target_weight, meta
        # return input, meta

    def get_warpmatrix(self, theta, size_input, size_dst, size_target):
        '''

        :param theta: angle
        :param size_input:[w,h]
        :param size_dst: [w,h]
        :param size_target: [w,h]/200.0
        :return:
        '''
        size_target = size_target * 200.0
        theta = theta / 180.0 * math.pi
        matrix = np.zeros((2, 3), dtype=np.float32)
        scale_x = size_target[0] / size_dst[0]
        scale_y = size_target[1] / size_dst[1]
        matrix[0, 0] = math.cos(theta) * scale_x
        matrix[0, 1] = math.sin(theta) * scale_y
        matrix[0, 2] = -0.5 * size_target[0] * math.cos(theta) - 0.5 * size_target[1] * math.sin(theta) + 0.5 * \
                       size_input[0]
        matrix[1, 0] = -math.sin(theta) * scale_x
        matrix[1, 1] = math.cos(theta) * scale_y
        matrix[1, 2] = 0.5 * size_target[0] * math.sin(theta) - 0.5 * size_target[1] * math.cos(theta) + 0.5 * \
                       size_input[1]
        return matrix
    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
            ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight

class PartsDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.target_type = cfg.MODEL.TARGET_TYPE
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        self.MASK_ON = cfg.MODEL.MASK_ON
        self.SEM_ON = cfg.MODEL.SEM_ON
        self.LIMB_ON = cfg.MODEL.LIMBS_ON
        # self.PART_RCNN_ON = cfg.MODEL.PARTRCNN_ON
        self.SPATIAL_POSE_ON = cfg.MODEL.SPATIAL_POSE_ON

        self.joints_weight = 1
        self.up_scale_factor = cfg.MODEL.UP_SCALE
        self.transform = transform
        self.db = []

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale

    def __len__(self, ):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        
        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']
        if 'interference' in db_rec.keys():
            interference_joints = db_rec['interference']
            interference_joints_vis = db_rec['interference_vis']
            interference_masks = db_rec['interference_segms']
        else:
            interference_joints = [joints]
            interference_joints_vis = [joints_vis]
            interference_masks = [db_rec['segms']]

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.MASK_ON:
            seg_poly = db_rec['segms']
            if len(seg_poly) != 0:
                mask = self.polys_to_mask(seg_poly, data_numpy.shape[0], data_numpy.shape[1])
            else:
                mask = np.zeros((data_numpy.shape[0], data_numpy.shape[1]), dtype=np.float32)
            inter_masks = np.zeros((data_numpy.shape[0], data_numpy.shape[1]), dtype=np.float32)
            for i in range(len(interference_masks)):
                seg_poly = interference_masks[i]
                if len(seg_poly) != 0:
                    inter_mask = self.polys_to_mask(seg_poly, data_numpy.shape[0], data_numpy.shape[1])
                    inter_masks = np.maximum(inter_masks, inter_mask)
        else:
            mask = np.zeros((data_numpy.shape[0], data_numpy.shape[1]), dtype=np.float32)
            inter_masks = np.zeros((data_numpy.shape[0], data_numpy.shape[1]), dtype=np.float32)

        if self.is_train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body \
                    and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                )

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1
                for i in range(len(interference_joints)):
                    interference_joints[i], interference_joints_vis[i] = fliplr_joints(
                        interference_joints[i], interference_joints_vis[i], data_numpy.shape[1], self.flip_pairs)
                if self.MASK_ON:
                    mask = mask[:, ::-1]
                    inter_masks = inter_masks[:, ::-1]

        trans = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)
        # cv2.imwrite('img.jpg', input)
        if self.MASK_ON:
            mask = cv2.warpAffine(
                mask,
                trans,
                (int(self.image_size[0]), int(self.image_size[1])),
                flags=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (int(self.heatmap_size[0]), int(self.heatmap_size[1])))
            mask = (mask > 0.5).astype(np.float32)
            #############
            inter_masks = cv2.warpAffine(
                inter_masks,
                trans,
                (int(self.image_size[0]), int(self.image_size[1])),
                flags=cv2.INTER_LINEAR)
            inter_masks = (inter_masks > 0.5).astype(np.float32)
        if self.transform:
            input = self.transform(input)

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        target, target_weight = self.generate_target(joints, joints_vis)

        inter_target = np.zeros_like(target)
        inter_target_weight = np.zeros_like(target_weight)
        for i in range(len(interference_joints)):
            inter_joints = interference_joints[i].copy()
            inter_joints_vis = interference_joints_vis[i].copy()
            for j in range(self.num_joints):
                if inter_joints_vis[j, 0] > 0.0:
                    inter_joints[j, 0:2] = affine_transform(inter_joints[j, 0:2], trans)
            _inter_target, _inter_target_weight = self.generate_target(inter_joints, inter_joints_vis)

            inter_target = np.maximum(inter_target, _inter_target)
            inter_target_weight = np.maximum(inter_target_weight, _inter_target_weight)

        all_ins_target = np.maximum(0.5 * inter_target, target)
        all_ins_target_weight = np.maximum(inter_target_weight, target_weight)

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score,
            'pose': target,
            'pose_weight': target_weight,
        }
        # if self.PART_RCNN_ON:
        #     if self.up_scale_factor > 1:
        #         self.heatmap_size *= 2
        #         self.sigma += 1
        #         part_target, part_target_weight = self.generate_part_target(joints, joints_vis)
        #         self.heatmap_size = self.heatmap_size // 2
        #         self.sigma -= 1
        #     else:
        #         part_target, part_target_weight = self.generate_part_target(joints, joints_vis)
        #     part_target_hm = part_target[0::3, :, :]
        #     part_target_x = part_target[1::3, :, :]
        #     part_target_y = part_target[2::3, :, :]
        #     # part_target_degree_x = part_target[3::5, :, :]
        #     # part_target_degree_y = part_target[4::5, :, :]
        #     part_target = np.concatenate([part_target_hm[:, None], part_target_x[:, None], part_target_y[:, None]], axis=1)
        #     # part_target = np.concatenate([part_target_x[:, None], part_target_y[:, None], part_target_degree_x[:, None],part_target_degree_y[:, None]], axis=1)
        #     spatial_size = part_target.shape[2:]
        #     part_target = np.reshape(part_target,(self.num_joints * 3, spatial_size[0], spatial_size[1]))
        #     meta['part_rcnn'] = part_target

        limbs_target, limbs_vis = self.generate_limbs_target(joints, (target_weight > 0).astype(np.float32))
        for conn_id, conn in enumerate(self.connection_rules):
            kpt1_hm, kpt2_hm = target[conn[0]], target[conn[1]]
            limbs_target[conn_id] = np.maximum(limbs_target[conn_id], np.maximum(kpt1_hm, kpt2_hm))
        if self.LIMB_ON:
            limbs_target = (limbs_target > 0.9).astype(np.float32)
            meta['limbs'] = [limbs_vis.astype(np.float32), limbs_target]
        if self.SEM_ON:
            sem_labels = (target_weight>0).astype(np.float32)
            sem_labels = np.reshape(sem_labels, (-1,))
            meta['semantics'] = [sem_labels]
        if self.MASK_ON:
            mask_target = np.max(target, axis=0)
            mask_target = (mask_target > 0.3).astype(np.float32)
            meta['masks'] = mask_target

        if self.SPATIAL_POSE_ON:
            meta['sp_keypoint'] = [all_ins_target, all_ins_target_weight]

        if self.up_scale_factor > 1:
            self.heatmap_size *= 2
            self.sigma += 1
            target, target_weight = self.generate_target(joints, joints_vis)
            inter_target = np.zeros_like(target)
            inter_target_weight = np.zeros_like(target_weight)
            for i in range(len(interference_joints)):
                inter_joints = interference_joints[i].copy()
                inter_joints_vis = interference_joints_vis[i].copy()
                for j in range(self.num_joints):
                    if inter_joints_vis[j, 0] > 0.0:
                        inter_joints[j, 0:2] = affine_transform(inter_joints[j, 0:2], trans)
                _inter_target, _inter_target_weight = self.generate_target(inter_joints, inter_joints_vis)
                inter_target = np.maximum(inter_target, _inter_target)
                inter_target_weight = np.maximum(inter_target_weight, _inter_target_weight)
            all_ins_target = np.maximum(0.5 * inter_target, target)
            all_ins_target_weight = np.maximum(inter_target_weight, target_weight)
            meta['sp_keypoint'] = [all_ins_target, all_ins_target_weight]

            self.heatmap_size = self.heatmap_size // 2
            self.sigma -= 1
            meta['pose'] = target
            meta['pose_weight'] = target_weight
            if self.SEM_ON:
                sem_labels = (target_weight[:, 0] > 0).astype(np.float32)
                sem_labels = np.reshape(sem_labels, (-1,))
                meta['semantics'] = [sem_labels]
            if self.MASK_ON:
                mask_target = np.max(target, axis=0)
                mask_target = (mask_target > 0.3).astype(np.float32)
                meta['masks'] = mask_target

        return input, meta

    def polys_to_mask(self, polygons, height, width):
        """Convert from the COCO polygon segmentation format to a binary mask
        encoded as a 2D array of data type numpy.float32. The polygon segmentation
        is understood to be enclosed inside a height x width image. The resulting
        mask is therefore of shape (height, width).
        """
        rle = mask_util.frPyObjects(polygons, height, width)
        mask = np.array(mask_util.decode(rle), dtype=np.float32)
        # Flatten in case polygons was a list
        mask = np.sum(mask, axis=2)
        mask = np.array(mask > 0, dtype=np.float32)
        return mask

    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std ** 2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center - bbox_center), 2)
            ks = np.exp(-1.0 * (diff_norm2 ** 2) / ((0.2) ** 2 * 2.0 * area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight

    def generate_specific_target(self, joints, joints_vis, heatmap_size=(48, 64), sigma=2):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               heatmap_size[1],
                               heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = int(sigma * 3)

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight

    def generate_limbs_target(self, joints, joints_vis):
        num_limbs = len(self.connection_rules)
        limbs_target = np.zeros((num_limbs, self.heatmap_size[1], self.heatmap_size[0]))
        feat_stride = self.image_size / self.heatmap_size
        limbs_vis = np.zeros((num_limbs,))
        for conn_id, conn in enumerate(self.connection_rules):
            kpt1, kpt2 = joints[conn[0]], joints[conn[1]]
            vis1, vis2 = joints_vis[conn[0], 0], joints_vis[conn[1], 0]

            if vis1 > 0 and vis2 > 0:
                kpt1 = np.asarray([int(kpt1[0] / feat_stride[0] + 0.5), int(kpt1[1] / feat_stride[1] + 0.5)])
                kpt2 = np.asarray([int(kpt2[0] / feat_stride[0] + 0.5), int(kpt2[1] / feat_stride[1] + 0.5)])
                limbs_target[conn_id] = self.generate_limb_from_two_point(kpt1,
                                                                          kpt2,
                                                                          self.heatmap_size[0],
                                                                          self.heatmap_size[1]
                                                                          )
                limbs_vis[conn_id] = 1
        return limbs_target, limbs_vis

    def generate_limb_from_two_point(self, pointA, pointB, hm_x, hm_y, thre=1):
        limb_maps = np.zeros((hm_y, hm_x))
        centerA = pointA.astype(float)
        centerB = pointB.astype(float)
        epis = 1e-10
        limb_vec = centerB - centerA
        norm = np.linalg.norm(limb_vec)
        limb_vec_unit = limb_vec / (norm + epis)

        # To make sure not beyond the border of this two points
        min_x = max(int(round(min(centerA[0], centerB[0]) - thre)), 0)
        max_x = min(int(round(max(centerA[0], centerB[0]) + thre)), hm_x)
        min_y = max(int(round(min(centerA[1], centerB[1]) - thre)), 0)
        max_y = min(int(round(max(centerA[1], centerB[1]) + thre)), hm_y)

        range_x = list(range(int(min_x), int(max_x), 1))
        range_y = list(range(int(min_y), int(max_y), 1))

        xx, yy = np.meshgrid(range_x, range_y)

        ba_x = xx - centerA[0]  # the vector from (x,y) to centerA

        ba_y = yy - centerA[1]

        limb_width = np.abs(ba_x * limb_vec_unit[1] - ba_y * limb_vec_unit[0])

        mask = limb_width < thre  # mask is 2D

        xx = xx.reshape((-1, 1))
        yy = yy.reshape((-1, 1))
        mask = mask.reshape(-1)
        limb_points = np.hstack([xx[mask], yy[mask]])
        limb_points = limb_points.astype(np.int32)
        limb_maps[limb_points[:, 1], limb_points[:, 0]] = 1

        return limb_maps

    def generate_part_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]
        # self.heatmap_size: [48,64] [w,h]
        target = np.zeros((self.num_joints,
                           3,
                           self.heatmap_size[1] *
                           self.heatmap_size[0]),
                          dtype=np.float32)

        feat_width = self.heatmap_size[0]
        feat_height = self.heatmap_size[1]
        feat_x_int = np.arange(0, feat_width)
        feat_y_int = np.arange(0, feat_height)
        feat_x_int, feat_y_int = np.meshgrid(feat_x_int, feat_y_int)
        feat_x_int = feat_x_int.reshape((-1,))
        feat_y_int = feat_y_int.reshape((-1,))
        if self.up_scale_factor > 1:
            kps_radius = (self.sigma-1) * 3
        else:
            kps_radius = self.sigma * 3
        feat_stride = self.image_size / self.heatmap_size
        for joint_id in range(self.num_joints):
            mu_x = joints[joint_id][0] / feat_stride[0]
            mu_y = joints[joint_id][1] / feat_stride[1]
            # Check that any part of the gaussian is in-bounds
            x_offset = (mu_x - feat_x_int) / kps_radius
            y_offset = (mu_y - feat_y_int) / kps_radius
            g = np.exp(- (x_offset ** 2 + y_offset ** 2) / 2**2)
            # vectors = np.hstack([(mu_x - feat_x_int).reshape((-1,1)), (mu_y - feat_y_int).reshape((-1,1))])
            # v_norms = np.linalg.norm(vectors, axis=1) #(vectors[:, 0] ** 2 + vectors[:, 1] ** 2)**0.5
            # vectors /= (v_norms[:, np.newaxis]+1e-10)
            # x_degree = vectors[:, 0] #np.degrees(np.arccos(vectors[:, 0]))/180.
            # y_degree = vectors[:, 1] #(np.degrees(np.arcsin(vectors[:, 1]))+90.)/180.
            dis = x_offset ** 2 + y_offset ** 2
            keep_pos = np.where((dis <= 2) & (dis >= 0))[0]

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id, 0, keep_pos] = g[keep_pos]
                target[joint_id, 1, keep_pos] = x_offset[keep_pos]
                target[joint_id, 2, keep_pos] = y_offset[keep_pos]
                # target[joint_id, 3, keep_pos] = x_degree[keep_pos]
                # target[joint_id, 4, keep_pos] = y_degree[keep_pos]

        target = target.reshape((self.num_joints * 3, self.heatmap_size[1], self.heatmap_size[0]))
        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)
        return target, target_weight
