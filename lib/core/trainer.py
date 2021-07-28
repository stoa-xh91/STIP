from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os

import numpy as np
import torch
from config.models import get_model_name
from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images, save_batch_maps, save_batch_stip_maps

logger = logging.getLogger(__name__)
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def do_train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    sem_acc = AverageMeter()
    # limb_acc = AverageMeter()
    poss_losses = AverageMeter()
    part_losses = AverageMeter()
    att_losses = AverageMeter()
    mask_losses = AverageMeter()
    kernel_losses = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # compute output
        outputs = model(input)
        ground_truth = {}
        predictions = {}
        if config.MODEL.POSE_ON:
            predictions['pose'] = outputs['keypoint']
            target = meta['pose'].cuda(non_blocking=True)
            target_weight = meta['pose_weight'].cuda(non_blocking=True)
            ground_truth['pose'] = [target, target_weight]
            if config.MODEL.SPATIAL_POSE_ON:
                predictions['spatial_pose'] = outputs['sp_keypoint']
                ground_truth['spatial_pose'] = [meta['pose'].cuda(non_blocking=True),
                                                meta['pose_weight'].cuda(non_blocking=True)]

        if config.MODEL.MASK_ON:
            predictions['mask'] = [outputs['mask'][0]]
            target = meta['masks'].cuda(non_blocking=True)
            ground_truth['mask'] = target.unsqueeze(1)
        if config.MODEL.SEM_ON:
            predictions['semantics'] = outputs['semantics']
            target = [meta['semantics'][0].cuda(non_blocking=True)]
            ground_truth['semantics'] = target

        kernel_loss = torch.mean(outputs['kernel'])

        loss = criterion(predictions, ground_truth)

        total_loss = loss['total loss']

        # compute gradient and do update step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(total_loss.item(), input.size(0))
        if config.MODEL.POSE_ON:
            if isinstance(loss['pose'], list):
                poss_losses.update(loss['pose'][0].item(), input.size(0))
            else:
                poss_losses.update(loss['pose'].item(), input.size(0))

            if config.MODEL.SPATIAL_POSE_ON:
                part_losses.update(loss['sp_pose'].item(), input.size(0))

        if config.MODEL.SEM_ON:
            if isinstance(loss['semantics'], list):
                att_losses.update(loss['semantics'][0].item(), input.size(0))
            else:
                att_losses.update(loss['semantics'].item(), input.size(0))
            sem_preds = predictions['semantics'][0].detach().cpu().numpy()
            sem_gt = ground_truth['semantics'][0].detach().cpu().numpy()
            avg_sem_acc = np.sum((sem_preds>=0.5).astype(np.float32) == sem_gt, axis=1) / sem_gt.shape[1]
            avg_sem_acc = np.mean(avg_sem_acc)
            sem_acc.update(avg_sem_acc, sem_gt.shape[0])

        if config.MODEL.MASK_ON:
            mask_losses.update(loss['mask'].item(), input.size(0))


        kernel_losses.update(kernel_loss.item(), input.size(0))
        _, avg_acc, cnt, pred = accuracy(predictions['pose'][0].detach().cpu().numpy(),
                                         ground_truth['pose'][0].detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            if isinstance(loss['pose'], list):
                loss_details = 'pose loss {poss_loss.val:.5f} '.format(poss_loss=poss_losses)
                for i_l in range(1, len(loss['pose'])):
                    l = loss['pose'][i_l].detach().cpu().numpy()
                    loss_details += '{l:.5f} '.format(l=l)
                loss_details += '\t'
            else:
                loss_details = 'pose loss {poss_loss.val:.5f}\t'.format(poss_loss=poss_losses)

            if config.MODEL.SPATIAL_POSE_ON:
                sp_losses_details = 'part heatmap loss {sp_hm_losses.val:.5f}\t'.format(sp_hm_losses=part_losses)
                loss_details += sp_losses_details

            if config.MODEL.SEM_ON:
                if isinstance(loss['semantics'], list):
                    sems_losses_details = 'keypoints aware loss: {sems_aware_loss.val:.5f}\t'.format(
                        sems_aware_loss=att_losses)
                    for i_l in range(1, len(loss['semantics'])):
                        l = loss['semantics'][i_l].detach().cpu().numpy()
                        sems_losses_details += '{l:.5f} '.format(l=l)
                    sems_losses_details += '\t'
                else:
                    sems_losses_details = 'keypoints aware loss: {sems_aware_loss.val:.5f}\t'.format(
                        sems_aware_loss=att_losses)
                loss_details += sems_losses_details

            if config.MODEL.MASK_ON:
                relation_loss_details = 'mask loss {relation_loss.val:.5f}\t'.format(relation_loss=mask_losses)
                loss_details += relation_loss_details
            kernel_loss_details = 'kernel loss {kernel_loss.val:.8f}\t'.format(kernel_loss=kernel_losses)
            loss_details += kernel_loss_details
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f}) ' \
                  'Loss details: {loss_details}' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f}) '\
                  .format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, loss_details=loss_details, acc=acc)

            if config.MODEL.SEM_ON:
                sems_aware_acc_details = 'Keypoints Aware Accuracy {sem_acc.val:.3f} ({sem_acc.avg:.3f}) '.format(
                    sem_acc=sem_acc
                )
                msg += sems_aware_acc_details
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, ground_truth['pose'][0], pred*4, outputs['keypoint'][0],
                              prefix)
            if config.MODEL.MASK_ON:
                save_batch_maps(input, ground_truth['mask'], '{}_mask_gt.jpg'.format(prefix))
                for i in range(len(outputs['mask'])):
                    save_batch_maps(input, outputs['mask'][i].clamp(min=0.5), '{}_mask_{}_pred.jpg'.format(prefix, i+1))
            if config.MODEL.SPATIAL_POSE_ON:
                save_batch_maps(input, outputs['sp_keypoint'], '{}_parts_pred.jpg'.format(prefix))
                save_batch_maps(input, meta['sp_keypoint'][0].cuda(non_blocking=True), '{}_parts_gt.jpg'.format(prefix))


def do_validate(config, val_loader, val_dataset, model, criterion, output_dir,
                tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, meta) in enumerate(val_loader):
            # compute output
            outputs = model(input)
            output = outputs['keypoint']
            if isinstance(output, list):
                output = output[0]
                # hm_output = 0
                # for o in output:
                #     hm_output += o
                # output = hm_output / len(output)
            else:
                output = output
            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped = model(input_flipped)

                output_flipped = outputs_flipped['keypoint']
                if isinstance(output_flipped, list):
                    output_flipped = output_flipped[0]
                    # hm_flipped = 0
                    # for o in output_flipped:
                    #     hm_flipped += o
                    # output_flipped = hm_flipped / len(output_flipped)
                else:
                    output_flipped = output_flipped
                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5
            gaussian_output = output
            target = meta['pose'].cuda(non_blocking=True)

            loss = torch.mean(gaussian_output[target == 1])
            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(gaussian_output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)
            if config.MODEL.SEM_ON:
                sem_preds = outputs['semantics'][0].detach().cpu().numpy()
                sem_gt = meta['semantics'][0].detach().cpu().numpy()
                avg_sem_acc = np.sum((sem_preds >= 0.5).astype(np.float32) == sem_gt, axis=1) / sem_gt.shape[1]
                avg_sem_acc = np.mean(avg_sem_acc)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
              config, gaussian_output.clone().cpu().numpy(), c, s)
            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s * 200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time,
                    loss=losses, acc=acc)
                if config.MODEL.SEM_ON:
                    msg += ' Keypoints Aware Accuracy %.4f'%(avg_sem_acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred * 4, gaussian_output,
                                  prefix)
                if config.MODEL.MASK_ON:
                    save_batch_maps(input, meta['masks'].unsqueeze(1), '{}_mask_gt.jpg'.format(prefix))
        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


def do_debug_validate(config, val_loader, val_dataset, model, criterion, output_dir,
                tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    errors = {}
    R=0
    TP=0
    for i in np.arange(0., 1.1, 0.1):
        errors[i] = []
    case_recall_errors = {'easy':{},'mediumn':{},'hard':{}}
    for k in case_recall_errors:
        for i in np.arange(0., 1.1, 0.1):
            case_recall_errors[k][i] = []
    overall_sem_acc = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, meta) in enumerate(val_loader):
            # compute output
            outputs = model(input)
            output = outputs['keypoint']
            if isinstance(output, list):
                output = output[0]
                # hm_output = 0
                # for o in output:
                #     hm_output += o
                # output = hm_output / len(output)
            else:
                output = output
            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped = model(input_flipped)

                output_flipped = outputs_flipped['keypoint']
                if isinstance(output_flipped, list):
                    output_flipped = output_flipped[0]
                else:
                    output_flipped = output_flipped
                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5
            gaussian_output = output
            target = meta['pose'].cuda(non_blocking=True)

            loss = torch.mean(gaussian_output[target == 1])
            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(gaussian_output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)
            sem_preds = outputs['semantics'][0].detach().cpu().numpy()
            sem_gt = meta['semantics'][0].detach().cpu().numpy()
            avg_sem_acc = np.sum((sem_preds >= 0.5).astype(np.float32) == sem_gt, axis=1) / sem_gt.shape[1]
            avg_sem_acc = np.mean(avg_sem_acc)
            overall_sem_acc.append(avg_sem_acc)
            for e in np.arange(0., 1.1, 0.1):
                tp = np.sum((sem_preds >= e).astype(np.float32)[sem_gt==1])
                if e==0:
                    TP += tp
                    R += np.sum(sem_gt==1)
                if not np.sum(sem_gt==1) == 0:
                    recal_tmp = tp / np.sum(sem_gt==1)
                    errors[e].append(recal_tmp)
            crowd_index = meta['crowd_index'].cpu().numpy()
            for case in case_recall_errors.keys():
                if case == 'easy':
                    index = crowd_index<=0.1
                elif case == 'mediumn':
                    index = np.logical_and(crowd_index>0.1,crowd_index<0.8)
                else:
                    index = crowd_index>=0.8
                index = index.reshape(-1)
                if np.sum(index)==0:
                    continue
                case_sem_preds = sem_preds[index, :]
                case_gt_preds = sem_gt[index, :]
                for e in np.arange(0., 1.1, 0.1):
                    tp = np.sum((case_sem_preds >= e).astype(np.float32)[case_gt_preds == 1])
                    if not np.sum(case_gt_preds == 0) == 0:
                        recal_tmp = tp / np.sum(case_gt_preds == 1)
                        case_recall_errors[case][e].append(recal_tmp)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()
            preds, maxvals = get_final_preds(
              config, gaussian_output.clone().cpu().numpy(), c, s)
            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s * 200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time,
                    loss=losses, acc=acc)
                msg += ' Keypoints Aware Accuracy %.4f'%(avg_sem_acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred * 4, gaussian_output,
                                  prefix)
                if config.MODEL.MASK_ON:
                    save_batch_maps(input, meta['masks'].unsqueeze(1), '{}_mask_gt.jpg'.format(prefix))

        print('everage accuracy')
        print(np.asarray(overall_sem_acc).mean())
        print('point-wise recall:', TP*1./R)
        for e in np.arange(0.,1.1,0.1):
            print(np.asarray(errors[e]).mean())

        print('case recall outputs')
        for case in case_recall_errors.keys():
            print(case)
            for e in np.arange(0., 1.1, 0.1):
                print(np.asarray(case_recall_errors[case][e]).mean())

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator

def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )