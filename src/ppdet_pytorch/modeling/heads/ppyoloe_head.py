# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2025 PyTorch Migration. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PPYOLOEHead - PyTorch Migration from PaddlePaddle

Complete port of Paddle's PPYOLOEHead implementation.
Reference: ppdet/modeling/heads/ppyoloe_head.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple

from ...core.workspace import register
from ..bbox_utils import batch_distance2bbox
from ..initializer import bias_init_with_prob, constant_, normal_
from ..assigners.utils import generate_anchors_for_grid_cell
from ..backbones.cspresnet import ConvBNLayer, RepVggBlock
from ..ops import get_static_shape, get_act_fn
from ..layers import MultiClassNMS
from ..losses import GIoULoss

# TODO: Import assigners when needed for full training implementation
# from ..assigners import ATSSAssigner, TaskAlignedAssigner

__all__ = ['PPYOLOEHead']


class ESEAttn(nn.Module):
    """Effective Squeeze-and-Excitation Attention"""

    def __init__(self, feat_channels, act='swish', attn_conv='convbn'):
        super(ESEAttn, self).__init__()
        self.fc = nn.Conv2d(feat_channels, feat_channels, 1)
        if attn_conv == 'convbn':
            self.conv = ConvBNLayer(feat_channels, feat_channels, 1, act=act)
        elif attn_conv == 'repvgg':
            self.conv = RepVggBlock(feat_channels, feat_channels, act=act)
        else:
            self.conv = None
        self._init_weights()

    def _init_weights(self):
        normal_(self.fc.weight, std=0.001)

    def forward(self, feat, avg_feat):
        weight = torch.sigmoid(self.fc(avg_feat))
        if self.conv:
            return self.conv(feat * weight)
        else:
            return feat * weight


@register
class PPYOLOEHead(nn.Module):
    """PPYOLOEHead detection head for auxiliary branch"""

    __shared__ = [
        'num_classes', 'eval_size', 'trt', 'exclude_nms',
        'exclude_post_process', 'use_shared_conv', 'for_distill'
    ]
    __inject__ = ['static_assigner', 'assigner', 'nms']

    def __init__(self,
                 in_channels=[1024, 512, 256],
                 num_classes=80,
                 act='swish',
                 fpn_strides=(32, 16, 8),
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 reg_max=16,
                 reg_range=None,
                 static_assigner_epoch=4,
                 use_varifocal_loss=True,
                 static_assigner='ATSSAssigner',
                 assigner='TaskAlignedAssigner',
                 nms='MultiClassNMS',
                 eval_size=None,
                 loss_weight={
                     'class': 1.0,
                     'iou': 2.5,
                     'dfl': 0.5,
                 },
                 trt=False,
                 attn_conv='convbn',
                 exclude_nms=False,
                 exclude_post_process=False,
                 use_shared_conv=True,
                 for_distill=False):
        super(PPYOLOEHead, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        if reg_range:
            self.sm_use = True
            self.reg_range = reg_range
        else:
            self.sm_use = False
            self.reg_range = (0, reg_max + 1)
        self.reg_channels = self.reg_range[1] - self.reg_range[0]
        self.iou_loss = GIoULoss()
        self.loss_weight = loss_weight
        self.use_varifocal_loss = use_varifocal_loss
        self.eval_size = eval_size

        self.static_assigner_epoch = static_assigner_epoch
        self.static_assigner = static_assigner
        self.assigner = assigner
        self.nms = nms
        if isinstance(self.nms, MultiClassNMS) and trt:
            self.nms.trt = trt
        self.exclude_nms = exclude_nms
        self.exclude_post_process = exclude_post_process
        self.use_shared_conv = use_shared_conv
        self.for_distill = for_distill
        self.is_teacher = False

        # stem
        self.stem_cls = nn.ModuleList()
        self.stem_reg = nn.ModuleList()
        # Get activation function - match Paddle's logic
        # Paddle: act = get_act_fn(act, trt=trt) if act is None or isinstance(act, (str, dict)) else act
        # For PyTorch: ESEAttn will handle act parameter internally
        for in_c in self.in_channels:
            self.stem_cls.append(ESEAttn(in_c, act=act, attn_conv=attn_conv))
            self.stem_reg.append(ESEAttn(in_c, act=act, attn_conv=attn_conv))

        # pred head
        self.pred_cls = nn.ModuleList()
        self.pred_reg = nn.ModuleList()
        for in_c in self.in_channels:
            self.pred_cls.append(
                nn.Conv2d(in_c, self.num_classes, 3, padding=1))
            self.pred_reg.append(
                nn.Conv2d(in_c, 4 * self.reg_channels, 3, padding=1))

        # projection conv
        self.proj_conv = nn.Conv2d(self.reg_channels, 1, 1, bias=False)
        self.proj_conv.skip_quant = True  # For compatibility with Paddle (quantization flag)
        self._init_weights()

        if self.for_distill:
            self.distill_pairs = {}

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        for cls_, reg_ in zip(self.pred_cls, self.pred_reg):
            constant_(cls_.weight)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.weight)
            constant_(reg_.bias, 1.0)

        proj = torch.linspace(self.reg_range[0], self.reg_range[1] - 1,
                              self.reg_channels).reshape(1, self.reg_channels, 1, 1)
        self.proj_conv.weight.data = proj
        self.proj_conv.weight.requires_grad = False

        if self.eval_size:
            anchor_points, stride_tensor = self._generate_anchors()
            self.register_buffer('anchor_points', anchor_points)
            self.register_buffer('stride_tensor', stride_tensor)

    def m_avg_pool2d(self, feat, w, h):
        """Manual average pooling for NPU compatibility"""
        batch_size, channels, _, _ = feat.shape
        feat_flat = feat.reshape(batch_size, channels, -1)
        feat_mean = feat_flat.mean(dim=2)
        feat_mean = feat_mean.reshape(batch_size, channels, w, h)
        return feat_mean

    def forward_train(self, feats, targets, aux_pred=None):
        """Training forward pass - matches Paddle implementation"""
        # Generate anchors for training
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        cls_score_list, reg_distri_list = [], []
        for i, feat in enumerate(feats):
            # Use m_avg_pool2d for NPU, otherwise use F.adaptive_avg_pool2d
            # Note: torch doesn't have device.get_device() like Paddle, so always use adaptive
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) + feat)
            reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            # cls and reg
            cls_score = torch.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).permute(0, 2, 1))
            reg_distri_list.append(reg_distri.flatten(2).permute(0, 2, 1))
        cls_score_list = torch.cat(cls_score_list, dim=1)
        reg_distri_list = torch.cat(reg_distri_list, dim=1)

        # Handle teacher mode or get_data mode
        if targets.get('is_teacher', False):
            pred_deltas, pred_dfls = self._bbox_decode_fake(reg_distri_list)
            return cls_score_list, pred_deltas * stride_tensor, pred_dfls

        if targets.get('get_data', False):
            pred_deltas, pred_dfls = self._bbox_decode_fake(reg_distri_list)
            return cls_score_list, pred_deltas * stride_tensor, pred_dfls

        return self.get_loss([
            cls_score_list, reg_distri_list, anchors, anchor_points,
            num_anchors_list, stride_tensor
        ], targets, aux_pred)

    def _generate_anchors(self, feats=None, dtype=torch.float32):
        """Generate anchors for evaluation"""
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(self.fpn_strides):
            if feats is not None:
                _, _, h, w = feats[i].shape
                device = feats[i].device
            else:
                h = int(self.eval_size[0] / stride)
                w = int(self.eval_size[1] / stride)
                device = None

            shift_x = (
                torch.arange(end=w, dtype=dtype, device=device)
                + self.grid_cell_offset)
            shift_y = (
                torch.arange(end=h, dtype=dtype, device=device)
                + self.grid_cell_offset)
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
            anchor_point = torch.stack([shift_x, shift_y], dim=-1)
            anchor_points.append(anchor_point.reshape(-1, 2))
            stride_tensor.append(
                torch.full(
                    (h * w, 1), stride, dtype=dtype, device=device))

        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)
        return anchor_points, stride_tensor

    def forward_eval(self, feats):
        """Evaluation forward pass"""
        if self.eval_size:
            anchor_points = self.anchor_points
            stride_tensor = self.stride_tensor
        else:
            anchor_points, stride_tensor = self._generate_anchors(feats)

        cls_score_list, reg_dist_list = [], []
        for i, feat in enumerate(feats):
            _, _, h, w = feat.shape
            l = h * w
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))

            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) + feat)
            reg_dist = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))

            reg_dist = reg_dist.reshape(-1, 4, self.reg_channels, l).permute(0, 2, 3, 1)
            if self.use_shared_conv:
                reg_dist = self.proj_conv(F.softmax(reg_dist, dim=1)).squeeze(1)
            else:
                reg_dist = F.softmax(reg_dist, dim=1)

            cls_score = torch.sigmoid(cls_logit)
            cls_score_list.append(cls_score.reshape(-1, self.num_classes, l))
            reg_dist_list.append(reg_dist)

        cls_score_list = torch.cat(cls_score_list, dim=-1)
        if self.use_shared_conv:
            reg_dist_list = torch.cat(reg_dist_list, dim=1)
        else:
            reg_dist_list = torch.cat(reg_dist_list, dim=2)
            reg_dist_list = self.proj_conv(reg_dist_list).squeeze(1)

        return cls_score_list, reg_dist_list, anchor_points, stride_tensor

    def forward(self, feats, targets=None, aux_pred=None):
        """Forward pass - matches Paddle's interface"""
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"

        if self.training:
            return self.forward_train(feats, targets, aux_pred)
        else:
            if targets is not None:
                # only for semi-det
                self.is_teacher = targets.get('is_teacher', False)
                if self.is_teacher:
                    return self.forward_train(feats, targets, aux_pred=None)
                else:
                    return self.forward_eval(feats)
            return self.forward_eval(feats)

    @staticmethod
    def _focal_loss(score, label, alpha=0.25, gamma=2.0):
        """Focal Loss"""
        weight = (score - label).pow(gamma)
        if alpha > 0:
            alpha_t = alpha * label + (1 - alpha) * (1 - label)
            weight = weight * alpha_t
        with torch.autocast(device_type=score.device.type, enabled=False):
            loss = F.binary_cross_entropy(
                score.float(), label.float(), reduction='none')
        return (loss * weight.float()).sum()

    @staticmethod
    def _varifocal_loss(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Varifocal Loss"""
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        with torch.autocast(device_type=pred_score.device.type, enabled=False):
            loss = F.binary_cross_entropy(
                pred_score.float(), gt_score.float(), reduction='none')
        return (loss * weight.float()).sum()

    def _bbox_decode(self, anchor_points, pred_dist):
        """Decode bbox from distribution"""
        _, l, _ = pred_dist.shape
        pred_dist = F.softmax(pred_dist.reshape(-1, l, 4, self.reg_channels), dim=-1)
        pred_dist = self.proj_conv(pred_dist.permute(0, 3, 1, 2)).squeeze(1)
        return batch_distance2bbox(anchor_points, pred_dist)

    def _bbox_decode_fake(self, pred_dist):
        """Decode bbox distribution without anchor points (for teacher/get_data mode)"""
        _, l, _ = pred_dist.shape
        pred_dist_dfl = F.softmax(pred_dist.reshape(-1, l, 4, self.reg_channels), dim=-1)
        pred_dist = self.proj_conv(pred_dist_dfl.permute(0, 3, 1, 2)).squeeze(1)
        return pred_dist, pred_dist_dfl

    def _bbox2distance(self, points, bbox):
        """Convert bbox to distance format"""
        x1y1, x2y2 = torch.split(bbox, 2, -1)
        lt = points - x1y1
        rb = x2y2 - points
        return torch.cat([lt, rb], -1).clamp(self.reg_range[0],
                                             self.reg_range[1] - 1 - 0.01)

    def _df_loss(self, pred_dist, target, lower_bound=0):
        """Distribution Focal Loss"""
        target_left = target.floor().long()
        target_right = target_left + 1
        weight_left = target_right.float() - target
        weight_right = 1 - weight_left

        pred_dist = pred_dist.reshape(-1, self.reg_channels)
        loss_left = F.cross_entropy(
            pred_dist,
            (target_left - lower_bound).reshape(-1),
            reduction='none').reshape_as(target) * weight_left
        loss_right = F.cross_entropy(
            pred_dist,
            (target_right - lower_bound).reshape(-1),
            reduction='none').reshape_as(target) * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)

    def _bbox_loss(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        assigned_labels,
        assigned_bboxes,
        assigned_scores,
        assigned_scores_sum,
    ):
        mask_positive = assigned_labels != self.num_classes

        if self.for_distill:
            self.distill_pairs['mask_positive_select'] = mask_positive

        if mask_positive.any():
            bbox_mask = mask_positive.unsqueeze(-1).expand(-1, -1, 4)
            pred_bboxes_pos = pred_bboxes.masked_select(bbox_mask).reshape(-1, 4)
            assigned_bboxes_pos = assigned_bboxes.masked_select(
                bbox_mask).reshape(-1, 4)
            bbox_weight = assigned_scores.sum(-1).masked_select(
                mask_positive).unsqueeze(-1)

            loss_l1 = F.l1_loss(pred_bboxes_pos, assigned_bboxes_pos)
            loss_iou = self.iou_loss(
                pred_bboxes_pos, assigned_bboxes_pos) * bbox_weight
            loss_iou = loss_iou.sum() / assigned_scores_sum

            dist_mask = mask_positive.unsqueeze(-1).expand(
                -1, -1, self.reg_channels * 4)
            pred_dist_pos = pred_dist.masked_select(dist_mask).reshape(
                -1, 4, self.reg_channels)
            assigned_ltrb = self._bbox2distance(
                anchor_points, assigned_bboxes)
            assigned_ltrb_pos = assigned_ltrb.masked_select(
                bbox_mask).reshape(-1, 4)
            loss_dfl = self._df_loss(
                pred_dist_pos,
                assigned_ltrb_pos,
                self.reg_range[0],
            ) * bbox_weight
            loss_dfl = loss_dfl.sum() / assigned_scores_sum

            if self.for_distill:
                self.distill_pairs['pred_bboxes_pos'] = pred_bboxes_pos
                self.distill_pairs['pred_dist_pos'] = pred_dist_pos
                self.distill_pairs['bbox_weight'] = bbox_weight
        else:
            loss_l1 = pred_bboxes.sum() * 0.0
            loss_iou = pred_bboxes.sum() * 0.0
            loss_dfl = pred_dist.sum() * 0.0

        return loss_l1, loss_iou, loss_dfl

    def get_loss(self, head_outs, gt_meta, aux_pred=None):
        (
            pred_scores,
            pred_distri,
            anchors,
            anchor_points,
            num_anchors_list,
            stride_tensor,
        ) = head_outs

        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self._bbox_decode(anchor_points_s, pred_distri)

        if aux_pred is not None:
            pred_scores_aux = aux_pred[0]
            pred_bboxes_aux = self._bbox_decode(anchor_points_s, aux_pred[1])

        if 'origin_gt_class' in gt_meta:
            gt_labels = gt_meta['origin_gt_class']
            gt_bboxes = gt_meta['origin_gt_bbox']
            pad_gt_mask = gt_meta['pad_origin_gt_mask']
        else:
            gt_labels = gt_meta['gt_class']
            gt_bboxes = gt_meta['gt_bbox']
            pad_gt_mask = gt_meta['pad_gt_mask']

        if gt_meta['epoch_id'] < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, assigned_scores = (
                self.static_assigner(
                    anchors,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes,
                    pred_bboxes=pred_bboxes.detach() * stride_tensor,
                )
            )
            alpha_l = 0.25
        else:
            assign_scores = pred_scores.detach()
            assign_bboxes = pred_bboxes.detach()
            if aux_pred is not None:
                assign_scores = pred_scores_aux.detach()
                assign_bboxes = pred_bboxes_aux.detach()
            assigned_labels, assigned_bboxes, assigned_scores = self.assigner(
                assign_scores,
                assign_bboxes * stride_tensor,
                anchor_points,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index=self.num_classes,
            )
            alpha_l = -1

        assigned_bboxes = assigned_bboxes / stride_tensor
        losses = self.get_loss_from_assign(
            pred_scores,
            pred_distri,
            pred_bboxes,
            anchor_points_s,
            assigned_labels,
            assigned_bboxes,
            assigned_scores,
            alpha_l,
        )

        if aux_pred is None:
            return losses

        aux_losses = self.get_loss_from_assign(
            aux_pred[0],
            aux_pred[1],
            pred_bboxes_aux,
            anchor_points_s,
            assigned_labels,
            assigned_bboxes,
            assigned_scores,
            alpha_l,
        )
        return {key: value + aux_losses[key] for key, value in losses.items()}

    def get_loss_from_assign(
        self,
        pred_scores,
        pred_distri,
        pred_bboxes,
        anchor_points_s,
        assigned_labels,
        assigned_bboxes,
        assigned_scores,
        alpha_l,
    ):
        if self.use_varifocal_loss:
            one_hot_label = F.one_hot(
                assigned_labels.long(), self.num_classes + 1)[..., :-1]
            loss_cls = self._varifocal_loss(
                pred_scores, assigned_scores, one_hot_label)
        else:
            loss_cls = self._focal_loss(
                pred_scores, assigned_scores, alpha_l)

        assigned_scores_sum = assigned_scores.sum()
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            torch.distributed.all_reduce(assigned_scores_sum)
            assigned_scores_sum /= torch.distributed.get_world_size()
        assigned_scores_sum = assigned_scores_sum.clamp(min=1.0)
        loss_cls = loss_cls / assigned_scores_sum

        if self.for_distill:
            self.distill_pairs['pred_cls_scores'] = pred_scores
            self.distill_pairs['pos_num'] = assigned_scores_sum
            self.distill_pairs['assigned_scores'] = assigned_scores
            self.distill_pairs['target_labels'] = F.one_hot(
                assigned_labels.long(), self.num_classes + 1)[..., :-1]

        loss_l1, loss_iou, loss_dfl = self._bbox_loss(
            pred_distri,
            pred_bboxes,
            anchor_points_s,
            assigned_labels,
            assigned_bboxes,
            assigned_scores,
            assigned_scores_sum,
        )
        loss = (
            self.loss_weight['class'] * loss_cls
            + self.loss_weight['iou'] * loss_iou
            + self.loss_weight['dfl'] * loss_dfl
        )
        return {
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_iou': loss_iou,
            'loss_dfl': loss_dfl,
            'loss_l1': loss_l1,
        }

    def post_process(self, head_outs, scale_factor):
        """Post-process predictions"""
        pred_scores, pred_dist, anchor_points, stride_tensor = head_outs
        pred_bboxes = batch_distance2bbox(anchor_points, pred_dist)
        pred_bboxes = pred_bboxes * stride_tensor

        if self.exclude_post_process:
            return torch.cat(
                [pred_bboxes, pred_scores.permute(0, 2, 1)],
                dim=-1), None, None
        else:
            # scale bbox to origin
            scale_y, scale_x = torch.split(scale_factor, 2, dim=-1)
            scale_factor = torch.cat(
                [scale_x, scale_y, scale_x, scale_y],
                dim=-1).reshape(-1, 1, 4)
            pred_bboxes = pred_bboxes / scale_factor

            if self.exclude_nms:
                return pred_bboxes, pred_scores, None
            else:
                # TODO: NMS needs to be fully implemented
                bbox_pred, bbox_num, nms_keep_idx = self.nms(pred_bboxes, pred_scores)
                return bbox_pred, bbox_num, nms_keep_idx
