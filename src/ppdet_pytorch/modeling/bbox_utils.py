# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
Bounding Box Utilities - PyTorch Migration from PaddlePaddle

Reference: ppdet/modeling/bbox_utils.py
"""

import math

import torch

__all__ = [
    "batch_distance2bbox",
    "iou_similarity",
    "batch_iou_similarity",
    "bbox_center",
    "bbox_iou",
]


def batch_distance2bbox(points, distance, max_shapes=None):
    """Decode distance prediction to bounding box for batch.

    Args:
        points (Tensor): [B, ..., 2], "xy" format
        distance (Tensor): [B, ..., 4], "ltrb" format
        max_shapes (Tensor): [B, 2], "h,w" format, Shape of the image.

    Returns:
        Tensor: Decoded bboxes, "x1y1x2y2" format.
    """
    lt, rb = torch.split(distance, 2, -1)
    # while tensor add parameters, parameters should be better placed on the second place
    x1y1 = -lt + points
    x2y2 = rb + points
    out_bbox = torch.cat([x1y1, x2y2], -1)

    if max_shapes is not None:
        max_shapes = max_shapes.flip(-1).tile([1, 2])
        delta_dim = out_bbox.ndim - max_shapes.ndim
        for _ in range(delta_dim):
            max_shapes = max_shapes.unsqueeze(1)
        out_bbox = torch.where(out_bbox < max_shapes, out_bbox, max_shapes)
        out_bbox = torch.where(out_bbox > 0, out_bbox, torch.zeros_like(out_bbox))

    return out_bbox


def iou_similarity(box1, box2, eps=1e-10):
    """Calculate iou of box1 and box2

    Args:
        box1 (Tensor): box with the shape [M1, 4]
        box2 (Tensor): box with the shape [M2, 4]

    Return:
        iou (Tensor): iou between box1 and box2 with the shape [M1, M2]
    """
    box1 = box1.unsqueeze(1)  # [M1, 4] -> [M1, 1, 4]
    box2 = box2.unsqueeze(0)  # [M2, 4] -> [1, M2, 4]
    px1y1, px2y2 = box1[:, :, 0:2], box1[:, :, 2:4]
    gx1y1, gx2y2 = box2[:, :, 0:2], box2[:, :, 2:4]
    x1y1 = torch.maximum(px1y1, gx1y1)
    x2y2 = torch.minimum(px2y2, gx2y2)
    overlap = (x2y2 - x1y1).clamp(min=0).prod(-1)
    area1 = (px2y2 - px1y1).clamp(min=0).prod(-1)
    area2 = (gx2y2 - gx1y1).clamp(min=0).prod(-1)
    union = area1 + area2 - overlap + eps
    return overlap / union


def batch_iou_similarity(box1, box2, eps=1e-9):
    """Calculate iou of box1 and box2 in batch

    Args:
        box1 (Tensor): box with the shape [N, M1, 4]
        box2 (Tensor): box with the shape [N, M2, 4]

    Return:
        iou (Tensor): iou between box1 and box2 with the shape [N, M1, M2]
    """
    box1 = box1.unsqueeze(2)  # [N, M1, 4] -> [N, M1, 1, 4]
    box2 = box2.unsqueeze(1)  # [N, M2, 4] -> [N, 1, M2, 4]
    px1y1, px2y2 = box1[:, :, :, 0:2], box1[:, :, :, 2:4]
    gx1y1, gx2y2 = box2[:, :, :, 0:2], box2[:, :, :, 2:4]
    x1y1 = torch.maximum(px1y1, gx1y1)
    x2y2 = torch.minimum(px2y2, gx2y2)
    overlap = (x2y2 - x1y1).clamp(min=0).prod(-1)
    area1 = (px2y2 - px1y1).clamp(min=0).prod(-1)
    area2 = (gx2y2 - gx1y1).clamp(min=0).prod(-1)
    union = area1 + area2 - overlap + eps
    return overlap / union


def bbox_center(boxes):
    """Get bbox centers from boxes.

    Args:
        boxes (Tensor): boxes with shape (..., 4), "xmin, ymin, xmax, ymax" format.
    Returns:
        Tensor: boxes centers with shape (..., 2), "cx, cy" format.
    """
    boxes_cx = (boxes[..., 0] + boxes[..., 2]) / 2
    boxes_cy = (boxes[..., 1] + boxes[..., 3]) / 2
    return torch.stack([boxes_cx, boxes_cy], dim=-1)


def bbox_iou(box1, box2, giou=False, diou=False, ciou=False, eps=1e-9):
    """Calculate the iou of box1 and box2

    Args:
        box1 (list): [x, y, w, h], all have the shape [b, na, h, w, 1]
        box2 (list): [x, y, w, h], all have the shape [b, na, h, w, 1]
        giou (bool): whether use giou or not, default False
        diou (bool): whether use diou or not, default False
        ciou (bool): whether use ciou or not, default False
        eps (float): epsilon to avoid divide by zero

    Return:
        iou (Tensor): iou of box1 and box1, with the shape [b, na, h, w, 1]
    """
    px1, py1, px2, py2 = box1
    gx1, gy1, gx2, gy2 = box2
    x1 = torch.maximum(px1, gx1)
    y1 = torch.maximum(py1, gy1)
    x2 = torch.minimum(px2, gx2)
    y2 = torch.minimum(py2, gy2)

    overlap = ((x2 - x1).clamp(min=0)) * ((y2 - y1).clamp(min=0))

    area1 = (px2 - px1) * (py2 - py1)
    area1 = area1.clamp(min=0)

    area2 = (gx2 - gx1) * (gy2 - gy1)
    area2 = area2.clamp(min=0)

    union = area1 + area2 - overlap + eps
    iou = overlap / union

    if giou or ciou or diou:
        # convex w, h
        cw = torch.maximum(px2, gx2) - torch.minimum(px1, gx1)
        ch = torch.maximum(py2, gy2) - torch.minimum(py1, gy1)
        if giou:
            c_area = cw * ch + eps
            return iou - (c_area - union) / c_area
        else:
            # convex diagonal squared
            c2 = cw**2 + ch**2 + eps
            # center distance
            rho2 = ((px1 + px2 - gx1 - gx2) ** 2 + (py1 + py2 - gy1 - gy2) ** 2) / 4
            if diou:
                return iou - rho2 / c2
            else:
                w1, h1 = px2 - px1, py2 - py1 + eps
                w2, h2 = gx2 - gx1, gy2 - gy1 + eps
                delta = torch.atan(w1 / h1) - torch.atan(w2 / h2)
                v = (4 / math.pi**2) * torch.pow(delta, 2)
                alpha = v / (1 + eps - iou + v)
                alpha = alpha.detach()
                return iou - (rho2 / c2 + v * alpha)
    else:
        return iou
