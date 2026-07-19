# Copyright (c) 2025 RT-DETRv3 PyTorch Authors. All Rights Reserved.
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
#
# Modified from PaddlePaddle RT-DETRv3
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

"""
Rotated Box (RBox) utilities for oriented object detection.

NOTE: RT-DETRv3 uses standard horizontal bounding boxes,
but this module is preserved for:
1. Compatibility with PaddleDetection ecosystem
2. Future extension to rotated object detection tasks
3. Completeness of the migration from Paddle to PyTorch
"""

import math

import cv2
import numpy as np
import torch


def norm_angle(angle, range=[-np.pi / 4, np.pi]):
    """Normalize angle to specified range"""
    return (angle - range[0]) % range[1] + range[0]


# rbox function implemented using numpy
def poly2rbox_le135_np(poly):
    """Convert polygon to rotated box with angle in [-pi/4, 3*pi/4]

    Args:
        poly: [x1, y1, x2, y2, x3, y3, x4, y4]

    Returns:
        rbox: [cx, cy, w, h, angle]
    """
    poly = np.array(poly[:8], dtype=np.float32)

    pt1 = (poly[0], poly[1])
    pt2 = (poly[2], poly[3])
    pt3 = (poly[4], poly[5])
    pt4 = (poly[6], poly[7])

    edge1 = np.sqrt(
        (pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1])
    )
    edge2 = np.sqrt(
        (pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) * (pt2[1] - pt3[1])
    )

    width = max(edge1, edge2)
    height = min(edge1, edge2)

    rbox_angle = 0
    if edge1 > edge2:
        rbox_angle = np.arctan2(float(pt2[1] - pt1[1]), float(pt2[0] - pt1[0]))
    elif edge2 >= edge1:
        rbox_angle = np.arctan2(float(pt4[1] - pt1[1]), float(pt4[0] - pt1[0]))

    rbox_angle = norm_angle(rbox_angle)

    x_ctr = float(pt1[0] + pt3[0]) / 2
    y_ctr = float(pt1[1] + pt3[1]) / 2
    return [x_ctr, y_ctr, width, height, rbox_angle]


def poly2rbox_oc_np(poly):
    """Convert polygon to rotated box with angle in (0, pi/2]

    Args:
        poly: [x1, y1, x2, y2, x3, y3, x4, y4]

    Returns:
        rbox: [cx, cy, w, h, angle]
    """
    points = np.array(poly, dtype=np.float32).reshape((-1, 2))
    (cx, cy), (w, h), angle = cv2.minAreaRect(points)
    # using the new OpenCV Rotated BBox definition since 4.5.1
    # if angle < 0, opencv is older than 4.5.1, angle is in [-90, 0)
    if angle < 0:
        angle += 90
        w, h = h, w

    # convert angle to [0, 90)
    if angle == -0.0:
        angle = 0.0
    if angle == 90.0:
        angle = 0.0
        w, h = h, w

    angle = angle / 180 * np.pi
    return [cx, cy, w, h, angle]


def poly2rbox_np(polys, rbox_type="oc"):
    """Convert polygons to rotated boxes

    Args:
        polys: [x0,y0,x1,y1,x2,y2,x3,y3]
        rbox_type: 'oc' or 'le135'

    Returns:
        rboxes: [x_ctr,y_ctr,w,h,angle]
    """
    assert rbox_type in ["oc", "le135"], "only oc or le135 is supported now"
    poly2rbox_fn = poly2rbox_oc_np if rbox_type == "oc" else poly2rbox_le135_np
    rboxes = []
    for poly in polys:
        x, y, w, h, angle = poly2rbox_fn(poly)
        rbox = np.array([x, y, w, h, angle], dtype=np.float32)
        rboxes.append(rbox)

    return np.array(rboxes)


def cal_line_length(point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt(
        math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2)
    )


def get_best_begin_point_single(coordinate):
    """Find the best starting point for a polygon to match axis-aligned box"""
    x1, y1, x2, y2, x3, y3, x4, y4 = coordinate
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combinate = [
        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
        [[x4, y4], [x1, y1], [x2, y2], [x3, y3]],
        [[x3, y3], [x4, y4], [x1, y1], [x2, y2]],
        [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
    ]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = (
            cal_line_length(combinate[i][0], dst_coordinate[0])
            + cal_line_length(combinate[i][1], dst_coordinate[1])
            + cal_line_length(combinate[i][2], dst_coordinate[2])
            + cal_line_length(combinate[i][3], dst_coordinate[3])
        )
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
    return np.array(combinate[force_flag]).reshape(8)


def rbox2poly_np(rboxes):
    """Convert rotated boxes to polygons

    Args:
        rboxes: [x_ctr,y_ctr,w,h,angle]

    Returns:
        poly: [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    polys = []
    for i in range(len(rboxes)):
        x_ctr, y_ctr, width, height, angle = rboxes[i][:5]
        tl_x, tl_y, br_x, br_y = -width / 2, -height / 2, width / 2, height / 2
        rect = np.array([[tl_x, br_x, br_x, tl_x], [tl_y, tl_y, br_y, br_y]])
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        poly = R.dot(rect)
        x0, x1, x2, x3 = poly[0, :4] + x_ctr
        y0, y1, y2, y3 = poly[1, :4] + y_ctr
        poly = np.array([x0, y0, x1, y1, x2, y2, x3, y3], dtype=np.float32)
        poly = get_best_begin_point_single(poly)
        polys.append(poly)
    polys = np.array(polys)
    return polys


# rbox function implemented using PyTorch
def box2corners(box: torch.Tensor) -> torch.Tensor:
    """Convert box coordinate to corners

    Args:
        box (Tensor): (B, N, 5) with (x, y, w, h, alpha) angle is in [0, 90)

    Returns:
        corners (Tensor): (B, N, 4, 2) with (x1, y1, x2, y2, x3, y3, x4, y4)
    """
    B = box.shape[0]
    x, y, w, h, alpha = torch.split(box, 1, dim=-1)
    x4 = torch.tensor(
        [0.5, 0.5, -0.5, -0.5], dtype=torch.float32, device=box.device
    ).reshape(1, 1, 4)
    x4 = x4 * w  # (B, N, 4)
    y4 = torch.tensor(
        [-0.5, 0.5, 0.5, -0.5], dtype=torch.float32, device=box.device
    ).reshape(1, 1, 4)
    y4 = y4 * h  # (B, N, 4)
    corners = torch.stack([x4, y4], dim=-1)  # (B, N, 4, 2)
    sin = torch.sin(alpha)
    cos = torch.cos(alpha)
    row1 = torch.cat([cos, sin], dim=-1)
    row2 = torch.cat([-sin, cos], dim=-1)  # (B, N, 2)
    rot_T = torch.stack([row1, row2], dim=-2)  # (B, N, 2, 2)
    rotated = torch.bmm(corners.reshape([-1, 4, 2]), rot_T.reshape([-1, 2, 2]))
    rotated = rotated.reshape([B, -1, 4, 2])  # (B*N, 4, 2) -> (B, N, 4, 2)
    rotated[..., 0] += x
    rotated[..., 1] += y
    return rotated


def check_points_in_polys(points: torch.Tensor, polys: torch.Tensor) -> torch.Tensor:
    """Check whether point is in rotated boxes

    Args:
        points (tensor): (1, L, 2) anchor points
        polys (tensor): [B, N, 4, 2] gt_polys

    Returns:
        is_in_polys (tensor): (B, N, L)
    """
    # [1, L, 2] -> [1, 1, L, 2]
    points = points.unsqueeze(0)
    # [B, N, 4, 2] -> [B, N, 1, 2]
    a, b, c, d = torch.split(polys, 1, dim=2)
    ab = b - a
    ad = d - a
    # [B, N, L, 2]
    ap = points - a
    # [B, N, 1]
    norm_ab = torch.sum(ab * ab, dim=-1)
    # [B, N, 1]
    norm_ad = torch.sum(ad * ad, dim=-1)
    # [B, N, L] dot product
    ap_dot_ab = torch.sum(ap * ab, dim=-1)
    # [B, N, L] dot product
    ap_dot_ad = torch.sum(ap * ad, dim=-1)
    # [B, N, L] <A, B> = |A|*|B|*cos(theta)
    is_in_polys = (
        (ap_dot_ab >= 0)
        & (ap_dot_ab <= norm_ab)
        & (ap_dot_ad >= 0)
        & (ap_dot_ad <= norm_ad)
    )
    return is_in_polys


def check_points_in_rotated_boxes(
    points: torch.Tensor, boxes: torch.Tensor
) -> torch.Tensor:
    """Check whether point is in rotated boxes

    Args:
        points (tensor): (1, L, 2) anchor points
        boxes (tensor): [B, N, 5] gt_bboxes

    Returns:
        is_in_box (tensor): (B, N, L)
    """
    # [B, N, 5] -> [B, N, 4, 2]
    corners = box2corners(boxes)
    # [1, L, 2] -> [1, 1, L, 2]
    points = points.unsqueeze(0)
    # [B, N, 4, 2] -> [B, N, 1, 2]
    a, b, c, d = torch.split(corners, 1, dim=2)
    ab = b - a
    ad = d - a
    # [B, N, L, 2]
    ap = points - a
    # [B, N, L]
    norm_ab = torch.sum(ab * ab, dim=-1)
    # [B, N, L]
    norm_ad = torch.sum(ad * ad, dim=-1)
    # [B, N, L] dot product
    ap_dot_ab = torch.sum(ap * ab, dim=-1)
    # [B, N, L] dot product
    ap_dot_ad = torch.sum(ap * ad, dim=-1)
    # [B, N, L] <A, B> = |A|*|B|*cos(theta)
    is_in_box = (
        (ap_dot_ab >= 0)
        & (ap_dot_ab <= norm_ab)
        & (ap_dot_ad >= 0)
        & (ap_dot_ad <= norm_ad)
    )
    return is_in_box


def rotated_iou_similarity(
    box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-9
) -> torch.Tensor:
    """Calculate IoU of rotated boxes

    NOTE: This is a placeholder implementation.
    For production use, consider using:
    - torchvision.ops (if available)
    - Custom CUDA kernel for better performance
    - Third-party libraries like mmcv

    Args:
        box1 (Tensor): box with the shape [N, M1, 5]
        box2 (Tensor): box with the shape [N, M2, 5]

    Return:
        iou (Tensor): iou between box1 and box2 with the shape [N, M1, M2]
    """
    # This is a simplified implementation
    # For accurate rotated IoU, you need Shapely or custom CUDA implementation
    raise NotImplementedError(
        "Rotated IoU requires specialized implementation. "
        "Please use torchvision.ops.box_iou or implement custom CUDA kernel."
    )


__all__ = [
    "norm_angle",
    "poly2rbox_le135_np",
    "poly2rbox_oc_np",
    "poly2rbox_np",
    "rbox2poly_np",
    "box2corners",
    "check_points_in_polys",
    "check_points_in_rotated_boxes",
    "rotated_iou_similarity",
]
