"""Distribution regression utilities shared by D-FINE model families."""

import torch

from .utils import bbox_xyxy_to_cxcywh


def _validate_reg_max(reg_max):
    if not isinstance(reg_max, int) or reg_max < 4 or reg_max % 2:
        raise ValueError("reg_max must be an even integer greater than or equal to 4")


def weighting_function(reg_max, up, reg_scale, deploy=False):
    _validate_reg_max(reg_max)
    upper_bound1 = abs(up[0]) * abs(reg_scale)
    upper_bound2 = upper_bound1 * 2
    if deploy:
        upper_bound1 = upper_bound1.item()
        upper_bound2 = upper_bound2.item()
    step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
    left = [-(step**i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
    right = [step**i - 1 for i in range(1, reg_max // 2)]
    if deploy:
        return torch.tensor(
            [-upper_bound2, *left, 0, *right, upper_bound2],
            dtype=up.dtype,
            device=up.device,
        )
    values = (
        [
            upper_bound2.reshape(1)
            if torch.is_tensor(upper_bound2)
            else up.new_tensor([upper_bound2])
        ]
        + [value.reshape(1) for value in left]
        + [torch.zeros_like(up[0:1])]
        + [value.reshape(1) for value in right]
        + [
            upper_bound2.reshape(1)
            if torch.is_tensor(upper_bound2)
            else up.new_tensor([upper_bound2])
        ]
    )
    values[0] = -values[0]
    return torch.cat(values)


def translate_gt(gt, reg_max, reg_scale, up):
    _validate_reg_max(reg_max)
    gt = gt.reshape(-1)
    values = weighting_function(reg_max, up, reg_scale)
    indices = ((values.unsqueeze(0) - gt.unsqueeze(1)) <= 0).sum(1).float() - 1
    weight_right = torch.zeros_like(indices)
    weight_left = torch.zeros_like(indices)
    valid = (indices >= 0) & (indices < reg_max)
    valid_indices = indices[valid].long()
    left_diff = (gt[valid] - values[valid_indices]).abs()
    right_diff = (values[valid_indices + 1] - gt[valid]).abs()
    weight_right[valid] = left_diff / (left_diff + right_diff)
    weight_left[valid] = 1 - weight_right[valid]
    negative = indices < 0
    weight_left[negative] = 1
    indices[negative] = 0
    positive = indices >= reg_max
    weight_right[positive] = 1
    indices[positive] = reg_max - 0.1
    return indices, weight_right, weight_left


def distance2bbox(points, distance, reg_scale):
    reg_scale = abs(reg_scale)
    x1 = points[..., 0] - (0.5 * reg_scale + distance[..., 0]) * (
        points[..., 2] / reg_scale
    )
    y1 = points[..., 1] - (0.5 * reg_scale + distance[..., 1]) * (
        points[..., 3] / reg_scale
    )
    x2 = points[..., 0] + (0.5 * reg_scale + distance[..., 2]) * (
        points[..., 2] / reg_scale
    )
    y2 = points[..., 1] + (0.5 * reg_scale + distance[..., 3]) * (
        points[..., 3] / reg_scale
    )
    return bbox_xyxy_to_cxcywh(torch.stack([x1, y1, x2, y2], -1))


def bbox2distance(points, bbox, reg_max, reg_scale, up, eps=0.1):
    _validate_reg_max(reg_max)
    reg_scale = abs(reg_scale)
    width = points[..., 2] / reg_scale + 1e-16
    height = points[..., 3] / reg_scale + 1e-16
    distances = torch.stack(
        [
            (points[:, 0] - bbox[:, 0]) / width - 0.5 * reg_scale,
            (points[:, 1] - bbox[:, 1]) / height - 0.5 * reg_scale,
            (bbox[:, 2] - points[:, 0]) / width - 0.5 * reg_scale,
            (bbox[:, 3] - points[:, 1]) / height - 0.5 * reg_scale,
        ],
        -1,
    )
    indices, weight_right, weight_left = translate_gt(distances, reg_max, reg_scale, up)
    indices = indices.clamp(min=0, max=reg_max - eps)
    return indices.reshape(-1).detach(), weight_right.detach(), weight_left.detach()


__all__ = [
    "bbox2distance",
    "distance2bbox",
    "translate_gt",
    "weighting_function",
]
