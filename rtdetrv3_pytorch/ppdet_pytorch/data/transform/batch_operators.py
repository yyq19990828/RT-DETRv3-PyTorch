"""
Batch-level data transforms for RT-DETRv3 PyTorch

Operations that process a batch of samples together.
Migrated from PaddlePaddle RT-DETRv3.
"""

from typing import List, Dict, Any, Tuple
import random
import numpy as np
import torch
import cv2

from ...core.workspace import register


__all__ = [
    'PadBatch', 'BatchRandomResize', 'PadGT', 'NormalizeImage',
    'NormalizeBox', 'BboxXYXY2XYWH', 'Permute'
]


@register
class PadBatch:
    """
    Pad a batch of samples so they can be divisible by a stride.
    The layout of each image should be 'CHW'.

    Args:
        pad_to_stride (int): If `pad_to_stride > 0`, pad zeros to ensure
            height and width is divisible by `pad_to_stride`.
    """

    def __init__(self, pad_to_stride: int = 0):
        self.pad_to_stride = pad_to_stride

    def __call__(self, samples: List[Dict[str, Any]], context: Dict = None) -> List[Dict[str, Any]]:
        """
        Args:
            samples (list): a batch of sample, each is dict.
        """
        coarsest_stride = self.pad_to_stride

        max_shape = np.array(
            [data['image'].shape for data in samples]).max(axis=0)

        if coarsest_stride > 0:
            max_shape[1] = int(
                np.ceil(max_shape[1] / coarsest_stride) * coarsest_stride)
            max_shape[2] = int(
                np.ceil(max_shape[2] / coarsest_stride) * coarsest_stride)

        for data in samples:
            im = data['image']
            im_c, im_h, im_w = im.shape[:]
            padding_im = np.zeros(
                (im_c, max_shape[1], max_shape[2]), dtype=np.float32)
            padding_im[:, :im_h, :im_w] = im
            data['image'] = padding_im

        return samples


@register
class BatchRandomResize:
    """
    Resize image to target size randomly. random target_size and interpolation method

    Args:
        target_size (int, list, tuple): image target size, if random size is True, must be list or tuple
        keep_ratio (bool): whether keep_ratio or not, default False
        interp (int): the interpolation method
        random_size (bool): whether random select target size of image
        random_interp (bool): whether random select interpolation method
    """

    def __init__(self,
                 target_size,
                 keep_ratio=False,
                 interp=cv2.INTER_LINEAR,
                 random_size=True,
                 random_interp=False):
        self.keep_ratio = keep_ratio
        self.interps = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ]
        self.interp = interp

        assert isinstance(target_size, (int, list, tuple)), "target_size must be int, list or tuple"
        if random_size and not isinstance(target_size, (list, tuple)):
            raise TypeError(
                f"Type of target_size is invalid when random_size is True. Must be List, now is {type(target_size)}")

        self.target_size = target_size
        self.random_size = random_size
        self.random_interp = random_interp

    def __call__(self, samples: List[Dict[str, Any]], context: Dict = None) -> List[Dict[str, Any]]:
        if self.random_size:
            target_size = random.choice(self.target_size)
        else:
            target_size = self.target_size

        if self.random_interp:
            interp = random.choice(self.interps)
        else:
            interp = self.interp

        # Resize each sample in the batch
        for sample in samples:
            img = sample['image']
            h, w = img.shape[1:]  # CHW format

            if isinstance(target_size, int):
                if self.keep_ratio:
                    scale = target_size / min(h, w)
                    new_h, new_w = int(h * scale), int(w * scale)
                else:
                    new_h, new_w = target_size, target_size
            else:
                new_h, new_w = target_size

            # Resize image (CHW -> HWC -> resize -> CHW)
            img = img.transpose(1, 2, 0)  # CHW -> HWC
            img = cv2.resize(img, (new_w, new_h), interpolation=interp)
            img = img.transpose(2, 0, 1)  # HWC -> CHW
            sample['image'] = img

            # Update image shape
            if 'im_shape' in sample:
                sample['im_shape'] = np.array([new_h, new_w], dtype=np.float32)

        return samples


@register
class PadGT:
    """
    Pad 0 to `gt_class`, `gt_bbox`, `gt_score`...
    The num_max_boxes is the largest for batch.

    Args:
        return_gt_mask (bool): If true, return `pad_gt_mask`,
                                1 means bbox, 0 means no bbox.
        pad_img (bool): whether to pad image to max shape in batch
        minimum_gtnum (int): minimum number of ground truth boxes
        only_origin_box (bool): only pad origin_gt_bbox and origin_gt_class
    """

    def __init__(self,
                 return_gt_mask=True,
                 pad_img=False,
                 minimum_gtnum=0,
                 only_origin_box=False):
        self.return_gt_mask = return_gt_mask
        self.pad_img = pad_img
        self.minimum_gtnum = minimum_gtnum
        self.only_origin_box = only_origin_box

    def __call__(self, samples: List[Dict[str, Any]], context: Dict = None) -> List[Dict[str, Any]]:
        # Find max number of boxes in batch
        num_max_boxes = max([len(s.get('gt_bbox', [])) for s in samples])
        num_max_boxes = max(self.minimum_gtnum, num_max_boxes)

        if self.only_origin_box:
            # Pad origin boxes (used in RT-DETR)
            for sample in samples:
                if self.return_gt_mask:
                    sample['pad_origin_gt_mask'] = np.zeros(
                        (num_max_boxes, 1), dtype=np.float32)

                if num_max_boxes == 0:
                    continue

                num_gt = len(sample.get('origin_gt_bbox', []))
                pad_origin_gt_class = np.zeros((num_max_boxes, 1), dtype=np.int32)
                pad_origin_gt_bbox = np.zeros((num_max_boxes, 4), dtype=np.float32)

                if num_gt > 0:
                    pad_origin_gt_class[:num_gt] = sample['origin_gt_class']
                    pad_origin_gt_bbox[:num_gt] = sample['origin_gt_bbox']

                sample['origin_gt_class'] = pad_origin_gt_class
                sample['origin_gt_bbox'] = pad_origin_gt_bbox

                if 'pad_origin_gt_mask' in sample:
                    sample['pad_origin_gt_mask'][:num_gt] = 1
        else:
            # Pad regular boxes
            for sample in samples:
                if self.return_gt_mask:
                    sample['pad_gt_mask'] = np.zeros(
                        (num_max_boxes, 1), dtype=np.float32)

                if num_max_boxes == 0:
                    continue

                num_gt = len(sample.get('gt_bbox', []))
                pad_gt_class = np.zeros((num_max_boxes, 1), dtype=np.int32)
                pad_gt_bbox = np.zeros((num_max_boxes, 4), dtype=np.float32)
                pad_gt_score = np.zeros((num_max_boxes, 1), dtype=np.float32)

                if num_gt > 0:
                    pad_gt_class[:num_gt] = sample['gt_class']
                    pad_gt_bbox[:num_gt] = sample['gt_bbox']
                    if 'gt_score' in sample:
                        pad_gt_score[:num_gt] = sample['gt_score']

                sample['gt_class'] = pad_gt_class
                sample['gt_bbox'] = pad_gt_bbox
                sample['gt_score'] = pad_gt_score

                if 'pad_gt_mask' in sample:
                    sample['pad_gt_mask'][:num_gt] = 1

        return samples


@register
class NormalizeImage:
    """
    Normalize image with mean and std.

    Args:
        mean (list): RGB mean values
        std (list): RGB std values
        norm_type (str): normalization type, 'mean_std' or 'none'
        is_scale (bool): whether to scale to [0, 1]
        is_channel_first (bool): whether the image is CHW format
    """

    def __init__(self,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 norm_type='mean_std',
                 is_scale=True,
                 is_channel_first=True):
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1) if is_channel_first else np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32).reshape(3, 1, 1) if is_channel_first else np.array(std, dtype=np.float32)
        self.is_scale = is_scale
        self.norm_type = norm_type
        self.is_channel_first = is_channel_first

    def __call__(self, samples: List[Dict[str, Any]], context: Dict = None) -> List[Dict[str, Any]]:
        for sample in samples:
            img = sample['image'].astype(np.float32, copy=False)

            if self.is_scale:
                img = img / 255.0

            if self.norm_type == 'mean_std':
                img = (img - self.mean) / self.std

            sample['image'] = img

        return samples


@register
class NormalizeBox:
    """
    Transform the bounding box's coordinates to [0,1].

    Args:
        retain_origin_box (bool): whether to retain original bbox
    """

    def __init__(self, retain_origin_box=False):
        self.retain_origin_box = retain_origin_box

    def __call__(self, samples: List[Dict[str, Any]], context: Dict = None) -> List[Dict[str, Any]]:
        for sample in samples:
            im = sample['image']

            if 'gt_bbox' in sample:
                if self.retain_origin_box:
                    sample['origin_gt_bbox'] = sample['gt_bbox'].copy()
                    sample['origin_gt_class'] = sample['gt_class'].copy()

                gt_bbox = sample['gt_bbox']
                height, width = im.shape[1:3]  # CHW format

                # Normalize to [0, 1]
                for i in range(gt_bbox.shape[0]):
                    gt_bbox[i][0] = gt_bbox[i][0] / width
                    gt_bbox[i][1] = gt_bbox[i][1] / height
                    gt_bbox[i][2] = gt_bbox[i][2] / width
                    gt_bbox[i][3] = gt_bbox[i][3] / height

                sample['gt_bbox'] = gt_bbox

        return samples


@register
class BboxXYXY2XYWH:
    """
    Convert bbox XYXY format to XYWH format.
    """

    def __init__(self):
        pass

    def __call__(self, samples: List[Dict[str, Any]], context: Dict = None) -> List[Dict[str, Any]]:
        for sample in samples:
            if 'gt_bbox' in sample:
                bbox = sample['gt_bbox']
                # Convert from [x1, y1, x2, y2] to [cx, cy, w, h]
                bbox[:, 2:4] = bbox[:, 2:4] - bbox[:, :2]  # width, height
                bbox[:, :2] = bbox[:, :2] + bbox[:, 2:4] / 2.  # center x, center y
                sample['gt_bbox'] = bbox

        return samples


@register
class Permute:
    """
    Permute image from HWC to CHW.
    Note: In our pipeline, images are already in CHW format after cv2.imread,
    but this is kept for compatibility with Paddle's pipeline.
    """

    def __init__(self, to_bgr=True, channel_first=True):
        self.to_bgr = to_bgr
        self.channel_first = channel_first

    def __call__(self, samples: List[Dict[str, Any]], context: Dict = None) -> List[Dict[str, Any]]:
        for sample in samples:
            img = sample['image']

            # If image is HWC and we need CHW
            if not self.channel_first and len(img.shape) == 3 and img.shape[-1] == 3:
                img = img.transpose(2, 0, 1)

            # RGB to BGR if needed
            if self.to_bgr and len(img.shape) == 3:
                if self.channel_first and img.shape[0] == 3:
                    img = img[[2, 1, 0], :, :]
                elif not self.channel_first and img.shape[-1] == 3:
                    img = img[:, :, ::-1]

            sample['image'] = img

        return samples


# Additional operators for completeness (used in other models)

@register
class Gt2YoloTarget:
    """
    Generate YOLOv3 targets by ground truth data.
    Note: Not used in RT-DETRv3, but preserved for compatibility.
    """
    __shared__ = ['num_classes']

    def __init__(self,
                 anchors,
                 anchor_masks,
                 downsample_ratios,
                 num_classes=80,
                 iou_thresh=1.):
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.downsample_ratios = downsample_ratios
        self.num_classes = num_classes
        self.iou_thresh = iou_thresh

    def __call__(self, samples: List[Dict[str, Any]], context: Dict = None) -> List[Dict[str, Any]]:
        # TODO: Implement YOLO target generation if needed
        # This is a placeholder for preserving Paddle logic branches
        raise NotImplementedError("Gt2YoloTarget not implemented yet. Not used in RT-DETRv3.")
