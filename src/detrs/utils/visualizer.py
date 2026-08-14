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
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

"""
Visualization utilities for detection results.

Paddle-compatible visualization functions.
"""

import logging
from typing import Any, Union

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["visualize_results", "save_result"]


def visualize_results(
    image,
    bbox_res,
    mask_res,
    segm_res,
    keypoint_res,
    pose3d_res,
    im_id,
    catid2name,
    threshold=0.5,
):
    """
    Visualize bbox and mask results (Paddle compatible API).

    Args:
        image: Input image (PIL Image or numpy array)
        bbox_res: Bounding box results
        mask_res: Mask results
        segm_res: Segmentation results
        keypoint_res: Keypoint results
        pose3d_res: 3D pose results
        im_id: Image ID
        catid2name: Category ID to name mapping
        threshold: Confidence threshold

    Returns:
        Visualized image
    """
    try:
        from PIL import Image
    except ImportError:
        logger.warning("PIL or cv2 not available, skipping visualization")
        return image

    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)

    # Draw bounding boxes
    if bbox_res is not None:
        image = draw_bbox(image, im_id, catid2name, bbox_res, threshold)

    # Draw masks
    if mask_res is not None:
        image = draw_mask(image, im_id, mask_res, threshold)

    # Draw segmentation
    if segm_res is not None:
        image = draw_segm(image, im_id, catid2name, segm_res, threshold)

    # Draw keypoints
    if keypoint_res is not None:
        image = draw_pose(image, keypoint_res, threshold)

    # Draw 3D pose
    if pose3d_res is not None:
        pose3d = np.array(pose3d_res[0]["pose3d"]) * 1000
        image = draw_pose3d(image, pose3d, visual_thread=threshold)

    return image


def draw_bbox(image, im_id, catid2name, bboxes, threshold):
    """
    Draw bounding boxes on image (Paddle compatible).

    Args:
        image: PIL Image
        im_id: Image ID
        catid2name: Category ID to name mapping
        bboxes: List of bounding box detections
        threshold: Confidence threshold

    Returns:
        Image with bboxes drawn
    """
    try:
        from PIL import ImageDraw, ImageFont
    except ImportError:
        logger.warning("PIL not available")
        return image

    draw = ImageDraw.Draw(image)

    # Use default font
    font: Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    # Generate colors for categories
    catid2color: dict[Any, np.ndarray] = {}
    color_list = _colormap(rgb=True)[:40]

    for dt in np.array(bboxes):
        if im_id != dt["image_id"]:
            continue

        catid, bbox, score = dt["category_id"], dt["bbox"], dt["score"]

        if score < threshold:
            continue

        # Get color for this category
        if catid not in catid2color:
            catid2color[catid] = color_list[len(catid2color) % len(color_list)]

        color = tuple(int(component) for component in catid2color[catid])

        # Draw box
        xmin, ymin, w, h = bbox
        xmax = xmin + w
        ymax = ymin + h

        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=2)

        # Draw label
        text = f"{catid2name.get(catid, 'unknown')}: {score:.2f}"
        draw.text((xmin, ymin - 18), text, fill=color, font=font)

    return image


def draw_mask(image, im_id, segms, threshold, alpha=0.7):
    """
    Draw segmentation mask on image.

    Args:
        image: PIL Image
        im_id: Image ID
        segms: Segmentation results
        threshold: Confidence threshold
        alpha: Mask transparency

    Returns:
        Image with masks drawn
    """
    try:
        import pycocotools.mask as mask_util
        from PIL import Image
    except ImportError:
        logger.warning("pycocotools not available")
        return image

    mask_color_id = 0
    w_ratio = 0.4
    color_list = _colormap(rgb=True)

    img_array = np.array(image).astype("float32")

    for dt in np.array(segms):
        if im_id != dt["image_id"]:
            continue

        segm, score = dt["segmentation"], dt["score"]
        if score < threshold:
            continue

        mask = mask_util.decode(segm) * 255
        color_mask = color_list[mask_color_id % len(color_list), 0:3]
        mask_color_id += 1

        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio * 255

        idx = np.nonzero(mask)
        img_array[idx[0], idx[1], :] *= 1.0 - alpha
        img_array[idx[0], idx[1], :] += alpha * color_mask

    from PIL import Image

    return Image.fromarray(img_array.astype("uint8"))


def draw_segm(image, im_id, catid2name, segms, threshold):
    """
    Draw semantic segmentation on image.

    Args:
        image: PIL Image
        im_id: Image ID
        catid2name: Category ID to name mapping
        segms: Segmentation results
        threshold: Confidence threshold

    Returns:
        Image with segmentation drawn
    """
    # Placeholder implementation
    return image


def draw_pose(image, keypoint_res, threshold):
    """
    Draw pose keypoints on image.

    Args:
        image: PIL Image
        keypoint_res: Keypoint detection results
        threshold: Confidence threshold

    Returns:
        Image with pose drawn
    """
    # Placeholder implementation
    return image


def draw_pose3d(image, pose3d, visual_thread=0.5):
    """
    Draw 3D pose on image.

    Args:
        image: PIL Image
        pose3d: 3D pose data
        visual_thread: Visualization threshold

    Returns:
        Image with 3D pose drawn
    """
    # Placeholder implementation
    return image


def save_result(save_path, results, catid2name, threshold):
    """
    Save result as txt (Paddle compatible).

    Args:
        save_path: Path to save the result txt file
        results: Results dict containing bbox_res or keypoint_res
        catid2name: Category ID to name mapping
        threshold: Score threshold for filtering results
    """
    img_id = int(results.get("im_id", 0))

    with open(save_path, "w") as f:
        if "bbox_res" in results:
            for dt in results["bbox_res"]:
                catid, bbox, score = dt["category_id"], dt["bbox"], dt["score"]
                if score < threshold:
                    continue
                # each bbox result as a line
                # for rbox: classname score x1 y1 x2 y2 x3 y3 x4 y4
                # for bbox: classname score x1 y1 w h
                bbox_pred = "{} {} ".format(catid2name[catid], score) + " ".join(
                    [str(e) for e in bbox]
                )
                f.write(bbox_pred + "\n")
        elif "keypoint_res" in results:
            for dt in results["keypoint_res"]:
                kpts = dt["keypoints"]
                scores = dt["score"]
                keypoint_pred = [img_id, scores, kpts]
                print(keypoint_pred, file=f)
        else:
            logger.info("No valid results found, skip txt save")


def _colormap(rgb=False):
    """
    Get colormap for visualization.

    Args:
        rgb: If True, return RGB colors, else BGR

    Returns:
        Colormap array
    """
    color_list = (
        np.array(
            [
                0.000,
                0.447,
                0.741,
                0.850,
                0.325,
                0.098,
                0.929,
                0.694,
                0.125,
                0.494,
                0.184,
                0.556,
                0.466,
                0.674,
                0.188,
                0.301,
                0.745,
                0.933,
                0.635,
                0.078,
                0.184,
                0.300,
                0.300,
                0.300,
                0.600,
                0.600,
                0.600,
                1.000,
                0.000,
                0.000,
                1.000,
                0.500,
                0.000,
                0.749,
                0.749,
                0.000,
                0.000,
                1.000,
                0.000,
                0.000,
                0.000,
                1.000,
                0.667,
                0.000,
                1.000,
                0.333,
                0.333,
                0.000,
                0.333,
                0.667,
                0.000,
                0.333,
                1.000,
                0.000,
                0.667,
                0.333,
                0.000,
                0.667,
                0.667,
                0.000,
                0.667,
                1.000,
                0.000,
                1.000,
                0.333,
                0.000,
                1.000,
                0.667,
                0.000,
                1.000,
                1.000,
                0.000,
                0.000,
                0.333,
                0.500,
                0.000,
                0.667,
                0.500,
                0.000,
                1.000,
                0.500,
                0.333,
                0.000,
                0.500,
                0.333,
                0.333,
                0.500,
                0.333,
                0.667,
                0.500,
                0.333,
                1.000,
                0.500,
                0.667,
                0.000,
                0.500,
                0.667,
                0.333,
                0.500,
                0.667,
                0.667,
                0.500,
                0.667,
                1.000,
                0.500,
                1.000,
                0.000,
                0.500,
                1.000,
                0.333,
                0.500,
                1.000,
                0.667,
                0.500,
                1.000,
                1.000,
                0.500,
                0.000,
                0.333,
                1.000,
                0.000,
                0.667,
                1.000,
                0.000,
                1.000,
                1.000,
                0.333,
                0.000,
                1.000,
                0.333,
                0.333,
                1.000,
                0.333,
                0.667,
                1.000,
                0.333,
                1.000,
                1.000,
                0.667,
                0.000,
                1.000,
                0.667,
                0.333,
                1.000,
                0.667,
                0.667,
                1.000,
                0.667,
                1.000,
                1.000,
                1.000,
                0.000,
                1.000,
                1.000,
                0.333,
                1.000,
                1.000,
                0.667,
                1.000,
                0.333,
                0.000,
                0.000,
                0.500,
                0.000,
                0.000,
                0.667,
                0.000,
                0.000,
                0.833,
                0.000,
                0.000,
                1.000,
                0.000,
                0.000,
                0.000,
                0.167,
                0.000,
                0.000,
                0.333,
                0.000,
                0.000,
                0.500,
                0.000,
                0.000,
                0.667,
                0.000,
                0.000,
                0.833,
                0.000,
                0.000,
                1.000,
                0.000,
                0.000,
                0.000,
                0.167,
                0.000,
                0.000,
                0.333,
                0.000,
                0.000,
                0.500,
                0.000,
                0.000,
                0.667,
                0.000,
                0.000,
                0.833,
                0.000,
                0.000,
                1.000,
                0.000,
                0.000,
                0.000,
                0.143,
                0.143,
                0.143,
                0.286,
                0.286,
                0.286,
                0.429,
                0.429,
                0.429,
                0.571,
                0.571,
                0.571,
                0.714,
                0.714,
                0.714,
                0.857,
                0.857,
                0.857,
                0.000,
                0.447,
                0.741,
                0.314,
                0.717,
                0.741,
                0.50,
                0.5,
                0,
            ]
        )
        .astype(np.float32)
        .reshape(-1, 3)
    )

    if not rgb:
        color_list = color_list[:, ::-1]

    return color_list * 255
