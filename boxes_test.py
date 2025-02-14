#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
visualize_nuscenes_bboxes.py

A script to load a NuScenes-based dataset and trained model,
generate 3D bounding box predictions, and visualize them on the BEV image.

Does NOT run the official NuScenesEval, so no assertion errors about splits.
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.ops import nms

# Import your classes from Model.py (adjust if needed)
from Model import (
    Config,
    NuScenesHybridDataset,
    HybridDetector
)

# If you want to load the NuScenes metadata:
from nuscenes.nuscenes import NuScenes


def decode_outputs(outputs, sample, class_list, device,
                   conf_threshold=0.1, nms_threshold=0.4):
    """
    Convert raw model outputs into bounding box dictionaries for visualization.
    """
    # 'reg_pred' shape is [1, max_objects, 7]
    pred_boxes = outputs['reg_pred'][0].cpu().numpy()

    # 'cls_logits' shape is [1, max_objects, num_classes]
    pred_cls = torch.softmax(outputs['cls_logits'], dim=2)[0].cpu().numpy()

    # Filter by confidence threshold
    conf_scores = np.max(pred_cls, axis=1)  # Max score per box
    valid_mask = conf_scores > conf_threshold
    pred_boxes = pred_boxes[valid_mask]
    pred_cls = pred_cls[valid_mask]

    # Construct dictionary list
    decoded_boxes = []
    for box, cls_prob in zip(pred_boxes, pred_cls):
        x, y, w, l, yaw, z, height = box
        class_id = int(np.argmax(cls_prob))
        decoded_boxes.append({
            'x': float(x),
            'y': float(y),
            'w': float(w),
            'l': float(l),
            'yaw': float(yaw),
            'z': float(z),
            'height': float(height),
            'class_id': class_id,
            'confidence': float(np.max(cls_prob))
        })

    # Apply NMS in BEV plane
    if len(decoded_boxes) > 0:
        boxes_array = np.array([
            [b['x'] - b['w'] / 2, b['y'] - b['l'] / 2,
             b['x'] + b['w'] / 2, b['y'] + b['l'] / 2]
            for b in decoded_boxes
        ], dtype=np.float32)

        scores = np.array([b['confidence'] for b in decoded_boxes], dtype=np.float32)
        keep_indices = nms(torch.tensor(boxes_array), torch.tensor(scores), nms_threshold)
        keep_indices = keep_indices.cpu().numpy() if isinstance(keep_indices, torch.Tensor) else keep_indices
        decoded_boxes = [decoded_boxes[idx] for idx in keep_indices]

    return decoded_boxes


def get_gt_boxes(sample):
    """
    Convert ground-truth boxes for visualization in the same format as predicted boxes.
    """
    gt_boxes = []
    for box, label in zip(sample['target']['boxes'], sample['target']['labels']):
        if label == -1:
            continue
        gt_boxes.append({
            'x': float(box[0].item()),
            'y': float(box[1].item()),
            'w': float(box[2].item()),
            'l': float(box[3].item()),
            'yaw': float(box[4].item()),
            'z': float(box[5].item()),
            'height': float(box[6].item()),
            'class_id': int(label.item())
        })
    return gt_boxes


def draw_boxes(bev, boxes, dataset, color='red', title='Predictions'):
    """
    Draw bounding boxes on a BEV image using matplotlib.
    'boxes' is a list of dicts with x, y, w, l, yaw, class_id, confidence, etc.
    """
    plt.imshow(bev[..., 0], cmap='viridis')
    plt.title(title)
    ax = plt.gca()

    for box in boxes:
        # Convert from world coordinates to pixel coords in the BEV image
        x_px = (box['x'] + (Config.bev_size[0]//2 * Config.bev_resolution)) / Config.bev_resolution
        y_px = (box['y'] + (Config.bev_size[1]//2 * Config.bev_resolution)) / Config.bev_resolution
        w_px = box['w'] / Config.bev_resolution
        l_px = box['l'] / Config.bev_resolution

        # Check if the rectangle would be on-screen
        if (0 <= x_px < bev.shape[1]) and (0 <= y_px < bev.shape[0]):
            rect = plt.Rectangle((x_px - w_px/2, y_px - l_px/2),
                                 w_px, l_px,
                                 angle=np.degrees(box['yaw']),
                                 fill=False, color=color, linewidth=2)
            ax.add_patch(rect)

            # Label (class name & confidence)
            class_id = box.get('class_id', -1)
            if class_id >= 0 and class_id < len(dataset.class_map):
                class_name = list(dataset.class_map.keys())[class_id]
            else:
                class_name = 'unknown'

            conf_str = ""
            if 'confidence' in box:
                conf_str = f"\n{box['confidence']:.2f}"
            plt.text(x_px, y_px, f"{class_name}{conf_str}",
                     color='white', fontsize=8, ha='center')


def main(args):
    """
    Main function to load model, dataset, predict, and visualize on BEV.
    """
    # 1. Load NuScenes object (optional if needed by the dataset)
    nusc = NuScenes(version=Config.version, dataroot=Config.dataroot, verbose=False)

    # 2. Create dataset
    dataset = NuScenesHybridDataset(nusc)
    print(f"Dataset length: {len(dataset)}")

    # Check sample_idx is in range
    if not (0 <= args.sample_idx < len(dataset)):
        raise ValueError(f"sample_idx {args.sample_idx} out of range [0, {len(dataset)-1}]")

    # 3. Load model
    device = Config.device
    model = HybridDetector(num_classes=Config.num_classes).to(device)
    print(f"Loading model weights from {args.model_path} ...")
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Get one sample from the dataset
    sample = dataset[args.sample_idx]
    bev = sample['bev'].unsqueeze(0).to(device)
    points = sample['points'].unsqueeze(0).to(device)

    # 5. Inference
    with torch.no_grad():
        outputs = model(bev, points)

    pred_boxes = decode_outputs(
        outputs,
        sample,
        class_list=list(dataset.class_map.keys()),
        device=device,
        conf_threshold=args.conf_thresh,
        nms_threshold=args.nms_thresh
    )

    # Debug: print how many boxes
    print(f"Number of predicted boxes: {len(pred_boxes)}")
    for i, box in enumerate(pred_boxes):
        print(f"Pred Box {i}: {box}")

    # 6. Plot Predictions vs. Ground Truth in BEV
    bev_np = sample['bev'].numpy().transpose(1, 2, 0)

    # A) Predictions
    plt.figure(figsize=(20, 8))

    plt.subplot(1, 2, 1)
    draw_boxes(bev_np, pred_boxes, dataset, color='red', title='Predictions')

    # B) Ground Truth
    gt_boxes = get_gt_boxes(sample)
    print(f"Number of ground-truth boxes: {len(gt_boxes)}")

    plt.subplot(1, 2, 2)
    draw_boxes(bev_np, gt_boxes, dataset, color='green', title='Ground Truth')

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize NuScenes bounding boxes on BEV.")
    parser.add_argument("--model_path", type=str, default="hybrid_detector.pth",
                        help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Index of sample to visualize from the dataset.")
    parser.add_argument("--conf_thresh", type=float, default=0.1,
                        help="Confidence threshold for predicted boxes.")
    parser.add_argument("--nms_thresh", type=float, default=0.4,
                        help="IOU threshold for non-maximum suppression.")
    args = parser.parse_args()

    main(args)
