"""
Test và so sánh hiệu suất của YOLOv8 và Faster R-CNN với RPL
Đánh giá trên folder ảnh test và lưu kết quả dự đoán
"""

import os
import time
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torchvision
from torchvision.transforms import functional as F
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from ultralytics import YOLO
import pandas as pd
import json
from datetime import datetime

# Import các module từ train_FasterRNN.py
import sys
sys.path.append(str(Path(__file__).parent))

from train_FasterRNN import (
    FasterRCNN_RPL,
    smart_resize_with_padding,
    transform_boxes,
    RPL_NUM_BLOCKS,
    USE_RPL
)

# ============ CONFIGURATION ============
# Paths to checkpoints
YOLOV8_CHECKPOINT = "UI/model/vietfood67_yolov8n/best.pt"
FASTERRCNN_CHECKPOINT = "UI/model/checkpoint_best_rpl.pth"

# Test data paths
TEST_IMG_DIR = "My First Project.v1i.yolov8/test/images"
TEST_LBL_DIR = "My First Project.v1i.yolov8/test/labels"

# Output paths
OUTPUT_BASE_DIR = Path("test_results")
YOLO_OUTPUT_DIR = OUTPUT_BASE_DIR / "yolov8_predictions"
FRCNN_OUTPUT_DIR = OUTPUT_BASE_DIR / "fasterrcnn_predictions"
METRICS_OUTPUT_DIR = OUTPUT_BASE_DIR / "metrics"

# Create output directories
for dir_path in [YOLO_OUTPUT_DIR, FRCNN_OUTPUT_DIR, METRICS_OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model settings
IMG_SIZE = 640
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
VISUALIZE_RESULTS = True # Set to False to speed up evaluation (skip saving images)

# Class names (68 classes + background for Faster R-CNN)
import yaml

# ============ LOAD CLASS MAPPINGS ============
# 1. Load Training Config (Source of Truth for Model Output)
TRAIN_YAML_PATH = "data.yaml" # Root yaml used for training
if not os.path.exists(TRAIN_YAML_PATH):
    print(f"Error: Training config not found at {TRAIN_YAML_PATH}")
    exit(1)

with open(TRAIN_YAML_PATH, 'r') as f:
    train_config = yaml.safe_load(f)
    # Ensure values are strings and strip whitespace
    TRAIN_CLASS_NAMES = [str(x).strip() for x in train_config['names']]

# 2. Load Test Config (Source of Truth for Ground Truth)
TEST_YAML_PATH = "My First Project.v1i.yolov8/data.yaml"
if not os.path.exists(TEST_YAML_PATH):
    print(f"Error: Test config not found at {TEST_YAML_PATH}")
    exit(1)

with open(TEST_YAML_PATH, 'r') as f:
    test_config = yaml.safe_load(f)
    TEST_CLASS_NAMES = [str(x).strip() for x in test_config['names']]

print(f"\nLoaded Class Configs:")
print(f"  Training Classes (Model Output): {len(TRAIN_CLASS_NAMES)} classes")
print(f"  Test Classes (Ground Truth):     {len(TEST_CLASS_NAMES)} classes")

# 3. Create Mapping: Train ID -> Name -> Test ID
# This maps the Model's prediction (Train ID) to the Dataset's expected label (Test ID)
train_id_to_test_id = {}
print("\nCreating Class Mapping:")
for train_idx, name in enumerate(TRAIN_CLASS_NAMES):
    # Case insensitive matching just in case
    try:
        # Find this name in the test list
        # Try exact match first
        if name in TEST_CLASS_NAMES:
            test_idx = TEST_CLASS_NAMES.index(name)
        else:
            # Try case-insensitive
            lower_test_names = [x.lower() for x in TEST_CLASS_NAMES]
            test_idx = lower_test_names.index(name.lower())
        
        train_id_to_test_id[train_idx] = test_idx
    except ValueError:
        print(f"  WARNING: Class '{name}' (Train ID {train_idx}) not found in Test Config!")
        train_id_to_test_id[train_idx] = -1 # Unmappable

# Validation: Check first few
print("  Mapping Sample (Train ID -> Test ID):")
for i in range(min(5, len(train_id_to_test_id))):
    print(f"    {i} ({TRAIN_CLASS_NAMES[i]}) -> {train_id_to_test_id[i]} ({TEST_CLASS_NAMES[train_id_to_test_id[i]]})")

# Use Test classes for final display/metrics since we map everything to them
CLASS_NAMES = TEST_CLASS_NAMES
NUM_CLASSES = len(CLASS_NAMES) + 1 

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============ UTILITY FUNCTIONS ============
def parse_yolo_label(label_path, img_w, img_h):
    """
    Parse YOLO format label file and convert to [x1, y1, x2, y2] format
    YOLO format: class x_center y_center width height (ALL NORMALIZED 0-1)
    Returns: boxes (N, 4), labels (N,)
    """
    boxes = []
    labels = []

    if not os.path.exists(label_path):
        return np.array([]), np.array([])

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls, xc_norm, yc_norm, bw_norm, bh_norm = map(float, parts)
            cls = int(cls)

            # Convert from normalized YOLO format (0-1) to pixel xyxy
            # YOLO: [x_center, y_center, width, height] all normalized
            # Output: [x1, y1, x2, y2] in pixels
            xc = xc_norm * img_w
            yc = yc_norm * img_h
            bw = bw_norm * img_w
            bh = bh_norm * img_h

            x1 = xc - bw / 2
            y1 = yc - bh / 2
            x2 = xc + bw / 2
            y2 = yc + bh / 2

            # Clip to image bounds
            x1 = max(0, min(x1, img_w))
            y1 = max(0, min(y1, img_h))
            x2 = max(0, min(x2, img_w))
            y2 = max(0, min(y2, img_h))

            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(cls)

    return np.array(boxes), np.array(labels)


def draw_predictions(image, boxes, labels, scores, class_names, color=(0, 255, 0)):
    """
    Draw bounding boxes and labels on image
    """
    img_draw = image.copy()

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)

        # Draw box
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label_idx = int(label)
        if label_idx < len(class_names):
            label_text = f"{class_names[label_idx]}: {score:.2f}"
        else:
            label_text = f"Class {label_idx}: {score:.2f}"

        # Background for text
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            img_draw, (x1, y1 - text_height - 5),
            (x1 + text_width, y1), color, -1
        )

        # Text
        cv2.putText(
            img_draw, label_text, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

    return img_draw


# ============ MODEL LOADING ============
def load_yolov8_model(checkpoint_path):
    """Load YOLOv8 model"""
    print(f"\nLoading YOLOv8 from {checkpoint_path}...")
    model = YOLO(checkpoint_path)
    print("YOLOv8 loaded successfully!")
    return model


def load_fasterrcnn_model(checkpoint_path, num_classes):
    """Load Faster R-CNN with RPL model"""
    print(f"\nLoading Faster R-CNN from {checkpoint_path}...")

    model = FasterRCNN_RPL(
        num_classes=num_classes,
        use_rpl=USE_RPL,
        num_rpl_blocks=RPL_NUM_BLOCKS
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    print("Faster R-CNN loaded successfully!")
    return model


# ============ ACCURACY CALCULATION ============
def calculate_bbox_stats(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5):
    """
    Calculate True Positives (TP) for the image using greedy matching.
    Returns: (tp_count, num_preds, num_gts)
    """
    num_preds = len(pred_boxes)
    num_gts = len(gt_boxes)
    
    if num_gts == 0:
        return 0, num_preds, 0
    if num_preds == 0:
        return 0, 0, num_gts
        
    gts = torch.tensor(gt_boxes, dtype=torch.float32)
    preds = torch.tensor(pred_boxes, dtype=torch.float32)
    
    # IoU Calculation
    area_gts = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])
    area_preds = (preds[:, 2] - preds[:, 0]) * (preds[:, 3] - preds[:, 1])
    
    lt = torch.max(gts[:, None, :2], preds[:, :2])
    rb = torch.min(gts[:, None, 2:], preds[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area_gts[:, None] + area_preds - inter
    iou = inter / (union + 1e-6) # [M, N]
    
    # Label matching [M, N]
    gt_l = torch.tensor(gt_labels)
    pred_l = torch.tensor(pred_labels)
    label_match = gt_l[:, None] == pred_l
    
    # Valid candidate matches
    valid_matrix = (iou > iou_threshold) & label_match
    
    # Greedy Matching
    # We want to maximize matches. Since this is evaluation, for each GT, finding the best Pred?
    # Or for each Pred, finding if it matches a GT?
    # Standard PASCAL VOC/COCO: Predicts are sorted by Key. 
    # Here predictions are unordered/random. We'll simply iterate GTs and match to best Pred, or vice versa.
    # Let's match Pair with Highest IoU first.
    
    # Get all indices where valid
    match_indices = torch.nonzero(valid_matrix, as_tuple=False) # [K, 2] (gt_idx, pred_idx)
    
    if len(match_indices) == 0:
        return 0, num_preds, num_gts
        
    # Get corresponding IoUs
    match_ious = iou[match_indices[:, 0], match_indices[:, 1]]
    
    # Sort by IoU descending
    sorted_idx = torch.argsort(match_ious, descending=True)
    match_indices = match_indices[sorted_idx]
    
    matched_gt = set()
    matched_pred = set()
    tp_count = 0
    
    for gt_idx, pred_idx in match_indices:
        gt_idx = gt_idx.item()
        pred_idx = pred_idx.item()
        
        if gt_idx not in matched_gt and pred_idx not in matched_pred:
            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)
            tp_count += 1
            
    return tp_count, num_preds, num_gts


# ============ TESTING FUNCTIONS ============
@torch.no_grad()
def test_yolov8(model, test_img_dir, test_lbl_dir, output_dir, valid_map, model_class_names, conf_threshold=0.25):
    """
    Test YOLOv8 model on test images
    """
    print("\n" + "="*60)
    print("TESTING YOLOv8")
    print("="*60)

    test_img_dir = Path(test_img_dir)
    test_lbl_dir = Path(test_lbl_dir)
    output_dir = Path(output_dir)

    # Get all test images
    image_files = sorted(list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png")))

    if len(image_files) == 0:
        print(f"No images found in {test_img_dir}")
        return None

    print(f"Found {len(image_files)} test images")

    # Metric calculator
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
    metric_95 = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", iou_thresholds=[0.95])

    predictions_list = []
    targets_list = []
    
    inference_times = []
    
    # Accuracy counters
    # Image Level
    correct_images_count = 0 
    total_valid_images = 0 
    
    # BBox Level
    total_tp = 0
    total_pred_boxes_count = 0
    total_gt_boxes_count = 0

    for img_path in tqdm(image_files, desc="Testing YOLOv8"):
        # Read image
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]

        # Get ground truth
        label_path = test_lbl_dir / img_path.name.replace('.jpg', '.txt').replace('.png', '.txt')
        gt_boxes, gt_labels = parse_yolo_label(label_path, img_w, img_h)

        # Predict
        start_time = time.time()
        # Pass original image (BGR) to YOLOv8 - it expects BGR for numpy arrays
        # Explicitly pass device to ensure GPU usage
        results = model.predict(img, imgsz=IMG_SIZE, conf=conf_threshold, verbose=False, device=device)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)

        # Parse results
        result = results[0]

        # Get predictions
        pred_scores = result.boxes.conf.cpu().numpy()
        pred_labels_raw = result.boxes.cls.cpu().numpy().astype(int)
        
        # YOLOv8 boxes - use xyxyn (normalized xyxy format)
        if len(result.boxes) > 0:
            # Get normalized boxes (xyxyn format: x1, y1, x2, y2 - all 0-1)
            boxes_xyxyn = result.boxes.xyxyn.cpu().numpy()

            # Convert to pixel coordinates
            pred_boxes = boxes_xyxyn.copy()
            pred_boxes[:, [0, 2]] *= img_w  # x1, x2
            pred_boxes[:, [1, 3]] *= img_h  # y1, y2

            # Clip to image bounds
            pred_boxes[:, [0, 2]] = np.clip(pred_boxes[:, [0, 2]], 0, img_w)
            pred_boxes[:, [1, 3]] = np.clip(pred_boxes[:, [1, 3]], 0, img_h)
        else:
            pred_boxes = np.array([])

        # Map labels (Train ID -> Test ID)
        pred_labels = []
        valid_indices = []
        for i, raw_label in enumerate(pred_labels_raw):
            mapped_label = valid_map.get(raw_label, -1)
            if mapped_label != -1:
                pred_labels.append(mapped_label)
                valid_indices.append(i)
        
        pred_labels = np.array(pred_labels)
        
        # Filter boxes/scores by valid mapping
        if len(valid_indices) < len(pred_boxes):
            pred_boxes = pred_boxes[valid_indices]
            pred_scores = pred_scores[valid_indices]
        
        # Add to metric
        if len(pred_boxes) > 0:
            predictions_list.append({
                'boxes': torch.tensor(pred_boxes, dtype=torch.float32),
                'scores': torch.tensor(pred_scores, dtype=torch.float32),
                'labels': torch.tensor(pred_labels, dtype=torch.int64)
            })
        else:
            predictions_list.append({
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'scores': torch.zeros((0,), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64)
            })

        if len(gt_boxes) > 0:
            targets_list.append({
                'boxes': torch.tensor(gt_boxes, dtype=torch.float32),
                'labels': torch.tensor(gt_labels, dtype=torch.int64)
            })
        else:
            targets_list.append({
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64)
            })


        # Draw and save (Optional)
        if VISUALIZE_RESULTS:
            img_draw = img_rgb.copy()

            # ... (drawing code) ...

            if len(pred_boxes) > 0:
                img_draw = draw_predictions(
                    img_draw, pred_boxes, pred_labels, pred_scores,
                    CLASS_NAMES, color=(0, 0, 255)  # Red for predictions
                )

            img_draw_bgr = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
            output_path = output_dir / img_path.name
            cv2.imwrite(str(output_path), img_draw_bgr)
            
        # Accuracy Check
        if len(gt_boxes) > 0:
            total_valid_images += 1
            # BBox Stats
            matches, n_pred, n_gt = calculate_bbox_stats(pred_boxes, pred_labels, gt_boxes, gt_labels, IOU_THRESHOLD)
            total_tp += matches
            total_pred_boxes_count += n_pred
            total_gt_boxes_count += n_gt
            
            # For Image Accuracy: if at least 1 match was found in this image
            if matches > 0:
                correct_images_count += 1
        else:
             # Even if no GT, preds are False Positives
             matches, n_pred, n_gt = calculate_bbox_stats(pred_boxes, pred_labels, gt_boxes, gt_labels, IOU_THRESHOLD)
             total_tp += matches
             total_pred_boxes_count += n_pred
             total_gt_boxes_count += n_gt

    # Calculate metrics
    metric.update(predictions_list, targets_list)
    metric_95.update(predictions_list, targets_list)
    
    metrics = metric.compute()
    metrics_95 = metric_95.compute()

    # Convert to dict
    results_dict = {
        'mAP': float(metrics['map'].item()),
        'mAP_50': float(metrics['map_50'].item()),
        'mAP_75': float(metrics['map_75'].item()),
        'mAP_95': float(metrics_95['map'].item()), # Only comprised of 0.95 threshold
        'mAP_small': float(metrics['map_small'].item()),
        'mAP_medium': float(metrics['map_medium'].item()),
        'mAP_large': float(metrics['map_large'].item()),
        'mAR_100': float(metrics['mar_100'].item()), # Mean Average Recall
        'avg_inference_time_ms': np.mean(inference_times) * 1000,
        'fps': len(image_files) / sum(inference_times)
    }
    
    # Process Per-Class Metrics
    per_class_map = metrics['map_per_class']
    # If metric doesn't return all classes (e.g. if some classes missing in GT), we need to handle alignment
    # But usually map_per_class length matches the number of classes seen or defined.
    # Note: MeanAveragePrecision usually returns tensor of size (C).
    
    # We will create a DataFrame for per-class results
    per_class_data = []
    
    # We iterate through the actual test classes (CLASS_NAMES)
    # The metric indices correspond to the label indices passed in targets/preds
    # Since we mapped everything to Test IDs, index i corresponds to CLASS_NAMES[i]
    
    device_cpu = torch.device('cpu')
    map_per_class_cpu = per_class_map.to(device_cpu)
    
    for i, class_name in enumerate(CLASS_NAMES):
        if i < len(map_per_class_cpu):
             ap = float(map_per_class_cpu[i].item())
        else:
             ap = -1.0 # Should not happen if configured correctly
        
        per_class_data.append({
            'Class_ID': i,
            'Class_Name': class_name,
            'AP': ap
        })
        
    per_class_df = pd.DataFrame(per_class_data)

    print("\nYOLOv8 Results:")
    print(f"  mAP: {results_dict['mAP']:.4f}")
    print(f"  mAP@50: {results_dict['mAP_50']:.4f}")
    print(f"  mAP@75: {results_dict['mAP_75']:.4f}")
    print(f"  mAP@95: {results_dict['mAP_95']:.4f}")
    print(f"  mAR@100: {results_dict['mAR_100']:.4f}")
    print(f"  FPS: {results_dict['fps']:.2f}")

    # Accuracy - Image Level
    image_accuracy = correct_images_count / total_valid_images if total_valid_images > 0 else 0.0
    results_dict['image_accuracy'] = image_accuracy
    print(f"  Image Accuracy (Correct Class Detected): {image_accuracy*100:.2f}% ({correct_images_count}/{total_valid_images})")
    
    # Accuracy - BBox Level
    precision = total_tp / total_pred_boxes_count if total_pred_boxes_count > 0 else 0.0
    recall = total_tp / total_gt_boxes_count if total_gt_boxes_count > 0 else 0.0
    results_dict['bbox_precision'] = precision
    results_dict['bbox_recall'] = recall
    print(f"  BBox Precision (TP/TotalPreds): {precision*100:.2f}% ({total_tp}/{total_pred_boxes_count})")
    print(f"  BBox Recall (TP/TotalGTs):      {recall*100:.2f}% ({total_tp}/{total_gt_boxes_count})")

    return results_dict, per_class_df



@torch.no_grad()
def test_fasterrcnn(model, test_img_dir, test_lbl_dir, output_dir, valid_map, model_class_names, conf_threshold=0.25):
    """
    Test Faster R-CNN model on test images
    """
    print("\n" + "="*60)
    print("TESTING FASTER R-CNN")
    print("="*60)

    test_img_dir = Path(test_img_dir)
    test_lbl_dir = Path(test_lbl_dir)
    output_dir = Path(output_dir)

    # Get all test images
    image_files = sorted(list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png")))

    if len(image_files) == 0:
        print(f"No images found in {test_img_dir}")
        return None

    print(f"Found {len(image_files)} test images")

    # Metric calculator
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
    metric_95 = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", iou_thresholds=[0.95])

    predictions_list = []
    targets_list = []
    
    inference_times = []
    
    # Accuracy counters
    # Image Level
    correct_images_count = 0 
    total_valid_images = 0 
    
    # BBox Level
    total_tp = 0
    total_pred_boxes_count = 0
    total_gt_boxes_count = 0

    for img_path in tqdm(image_files, desc="Testing Faster R-CNN"):
        # Read image
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]

        # Get ground truth
        label_path = test_lbl_dir / img_path.name.replace('.jpg', '.txt').replace('.png', '.txt')
        gt_boxes_orig, gt_labels = parse_yolo_label(label_path, orig_w, orig_h)

        # Resize image with padding
        img_resized, scale, offset = smart_resize_with_padding(img_rgb, IMG_SIZE)
        
        # Transform GT boxes (only for visualization if needed, but we use orig for metrics now)
        # Note: We kept the original boxes in gt_boxes_orig


        # Convert to tensor
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().div(255.0)
        img_tensor = img_tensor.to(device)

        # Predict
        start_time = time.time()
        predictions = model([img_tensor])
        inference_time = time.time() - start_time
        inference_times.append(inference_time)

        pred = predictions[0]

        # Filter by confidence
        keep_idx = pred['scores'] >= conf_threshold
        pred_boxes = pred['boxes'][keep_idx].cpu().numpy()
        pred_scores = pred['scores'][keep_idx].cpu().numpy()
        pred_labels_raw = pred['labels'][keep_idx].cpu().numpy() 
        
        # Map labels
        # Faster R-CNN returns 1-indexed (0=BG), so subtract 1 first to get Train ID
        pred_labels = []
        valid_indices = []
        for i, raw_label in enumerate(pred_labels_raw):
            train_id = raw_label - 1 # Convert to 0-indexed Train ID
            if 0 <= train_id < len(model_class_names):
                mapped_label = valid_map.get(train_id, -1)
                if mapped_label != -1:
                    pred_labels.append(mapped_label)
                    valid_indices.append(i)
        
        pred_labels = np.array(pred_labels)

        # Filter boxes/scores
        if len(valid_indices) < len(pred_boxes):
            pred_boxes = pred_boxes[valid_indices]
            pred_scores = pred_scores[valid_indices]

        # Transform predictions back to original image space
        if len(pred_boxes) > 0:
            # Reverse padding
            left_pad, top_pad = offset
            pred_boxes[:, [0, 2]] = (pred_boxes[:, [0, 2]] - left_pad) / scale
            pred_boxes[:, [1, 3]] = (pred_boxes[:, [1, 3]] - top_pad) / scale

            # Clip to image bounds
            pred_boxes[:, [0, 2]] = np.clip(pred_boxes[:, [0, 2]], 0, orig_w)
            pred_boxes[:, [1, 3]] = np.clip(pred_boxes[:, [1, 3]], 0, orig_h)

        # Add to metric (using ORIGINAL coordinates)
        if len(gt_boxes_orig) > 0:
            gt_boxes_tensor = torch.tensor(gt_boxes_orig, dtype=torch.float32)
        else:
            gt_boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)

        targets_list.append({
            'boxes': gt_boxes_tensor,
            'labels': torch.tensor(gt_labels, dtype=torch.int64) if len(gt_labels) > 0 else torch.zeros((0,), dtype=torch.int64)
        })

        # Predictions in ORIGINAL space
        if len(pred_boxes) > 0:
            predictions_list.append({
                'boxes': torch.tensor(pred_boxes, dtype=torch.float32),
                'scores': torch.tensor(pred_scores, dtype=torch.float32),
                'labels': torch.tensor(pred_labels, dtype=torch.int64)
            })
        else:
            predictions_list.append({
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'scores': torch.zeros((0,), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64)
            })


        # Draw and save (Optional)
        if VISUALIZE_RESULTS:
            img_draw = img_rgb.copy()

            # Draw Ground Truth (GREEN)
            if len(gt_boxes_orig) > 0:
                for box, label in zip(gt_boxes_orig, gt_labels):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_text = f"GT: {CLASS_NAMES[label]}"
                    cv2.putText(img_draw, label_text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw Predictions (RED)
            if len(pred_boxes) > 0:
                img_draw = draw_predictions(
                    img_rgb, pred_boxes, pred_labels, pred_scores,
                    CLASS_NAMES, color=(0, 0, 255)  # Red for predictions
                )

            img_draw_bgr = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
            output_path = output_dir / img_path.name
            cv2.imwrite(str(output_path), img_draw_bgr)
            
        # Accuracy Check (Use original boxes for GT and predictions mapped to valid_map)
        # Note: gt_boxes_orig is in original scale. pred_boxes (at this point) is back in original scale.
        if len(gt_boxes_orig) > 0:
            total_valid_images += 1
            
            # BBox Stats
            matches, n_pred, n_gt = calculate_bbox_stats(pred_boxes, pred_labels, gt_boxes_orig, gt_labels, IOU_THRESHOLD)
            total_tp += matches
            total_pred_boxes_count += n_pred
            total_gt_boxes_count += n_gt
            
            if matches > 0:
                correct_images_count += 1
        else:
             # Even if no GT, preds are False Positives
             matches, n_pred, n_gt = calculate_bbox_stats(pred_boxes, pred_labels, gt_boxes_orig, gt_labels, IOU_THRESHOLD)
             total_tp += matches
             total_pred_boxes_count += n_pred
             total_gt_boxes_count += n_gt

    # Calculate metrics
    metric.update(predictions_list, targets_list)
    metric_95.update(predictions_list, targets_list)
    metrics = metric.compute()
    metrics_95 = metric_95.compute()

    # Convert to dict
    results_dict = {
        'mAP': float(metrics['map'].item()),
        'mAP_50': float(metrics['map_50'].item()),
        'mAP_75': float(metrics['map_75'].item()),
        'mAP_95': float(metrics_95['map'].item()), # Only comprised of 0.95 threshold
        'mAP_small': float(metrics['map_small'].item()),
        'mAP_medium': float(metrics['map_medium'].item()),
        'mAP_large': float(metrics['map_large'].item()),
        'mAR_100': float(metrics['mar_100'].item()),
        'avg_inference_time_ms': np.mean(inference_times) * 1000,
        'fps': len(image_files) / sum(inference_times)
    }
    
    # Process Per-Class Metrics
    per_class_map = metrics['map_per_class']
    per_class_data = []
    
    device_cpu = torch.device('cpu')
    map_per_class_cpu = per_class_map.to(device_cpu)
    
    for i, class_name in enumerate(CLASS_NAMES):
        if i < len(map_per_class_cpu):
             ap = float(map_per_class_cpu[i].item())
        else:
             ap = -1.0 # Should not happen if configured correctly
        
        per_class_data.append({
            'Class_ID': i,
            'Class_Name': class_name,
            'AP': ap
        })
        
    per_class_df = pd.DataFrame(per_class_data)

    print("\nFaster R-CNN Results:")
    print(f"  mAP: {results_dict['mAP']:.4f}")
    print(f"  mAP@50: {results_dict['mAP_50']:.4f}")
    print(f"  mAP@75: {results_dict['mAP_75']:.4f}")
    print(f"  mAP@95: {results_dict['mAP_95']:.4f}")
    print(f"  mAR@100: {results_dict['mAR_100']:.4f}")
    print(f"  Avg Inference Time: {results_dict['avg_inference_time_ms']:.2f}ms")
    print(f"  FPS: {results_dict['fps']:.2f}")

    # Accuracy - Image Level
    image_accuracy = correct_images_count / total_valid_images if total_valid_images > 0 else 0.0
    results_dict['image_accuracy'] = image_accuracy
    print(f"  Image Accuracy (Correct Class Detected): {image_accuracy*100:.2f}% ({correct_images_count}/{total_valid_images})")
    
    # Accuracy - BBox Level
    precision = total_tp / total_pred_boxes_count if total_pred_boxes_count > 0 else 0.0
    recall = total_tp / total_gt_boxes_count if total_gt_boxes_count > 0 else 0.0
    results_dict['bbox_precision'] = precision
    results_dict['bbox_recall'] = recall
    print(f"  BBox Precision (TP/TotalPreds): {precision*100:.2f}% ({total_tp}/{total_pred_boxes_count})")
    print(f"  BBox Recall (TP/TotalGTs):      {recall*100:.2f}% ({total_tp}/{total_gt_boxes_count})")

    return results_dict, per_class_df



def save_comparison_results(yolo_results, frcnn_results, yolo_per_class_df, frcnn_per_class_df, output_dir):
    """
    Save comparison results to CSV and JSON
    """
    output_dir = Path(output_dir)

    # Create comparison dataframe
    comparison_data = {
        'Model': ['YOLOv8', 'Faster R-CNN'],
        'mAP': [yolo_results['mAP'], frcnn_results['mAP']],
        'mAP@50': [yolo_results['mAP_50'], frcnn_results['mAP_50']],
        'mAP@75': [yolo_results['mAP_75'], frcnn_results['mAP_75']],
        'mAP@95': [yolo_results['mAP_95'], frcnn_results['mAP_95']],
        'mAP_small': [yolo_results['mAP_small'], frcnn_results['mAP_small']],
        'mAP_medium': [yolo_results['mAP_medium'], frcnn_results['mAP_medium']],
        'mAP_large': [yolo_results['mAP_large'], frcnn_results['mAP_large']],
        'mAP_large': [yolo_results['mAP_large'], frcnn_results['mAP_large']],
        'Avg_Inference_Time_ms': [yolo_results['avg_inference_time_ms'], frcnn_results['avg_inference_time_ms']],
        'FPS': [yolo_results['fps'], frcnn_results['fps']],
        'Image_Accuracy': [yolo_results['image_accuracy'], frcnn_results['image_accuracy']],
        'BBox_Precision': [yolo_results['bbox_precision'], frcnn_results['bbox_precision']],
        'BBox_Recall': [yolo_results['bbox_recall'], frcnn_results['bbox_recall']]
    }

    df = pd.DataFrame(comparison_data)

    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"comparison_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nComparison results saved to: {csv_path}")

    # Save Per-Class Results
    yolo_class_path = output_dir / f"yolo_per_class_{timestamp}.csv"
    yolo_per_class_df.to_csv(yolo_class_path, index=False)
    
    frcnn_class_path = output_dir / f"fasterrcnn_per_class_{timestamp}.csv"
    frcnn_per_class_df.to_csv(frcnn_class_path, index=False)
    print(f"Per-class results saved to: {yolo_class_path} and {frcnn_class_path}")

    # Save to JSON
    json_data = {
        'timestamp': timestamp,
        'yolov8': yolo_results,
        'faster_rcnn': frcnn_results
    }
    json_path = output_dir / f"comparison_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    print(f"Detailed results saved to: {json_path}")

    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)


# ============ HELPER: CREATE CLASS MAPPING ============
def create_class_mapping(source_names, target_names, model_name="Model"):
    """
    Create a mapping from Source IDs (Model) to Target IDs (Dataset)
    """
    mapping = {}
    print(f"\nCreating Class Mapping for {model_name}:")
    
    # Normalize target names for case-insensitive matching
    lower_target_names = [x.lower() for x in target_names]
    
    for src_idx, name in enumerate(source_names):
        # Handle dictionary source (YOLO model.names is dict: {0: 'name'})
        if isinstance(source_names, dict):
            name = source_names[src_idx]
            
        test_idx = -1
        # Try exact match
        if name in target_names:
            test_idx = target_names.index(name)
        # Try case-insensitive
        elif name.lower() in lower_target_names:
            test_idx = lower_target_names.index(name.lower())
            
        mapping[src_idx] = test_idx
        
        if test_idx == -1:
             print(f"  WARNING: Class '{name}' (ID {src_idx}) not found in Test Config!")

    # Debug sample
    # print(f"  Mapping Sample ({model_name}):")
    # keys = list(mapping.keys())
    # for i in keys[:5]:
    #     src_name = source_names[i]
    #     tgt_name = target_names[mapping[i]] if mapping[i] != -1 else "UNMAPPED"
    #     print(f"    {i} ({src_name}) -> {mapping[i]} ({tgt_name})")
        
    return mapping


# ============ MAIN FUNCTION ============
def main():
    print("\n" + "="*80)
    print("MODEL COMPARISON: YOLOv8 vs Faster R-CNN + RPL")
    print("="*80)

    # Check if test images exist
    test_img_path = Path(TEST_IMG_DIR)
    if not test_img_path.exists():
        print(f"Error: Test image directory not found: {test_img_path}")
        return

    # 1. Load YOLOv8 Model first to get its class names
    yolo_model = load_yolov8_model(YOLOV8_CHECKPOINT)
    
    # 2. Create specific mapping for YOLOv8 using its INTERNAL names
    # YOLO model.names is a dictionary {0: 'class0', 1: 'class1', ...}
    # We maintain order by iterating keys 0..N
    yolo_names_list = [yolo_model.names[i] for i in range(len(yolo_model.names))]
    yolo_map = create_class_mapping(yolo_names_list, TEST_CLASS_NAMES, "YOLOv8")

    # 3. Load Faster R-CNN Model
    frcnn_model = load_fasterrcnn_model(FASTERRCNN_CHECKPOINT, NUM_CLASSES)
    
    # 4. Create mapping for Faster R-CNN (using global TRAIN_CLASS_NAMES from data.yaml)
    frcnn_map = create_class_mapping(TRAIN_CLASS_NAMES, TEST_CLASS_NAMES, "Faster R-CNN")

    # Test YOLOv8
    yolo_results, yolo_per_class_df = test_yolov8(
        yolo_model,
        TEST_IMG_DIR,
        TEST_LBL_DIR,
        YOLO_OUTPUT_DIR,
        yolo_map, # PASS THE MAP
        yolo_names_list,
        CONFIDENCE_THRESHOLD
    )

    # Test Faster R-CNN
    frcnn_results, frcnn_per_class_df = test_fasterrcnn(
        frcnn_model,
        TEST_IMG_DIR,
        TEST_LBL_DIR,
        FRCNN_OUTPUT_DIR,
        frcnn_map, # PASS THE MAP
        TRAIN_CLASS_NAMES,
        CONFIDENCE_THRESHOLD
    )

    # Save comparison results
    if yolo_results and frcnn_results:
        save_comparison_results(yolo_results, frcnn_results, yolo_per_class_df, frcnn_per_class_df, METRICS_OUTPUT_DIR)

    print("\n" + "="*80)
    print("TESTING COMPLETED!")
    print("="*80)
    print(f"YOLOv8 predictions saved to: {YOLO_OUTPUT_DIR}")
    print(f"Faster R-CNN predictions saved to: {FRCNN_OUTPUT_DIR}")
    print(f"Metrics saved to: {METRICS_OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()

