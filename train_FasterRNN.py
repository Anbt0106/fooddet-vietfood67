"""
Faster R-CNN với Residual Pattern Learning (RPL)
Chạy trên môi trường local (Cadence) với dataset từ Kaggle
"""

import os
import shutil
import time
from pathlib import Path
from glob import glob
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
import torchvision
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import numpy as np
import pandas as pd
import gc
from PIL import Image


# ============ DOWNLOAD DATASET FROM KAGGLE ============
def download_dataset():
    """
    Downloads the vietfood68 dataset from Kaggle.
    """
    local_kaggle_json = Path("kaggle.json")
    kaggle_config_dir = Path.home() / ".kaggle"
    kaggle_config_file = kaggle_config_dir / "kaggle.json"

    # Copy kaggle.json if exists in project root
    if local_kaggle_json.exists():
        print(f"Found kaggle.json in project root. Copying to {kaggle_config_dir}...")
        kaggle_config_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(local_kaggle_json, kaggle_config_file)
        try:
            os.chmod(kaggle_config_file, 0o600)
        except Exception:
            pass

    # Check credentials
    if 'KAGGLE_USERNAME' not in os.environ or 'KAGGLE_KEY' not in os.environ:
        if not kaggle_config_file.exists():
            print("DEBUG: Environment variables:")
            for k, v in os.environ.items():
                if 'KAGGLE' in k:
                    print(f"{k}: {v}")
            print("WARNING: KAGGLE_USERNAME or KAGGLE_KEY not found in environment.")
            print("AND kaggle.json not found in ~/.kaggle/")
            print("\nPlease either:")
            print("1. Set environment variables: KAGGLE_USERNAME and KAGGLE_KEY")
            print("2. Place kaggle.json in ~/.kaggle/ or project root")
            return False

    try:
        import kaggle
    except ImportError:
        print("Error: 'kaggle' library not found. Installing...")
        os.system("pip install kaggle")
        import kaggle
    except Exception as e:
        print(f"Error importing kaggle: {e}")
        return False

    dataset_slug = "thomasnguyen6868/vietfood68"
    target_dir = Path("datasets/vietfood68")

    # Check if already downloaded
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"Dataset already exists in {target_dir.absolute()}")
        print("Skipping download. Delete the folder to re-download.")
        return True

    print(f"Starting download of {dataset_slug}...")

    try:
        kaggle.api.authenticate()
        target_dir.mkdir(parents=True, exist_ok=True)
        kaggle.api.dataset_download_files(dataset_slug, path=target_dir, unzip=True)

        print(f"\nDataset downloaded successfully to {target_dir.absolute()}")
        print("\nDirectory structure:")
        for root, dirs, files in os.walk(target_dir):
            level = root.replace(str(target_dir), '').count(os.sep)
            indent = ' ' * 4 * level
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 4 * (level + 1)
            if len(files) > 5:
                print(f'{subindent}({len(files)} files)')
            else:
                for f in files[:5]:
                    print(f'{subindent}{f}')
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False


# ============ LOCAL CONFIG ============
# Paths - adjust based on your local structure
BASE_DATA_DIR = Path("datasets/vietfood68/dataset")

TRAIN_IMG_DIR = str(BASE_DATA_DIR / "images/train")
TRAIN_LBL_DIR = str(BASE_DATA_DIR / "labels/train")
VALID_IMG_DIR = str(BASE_DATA_DIR / "images/valid")
VALID_LBL_DIR = str(BASE_DATA_DIR / "labels/valid")
TEST_IMG_DIR = str(BASE_DATA_DIR / "images/test")
TEST_LBL_DIR = str(BASE_DATA_DIR / "labels/test")

# Training config
TRAIN_SUBSET_SIZE = 30000  # Use full dataset or limit for faster training
VALID_SUBSET_SIZE = 2000
USE_STRATIFIED_SAMPLING = True
MIN_SAMPLES_PER_CLASS = 50

NUM_EPOCHS = 50
BATCH_SIZE = 12
ACCUMULATION_STEPS = 1
LEARNING_RATE = 0.0025
IMG_SIZE = 640
NUM_WORKERS = 16


# RPL Config
USE_RPL = True
RPL_NUM_BLOCKS = 1
RPL_FREEZE_EPOCHS = 1

# Early stopping
EARLY_STOP_PATIENCE = 25
VAL_EVERY_N_EPOCHS = 1

# Memory optimization
CLEAR_CACHE_EVERY_N_BATCHES = 50
USE_MIXED_PRECISION = True

# Output paths
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

CHECKPOINT_BEST_PATH = CHECKPOINT_DIR / "checkpoint_best_rpl.pth"
CHECKPOINT_LAST_PATH = CHECKPOINT_DIR / "checkpoint_last_rpl.pth"
LOG_PATH = OUTPUT_DIR / "training_log_rpl.csv"
TENSORBOARD_DIR = OUTPUT_DIR / "tensorboard"

# ============ DEVICE SETUP ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'=' * 50}")
print("SYSTEM INFO")
print("=" * 50)
print(f"Device: {device}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"PyTorch Version: {torch.__version__}")
print(f"Torchvision Version: {torchvision.__version__}")
print("=" * 50)


# ============ LIGHTWEIGHT RPL MODULES ============
class LightweightChannelAttention(nn.Module):
    """Lightweight Channel Attention"""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


class LightweightResidualBlock(nn.Module):
    """Lightweight Residual Block"""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.channel_att = LightweightChannelAttention(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.channel_att(out)
        return self.relu(out + identity)


class ResidualPatternLearning(nn.Module):
    """Lightweight RPL Module"""

    def __init__(self, in_channels=256, num_blocks=1):
        super().__init__()
        self.domain_adapt = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.residual_blocks = nn.ModuleList([
            LightweightResidualBlock(in_channels) for _ in range(num_blocks)
        ])
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        identity = x
        out = self.domain_adapt(x)
        for block in self.residual_blocks:
            out = block(out)
        return identity + self.gate * out


class FasterRCNN_RPL(nn.Module):
    """Faster R-CNN với Lightweight RPL"""

    def __init__(self, num_classes, use_rpl=True, num_rpl_blocks=1):
        super().__init__()

        # Load base model
        self.base_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT"
        )

        # Config - Set proper image size range
        self.base_model.transform.min_size = (IMG_SIZE,)
        self.base_model.transform.max_size = IMG_SIZE

        # Replace ROI head
        in_features = self.base_model.roi_heads.box_predictor.cls_score.in_features
        self.base_model.roi_heads.box_predictor = \
            torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                in_features, num_classes
            )

        self.use_rpl = use_rpl

        if use_rpl:
            self.rpl_modules = nn.ModuleDict({
                f'rpl_layer{i}': ResidualPatternLearning(256, num_rpl_blocks)
                for i in range(3)
            })
            self._inject_rpl_into_fpn()

    def _inject_rpl_into_fpn(self):
        """Inject RPL vào FPN"""
        fpn = self.base_model.backbone.fpn
        original_forward = fpn.forward

        def new_forward(x):
            fpn_outputs = original_forward(x)
            enhanced_outputs = {}
            for idx, (name, feat) in enumerate(fpn_outputs.items()):
                if idx < 3 and f'rpl_layer{idx}' in self.rpl_modules:
                    enhanced_outputs[name] = self.rpl_modules[f'rpl_layer{idx}'](feat)
                else:
                    enhanced_outputs[name] = feat
            return enhanced_outputs

        fpn.forward = new_forward

    def freeze_rpl(self):
        if self.use_rpl:
            for param in self.rpl_modules.parameters():
                param.requires_grad = False

    def unfreeze_rpl(self):
        if self.use_rpl:
            for param in self.rpl_modules.parameters():
                param.requires_grad = True

    def forward(self, images, targets=None):
        return self.base_model(images, targets)


# ============ SMART RESIZE FUNCTION ============
def smart_resize_with_padding(image, target_size=640):
    """
    Resize ảnh về target_size với padding để giữ aspect ratio
    """
    h, w = image.shape[:2]

    # Tính scale factor
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Padding
    top = (target_size - new_h) // 2
    bottom = target_size - new_h - top
    left = (target_size - new_w) // 2
    right = target_size - new_w - left

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=[114, 114, 114]
    )

    return padded, scale, (left, top)


def transform_boxes(boxes, scale, offset):
    """
    Transform bounding boxes sau khi resize và padding
    """
    if len(boxes) == 0:
        return boxes

    left_pad, top_pad = offset
    transformed_boxes = []

    for box in boxes:
        x1, y1, x2, y2 = box
        x1, x2 = x1 * scale + left_pad, x2 * scale + left_pad
        y1, y2 = y1 * scale + top_pad, y2 * scale + top_pad
        transformed_boxes.append([x1, y1, x2, y2])

    return transformed_boxes


# ============ STRATIFIED DATASET SAMPLING ============
def get_stratified_indices(labels_dir, image_files, num_samples, min_per_class=50):
    """Lấy indices sao cho mỗi class đều có đủ samples"""
    class_samples = {}

    print("Analyzing class distribution...")
    for idx, img_file in enumerate(tqdm(image_files)):
        label_path = os.path.join(
            labels_dir,
            img_file.replace(".jpg", ".txt").replace(".png", ".txt")
        )

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls = int(parts[0])
                        if cls not in class_samples:
                            class_samples[cls] = []
                        class_samples[cls].append(idx)
                        break

    # Remove duplicates
    for cls in class_samples:
        class_samples[cls] = list(set(class_samples[cls]))

    print(f"Found {len(class_samples)} classes")

    # Stratified sampling
    selected_indices = []
    samples_per_class = max(min_per_class, num_samples // len(class_samples))

    for cls, indices in class_samples.items():
        n = min(samples_per_class, len(indices))
        selected = random.sample(indices, n)
        selected_indices.extend(selected)

    selected_indices = list(set(selected_indices))
    random.shuffle(selected_indices)

    if len(selected_indices) > num_samples:
        selected_indices = selected_indices[:num_samples]

    print(f"Selected {len(selected_indices)} samples")
    return selected_indices


# ============ OPTIMIZED DATASET WITH SMART RESIZE ============
class OptimizedMosaicDataset(Dataset):
    """Dataset tối ưu với smart resize cho mọi kích thước ảnh"""

    def __init__(self, images_dir, labels_dir, target_size=640, augment=False):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.target_size = target_size
        self.augment = augment
        self.image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.endswith((".jpg", ".png", ".jpeg"))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(
            self.labels_dir,
            img_name.replace(".jpg", ".txt").replace(".png", ".txt").replace(".jpeg", ".txt")
        )

        # Load original image
        orig = cv2.imread(img_path)
        if orig is None:
            orig = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        else:
            orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = orig.shape[:2]

        # Resize + padding (sau khi convert box)
        img, scale, offset = smart_resize_with_padding(orig, self.target_size)

        boxes, labels = [], []

        # Parse labels
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, xc, yc, bw, bh = parts
                    cls = int(cls)
                    xc, yc, bw, bh = map(float, [xc, yc, bw, bh])

                    # Convert YOLO format
                    x1 = (xc - bw / 2) * orig_w
                    y1 = (yc - bh / 2) * orig_h
                    x2 = (xc + bw / 2) * orig_w
                    y2 = (yc + bh / 2) * orig_h

                    if x2 > x1 + 1 and y2 > y1 + 1:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(cls + 1)

        # Transform boxes
        if len(boxes) > 0:
            boxes = transform_boxes(boxes, scale, offset)

            # Filter invalid boxes
            valid_boxes, valid_labels = [], []
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                x1 = max(0, min(x1, self.target_size - 1))
                y1 = max(0, min(y1, self.target_size - 1))
                x2 = max(0, min(x2, self.target_size))
                y2 = max(0, min(y2, self.target_size))

                if x2 > x1 and y2 > y1:
                    valid_boxes.append([x1, y1, x2, y2])
                    valid_labels.append(label)

            boxes = valid_boxes
            labels = valid_labels

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img_tensor, {"boxes": boxes, "labels": labels}


# ============ EVALUATION FUNCTION ============
@torch.no_grad()
def evaluate_model(model, data_loader, device):
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    total_loss = 0.0
    num_batches = 0

    for imgs, targets in tqdm(data_loader, desc="Evaluating", leave=False):
        imgs_device = [img.to(device, non_blocking=True) for img in imgs]
        targets_device = [{k: v.to(device, non_blocking=True) for k, v in t.items()}
                          for t in targets]

        # Calculate loss
        model.train()
        loss_dict = model(imgs_device, targets_device)
        batch_loss = sum(loss for loss in loss_dict.values())
        total_loss += batch_loss.item()
        num_batches += 1

        # Inference
        model.eval()
        preds = model(imgs_device)

        preds_for_metric = [
            {"boxes": p["boxes"].detach(), "scores": p["scores"].detach(),
             "labels": p["labels"].detach()} for p in preds
        ]
        targets_for_metric = [
            {"boxes": t["boxes"], "labels": t["labels"]} for t in targets_device
        ]

        metric.update(preds_for_metric, targets_for_metric)

        if num_batches % 20 == 0:
            torch.cuda.empty_cache()

    result = metric.compute()
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    result_cpu = {
        k: float(v.cpu().item()) if torch.is_tensor(v) and v.numel() == 1 else v
        for k, v in result.items()
    }
    result_cpu['loss'] = avg_loss

    model.train()
    return result_cpu


# ============ BENCHMARK FPS ============
def measure_fps(model, source_dir, imgsz=640, max_frames=100):
    """
    Measure FPS of the model on images in source_dir
    """
    source_dir = Path(source_dir)
    if not source_dir.exists():
        print(f"Source directory {source_dir} does not exist for benchmarking.")
        return 0, 0, 0

    image_files = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))
    if not image_files:
        print(f"No images found in {source_dir}")
        return 0, 0, 0

    image_files = image_files[:max_frames] if max_frames else image_files
    print(f"\nBenchmarking FPS on {len(image_files)} images...")

    model.eval()
    times = []

    with torch.no_grad():
        # Warmup
        for _ in range(5):
            img = cv2.imread(str(image_files[0]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img, _, _ = smart_resize_with_padding(img, imgsz)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)
            _ = model([img_tensor.to(device)])

        # Actual measurement
        for img_path in tqdm(image_files, desc="FPS Benchmark"):
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img, _, _ = smart_resize_with_padding(img, imgsz)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)

            t0 = time.time()
            _ = model([img_tensor.to(device)])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()

            times.append(t1 - t0)

    frames = len(times)
    duration = sum(times)
    fps = frames / duration if duration > 0 else 0

    print(f"Processed {frames} frames in {duration:.2f}s")
    print(f"Average FPS: {fps:.2f}")
    print(f"Average inference time: {duration / frames * 1000:.2f}ms per frame")

    return frames, duration, fps
# ============================================================
#                           MAIN
# ============================================================

def main():
    print("\n" + "="*60)
    print("FASTER R-CNN + RPL TRAINING PIPELINE")
    print("="*60)

    # 1) Download dataset (skip if already downloaded)
    print("\nChecking dataset")
    ok = download_dataset()
    if not ok:
        print("Dataset not found and cannot download.")
        return

    # 2) Build datasets
    print("\nBuilding datasets")

    train_dataset_full = OptimizedMosaicDataset(
        TRAIN_IMG_DIR, TRAIN_LBL_DIR,
        target_size=IMG_SIZE, augment=True
    )
    valid_dataset_full = OptimizedMosaicDataset(
        VALID_IMG_DIR, VALID_LBL_DIR,
        target_size=IMG_SIZE, augment=False
    )

    print(f"Full train size: {len(train_dataset_full)}")
    print(f"Full valid size: {len(valid_dataset_full)}")

    # 3) Stratified sampling
    if USE_STRATIFIED_SAMPLING:
        train_indices = get_stratified_indices(
            TRAIN_LBL_DIR,
            train_dataset_full.image_files,
            TRAIN_SUBSET_SIZE,
            MIN_SAMPLES_PER_CLASS
        )
        valid_indices = random.sample(
            range(len(valid_dataset_full)),
            min(VALID_SUBSET_SIZE, len(valid_dataset_full))
        )
    else:
        train_indices = random.sample(range(len(train_dataset_full)), TRAIN_SUBSET_SIZE)
        valid_indices = random.sample(range(len(valid_dataset_full)), VALID_SUBSET_SIZE)

    train_dataset = Subset(train_dataset_full, train_indices)
    valid_dataset = Subset(valid_dataset_full, valid_indices)

    # 4) Dataloaders
    print("\nBuilding dataloaders")
    def collate_fn(batch):
        return tuple(zip(*batch))

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn,
        pin_memory=True
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Valid batches: {len(valid_loader)}")

    # 5) Build model
    print("\nCreating model")
    num_classes = 69
    model = FasterRCNN_RPL(
        num_classes=num_classes,
        use_rpl=USE_RPL,
        num_rpl_blocks=RPL_NUM_BLOCKS
    ).to(device)

    # 6) Optimizer
    params = [
        {'params': model.base_model.backbone.parameters(), 'lr': LEARNING_RATE * 0.1},
        {'params': model.base_model.rpn.parameters(), 'lr': LEARNING_RATE},
        {'params': model.base_model.roi_heads.parameters(), 'lr': LEARNING_RATE},
    ]
    if USE_RPL:
        params.append({'params': model.rpl_modules.parameters(), 'lr': LEARNING_RATE * 0.5})

    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[p['lr'] for p in params],
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )

    scaler = torch.cuda.amp.GradScaler(enabled=USE_MIXED_PRECISION)

    # 7) TRAINING LOOP
    print("\nTRAINING START\n")
    best_map = 0
    patience = 0
    logs = []

    for epoch in range(NUM_EPOCHS):
        model.train()

        # Freeze / unfreeze RPL
        if USE_RPL:
            if epoch < RPL_FREEZE_EPOCHS:
                model.freeze_rpl()
                rpl_state = "FROZEN"
            else:
                model.unfreeze_rpl()
                rpl_state = "ACTIVE"
        else:
            rpl_state = "OFF"

        epoch_loss = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [{rpl_state}]")

        for i, (images, targets) in enumerate(pbar):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            if USE_MIXED_PRECISION:
                with torch.cuda.amp.autocast():
                    loss_dict = model(images, targets)
                    loss = sum(loss_dict.values()) / ACCUMULATION_STEPS
                scaler.scale(loss).backward()
                if (i+1) % ACCUMULATION_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values()) / ACCUMULATION_STEPS
                loss.backward()
                if (i+1) % ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

            epoch_loss += loss.item() * ACCUMULATION_STEPS
            pbar.set_postfix({"loss": f"{loss.item() * ACCUMULATION_STEPS:.4f}"})

            if i % CLEAR_CACHE_EVERY_N_BATCHES == 0:
                torch.cuda.empty_cache()

        # Validation
        print(f"\nValidating")
        eval_res = evaluate_model(model, valid_loader, device)
        mAP = eval_res.get("map", 0)
        mAP50 = eval_res.get("map_50", 0)

        # Save logs
        logs.append({
            "epoch": epoch+1,
            "train_loss": epoch_loss / len(train_loader),
            "val_loss": eval_res["loss"],
            "mAP": mAP,
            "mAP50": mAP50,
            "lr": optimizer.param_groups[0]["lr"]
        })

        # CSV SAVE
        df = pd.DataFrame(logs)
        df.to_csv(LOG_PATH, index=False)

        # Save checkpoints
        torch.save(model.state_dict(), CHECKPOINT_LAST_PATH)

        if mAP > best_map:
            best_map = mAP
            patience = 0
            torch.save(model.state_dict(), CHECKPOINT_BEST_PATH)
            print(f"New best mAP: {best_map:.4f}")
        else:
            patience += 1

        if patience >= EARLY_STOP_PATIENCE:
            print("\nEARLY STOPPING TRIGGERED!")
            break

    print("\nTraining complete!")
    print(f"Best mAP = {best_map:.4f}")
    print(f"Best checkpoint saved at: {CHECKPOINT_BEST_PATH}")


if __name__ == "__main__":
    main()
