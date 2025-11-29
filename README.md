# FoodDetect Viet67

A comprehensive pipeline for training and evaluating YOLOv8 on the VietFood67 dataset.

> **Note:** This project is designed and trained using **JetBrains Cadence**, leveraging its powerful remote development and computation capabilities.

## Features

- **Automated Data Pipeline**: Downloads dataset from Kaggle automatically.
- **YOLOv8 Training**: Trains YOLOv8n model from scratch or resumes from checkpoints.
- **Validation & Benchmarking**: Evaluates model performance (mAP) and inference speed (FPS).
- **One-Click Execution**: All steps integrated into a single `main.py` script.

## Prerequisites

1.  **Python 3.8+**
2.  **Kaggle API Credentials**:
    - Place `kaggle.json` in the project root OR
    - Set `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the main pipeline:

```bash
python train_Yolov8.py
```

This script will:
1.  Download `thomasnguyen6868/vietfood68` if not present.
2.  Train YOLOv8n for 50 epochs.
3.  Validate the best model.
4.  Benchmark inference speed on test images.

## Project Structure

- `main.py`: Core pipeline script.
- `data.yaml`: YOLO dataset configuration.
- `runs/`: Training artifacts (weights, logs, plots).
- `datasets/`: Downloaded dataset directory.

## Results

Training results (weights, confusion matrices, curves) are saved in `runs/train/vietfood67_yolov8n`.
