# FoodDetect Viet67

This project implements a food detection system using YOLOv11, specifically trained on the VietFood67 dataset.

## Project Structure

- `train.py`: Script to train the YOLO model.
- `predict.py`: Script to run inference on new images/videos.
- `val.py`: Script to validate the trained model.
- `download_data.py`: Script to download the VietFood67 dataset from Kaggle.
- `data.yaml`: Dataset configuration file.
- `requirements.txt`: List of dependencies.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download Dataset**:
    Ensure you have your Kaggle credentials set up (env vars `KAGGLE_USERNAME` and `KAGGLE_KEY` or `~/.kaggle/kaggle.json`).
    ```bash
    python download_data.py
    ```

## Training

To train the model from scratch (using YOLOv11n pretrained weights):

```bash
python train.py
```

This will train for 10 epochs (default) and save results to `runs/train/vietfood67_yolov11`.

## Inference

To detect food in an image or video:

```bash
python predict.py --source path/to/image.jpg
```

Optional arguments:
- `--model`: Path to the trained model (default: `runs/train/vietfood67_yolov11/weights/best.pt`)
- `--conf`: Confidence threshold (default: 0.25)

## Validation

To evaluate the model on the validation set:

```bash
python val.py
```

Optional arguments:
- `--model`: Path to the trained model.
- `--data`: Path to data config (default: `data.yaml`).
