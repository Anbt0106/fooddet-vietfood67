# FoodDetect Viet67 🍲🥢

A comprehensive AI-powered system for detecting Vietnamese dishes, featuring a robust training pipeline and an interactive web application.

**Live Demo:** [FoodDetect Viet67 App](https://fooddet-vietfood67-anbt2k4.streamlit.app/)

## Features

### Web Application
Built with **Streamlit**, the user interface allows for real-time detection through various inputs:
-   **Image Upload**: Detect food in uploaded images or via URL.
-   **Video Analysis**: Process uploaded video files or YouTube links.
-   **Webcam**: Live detection using your local webcam.
-   **IP Camera**: Connect to RTSP streams for remote monitoring.

### Advanced Training Pipelines
The project supports training state-of-the-art object detection models on the **VietFood67** dataset.
> **Note:** The models in this project were trained using **JetBrains Cadence**, leveraging its powerful remote development and computation capabilities.

-   **YOLOv8**: Efficient and accurate real-time detection.
-   **Faster R-CNN + RPL**: Enhanced architecture with Residual Pattern Learning for improved feature extraction.

### Automated Workflow
-   **Auto-Download**: Automatically fetches the [`vietfood68`](https://www.kaggle.com/datasets/thomasnguyen6868/vietfood68) dataset from Kaggle.
-   **Smart Preprocessing**: Includes optimized resizing and stratified sampling.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Anbt0106/fooddet-vietfood67.git
    cd "FoodDetect Viet67"
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Kaggle Credentials (for training):**
    -   Place your `kaggle.json` in the project root.
    -   Or set `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables.

## Usage

### Run the Web App
Launch the interactive interface locally:
```bash
streamlit run streamlit_app.py
```

### Train Models
**Train YOLOv8:**
```bash
python train_Yolov8.py
```

**Train Faster R-CNN (with RPL):**
```bash
python train_FasterRNN.py
```

## Project Structure

-   `streamlit_app.py`: Main entry point for the Web UI.
-   `train_Yolov8.py`: Script to train the YOLOv8 model.
-   `train_FasterRNN.py`: Script to train Faster R-CNN with RPL.
-   `UI/`: Contains assets and utility functions for the frontend.
-   `datasets/`: Directory where the dataset is downloaded.
-   `runs/` & `outputs/`: Stores training artifacts, logs, and weights.

---
*Developed with using Streamlit and PyTorch.*
