import os
import time
import shutil
from pathlib import Path
from ultralytics import YOLO


def download_dataset():
    """
    Downloads the vietfood68 dataset from Kaggle.
    """

    local_kaggle_json = Path("kaggle.json")
    kaggle_config_dir = Path.home() / ".kaggle"
    kaggle_config_file = kaggle_config_dir / "kaggle.json"

    if local_kaggle_json.exists():
        print(f"Found kaggle.json in project root. Copying to {kaggle_config_dir}...")
        kaggle_config_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(local_kaggle_json, kaggle_config_file)
        try:
            os.chmod(kaggle_config_file, 0o600)
        except Exception:
            pass  
            
    if 'KAGGLE_USERNAME' not in os.environ or 'KAGGLE_KEY' not in os.environ:
        if not kaggle_config_file.exists():
            print("DEBUG: Environment variables:")
            for k, v in os.environ.items():
                if 'KAGGLE' in k:
                    print(f"{k}: {v}")
            print("WARNING: KAGGLE_USERNAME or KAGGLE_KEY not found in environment.")
            print("AND kaggle.json not found in ~/.kaggle/")

    try:
        import kaggle
    except ImportError:
        print("Error: 'kaggle' library not found. Please install it via pip.")
        return
    except Exception as e:
        print(f"Error importing kaggle: {e}")
        print("Please ensure KAGGLE_USERNAME and KAGGLE_KEY are set or kaggle.json is in ~/.kaggle/")
        return

    dataset_slug = "thomasnguyen6868/vietfood68"
    target_dir = Path("datasets/vietfood68")
    
    print(f"Starting download of {dataset_slug}...")
    
    try:
        kaggle.api.authenticate()
        
        kaggle.api.dataset_download_files(dataset_slug, path=target_dir, unzip=True)
        print(f"Dataset downloaded successfully to {target_dir.absolute()}")

        print("Directory structure:")
        for root, dirs, files in os.walk(target_dir):
            level = root.replace(str(target_dir), '').count(os.sep)
            indent = ' ' * 4 * (level)
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 4 * (level + 1)
            if len(files) > 5:
                print(f'{subindent}({len(files)} files)')
            else:
                for f in files:
                    print(f'{subindent}{f}')

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise


def measure_fps(model, source_dir, imgsz=640, max_frames=None):
    """
    Measure FPS of the model on images in source_dir.
    """
    source_dir = Path(source_dir)
    if not source_dir.exists():
        print(f"Source directory {source_dir} does not exist for benchmarking.")
        return 0, 0, 0

    image_files = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))
    if not image_files:
        print(f"No images found in {source_dir}")
        return 0, 0, 0

    print(f"Benchmarking on {len(image_files)} images from {source_dir}...")

    t0 = time.time()
    frames = 0

    results = model.predict(source=source_dir, imgsz=imgsz, stream=True, verbose=False)

    for r in results:
        frames += 1
        if max_frames is not None and frames >= max_frames:
            break

    t1 = time.time()
    duration = t1 - t0
    fps = frames / duration if duration > 0 else 0

    print(f"Processed {frames} frames in {duration:.2f}s. FPS: {fps:.2f}")
    return frames, duration, fps


def main():
    print("=== Starting FoodDetect Viet67 Pipeline ===")

    # 1. Download Data
    print("\n--- Step 1: Data Preparation ---")
    dataset_path = Path("datasets/vietfood68")
    if not dataset_path.exists():
        print("Dataset not found. Downloading...")
        download_dataset()
    else:
        print("Dataset already exists.")

    data_yaml = "data.yaml"

    print("\n--- Step 2: Training & Comparison ---")
    models_to_train = ["yolov8s.pt"]

    for model_name in models_to_train:
        print(f"\n>>> Processing Model: {model_name} <<<")
        run_name = f"vietfood67_{Path(model_name).stem}"
        run_dir = Path("runs/train") / run_name
        
        if run_dir.exists():
            print(f"Deleting existing run directory {run_dir} for fresh start...")
            shutil.rmtree(run_dir)
            
        last_ckpt_path = run_dir / "weights" / "last.pt"
        
        target_epochs = 50
        
        if last_ckpt_path.exists():
            print(f"Found checkpoint at {last_ckpt_path}. Resuming training...")
            model = YOLO(last_ckpt_path)
            results = model.train(resume=True)
        else:
            print(f"Starting fresh training for {model_name}...")
            model = YOLO(model_name)

            results = model.train(
                data=data_yaml,
                epochs=target_epochs,
                imgsz=640,
                batch=16,
                fraction=0.5,
                project="runs/train",
                name=run_name,
                exist_ok=True
            )
        print(f"Training completed for {model_name}.")
        
        best_model_path = Path(results.save_dir) / "weights" / "best.pt"
        print(f"Best model saved at: {best_model_path}")

        print(f"\n--- Validation: {model_name} ---")
        best_model = YOLO(best_model_path)
        metrics = best_model.val(data=data_yaml, split='test')
        print(f"Validation mAP50-95 ({model_name}): {metrics.box.map:.4f}")

        print(f"\n--- Benchmarking: {model_name} ---")
        test_images_dir = Path("datasets/vietfood68/dataset/images/test")
        if not test_images_dir.exists():
            test_images_dir = Path("datasets/vietfood68/images/test")
        
        if test_images_dir.exists():
            frames, duration, fps = measure_fps(best_model, test_images_dir)
            print(f"Benchmark Result ({model_name}): {fps:.2f} FPS")
        else:
            print("Could not find test images directory for benchmarking.")

    print("\n=== Pipeline Completed Successfully ===")

if __name__ == "__main__":
    main()
