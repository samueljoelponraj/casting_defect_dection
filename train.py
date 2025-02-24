import argparse
import torch
import os
import shutil
from ultralytics import YOLO

def check_cuda():
    """Check if CUDA is available and return the device"""
    return "cuda" if torch.cuda.is_available() else "cpu"

def validate_paths(dataset_path):
    """Ensure the dataset path exists and contains necessary subdirectories"""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path '{dataset_path}' does not exist.")
    
    required_dirs = ["train", "val"]
    for sub_dir in required_dirs:
        if not os.path.exists(os.path.join(dataset_path, sub_dir)):
            raise FileNotFoundError(f"Required subdirectory '{sub_dir}' is missing in '{dataset_path}'.")

def train_model(data_path, model_name, epochs, img_size, batch_size, device, output_model):
    """Train a YOLOv8 classification model"""
    try:
        validate_paths(data_path)

        print(f"\n🚀 Training {model_name} on dataset: {data_path}")
        print(f"🔍 Device: {device}")

        # Load pre-trained YOLO classification model
        model = YOLO(model_name)

        # Start training
        model.train(
            data=data_path,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=device
        )

        # Evaluate model performance
        results = model.val()
        print("\n📊 Training Complete! Model Evaluation Results:")
        print(results)

        # Locate and save the best-trained model as .pt
        trained_model_path = "runs/classify/train/weights/best.pt"
        if os.path.exists(trained_model_path):
            shutil.copy(trained_model_path, output_model)
            print(f"✅ Model saved as {output_model}")
        else:
            print("❌ Error: Trained model file not found.")

        print("\n🎯 Training completed successfully!")

    except Exception as e:
        print(f"❌ Error during training: {e}")

if __name__ == "__main__":
    # Argument parser for user input
    parser = argparse.ArgumentParser(description="Train a YOLOv8 Classification Model")

    parser.add_argument("--data", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--model", type=str, default="yolov8n-cls.pt", help="Pretrained model file (default: yolov8n-cls.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--device", type=str, default=check_cuda(), help="Device to use (cpu/cuda)")
    parser.add_argument("--output_model", type=str, default="yolov8_custom.pt", help="Output model filename (default: yolov8_custom.pt)")

    args = parser.parse_args()

    # Run training
    train_model(args.data, args.model, args.epochs, args.img_size, args.batch_size, args.device, args.output_model)
