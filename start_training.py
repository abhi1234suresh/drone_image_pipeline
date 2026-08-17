import os
import torch
from ultralytics import YOLO

def main():
    # 1. Fix classes.txt
    txt_path = r"D:\Project\Drone_image_labelling\labels\classes.txt"
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            content = f.read().strip()
        with open(txt_path, "w") as f:
            f.write(content)
        print(f"File classes.txt fixed. Final character count: {len(content)}")
    
    # 2. Initialize YOLO Model
    model = YOLO('yolov8m.pt')
    
    # 3. Create callback for every 10 epochs
    def on_fit_epoch_end(trainer):
        # trainer.epoch is 0-indexed, so 9 is epoch 10
        epoch = trainer.epoch + 1
        if epoch % 10 == 0:
            metrics = getattr(trainer, 'metrics', None)
            if metrics and 'metrics/mAP50(B)' in metrics:
                map50 = metrics['metrics/mAP50(B)']
                print("\n" + "="*50)
                print(f"[CALLBACK] Epoch {epoch}/100 Completed. Current mAP50: {map50:.4f}")
                print("="*50 + "\n")
            else:
                print("\n" + "="*50)
                print(f"[CALLBACK] Epoch {epoch}/100 Completed. (mAP50 calculation deferred)")
                print("="*50 + "\n")
    
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    
    device = "0" if torch.cuda.is_available() else "cpu"
    
    print(f"Starting YOLOv8m Training on: {device}...")
    # Start Training
    model.train(
        data='D:/Project/Drone_image_labelling/dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=4,
        project='D:/Project/Drone_image_labelling/train_results',
        name='drone_vehicles',
        save=True,
        plots=True,
        device=device
    )

if __name__ == "__main__":
    main()
