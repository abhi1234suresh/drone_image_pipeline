import os
import glob
import random
import yaml
import shutil
import pandas as pd
import torch
from ultralytics import YOLO

def main():
    base_dir = r"D:\Project\Drone_image_labelling"
    images_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "labels")
    
    # 1. Split the 53 images
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"Error: Required directories {images_dir} or {labels_dir} do not exist.")
        return
        
    all_images = glob.glob(os.path.join(images_dir, "*.*"))
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.JPG')
    all_images = [img for img in all_images if img.endswith(valid_extensions)]
    
    valid_images = []
    for img_path in all_images:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, f"{base_name}.txt")
        if os.path.exists(label_path):
            valid_images.append(img_path)
            
    if not valid_images:
        print("No valid image/label pairs found in labeling folder.")
        return
        
    random.seed(42)
    random.shuffle(valid_images)
    
    split_idx = int(len(valid_images) * 0.8)  # 80/20 split
    train_images = [p.replace("\\", "/") for p in valid_images[:split_idx]]
    val_images = [p.replace("\\", "/") for p in valid_images[split_idx:]]
    
    print(f"Data Split Success: {len(train_images)} training, {len(val_images)} validation.")
    
    train_txt = os.path.join(base_dir, "train.txt").replace("\\", "/")
    val_txt = os.path.join(base_dir, "val.txt").replace("\\", "/")
    
    with open(train_txt, "w") as f:
        f.write("\n".join(train_images))
    with open(val_txt, "w") as f:
        f.write("\n".join(val_images))
        
    # 2. Create required dataset.yaml
    classes_txt = os.path.join(labels_dir, "classes.txt")
    names_dict = {}
    if os.path.exists(classes_txt):
        with open(classes_txt, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            names_dict = {i: name for i, name in enumerate(lines)}
    else:
        names_dict = {0: 'LMV', 1: 'heavy_equipment'}

    dataset_yaml_path = os.path.join(base_dir, "dataset.yaml").replace("\\", "/")
    dataset_dict = {
        "train": train_txt,
        "val": val_txt,
        "nc": len(names_dict),
        "names": names_dict
    }
    
    with open(dataset_yaml_path, "w") as f:
        yaml.dump(dataset_dict, f, default_flow_style=False)
    print(f"Created YAML specification at {dataset_yaml_path}")
    
    # 3. Train YOLOv8m for 100 epochs, imgsz 640
    print("Initializing YOLOv8m training loop...")
    model = YOLO("yolov8m.pt")
    
    device = "0" if torch.cuda.is_available() else "cpu"
    
    results = model.train(
        data=dataset_yaml_path,
        epochs=100,
        imgsz=640,
        project=base_dir,
        name="train_results",
        exist_ok=True,
        device=device
    )
    
    # 4. Save best model securely
    best_model_src = os.path.join(base_dir, "train_results", "weights", "best.pt")
    best_model_dst_dir = r"D:\drone_labeling\model"
    os.makedirs(best_model_dst_dir, exist_ok=True)
    best_model_dst = os.path.join(best_model_dst_dir, "best.pt")
    
    if os.path.exists(best_model_src):
        shutil.copy2(best_model_src, best_model_dst)
        print(f"Best weight matrix securely exported to: {best_model_dst}")
    else:
        print(f"Warning: Training output best.pt not located at {best_model_src}")
        
    # 5. Extract and print mAP50 Validation Scores
    results_csv = os.path.join(base_dir, "train_results", "results.csv")
    if os.path.exists(results_csv):
        try:
            df = pd.read_csv(results_csv)
            df.columns = df.columns.str.strip() 
            map50_col = [col for col in df.columns if 'mAP50' in col and '(B)' in col]
            if not map50_col:
                map50_col = [col for col in df.columns if 'mAP50' in col]
                
            if map50_col:
                best_map50 = df[map50_col[0]].max()
                print("\n" + "="*40)
                print(f"TRAINING COMPLETE (100 Epochs)")
                print(f"Final Best mAP50 Score: {best_map50:.4f}")
                print("="*40)
            else:
                print("Metric extraction error: mAP50 column hidden.")
        except Exception as e:
            print(f"Could not parse results.csv for final mAP: {e}")

if __name__ == "__main__":
    main()
