import os
import glob
import yaml
import shutil
import random

def main():
    base_dir = r"D:\Project\Drone_image_labelling"
    images_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "labels")
    
    # 3. Delete old files
    dataset_yaml_path = os.path.join(base_dir, "dataset.yaml").replace("\\", "/")
    train_results_dir = os.path.join(base_dir, "train_results")
    
    if os.path.exists(dataset_yaml_path):
        os.remove(dataset_yaml_path)
    if os.path.exists(train_results_dir):
        shutil.rmtree(train_results_dir, ignore_errors=True)
        
    # Find all images
    all_images = glob.glob(os.path.join(images_dir, "*.*"))
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.JPG')
    all_images = [img for img in all_images if img.endswith(valid_extensions)]
    
    valid_images = []
    for img_path in all_images:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, f"{base_name}.txt")
        if os.path.exists(label_path):
            valid_images.append(img_path)
            
    random.seed(42)
    random.shuffle(valid_images)
    
    split_idx = int(len(valid_images) * 0.8)
    train_images = [p.replace("\\", "/") for p in valid_images[:split_idx]]
    val_images = [p.replace("\\", "/") for p in valid_images[split_idx:]]
    
    print(f"Split {len(valid_images)} total images -> {len(train_images)} train, {len(val_images)} val")
    
    train_txt = os.path.join(base_dir, "train.txt").replace("\\", "/")
    val_txt = os.path.join(base_dir, "val.txt").replace("\\", "/")
    
    with open(train_txt, "w") as f:
        f.write("\n".join(train_images))
    with open(val_txt, "w") as f:
        f.write("\n".join(val_images))
        
    # 4. Regenerate dataset.yaml EXACTLY with nc: 2
    # Ensure keys are directly formatted as requested:
    # images path: .../images/
    # labels path: .../labels/
    # (Note: YOLOv8 usually uses 'path', 'train', 'val' but user says "images path...". 
    # Standard YOLOv8 format: python dict -> yaml)
    
    dataset_dict = {
        "path": base_dir.replace("\\", "/"),
        "train": "train.txt",
        "val": "val.txt",
        "nc": 2,
        "names": ['LMV', 'heavy_equipment']
    }
    
    with open(dataset_yaml_path, "w") as f:
        yaml.dump(dataset_dict, f, default_flow_style=None, sort_keys=False)
        
    # 5. Print the full contents of the new dataset.yaml
    print("\n--- NEW dataset.yaml CONTENTS ---")
    with open(dataset_yaml_path, "r") as f:
        print(f.read())
    print("---------------------------------")

if __name__ == "__main__":
    main()
