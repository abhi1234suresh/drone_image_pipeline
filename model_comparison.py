import os
import glob
import cv2
import torch
from collections import defaultdict
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

def main():
    models_config = [
        {
            "name": "yolov8m_conf_0.15",
            "model_path": "yolov8m.pt",
            "conf": 0.15,
            "slice_size": 512,
            "overlap": 0.2,
            "classes": [2, 5, 7],  # COCO: 2=car, 5=bus, 7=truck
            "output_dir": "results_conf_015"
        },
        {
            "name": "yolov8m_conf_0.20",
            "model_path": "yolov8m.pt",
            "conf": 0.20,
            "slice_size": 512,
            "overlap": 0.2,
            "classes": [2, 5, 7],
            "output_dir": "results_conf_020"
        },
        {
            "name": "yolov8m_conf_0.25",
            "model_path": "yolov8m.pt",
            "conf": 0.25,
            "slice_size": 512,
            "overlap": 0.2,
            "classes": [2, 5, 7],
            "output_dir": "results_conf_025"
        }
    ]

    test_images_dir = "test_images"
    
    image_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
        image_paths.extend(glob.glob(os.path.join(test_images_dir, ext)))
        
    if not image_paths:
        print(f"No images found in '{test_images_dir}' folder.")
        print("Please add some images before running.")
        return

    # Dictionary to keep track of detection counts
    # format: counts[image_name][model_name] = count
    counts = defaultdict(dict)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    for config in models_config:
        model_name = config["name"]
        output_dir = config["output_dir"]
        print(f"\n--- Loading and Testing Model: {model_name} ---")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize SAHI auto detection model
        try:
            detection_model = AutoDetectionModel.from_pretrained(
                model_type='yolov8',
                model_path=config["model_path"],
                confidence_threshold=config["conf"],
                device=device
            )
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}")
            continue
        
        for image_path in image_paths:
            img_name = os.path.basename(image_path)
            
            # Sliced inference
            result = get_sliced_prediction(
                image_path,
                detection_model,
                slice_height=config["slice_size"],
                slice_width=config["slice_size"],
                overlap_height_ratio=config["overlap"],
                overlap_width_ratio=config["overlap"]
            )
            
            # Filter exactly the requested classes
            filtered_predictions = []
            for pred in result.object_prediction_list:
                if pred.category.id in config["classes"]:
                    filtered_predictions.append(pred)
                    
            counts[img_name][model_name] = len(filtered_predictions)
            
            # Draw bounding boxes
            img_cv2 = cv2.imread(image_path)
            for pred in filtered_predictions:
                bbox = pred.bbox
                x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
                # Draw green box
                cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
            # Save the result image
            output_name = f"{model_name}_{img_name}"
            output_path = os.path.join(output_dir, output_name)
            cv2.imwrite(output_path, img_cv2)
            
            print(f"Processed {img_name} -> Found {len(filtered_predictions)} vehicles")

    # Print summary table
    print("\n" + "="*90)
    print("DETECTION SUMMARY TABLE".center(90))
    print("="*90)
    
    model_names = [config["name"] for config in models_config]
    
    # Create formatting strings based on max widths
    col_width = max([len(n) for n in model_names] + [15])
    
    header = f"{'Image Name':<30} | " + " | ".join([f"{name:<{col_width}}" for name in model_names])
    print(header)
    print("-" * len(header))
    
    for img_name in sorted(counts.keys()):
        row = f"{img_name[:30]:<30} | "
        
        # Add a column for each model
        row_counts = []
        for name in model_names:
            c = counts[img_name].get(name, 0)
            row_counts.append(f"{c:<{col_width}}")
            
        row += " | ".join(row_counts)
        print(row)
    print("="*90 + "\n")

if __name__ == '__main__':
    main()
