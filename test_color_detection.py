import os
import cv2
import torch
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

def main():
    img_path = os.path.join("input_images", "Miss1_Flgt1_073.JPG")
    output_path = os.path.join("test_results", "test_color_detection.jpg")
    
    if not os.path.exists(img_path):
        print(f"Error: image {img_path} not found.")
        return
        
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Standard YOLOv8m (COCO)
    print("Loading YOLOv8m...")
    model_standard = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path='yolov8m.pt',
        confidence_threshold=0.25,
        device=device
    )
    
    # Run SAHI for standard
    print("Running SAHI for YOLOv8m...")
    res_std = get_sliced_prediction(
        img_path,
        model_standard,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        verbose=False
    )
    
    target_classes = [2, 5, 7] # car, bus, truck in COCO
    std_vehicles = [p for p in res_std.object_prediction_list if p.category.id in target_classes]
    
    img = cv2.imread(img_path)
    
    # 2. Color Detection
    print("Running HSV Color Detection...")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Calibrated Range based on Miss1_Flgt1_073 analysis:
    # Most machines have H~17, S~45, V~170
    lower_yellow = np.array([12, 35, 50])
    upper_yellow = np.array([38, 255, 255])
    color_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(color_mask, connectivity=8)
    machinery_boxes = []
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # Filter blobs by size: 200 to 50,000 pixels
        if 200 <= area <= 50000:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            machinery_boxes.append((x, y, x+w, y+h))
            
    print(f"Standard YOLOv8m found {len(std_vehicles)} vehicles.")
    print(f"HSV Color detection found {len(machinery_boxes)} construction vehicles.")
    
    # Draw standard detections (Green)
    for p in std_vehicles:
        b = p.bbox
        x1, y1, x2, y2 = int(b.minx), int(b.miny), int(b.maxx), int(b.maxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img, f"YOLO:{p.category.name}", (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
    # Draw Color detections (Yellow)
    for (x, y, x2, y2) in machinery_boxes:
        cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 255), 3) # Yellow color in BGR
        cv2.putText(img, "COLOR:Machinery", (x, max(0, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
    os.makedirs("test_results", exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Saved debug image to {output_path}")

if __name__ == '__main__':
    main()
