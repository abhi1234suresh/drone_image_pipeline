import os
import cv2
import torch
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

def main():
    img_path = os.path.join("input_images", "Miss1_Flgt1_073.JPG")
    
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        return
        
    os.makedirs("test_results", exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    
    # Target COCO classes: 2:car, 3:motorcycle, 5:bus, 6:train, 7:truck, 8:boat
    target_classes = [2, 3, 5, 6, 7, 8]
    
    # ---------------------------------------------------------
    # METHOD A: YOLOv8m + SAHI (conf 0.10, slice 416)
    # ---------------------------------------------------------
    print("\nStarting Method A (YOLOv8m, conf=0.10, slice=416)...")
    model_a = AutoDetectionModel.from_pretrained(
        model_type='yolov8', model_path='yolov8m.pt', confidence_threshold=0.10, device=device
    )
    res_a = get_sliced_prediction(img_path, model_a, slice_height=416, slice_width=416, 
                                  overlap_height_ratio=0.2, overlap_width_ratio=0.2, verbose=False)
    preds_a = [p for p in res_a.object_prediction_list if p.category.id in target_classes]
    
    img_a = cv2.imread(img_path)
    for p in preds_a:
        b = p.bbox
        x1, y1, x2, y2 = int(b.minx), int(b.miny), int(b.maxx), int(b.maxy)
        cv2.rectangle(img_a, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{p.category.name} {p.score.value:.2f}"
        cv2.putText(img_a, label, (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imwrite(os.path.join("test_results", "method_a.jpg"), img_a)
    
    del model_a; torch.cuda.empty_cache() if device.startswith("cuda") else None
    
    # ---------------------------------------------------------
    # METHOD B: YOLOv8x + SAHI (conf 0.08, slice 416)
    # ---------------------------------------------------------
    print("Starting Method B (YOLOv8x, conf=0.08, slice=416)...")
    # Will gracefully download yolov8x.pt from Ultralytics
    model_b = AutoDetectionModel.from_pretrained(
        model_type='yolov8', model_path='yolov8x.pt', confidence_threshold=0.08, device=device
    )
    res_b = get_sliced_prediction(img_path, model_b, slice_height=416, slice_width=416, 
                                  overlap_height_ratio=0.2, overlap_width_ratio=0.2, verbose=False)
    preds_b = [p for p in res_b.object_prediction_list if p.category.id in target_classes]
    
    img_b = cv2.imread(img_path)
    for p in preds_b:
        b = p.bbox
        x1, y1, x2, y2 = int(b.minx), int(b.miny), int(b.maxx), int(b.maxy)
        cv2.rectangle(img_b, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue boxes for Method B
        label = f"{p.category.name} {p.score.value:.2f}"
        cv2.putText(img_b, label, (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.imwrite(os.path.join("test_results", "method_b.jpg"), img_b)
    
    del model_b; torch.cuda.empty_cache() if device.startswith("cuda") else None
    
    # ---------------------------------------------------------
    # METHOD C: YOLOv8m + SAHI (conf 0.10, slice 320)
    # ---------------------------------------------------------
    print("Starting Method C (YOLOv8m, conf=0.10, slice=320)...")
    model_c = AutoDetectionModel.from_pretrained(
        model_type='yolov8', model_path='yolov8m.pt', confidence_threshold=0.10, device=device
    )
    res_c = get_sliced_prediction(img_path, model_c, slice_height=320, slice_width=320, 
                                  overlap_height_ratio=0.2, overlap_width_ratio=0.2, verbose=False)
    preds_c = [p for p in res_c.object_prediction_list if p.category.id in target_classes]
    
    img_c = cv2.imread(img_path)
    for p in preds_c:
        b = p.bbox
        x1, y1, x2, y2 = int(b.minx), int(b.miny), int(b.maxx), int(b.maxy)
        cv2.rectangle(img_c, (x1, y1), (x2, y2), (0, 165, 255), 2) # Orange boxes for Method C
        label = f"{p.category.name} {p.score.value:.2f}"
        cv2.putText(img_c, label, (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    cv2.imwrite(os.path.join("test_results", "method_c.jpg"), img_c)
    
    del model_c; torch.cuda.empty_cache() if device.startswith("cuda") else None
    
    # ---------------------------------------------------------
    # METHOD D: HSV Color Only
    # ---------------------------------------------------------
    print("Starting Method D (HSV Color Only)...")
    img_d = cv2.imread(img_path)
    hsv = cv2.cvtColor(img_d, cv2.COLOR_BGR2HSV)
    
    # Yellow/Orange Rule (H=15-35)
    lower_yellow = np.array([12, 35, 50])  
    upper_yellow = np.array([38, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # White Rule (V>200, low S) for bright machines/trucks
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([179, 60, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    combined_color_mask = cv2.bitwise_or(yellow_mask, white_mask)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_color_mask, connectivity=8)
    preds_d = []
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        if 200 <= area <= 50000:
            x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
            w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            
            aspect_ratio = float(w) / h if h > 0 else 0
            extent = area / float(w * h)
            
            # Simple shape filter constraints for rectangles that look like vehicles
            if 0.2 < aspect_ratio < 5.0 and extent > 0.3:
                preds_d.append((x, y, x+w, y+h))
                cv2.rectangle(img_d, (x, y), (x+w, y+h), (0, 255, 255), 2)
                cv2.putText(img_d, "Color Blob", (x, max(0, y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
    cv2.imwrite(os.path.join("test_results", "method_d.jpg"), img_d)
    
    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------
    print("\n" + "="*40)
    print("DETECTION SUMMARY ON Miss1_Flgt1_073.JPG")
    print("="*40)
    print(f"Method A (YOLOv8m, slice 416) found: {len(preds_a)} objects")
    print(f"Method B (YOLOv8x, slice 416) found: {len(preds_b)} objects")
    print(f"Method C (YOLOv8m, slice 320) found: {len(preds_c)} objects")
    print(f"Method D (HSV Color blobs)    found: {len(preds_d)} objects")
    print("="*40)

if __name__ == '__main__':
    main()
