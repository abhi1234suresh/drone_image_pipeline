import os
import cv2
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLOWorld

def main():
    img_path = os.path.join("input_images", "Miss1_Flgt1_043.JPG")
    
    # --- Step 1: COCO Detection with SAHI (conf=0.10) ---
    print("\n--- STEP 1: YOLOv8m with SAHI (conf=0.10) ---")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_standard = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path='yolov8m.pt',
        confidence_threshold=0.10,
        device=device
    )
    
    res_std = get_sliced_prediction(
        img_path,
        model_standard,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        verbose=False
    )
    
    target_classes = [2, 5, 7] # car, bus, truck
    std_vehicles = [p for p in res_std.object_prediction_list if p.category.id in target_classes]
    
    print(f"YOLOv8m SAHI found: {len(std_vehicles)} objects")
    img = cv2.imread(img_path)
    if img is not None:
        for p in std_vehicles:
            b = p.bbox
            x1, y1, x2, y2 = int(b.minx), int(b.miny), int(b.maxx), int(b.maxy)
            print(f"  - [{p.category.name}] bbox: ({x1}, {y1}), ({x2}, {y2}), conf: {p.score.value:.3f}")
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{p.category.name} {p.score.value:.2f}", (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
        cv2.imwrite(os.path.join("test_results", "step1_coco_debug.jpg"), img)
    else:
        print("Failed to read image using cv2.imread")
        
    # --- Step 2: YOLO-World natively ---
    print("\n--- STEP 2: YOLO-World without SAHI (conf=0.05) ---")
    model_world = YOLOWorld('yolov8m-world.pt')
    model_world.set_classes(['excavator', 'crane', 'bulldozer', 'truck', 'vehicle'])
    results = model_world.predict(img_path, conf=0.05)
    
    print(f'YOLO-World found: {len(results[0].boxes)} objects')
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        cls_name = model_world.names[cls_id]
        conf = box.conf[0].item()
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        print(f"  - [{cls_name}] bbox: ({x1}, {y1}), ({x2}, {y2}), conf: {conf:.3f}")
        
    results[0].save(os.path.join('test_results', 'yoloworld_test.jpg'))
    
if __name__ == "__main__":
    main()
