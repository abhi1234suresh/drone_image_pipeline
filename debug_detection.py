import os
import cv2
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

def main():
    img_path = os.path.join("input_images", "Miss1_Flgt1_043.JPG")
    output_path = os.path.join("test_results", "debug_detection.jpg")
    
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
    
    # 2. YOLO-World
    print("Loading YOLO-World...")
    model_world = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path='yolov8m-world.pt',
        confidence_threshold=0.10,
        device=device
    )
    
    # Set custom classes on the underlying ultralytics YOLO object
    custom_classes = ['excavator', 'crane', 'bulldozer', 'heavy machinery', 'construction vehicle', 'dump truck aerial view']
    model_world.model.set_classes(custom_classes)
    
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
    
    # Run SAHI for World
    print("Running SAHI for YOLO-World...")
    res_world = get_sliced_prediction(
        img_path,
        model_world,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        verbose=False
    )
    
    target_classes = [2, 5, 7] # car, bus, truck in COCO
    std_vehicles = [p for p in res_std.object_prediction_list if p.category.id in target_classes]
    
    world_vehicles = res_world.object_prediction_list
    
    print(f"Standard YOLOv8m found {len(std_vehicles)} vehicles.")
    print(f"YOLO-World found {len(world_vehicles)} construction vehicles.")
    
    img = cv2.imread(img_path)
    
    # Draw standard detections (Green)
    for p in std_vehicles:
        b = p.bbox
        x1, y1, x2, y2 = int(b.minx), int(b.miny), int(b.maxx), int(b.maxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"COCO: {p.category.name}", (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
    # Draw World detections (Red)
    for p in world_vehicles:
        b = p.bbox
        x1, y1, x2, y2 = int(b.minx), int(b.miny), int(b.maxx), int(b.maxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # category.name is evaluated intrinsically by ultralytics zero-shot prompts
        cv2.putText(img, f"World: {str(p.category.name)}", (x1, y2+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
    cv2.imwrite(output_path, img)
    print(f"Saved debug image to {output_path}")

if __name__ == '__main__':
    main()
