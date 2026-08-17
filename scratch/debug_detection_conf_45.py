import os
import cv2
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

def main():
    # Configuration
    model_path = 'D:/Project/Drone_image_labelling/train_results/drone_vehicles/weights/best.pt'
    img_path = 'input_images/Miss1_Flgt1_001.JPG'
    output_path = 'test_results/debug_001.jpg'
    conf_threshold = 0.45
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    target_classes = [0, 1] # LMV, heavy_equipment

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Initializing model with confidence {conf_threshold}...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_path,
        confidence_threshold=conf_threshold,
        device=device
    )

    print(f"Running sliced prediction on {img_path}...")
    result = get_sliced_prediction(
        img_path,
        detection_model,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        verbose=False
    )

    # Filter by target classes
    filtered_preds = [p for p in result.object_prediction_list if p.category.id in target_classes]
    
    print(f"Detections found: {len(filtered_preds)}")

    # Draw boxes
    img = cv2.imread(img_path)
    for pred in filtered_preds:
        bbox = pred.bbox
        x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
        label = f"{pred.category.name} {pred.score.value:.2f}"
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        # Draw label
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save debug image
    cv2.imwrite(output_path, img)
    print(f"Debug image saved to {output_path}")

if __name__ == "__main__":
    main()
