import os
import cv2
import torch
import numpy as np
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from simple_lama_inpainting import SimpleLama

def main():
    image_path = os.path.join("test_images", "Miss1_Flgt1_001.JPG")
    output_path = os.path.join("test_results", "shadow_fix_comparison.jpg")
    
    if not os.path.exists(image_path):
        print(f"Error: {image_path} does not exist.")
        return
        
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Running detection...")
    try:
        detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path="yolov8m.pt",
            confidence_threshold=0.25,
            device=device
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )

    target_classes = [2, 5, 7]
    filtered_predictions = [p for p in result.object_prediction_list if p.category.id in target_classes]
    
    print(f"Found {len(filtered_predictions)} vehicles.")
    
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to read image")
        return
        
    H, W, C = img.shape
    
    mask_A = np.zeros((H, W), dtype=np.uint8)
    mask_B = np.zeros((H, W), dtype=np.uint8)
    
    for pred in filtered_predictions:
        bbox = pred.bbox
        x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
        
        # Mask A: expand 35px uniformly
        x1_A, y1_A = max(0, x1 - 35), max(0, y1 - 35)
        x2_A, y2_A = min(W, x2 + 35), min(H, y2 + 35)
        cv2.rectangle(mask_A, (x1_A, y1_A), (x2_A, y2_A), 255, -1)
        
        # Mask B: expand 35px + extra 40px right/down
        x1_B, y1_B = max(0, x1 - 35), max(0, y1 - 35)
        x2_B, y2_B = min(W, x2 + 35 + 40), min(H, y2 + 35 + 40)
        cv2.rectangle(mask_B, (x1_B, y1_B), (x2_B, y2_B), 255, -1)
        
    print("Running LaMa Inpainting for Mask A...")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    lama = SimpleLama()
    
    pil_mask_A = Image.fromarray(mask_A)
    lama_inpainted_A_pil = lama(pil_img, pil_mask_A)
    lama_inpainted_A_rgb = np.array(lama_inpainted_A_pil)
    lama_inpainted_A = cv2.cvtColor(lama_inpainted_A_rgb, cv2.COLOR_RGB2BGR)
    
    print("Running LaMa Inpainting for Mask B...")
    pil_mask_B = Image.fromarray(mask_B)
    lama_inpainted_B_pil = lama(pil_img, pil_mask_B)
    lama_inpainted_B_rgb = np.array(lama_inpainted_B_pil)
    lama_inpainted_B = cv2.cvtColor(lama_inpainted_B_rgb, cv2.COLOR_RGB2BGR)

    print("Creating comparison image...")
    comparison = cv2.hconcat([lama_inpainted_A, lama_inpainted_B])
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(H / 1000.0, 1.0)
    thickness = max(int(scale * 2), 2)
    
    text_A = "Mask A - uniform expand"
    text_B = "Mask B - shadow direction expand"
    
    size_A, _ = cv2.getTextSize(text_A, font, scale, thickness)
    size_B, _ = cv2.getTextSize(text_B, font, scale, thickness)
    
    x_A = (W - size_A[0]) // 2
    x_B = W + (W - size_B[0]) // 2
    y_text = int(size_A[1] * 2)
    
    def draw_text_with_bg(target_img, text, pos, txt_scale, txt_thickness):
        x, y = pos
        (w, h), baseline = cv2.getTextSize(text, font, txt_scale, txt_thickness)
        cv2.rectangle(target_img, (x - 10, y - h - 10), (x + w + 10, y + baseline + 10), (0, 0, 0), -1)
        cv2.putText(target_img, text, (x, y), font, txt_scale, (255, 255, 255), txt_thickness, cv2.LINE_AA)
        
    draw_text_with_bg(comparison, text_A, (x_A, y_text), scale, thickness)
    draw_text_with_bg(comparison, text_B, (x_B, y_text), scale, thickness)
    
    cv2.imwrite(output_path, comparison)
    print(f"Saved comparison to: {output_path}")

if __name__ == '__main__':
    main()
