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
    output_path = os.path.join("test_results", "final_mask_test.jpg")
    
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
    
    mask_1 = np.zeros((H, W), dtype=np.uint8) # 50px uniform
    mask_2 = np.zeros((H, W), dtype=np.uint8) # 60px uniform
    
    for pred in filtered_predictions:
        bbox = pred.bbox
        x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
        
        # Mask 1: expand 50px uniformly
        x1_1, y1_1 = max(0, x1 - 50), max(0, y1 - 50)
        x2_1, y2_1 = min(W, x2 + 50), min(H, y2 + 50)
        cv2.rectangle(mask_1, (x1_1, y1_1), (x2_1, y2_1), 255, -1)
        
        # Mask 2: expand 60px uniformly
        x1_2, y1_2 = max(0, x1 - 60), max(0, y1 - 60)
        x2_2, y2_2 = min(W, x2 + 60), min(H, y2 + 60)
        cv2.rectangle(mask_2, (x1_2, y1_2), (x2_2, y2_2), 255, -1)
        
    print("Running LaMa Inpainting for Panel 1 (50px)...")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    lama = SimpleLama()
    
    pil_mask_1 = Image.fromarray(mask_1)
    lama_inpainted_1_pil = lama(pil_img, pil_mask_1)
    lama_inpainted_1_rgb = np.array(lama_inpainted_1_pil)
    panel_1 = cv2.cvtColor(lama_inpainted_1_rgb, cv2.COLOR_RGB2BGR)
    
    print("Running LaMa Inpainting for Panel 2 (60px)...")
    pil_mask_2 = Image.fromarray(mask_2)
    lama_inpainted_2_pil = lama(pil_img, pil_mask_2)
    lama_inpainted_2_rgb = np.array(lama_inpainted_2_pil)
    panel_2 = cv2.cvtColor(lama_inpainted_2_rgb, cv2.COLOR_RGB2BGR)
    
    print("Creating Panel 3 with Brightness/Contrast Normalization...")
    panel_3 = panel_2.copy()
    
    # Border mask 30px around mask_2
    # 30px dilation -> 30*2+1 = 61 kernel
    kernel_30 = np.ones((61, 61), np.uint8)
    dilated_mask_2 = cv2.dilate(mask_2, kernel_30, iterations=1)
    border_mask = cv2.subtract(dilated_mask_2, mask_2)
    
    # Calculate stats
    mean_bg, std_bg = cv2.meanStdDev(img, mask=border_mask)
    mean_in, std_in = cv2.meanStdDev(panel_3, mask=mask_2)
    
    # Apply color mapping in the filled area
    for i in range(3):
        mean_bg_c = mean_bg[i][0]
        std_bg_c = std_bg[i][0]
        
        mean_in_c = mean_in[i][0]
        std_in_c = std_in[i][0]
        
        if std_in_c == 0:
            std_in_c = 1.0
            
        patch = panel_3[:, :, i].astype(np.float32)
        patch_norm = (patch - mean_in_c) * (std_bg_c / std_in_c) + mean_bg_c
        patch_norm = np.clip(patch_norm, 0, 255).astype(np.uint8)
        
        np.copyto(panel_3[:, :, i], patch_norm, where=mask_2 > 0)
        
    print("Creating comparison image...")
    comparison = cv2.hconcat([panel_1, panel_2, panel_3])
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(H / 1000.0, 1.0)
    thickness = max(int(scale * 2), 2)
    
    text_1 = "Panel 1 (50px dilate)"
    text_2 = "Panel 2 (60px dilate)"
    text_3 = "Panel 3 (60px + blend)"
    
    size_1, _ = cv2.getTextSize(text_1, font, scale, thickness)
    size_2, _ = cv2.getTextSize(text_2, font, scale, thickness)
    size_3, _ = cv2.getTextSize(text_3, font, scale, thickness)
    
    x_1 = (W - size_1[0]) // 2
    x_2 = W + (W - size_2[0]) // 2
    x_3 = 2 * W + (W - size_3[0]) // 2
    y_text = int(size_1[1] * 2)
    
    def draw_text_with_bg(target_img, text, pos, txt_scale, txt_thickness):
        x, y = pos
        (w, h), baseline = cv2.getTextSize(text, font, txt_scale, txt_thickness)
        cv2.rectangle(target_img, (x - 10, y - h - 10), (x + w + 10, y + baseline + 10), (0, 0, 0), -1)
        cv2.putText(target_img, text, (x, y), font, txt_scale, (255, 255, 255), txt_thickness, cv2.LINE_AA)
        
    draw_text_with_bg(comparison, text_1, (x_1, y_text), scale, thickness)
    draw_text_with_bg(comparison, text_2, (x_2, y_text), scale, thickness)
    draw_text_with_bg(comparison, text_3, (x_3, y_text), scale, thickness)
    
    cv2.imwrite(output_path, comparison)
    print(f"Saved final comparison to: {output_path}")

if __name__ == '__main__':
    main()
