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
    output_path = os.path.join("test_results", "patch_fix_test.jpg")
    
    if not os.path.exists(image_path):
        print(f"Error: {image_path} does not exist.")
        return
        
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Running SAHI detection...")
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
        overlap_width_ratio=0.2,
        verbose=False
    )

    target_classes = [2, 5, 7]
    filtered_predictions = [p for p in result.object_prediction_list if p.category.id in target_classes]
    
    print(f"Found {len(filtered_predictions)} vehicles.")
    
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to read image")
        return
        
    H, W, C = img.shape
    
    # Create base binary mask
    mask = np.zeros((H, W), dtype=np.uint8)
    for pred in filtered_predictions:
        bbox = pred.bbox
        x1, y1 = max(0, int(bbox.minx) - 60), max(0, int(bbox.miny) - 60)
        x2, y2 = min(W, int(bbox.maxx) + 60), min(H, int(bbox.maxy) + 60)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
    # Fix 1: Soft mask edges (Gaussian blur kernel 51)
    print("Fix 1: Softening mask edges...")
    mask_blurred = cv2.GaussianBlur(mask, (51, 51), 0)
    
    # Run LaMa Inpainting
    print("Running LaMa Inpainting...")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    pil_mask = Image.fromarray(mask_blurred)
    
    lama = SimpleLama()
    inpainted_pil = lama(pil_img, pil_mask)
    inpainted = cv2.cvtColor(np.array(inpainted_pil), cv2.COLOR_RGB2BGR)
    
    print("Fix 2 & 3: Brightness matching and Poisson blending per vehicle patch...")
    final_result = img.copy()
    
    # Process each disconnected mask component independently for optimal results
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        # Create mask for this specific patch
        c_mask = np.zeros_like(mask)
        cv2.drawContours(c_mask, [cnt], -1, 255, -1)
        
        # Fix 2: Better brightness matching using 50px border ring
        # 50px dilation = 50*2+1 = 101 kernel
        kernel_50 = np.ones((101, 101), np.uint8)
        dilated_c_mask = cv2.dilate(c_mask, kernel_50, iterations=1)
        border_c_mask = cv2.subtract(dilated_c_mask, c_mask)
        
        mean_bg, std_bg = cv2.meanStdDev(img, mask=border_c_mask)
        mean_in, std_in = cv2.meanStdDev(inpainted, mask=c_mask)
        
        inpainted_corrected = inpainted.copy()
        
        for c in range(3):
            m_bg, s_bg = mean_bg[c][0], std_bg[c][0]
            m_in, s_in = mean_in[c][0], std_in[c][0]
            
            if s_in == 0: 
                s_in = 1.0
                
            patch = inpainted[:, :, c].astype(np.float32)
            patch_norm = (patch - m_in) * (s_bg / s_in) + m_bg
            patch_norm = np.clip(patch_norm, 0, 255).astype(np.uint8)
            
            np.copyto(inpainted_corrected[:, :, c], patch_norm, where=c_mask > 0)
            
        # Fix 3: Poisson blending (cv2.seamlessClone)
        x_, y_, w_, h_ = cv2.boundingRect(cnt)
        center_ = (x_ + w_ // 2, y_ + h_ // 2)
        
        try:
            # NORMAL_CLONE is used to preserve texture while matching boundary colors perfectly
            final_result = cv2.seamlessClone(
                inpainted_corrected, 
                final_result, 
                c_mask, 
                center_, 
                cv2.NORMAL_CLONE
            )
        except Exception as e:
            print(f"seamlessClone failed for a patch at {center_}: {e}")
            # Fallback to direct replacement if seamless clone fails (e.g. edge constraint issues)
            np.copyto(final_result, inpainted_corrected, where=c_mask[..., None] > 0)

    cv2.imwrite(output_path, final_result)
    print(f"Saved seamlessly healed result to: {output_path}")

if __name__ == '__main__':
    main()
