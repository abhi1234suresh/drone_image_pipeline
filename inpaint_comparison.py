import os
import cv2
import torch
import numpy as np
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

def main():
    image_path = os.path.join("test_images", "Miss1_Flgt1_001.JPG")
    output_path = os.path.join("test_results", "inpaint_comparison.jpg")
    
    if not os.path.exists(image_path):
        print(f"Error: {image_path} does not exist.")
        return
        
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Detection Phase
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

    # Filter for cars, buses, trucks (2, 5, 7)
    target_classes = [2, 5, 7]
    filtered_predictions = [p for p in result.object_prediction_list if p.category.id in target_classes]
    
    print(f"Found {len(filtered_predictions)} vehicles.")
    
    # Load original image
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to read image")
        return
        
    H, W, C = img.shape
    
    # 2. Create Mask
    mask = np.zeros((H, W), dtype=np.uint8)
    for pred in filtered_predictions:
        bbox = pred.bbox
        x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
        # fill bounding box with 255
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
    # Dilate mask by 20 pixels
    # Creating a structuring element of size (41, 41) for a 20 pixel dilation in all directions
    kernel = np.ones((41, 41), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    
    # 3. Method A: OpenCV INPAINT_TELEA
    print("Running OpenCV Inpainting...")
    opencv_inpainted = cv2.inpaint(img, dilated_mask, 3, cv2.INPAINT_TELEA)
    
    # 4. Method B: LaMa Inpainting
    print("Running LaMa Inpainting...")
    import simple_lama_inpainting
    from simple_lama_inpainting import SimpleLama
    
    # LaMa needs exactly PIL RGB images for simple-lama-inpainting
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    pil_mask = Image.fromarray(dilated_mask)
    
    lama = SimpleLama()
    lama_inpainted_pil = lama(pil_img, pil_mask)
    
    # Convert LaMa result back to BGR for OpenCV
    lama_inpainted_rgb = np.array(lama_inpainted_pil)
    lama_inpainted = cv2.cvtColor(lama_inpainted_rgb, cv2.COLOR_RGB2BGR)

    # 5. Side-by-Side Comparison
    print("Creating comparison image at full resolution...")
    
    # Hconcat the two images
    comparison = cv2.hconcat([opencv_inpainted, lama_inpainted])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Dynamic scale based on image height; typical drone photo ~3000-4000px
    scale = max(H / 1000.0, 1.0)
    thickness = max(int(scale * 2), 2)
    
    text_cv = "OpenCV"
    text_lama = "LaMa"
    
    size_cv, _ = cv2.getTextSize(text_cv, font, scale, thickness)
    size_lama, _ = cv2.getTextSize(text_lama, font, scale, thickness)
    
    x_cv = (W - size_cv[0]) // 2
    x_lama = W + (W - size_lama[0]) // 2
    y_text = int(size_cv[1] * 2)
    
    def draw_text_with_bg(target_img, text, pos, txt_scale, txt_thickness):
        x, y = pos
        (w, h), baseline = cv2.getTextSize(text, font, txt_scale, txt_thickness)
        cv2.rectangle(target_img, (x - 10, y - h - 10), (x + w + 10, y + baseline + 10), (0, 0, 0), -1)
        cv2.putText(target_img, text, (x, y), font, txt_scale, (255, 255, 255), txt_thickness, cv2.LINE_AA)
        
    draw_text_with_bg(comparison, text_cv, (x_cv, y_text), scale, thickness)
    draw_text_with_bg(comparison, text_lama, (x_lama, y_text), scale, thickness)
    
    cv2.imwrite(output_path, comparison)
    print(f"Saved comparison to: {output_path}")

if __name__ == '__main__':
    main()
