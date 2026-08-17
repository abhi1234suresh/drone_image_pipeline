import os
import glob
import time
import shutil
import datetime
import traceback
import argparse
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from simple_lama_inpainting import SimpleLama

# Enable max threaded performance
torch.set_num_threads(8)

def main():
    parser = argparse.ArgumentParser(description="Run full detection and inpainting pipeline.")
    parser.add_argument("--batch", type=int, default=None, help="Number of pending images to process in this run.")
    args = parser.parse_args()

    start_time = time.time()
    
    # Configuration
    input_dir = "input_images"
    output_dir = "output_images"
    logs_dir = "logs"
    log_file = os.path.join(logs_dir, "pipeline_log.txt")
    
    # Ensure directories exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    def log_message(msg):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {msg}\n")
    
    log_message("=== Pipeline Started ===")
    
    # Fix 1: Properly glob without double counting on Windows
    all_files = glob.glob(os.path.join(input_dir, "*"))
    image_paths = list({os.path.abspath(p) for p in all_files if p.lower().endswith(('.jpg', '.jpeg'))})
    
    if not image_paths:
        log_message("No JPG files found in input_images/.")
        print(f"No input images found in '{input_dir}' folder.")
        return
        
    pending_images = []
    for p in sorted(image_paths):
        fname = os.path.basename(p)
        if not os.path.exists(os.path.join(output_dir, fname)):
            pending_images.append(p)
            
    skipped = len(image_paths) - len(pending_images)
    print(f"Found {len(image_paths)} images total. Skipping {skipped} already processed.")
    
    if args.batch is not None and args.batch > 0:
        if len(pending_images) > args.batch:
            pending_images = pending_images[:args.batch]
            print(f"Applied --batch limit: processing exactly {len(pending_images)} images this run.")
            
    log_message(f"Images remaining to process: {len(pending_images)}. Skipped: {skipped}.")
    
    if not pending_images:
        print("All pending images are already processed.")
        log_message("=== Pipeline Completed ===")
        return

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Initializing models on {device}...")
    
    try:
        # Load standard YOLOv8m (Standard Vehicles)
        detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path='D:/Project/Drone_image_labelling/train_results/drone_vehicles/weights/best.pt',
            confidence_threshold=0.45,
            device=device
        )
        lama = SimpleLama()
    except Exception as e:
        err = f"Failed to load models: {str(e)}"
        print(err)
        log_message(err)
        return
        
    target_classes = [0, 1] # LMV, heavy_equipment
    total_vehicles_removed = 0
    total_processed = 0
    
    pbar = tqdm(pending_images, desc="Processing Images", unit="img")
    for img_path in pbar:
        fname = os.path.basename(img_path)
        out_path = os.path.join(output_dir, fname)
        
        try:
            # 1. Standard YOLO Detection
            res_std = get_sliced_prediction(
                img_path,
                detection_model,
                slice_height=512,
                slice_width=512,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                verbose=False
            )
            std_preds = [p for p in res_std.object_prediction_list if p.category.id in target_classes]
            
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError("cv2.imread failed to load the image.")
            H, W, _ = img.shape

            # Filter 1: Remove detections larger than 30% of image (buildings)
            std_preds = [p for p in std_preds 
                        if (p.bbox.maxx - p.bbox.minx) < W * 0.3
                        and (p.bbox.maxy - p.bbox.miny) < H * 0.3]
            total_targets = len(std_preds)
            tqdm.write(f"[{fname}] Detected {total_targets} vehicles")
            
            if total_targets == 0:
                shutil.copy2(img_path, out_path)
                log_message(f"{fname} -> 0 vehicles found. Copied copy directly.")
                total_processed += 1
                continue
                
            # Create Combined Mask
            mask = np.zeros((H, W), dtype=np.uint8)
            for v in std_preds:
                b = v.bbox
                x1, y1 = max(0, int(b.minx) - 60), max(0, int(b.miny) - 60)
                x2, y2 = min(W, int(b.maxx) + 60), min(H, int(b.maxy) + 60)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                
            # Blur mask edges for better blending
            mask_blurred = cv2.GaussianBlur(mask, (51, 51), 0)
            
            # 3. LaMa Inpainting (25% scale for speed)
            start_inpaint = time.time()
            img_small = cv2.resize(img, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            mask_small = cv2.resize(mask_blurred, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
            
            img_rgb_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
            pil_img_small = Image.fromarray(img_rgb_small)
            pil_mask_small = Image.fromarray(mask_small)
            
            inpainted_pil_small = lama(pil_img_small, pil_mask_small)
            inpainted_bgr_small = cv2.cvtColor(np.array(inpainted_pil_small), cv2.COLOR_RGB2BGR)
            
            # Scale back up
            inpainted_full = cv2.resize(inpainted_bgr_small, (W, H), interpolation=cv2.INTER_CUBIC)
            
            # 4. Final Blending per unique contour
            final_result = img.copy()
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                c_mask = np.zeros_like(mask)
                cv2.drawContours(c_mask, [cnt], -1, 255, -1)
                
                # Brightness normalization using 50px border ring
                kernel_50 = np.ones((101, 101), np.uint8)
                dilated_c_mask = cv2.dilate(c_mask, kernel_50, iterations=1)
                border_c_mask = cv2.subtract(dilated_c_mask, c_mask)
                
                mean_bg, std_bg = cv2.meanStdDev(img, mask=border_c_mask)
                mean_in, std_in = cv2.meanStdDev(inpainted_full, mask=c_mask)
                
                inpainted_corrected = inpainted_full.copy()
                for ch in range(3):
                    m_bg, s_bg = mean_bg[ch][0], std_bg[ch][0]
                    m_in, s_in = mean_in[ch][0], std_in[ch][0]
                    s_in = max(s_in, 1.0)
                    
                    patch_norm = (inpainted_full[:, :, ch].astype(np.float32) - m_in) * (s_bg / s_in) + m_bg
                    patch_norm = np.clip(patch_norm, 0, 255).astype(np.uint8)
                    np.copyto(inpainted_corrected[:, :, ch], patch_norm, where=c_mask > 0)
                    
                # Poisson blending
                x_, y_, w_, h_ = cv2.boundingRect(cnt)
                center_ = (x_ + w_ // 2, y_ + h_ // 2)
                try:
                    final_result = cv2.seamlessClone(inpainted_corrected, final_result, c_mask, center_, cv2.NORMAL_CLONE)
                except:
                    np.copyto(final_result, inpainted_corrected, where=c_mask[..., None] > 0)
            
            cv2.imwrite(out_path, final_result)
            total_vehicles_removed += total_targets
            total_processed += 1
            log_message(f"{fname} -> OK. Removed {total_targets}")
            
        except Exception as e:
            log_message(f"{fname} -> FAIL: {str(e)}")
            tqdm.write(f"Error processing {fname}: {str(e)}")
            
    elapsed = time.time() - start_time
    sec_per_img = elapsed / total_processed if total_processed > 0 else 0
    summary = f"\nProcessed {total_processed} images in {elapsed:.2f}s ({sec_per_img:.2f}s/img). Removed {total_vehicles_removed} targets."
    print(summary)
    log_message(summary)

if __name__ == '__main__':
    main()
