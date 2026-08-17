import cv2
import numpy as np
import os

def main():
    img_path = os.path.join("input_images", "Miss1_Flgt1_073.JPG")
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        return
        
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Let's look for a broader "yellow" range to see what we find
    # H 10-40, S 40-255, V 40-255
    lower_broad = np.array([10, 40, 40])
    upper_broad = np.array([45, 255, 255])
    mask = cv2.inRange(hsv, lower_broad, upper_broad)
    
    # Save a mask visualization
    os.makedirs("test_results", exist_ok=True)
    cv2.imwrite("test_results/broad_yellow_mask.jpg", mask)
    
    # Find the largest blobs and print their average HSV
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    print(f"Found {num_labels - 1} blobs in broad yellow range.")
    
    # Sort by area descending
    blobs = []
    for i in range(1, num_labels):
        blobs.append((stats[i, cv2.CC_STAT_AREA], i))
    blobs.sort(key=lambda x: x[0], reverse=True)
    
    for area, idx in blobs[:10]:
        blob_mask = (labels == idx).astype(np.uint8)
        avg_hsv = cv2.mean(hsv, mask=blob_mask)
        print(f"Blob index {idx}: Area={area}, Avg HSV={avg_hsv[:3]}")
        
if __name__ == "__main__":
    main()
