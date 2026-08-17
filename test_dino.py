import os
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image, ImageDraw

def main():
    img_path = os.path.join("input_images", "Miss1_Flgt1_043.JPG")
    if not os.path.exists(img_path):
        print("Image not found:", img_path)
        return
        
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading Grounding DINO on {device}...")
    
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    
    image = Image.open(img_path).convert("RGB")
    text = "excavator . crane . construction machine . vehicle ."
    
    print(f"Running Grounding DINO with prompt: '{text}' at conf=0.05")
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    try:
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            target_sizes=[image.size[::-1]]
        )[0]
    except Exception as e:
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.05,
            text_threshold=0.05,
            target_sizes=[image.size[::-1]]
        )[0]
    
    threshold = 0.05
    keep = results["scores"] > threshold
    
    boxes = results["boxes"][keep].cpu()
    scores = results["scores"][keep].cpu()
    labels = [results["labels"][i] for i, k in enumerate(keep) if k]
    
    print(f"Grounding DINO found: {len(boxes)} objects")
    
    draw = ImageDraw.Draw(image)
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        print(f"  - [{label}] bbox: {box}, conf: {score.item():.3f}")
        
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], max(0, box[1]-10)), f"{label} {score.item():.2f}", fill="red")
        
    out_path = os.path.join("test_results", "dino_test.jpg")
    image.save(out_path)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
