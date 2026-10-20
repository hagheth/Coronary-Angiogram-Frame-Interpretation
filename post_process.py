import os
import cv2
import json
from ultralytics import YOLO

def analyze_angiogram(image_path, model_path, confidence_threshold=0.5):
    """
    Post-Processing Layer:
    Runs the trained YOLOv8 model on a single frame, filters low confidence,
    and returns a structured JSON dictionary for the Report Generation Layer.
    """
    # 1. Load your newly trained AI Brain!
    model = YOLO(model_path)
    
    # 2. Run inference (Detection) on the image
    # We set conf=confidence_threshold to automatically filter out bad guesses
    results = model.predict(source=image_path, conf=confidence_threshold, save=True, save_txt=False)
    
    # Grab the first (and only) image result
    result = results[0]
    
    frame_data = {
        "frame_id": os.path.basename(image_path),
        "stenosis_detected": False,
        "detections": [],
        "overall_max_confidence": 0.0,
        "estimated_severity_percentage": 0.0
    }
    
    # 3. Post-Process the bounding boxes
    boxes = result.boxes
    if len(boxes) > 0:
        frame_data["stenosis_detected"] = True
        
        for box in boxes:
            # Extract coordinates, confidence, and class
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            
            # Update max confidence
            if conf > frame_data["overall_max_confidence"]:
                frame_data["overall_max_confidence"] = round(conf, 3)
            
            # --- Stenosis Percentage Logic ---
            # NOTE: Bounding boxes just give location. True clinical "stenosis percentage" 
            # requires comparing the narrowest diameter to a healthy reference diameter. 
            # For now, we will assign a mock percentage based on confidence, but you can 
            # replace this with your project's specific mathematical formula!
            mock_percentage = round(conf * 0.85, 2) # e.g. 0.91 conf -> 77% stenosis
            frame_data["estimated_severity_percentage"] = mock_percentage
            
            frame_data["detections"].append({
                "confidence": round(conf, 3),
                "bbox_pixels": [int(x1), int(y1), int(x2), int(y2)],
                "severity_percentage": mock_percentage
            })
            
    return frame_data

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Path to your finished brain file
    MODEL_PATH = os.path.join(BASE_DIR, "runs", "detect", "stenosis_detector", "weights", "best.pt")
    
    # Let's test it on a random image from your validation set
    TEST_IMAGE = os.path.join(BASE_DIR, "dataset_augmented", "orig_14_002_5_0018.bmp") # change this to any image!
    
    if os.path.exists(TEST_IMAGE):
        print(f"--- Running Post-Processing on {os.path.basename(TEST_IMAGE)} ---")
        structured_output = analyze_angiogram(TEST_IMAGE, MODEL_PATH, confidence_threshold=0.60)
        
        print("\n--- FINAL STRUCTURED DATA FOR FLAN-T5 ---")
        print(json.dumps(structured_output, indent=4))
        
        print("\nNote: YOLO automatically saved a picture with the box drawn on it!")
        print("Check the newest folder inside 'runs/detect/' to see the visual result.")
    else:
        print("Please provide a valid test image path!")
