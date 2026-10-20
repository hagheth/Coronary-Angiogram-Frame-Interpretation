import os
import cv2
import torch
from ultralytics import YOLO
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def preprocess_frame(frame, target_size=(512, 512)):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    
    # Resize to standardized format
    resized_img = cv2.resize(clahe_img, target_size, interpolation=cv2.INTER_AREA)
    
    # Convert back to BGR so we can draw colored boxes on it later
    processed_bgr = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)
    return processed_bgr

def process_angiogram_video(video_path, output_path, model_path):
    print(f"\n--- Phase 1: Video Processing ({os.path.basename(video_path)}) ---")
    
    # 1. Load YOLOv8 AI Brain
    print("Loading YOLOv8 Object Detection Brain...")
    yolo_model = YOLO(model_path)
    
    # 2. Open Video using OpenCV
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 3. Create Video Writer to stitch frames back together
    # We force the video size to 512x512 because we are resizing all frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (512, 512))
    
    # Statistics for the Flan-T5 Report
    frames_with_stenosis = 0
    max_confidence = 0.0
    
    print(f"Analyzing {total_frames} frames individually...")
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. PREPROCESS THE FRAME FIRST (Grayscale, CLAHE, Resize)
        preprocessed_frame = preprocess_frame(frame)
        
        # 2. Run YOLO Detection on the PREPROCESSED frame
        results = yolo_model.predict(source=preprocessed_frame, conf=0.5, verbose=False)
        result = results[0]
        
        annotated_frame = preprocessed_frame.copy()
        boxes = result.boxes
        if len(boxes) > 0:
            frames_with_stenosis += 1
            for box in boxes:
                # Track max confidence
                conf = float(box.conf[0])
                if conf > max_confidence:
                    max_confidence = conf
                
                # --- Custom Drawing for Percentage ---
                # Get pixel coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Format text: e.g. "Stenosis: 92%"
                percent_str = f"{int(conf * 100)}%"
                label = f"Stenosis: {percent_str}"
                
                color = (0, 255, 150) # Beautiful Neon Green
                thickness = 2
                
                # Draw the bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Draw a sleek filled background for the text
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + tw + 10, y1), color, -1)
                
                # Draw the percentage text inside the box in black
                cv2.putText(annotated_frame, label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
        # Write the custom drawn frame into the new video file
        out.write(annotated_frame)
        
        frame_idx += 1
        if frame_idx % 30 == 0: # Print update every 30 frames
            print(f"Processed {frame_idx}/{total_frames} frames...")
            
    cap.release()
    out.release()
    print(f"Finished processing video! New video saved to: {output_path}")
    
    return total_frames, frames_with_stenosis, max_confidence

def generate_video_report(total_frames, frames_with_stenosis, max_confidence, video_name):
    print("\n--- Phase 2: Generating AI Clinical Report ---")
    print("Loading Flan-T5 Language Model...")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_name = "google/flan-t5-base"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    flan_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    
    # Formulate clinical findings based on the entire video timeline
    if frames_with_stenosis > 0:
        findings = (
            f"Stenosis was continuously detected across {frames_with_stenosis} out of {total_frames} frames in the angiogram video. "
            f"The peak AI confidence score during the video sequence was {max_confidence * 100:.2f}%. "
            f"This suggests a persistent narrowing of the vessel across multiple cardiac phases."
        )
    else:
        findings = f"No stenosis was detected in any of the {total_frames} frames of the angiogram video sequence."
        
    prompt = (
        "Generate a formal, highly structured clinical radiology report based strictly on the provided data.\n"
        "Do not invent patient names, dates, or specific arteries. Use a professional, objective medical tone.\n\n"
        "DATA:\n"
        f"Procedure: Coronary Angiography\n"
        f"Video File: {video_name}\n"
        f"AI Detection Results: {findings}\n\n"
        "REQUIRED FORMAT:\n"
        "EXAM: Coronary Angiography\n"
        "FINDINGS: [Clinical description of the AI results]\n"
        "IMPRESSION: [Brief diagnostic summary]\n\n"
        "REPORT:"
    )
    
    print("Writing the final report...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # We use beam search instead of random sampling for highly consistent, non-hallucinated clinical reports
    outputs = flan_model.generate(**inputs, max_new_tokens=200, num_beams=4, early_stopping=True)
    report_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return report_text

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    VIDEO_DIR = os.path.join(BASE_DIR, "video_test")
    YOLO_WEIGHTS = os.path.join(BASE_DIR, "runs", "detect", "stenosis_detector", "weights", "best.pt")
    
    # Create an output folder for the final products
    OUTPUT_DIR = os.path.join(BASE_DIR, "video_output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(VIDEO_DIR):
        print(f"Error: Could not find the folder '{VIDEO_DIR}'")
        exit()
        
    # ONLY look for the raw #input videos, ignoring the previously tested model outputs
    test_videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith('#input.avi')]
    
    if len(test_videos) == 0:
        print(f"Error: No '#input.avi' files found in '{VIDEO_DIR}'.")
    else:
        print("\n=== AVAILABLE RAW ANGIOGRAM VIDEOS ===")
        for i, video in enumerate(test_videos):
            print(f"[{i+1}] {video}")
            
        # Get user input interactively
        try:
            choice_str = input(f"\nEnter the number of the video you want to process (1-{len(test_videos)}): ")
            choice = int(choice_str) - 1
            if choice < 0 or choice >= len(test_videos):
                print("Invalid choice. Exiting script.")
                exit()
        except ValueError:
            print("Please enter a valid number. Exiting script.")
            exit()
            
        video_filename = test_videos[choice]
        input_video_path = os.path.join(VIDEO_DIR, video_filename)
        output_video_path = os.path.join(OUTPUT_DIR, f"yolov8_detected_{video_filename}")
        
        # Step 1: Video Processing (OpenCV -> YOLO -> OpenCV)
        total, stenosis_frames, max_conf = process_angiogram_video(input_video_path, output_video_path, YOLO_WEIGHTS)
        
        # Step 2: Language Processing (Flan-T5)
        final_report = generate_video_report(total, stenosis_frames, max_conf, video_filename)
        
        # Step 3: Output the Results
        print("\n" + "="*70)
        print("                 AI CLINICAL ANGIOGRAM VIDEO REPORT")
        print("="*70)
        print(final_report)
        print("="*70)
        
        # Save the report to a text file next to the video
        report_path = os.path.join(OUTPUT_DIR, f"report_{os.path.splitext(video_filename)[0]}.txt")
        with open(report_path, 'w') as f:
            f.write("AI CLINICAL ANGIOGRAM VIDEO REPORT\n")
            f.write("="*50 + "\n")
            f.write(final_report + "\n")
        print(f"Text report saved to: {report_path}")
