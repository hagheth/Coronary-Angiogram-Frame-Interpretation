import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def generate_clinical_report(json_data):
    """
    Report Generation Layer using Flan-T5.
    Converts structured JSON detection data into a clinical-style summary.
    """
    print("Loading Flan-T5 model... (This might take a minute the very first time to download the model weights)")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_name = "google/flan-t5-base"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    
    # 1. Translate our Robot JSON into a human-readable prompt
    frame_id = json_data.get("frame_id", "Unknown")
    detected = json_data.get("stenosis_detected", False)
    max_conf = json_data.get("overall_max_confidence", 0.0)
    severity = json_data.get("estimated_severity_percentage", 0.0)
    
    if detected:
        findings = f"stenosis detected with an AI confidence of {max_conf*100}%. The estimated severity of the blockage is {severity*100}%."
    else:
        findings = "no clear signs of stenosis were detected in this frame."
        
    # We construct a prompt commanding the AI how to behave
    prompt = (
        f"You are a professional medical radiologist.\n"
        f"Write a brief, professional clinical medical report summary based on the following angiogram findings:\n"
        f"Frame ID: {frame_id}\n"
        f"Findings: {findings}\n"
        f"Report:"
    )
    
    # 2. Generate the report!
    print("Generating clinical report...\n")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # max_new_tokens controls how long the report is allowed to be
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
    report_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return report_text

if __name__ == "__main__":
    # Test Data from our YOLOv8 Post-Processing script
    sample_json = {
        "frame_id": "orig_14_002_5_0018.bmp",
        "stenosis_detected": True,
        "detections": [
            {
                "confidence": 0.758,
                "bbox_pixels": [226, 174, 246, 188],
                "severity_percentage": 0.64
            }
        ],
        "overall_max_confidence": 0.758,
        "estimated_severity_percentage": 0.64
    }
    
    final_report = generate_clinical_report(sample_json)
    
    print("="*60)
    print("                 AI CLINICAL ANGIOGRAM REPORT")
    print("="*60)
    print(final_report)
    print("="*60)
