# pyrefly: ignore [missing-import]
import streamlit as st
import cv2
import os
import torch
import numpy as np
import tempfile
from ultralytics import YOLO
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except Exception as e:
    import traceback
    with open("import_error.log", "w") as f:
        f.write(traceback.format_exc())
    raise e

st.set_page_config(page_title="Coronary Angiogram Analysis", layout="wide")

# --- Custom Styling for Premium Look ---
st.markdown("""
    <style>
    .stat-box {
        font-size: 26px;
        font-weight: 500;
        margin-bottom: 20px;
        font-family: sans-serif;
    }
    .highlight {
        color: #E2B93B; /* Matches the yellow highlight in your mockup */
        font-weight: bold;
    }
    .header-title {
        text-align: center;
        color: #1E88E5;
        margin-bottom: 40px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="header-title">Coronary Angiogram AI Interpretation</h1>', unsafe_allow_html=True)

@st.cache_resource
def load_ai_models():
    """Load models once so the UI stays fast."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "runs", "detect", "stenosis_detector", "weights", "best.pt")
    yolo_model = YOLO(model_path)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    flan_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(flan_name)
    flan_model = AutoModelForSeq2SeqLM.from_pretrained(flan_name).to(device)
    
    return yolo_model, tokenizer, flan_model, device

def preprocess_frame(frame, target_size=(512, 512)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    resized_img = cv2.resize(clahe_img, target_size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)

def process_single_frame(frame, yolo_model):
    preprocessed = preprocess_frame(frame)
    results = yolo_model.predict(source=preprocessed, conf=0.5, verbose=False)
    result = results[0]
    
    annotated = preprocessed.copy()
    boxes = result.boxes
    
    data = {
        "stenosis_yn": "N",
        "coords": "-",
        "lumen_dia": "-",
        "percentage": "-",
        "raw_conf": 0.0
    }
    
    if len(boxes) > 0:
        data["stenosis_yn"] = "Y"
        # Take the box with highest confidence
        best_box = max(boxes, key=lambda x: float(x.conf[0]))
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
        conf = float(best_box.conf[0])
        
        data["coords"] = f"[{x1}, {y1}]"
        data["lumen_dia"] = f"{abs(x2 - x1)} pixels"
        data["percentage"] = f"{int(conf * 100)}%"
        data["raw_conf"] = conf

        # Draw bounding box
        color = (0, 215, 255)  # BGR: golden yellow
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        label = f"Stenosis: {data['percentage']}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (x1, y1 - 26), (x1 + tw + 10, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 5, y1 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return annotated, data

def generate_report(tokenizer, flan_model, device, file_name, all_detections, total_frames=None, is_video=False):
    """
    Hybrid report generator:
    - Python template guarantees all 5 sections always appear.
    - Flan-T5 is only asked to write a short FINDINGS prose paragraph
      (a focused task it can reliably handle).
    """
    # ── 1. Severity Classification ────────────────────────────────────────────
    if not all_detections:
        severity_class = "None"
        raw_findings   = "No stenotic lesions were identified in this study."
        impression     = "No hemodynamically significant coronary artery stenosis detected."
        recommendation = "Routine clinical follow-up as clinically indicated."
        lesion_detail  = "  None."
        video_stat     = ""
    else:
        max_conf     = max(d["raw_conf"] for d in all_detections)
        max_conf_pct = max_conf * 100

        if max_conf_pct < 50:
            severity_class = "Mild"
            impression     = (f"Mild coronary artery stenosis identified "
                              f"(peak AI confidence {max_conf_pct:.1f}%). "
                              f"Likely non-obstructive coronary artery disease.")
            recommendation = ("Medical management and lifestyle modification advised. "
                              "Repeat imaging in 6-12 months.")
        elif max_conf_pct < 70:
            severity_class = "Moderate"
            impression     = (f"Moderate coronary artery stenosis identified "
                              f"(peak AI confidence {max_conf_pct:.1f}%). "
                              f"Clinical correlation with patient symptoms is strongly advised.")
            recommendation = ("Cardiology consultation recommended. Consider functional "
                              "assessment (FFR or iFR) and optimisation of anti-anginal therapy.")
        elif max_conf_pct < 90:
            severity_class = "Severe"
            impression     = (f"Severe coronary artery stenosis identified "
                              f"(peak AI confidence {max_conf_pct:.1f}%). "
                              f"Significant lumen compromise is present.")
            recommendation = ("Urgent cardiology review required. Percutaneous coronary "
                              "intervention (PCI) or coronary artery bypass grafting (CABG) "
                              "should be considered.")
        else:
            severity_class = "Critical"
            impression     = (f"Critical coronary artery stenosis identified "
                              f"(peak AI confidence {max_conf_pct:.1f}%). "
                              f"Near-total or total occlusion cannot be excluded.")
            recommendation = ("Immediate cardiology intervention required. Emergency PCI or "
                              "surgical referral should be arranged without delay.")

        # Per-lesion bullet lines
        lesion_lines = []
        for i, d in enumerate(all_detections[:5], 1):
            lesion_lines.append(
                f"  Lesion {i}: pixel location {d['coords']}, "
                f"lumen width {d['lumen_dia']}, AI confidence {d['percentage']}"
            )
        lesion_detail = "\n".join(lesion_lines)

        # Video temporal stat
        video_stat = ""
        if is_video and total_frames:
            pct_frames = (len(all_detections) / total_frames) * 100
            video_stat = (
                f"Stenosis was detected in {len(all_detections)} of {total_frames} "
                f"frames ({pct_frames:.1f}% of study duration). "
            )

        raw_findings = (
            f"{len(all_detections)} region(s) of coronary stenosis detected. "
            f"Severity: {severity_class} (peak confidence {max_conf_pct:.1f}%). "
            f"{video_stat}"
            f"Peak lumen narrowing at {all_detections[0]['coords']} "
            f"with bounding box width {all_detections[0]['lumen_dia']}."
        )

    # ── 2. Ask Flan-T5 only for a short FINDINGS prose paragraph ─────────────
    findings_prompt = (
        f"Write 2 professional clinical sentences describing these angiogram findings: "
        f"{raw_findings}"
    )
    inputs  = tokenizer(findings_prompt, return_tensors="pt").to(device)
    outputs = flan_model.generate(
        **inputs,
        max_new_tokens=120,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    findings_paragraph = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ── 3. Assemble the full structured report using a Python template ─────────
    report = (
        f"EXAM\n"
        f"{'─'*60}\n"
        f"Procedure : Coronary Angiography\n"
        f"File      : {file_name}\n"
        f"Severity  : {severity_class}\n"
        f"\n"
        f"TECHNIQUE\n"
        f"{'─'*60}\n"
        f"Coronary angiography frames were analyzed using an AI-assisted\n"
        f"YOLOv8 stenosis detection model. Images were preprocessed with\n"
        f"CLAHE contrast enhancement and resized to 512×512 pixels prior\n"
        f"to inference (confidence threshold ≥ 0.50).\n"
        f"\n"
        f"FINDINGS\n"
        f"{'─'*60}\n"
        f"{findings_paragraph}\n"
        f"\n"
        f"Detected lesion(s):\n"
        f"{lesion_detail}\n"
        f"\n"
        f"IMPRESSION\n"
        f"{'─'*60}\n"
        f"{impression}\n"
        f"\n"
        f"RECOMMENDATION\n"
        f"{'─'*60}\n"
        f"{recommendation}\n"
        f"\n"
        f"{'═'*60}\n"
        f"⚠  AI-GENERATED REPORT — For decision support only.\n"
        f"   Must be reviewed by a qualified cardiologist before\n"
        f"   use in patient care.\n"
        f"{'═'*60}"
    )
    return report

# 1. Load Models automatically
with st.spinner("Waking up AI Models..."):
    yolo_model, tokenizer, flan_model, device = load_ai_models()

# 2. File Uploader
uploaded_file = st.file_uploader("Upload Angiogram Video or Image", type=["mp4", "avi", "mov", "bmp", "jpg", "png"])

if uploaded_file is not None:
    is_video = uploaded_file.name.lower().endswith(('.mp4', '.avi', '.mov'))
    
    with st.spinner("Analyzing Angiogram..."):
        if not is_video:
            # Process single image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
            annotated_img, data = process_single_frame(frame, yolo_model)

            # Collect detections for the enhanced report generator
            all_detections = [data] if data["stenosis_yn"] == "Y" else []
            total_frames = 1

            final_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
        else:
            # Process video - find the frame with the worst stenosis to display on the UI
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Setup output video writer
            out_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_video_file.name, fourcc, fps, (512, 512))
            
            best_annotated_img = None
            best_data = {
                "stenosis_yn": "N",
                "coords": "-",
                "lumen_dia": "-",
                "percentage": "-",
                "raw_conf": 0.0
            }
            # Accumulate all positive detections across every frame
            all_detections = []
            total_frames = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                annotated_img, data = process_single_frame(frame, yolo_model)
                out.write(annotated_img)  # Save to compiled video
                total_frames += 1

                # Collect every frame that had a positive detection
                if data["stenosis_yn"] == "Y":
                    all_detections.append(data)

                # Keep the highest-confidence frame for the UI thumbnail
                if data["raw_conf"] > best_data["raw_conf"]:
                    best_data = data
                    best_annotated_img = annotated_img

            cap.release()
            out.release()

            if best_annotated_img is None:
                best_annotated_img = annotated_img  # Fallback to last frame

            final_img = cv2.cvtColor(best_annotated_img, cv2.COLOR_BGR2RGB)
            data = best_data

        # --- Display Layout (Matching the user's requested Mockup) ---
        col1, col2 = st.columns([0.5, 0.5])
        
        with col1:
            st.image(final_img, caption="AI Highlighted View", use_container_width=True)
            
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<div class='stat-box'>Stenosis : <span class='highlight'>{data['stenosis_yn']}</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='stat-box'>Coordinate : <span class='highlight'>{data['coords']}</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='stat-box'>lumen dia : <span class='highlight'>{data['lumen_dia']}</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='stat-box'>Percentage : <span class='highlight'>{data['percentage']}</span></div>", unsafe_allow_html=True)
            
            # Generate enhanced AI Clinical Report
            st.markdown("---")
            st.markdown("### 📝 AI Clinical Report")

            report_text = generate_report(
                tokenizer, flan_model, device,
                uploaded_file.name,
                all_detections,
                total_frames=total_frames,
                is_video=is_video,
            )

            st.text_area(
                label="",
                value=report_text,
                height=380,
            )
            
            # Provide Download Button for Video
            if is_video:
                st.markdown("---")
                with open(out_video_file.name, "rb") as vf:
                    st.download_button(
                        label="⬇️ Download Processed Angiogram Video",
                        data=vf,
                        file_name=f"AI_Analyzed_{uploaded_file.name}",
                        mime="video/mp4"
                    )
