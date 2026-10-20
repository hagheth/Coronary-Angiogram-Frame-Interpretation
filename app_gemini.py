# pyrefly: ignore [missing-import]
import streamlit as st
import cv2
import os
import torch
import numpy as np
import tempfile
from ultralytics import YOLO
from google import genai

st.set_page_config(page_title="Coronary Angiogram Analysis (Gemini)", layout="wide")

# --- Custom Styling ---
st.markdown("""
    <style>
    .stat-box {
        font-size: 26px;
        font-weight: 500;
        margin-bottom: 20px;
        font-family: sans-serif;
    }
    .highlight {
        color: #E2B93B;
        font-weight: bold;
    }
    .header-title {
        text-align: center;
        color: #1E88E5;
        margin-bottom: 40px;
    }
    .report-box {
        background-color: #1a1a2e;
        border: 1px solid #0f3460;
        border-radius: 8px;
        padding: 20px;
        font-family: 'Courier New', monospace;
        font-size: 14px;
        line-height: 1.7;
        white-space: pre-wrap;
        color: #e0e0e0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="header-title">Coronary Angiogram AI Interpretation — Gemini Edition</h1>',
            unsafe_allow_html=True)

# ── Sidebar: API Key Input ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        placeholder="Paste your API key here...",
        help="Get a free key at https://aistudio.google.com/app/apikey"
    )
    st.markdown(
        "🔑 [Get a free API key](https://aistudio.google.com/app/apikey)",
        unsafe_allow_html=False
    )
    st.markdown("---")
    st.markdown("**Model:** `gemini-1.5-flash`")
    st.markdown("**Detector:** YOLOv8 (local)")

# ── Load YOLO only — no Flan-T5 needed ────────────────────────────────────────
@st.cache_resource
def load_yolo():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "runs", "detect", "stenosis_detector", "weights", "best.pt")
    return YOLO(model_path)

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

# ── Report Generation — Gemini API ────────────────────────────────────────────
def classify_severity(all_detections):
    """Return (severity_class, max_conf_pct, impression, recommendation)."""
    if not all_detections:
        return (
            "None",
            0.0,
            "No hemodynamically significant coronary artery stenosis detected.",
            "Routine clinical follow-up as clinically indicated."
        )
    max_conf     = max(d["raw_conf"] for d in all_detections)
    max_conf_pct = max_conf * 100

    if max_conf_pct < 50:
        return (
            "Mild", max_conf_pct,
            (f"Mild coronary artery stenosis identified (peak AI confidence {max_conf_pct:.1f}%). "
             f"Likely non-obstructive coronary artery disease."),
            "Medical management and lifestyle modification advised. Repeat imaging in 6-12 months."
        )
    elif max_conf_pct < 70:
        return (
            "Moderate", max_conf_pct,
            (f"Moderate coronary artery stenosis identified (peak AI confidence {max_conf_pct:.1f}%). "
             f"Clinical correlation with patient symptoms is strongly advised."),
            ("Cardiology consultation recommended. Consider functional assessment (FFR or iFR) "
             "and optimisation of anti-anginal therapy.")
        )
    elif max_conf_pct < 90:
        return (
            "Severe", max_conf_pct,
            (f"Severe coronary artery stenosis identified (peak AI confidence {max_conf_pct:.1f}%). "
             f"Significant lumen compromise is present."),
            ("Urgent cardiology review required. Percutaneous coronary intervention (PCI) or "
             "coronary artery bypass grafting (CABG) should be considered.")
        )
    else:
        return (
            "Critical", max_conf_pct,
            (f"Critical coronary artery stenosis identified (peak AI confidence {max_conf_pct:.1f}%). "
             f"Near-total or total occlusion cannot be excluded."),
            ("Immediate cardiology intervention required. Emergency PCI or surgical referral "
             "should be arranged without delay.")
        )

def generate_report_gemini(api_key, file_name, all_detections, total_frames=None, is_video=False):
    """
    Generate a detailed clinical report using the Gemini API (new google-genai SDK).
    Forces API version v1 and tries multiple model names as fallbacks.
    """
    # Force v1 API — the SDK defaults to v1beta which has model availability issues
    client = genai.Client(
        api_key=api_key,
        http_options={"api_version": "v1"}
    )

    severity_class, max_conf_pct, impression, recommendation = classify_severity(all_detections)

    # Build structured data summary for the prompt
    if not all_detections:
        lesion_summary = "No stenotic lesions were detected."
        video_stat = ""
    else:
        lesion_lines = []
        for i, d in enumerate(all_detections[:5], 1):
            lesion_lines.append(
                f"  - Lesion {i}: pixel location {d['coords']}, "
                f"lumen width {d['lumen_dia']}, AI confidence {d['percentage']}"
            )
        lesion_summary = "\n".join(lesion_lines)

        video_stat = ""
        if is_video and total_frames:
            pct_frames = (len(all_detections) / total_frames) * 100
            video_stat = (
                f"\n- Temporal extent: stenosis detected in {len(all_detections)} of "
                f"{total_frames} frames ({pct_frames:.1f}% of study duration)"
            )

    prompt = f"""You are a senior interventional cardiologist writing a formal clinical radiology report.

STUDY DATA PROVIDED BY AI DETECTION SYSTEM:
- Procedure: Coronary Angiography
- File: {file_name}
- Severity Classification: {severity_class}
- Peak AI Confidence Score: {max_conf_pct:.1f}%
- Detected Lesions:
{lesion_summary}{video_stat}
- Pre-computed Impression: {impression}
- Pre-computed Recommendation: {recommendation}

YOUR TASK:
Write a complete, detailed, and professional clinical radiology report using the above data.
Structure it with these clearly labelled sections:

1. EXAMINATION
2. CLINICAL INDICATION
3. TECHNIQUE
4. FINDINGS
5. IMPRESSION
6. RECOMMENDATION

Guidelines:
- Write in formal, objective medical prose (3rd person, past tense for technique).
- For FINDINGS, expand the lesion data into 3-5 clinical sentences describing location, morphology, and haemodynamic significance.
- Do NOT invent patient names, dates, referring physicians, or specific artery names beyond what the data supports.
- End with a one-line disclaimer that this is AI-assisted analysis requiring physician review.
- Do not use bullet points inside the sections — write full paragraphs.
"""

    # Try models in order of preference until one works
    models_to_try = [
        "gemini-2.0-flash-lite",   # lightest, most quota-friendly
        "gemini-2.0-flash",
        "gemini-2.5-flash",
    ]

    last_error = None
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            last_error = e
            continue

    raise last_error


# ── Main App ───────────────────────────────────────────────────────────────────
with st.spinner("Loading YOLO model..."):
    yolo_model = load_yolo()

uploaded_file = st.file_uploader(
    "Upload Angiogram Video or Image",
    type=["mp4", "avi", "mov", "bmp", "jpg", "png"]
)

if uploaded_file is not None:
    if not api_key:
        st.warning("⚠️ Please enter your Gemini API key in the sidebar to generate the report.")

    is_video = uploaded_file.name.lower().endswith(('.mp4', '.avi', '.mov'))

    with st.spinner("Analysing Angiogram with YOLO..."):
        if not is_video:
            # --- Single image ---
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
            annotated_img, data = process_single_frame(frame, yolo_model)

            all_detections = [data] if data["stenosis_yn"] == "Y" else []
            total_frames = 1
            final_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        else:
            # --- Video ---
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())

            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS)

            out_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_video_file.name, fourcc, fps, (512, 512))

            best_annotated_img = None
            best_data = {"stenosis_yn": "N", "coords": "-", "lumen_dia": "-",
                         "percentage": "-", "raw_conf": 0.0}
            all_detections = []
            total_frames = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                annotated_img, data = process_single_frame(frame, yolo_model)
                out.write(annotated_img)
                total_frames += 1

                if data["stenosis_yn"] == "Y":
                    all_detections.append(data)

                if data["raw_conf"] > best_data["raw_conf"]:
                    best_data = data
                    best_annotated_img = annotated_img

            cap.release()
            out.release()

            if best_annotated_img is None:
                best_annotated_img = annotated_img

            final_img = cv2.cvtColor(best_annotated_img, cv2.COLOR_BGR2RGB)
            data = best_data

    # --- Display Layout ---
    col1, col2 = st.columns([0.5, 0.5])

    with col1:
        st.image(final_img, caption="AI Highlighted View", use_container_width=True)

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div class='stat-box'>Stenosis : <span class='highlight'>{data['stenosis_yn']}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='stat-box'>Coordinate : <span class='highlight'>{data['coords']}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='stat-box'>Lumen Dia : <span class='highlight'>{data['lumen_dia']}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='stat-box'>Percentage : <span class='highlight'>{data['percentage']}</span></div>", unsafe_allow_html=True)

        # --- Gemini Report ---
        st.markdown("---")
        st.markdown("### 📝 AI Clinical Report (Gemini)")

        if api_key:
            with st.spinner("Generating detailed report with Gemini..."):
                try:
                    report_text = generate_report_gemini(
                        api_key,
                        uploaded_file.name,
                        all_detections,
                        total_frames=total_frames,
                        is_video=is_video,
                    )
                    st.markdown(
                        f"<div class='report-box'>{report_text}</div>",
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.error(f"❌ Gemini API error: {e}")
                    st.info("Check your API key in the sidebar or visit https://aistudio.google.com/app/apikey")
        else:
            st.info("Enter your Gemini API key in the sidebar to generate the report.")

        # --- Video Download ---
        if is_video:
            st.markdown("---")
            with open(out_video_file.name, "rb") as vf:
                st.download_button(
                    label="⬇️ Download Processed Angiogram Video",
                    data=vf,
                    file_name=f"AI_Analyzed_{uploaded_file.name}",
                    mime="video/mp4"
                )
