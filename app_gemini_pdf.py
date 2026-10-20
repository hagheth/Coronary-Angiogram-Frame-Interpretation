# pyrefly: ignore [missing-import]
import streamlit as st
import cv2
import os
import io
import datetime
import tempfile
import numpy as np
from ultralytics import YOLO
from google import genai

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    HRFlowable, Table, TableStyle
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

st.set_page_config(page_title="Coronary Angiogram Frame Interpretation", layout="wide")

# --- Custom Styling ---
st.markdown("""
    <style>
    /* ── Hide Streamlit anchor links on headings ── */
    h1 a, h2 a, h3 a { display: none !important; }

    /* ── Global ── */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0d1117 0%, #0f1e33 60%, #0d1117 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1e33 0%, #0a1628 100%);
        border-right: 1px solid #1e3a5f;
    }
    [data-testid="stSidebar"] * { color: #c9d8ea !important; }

    /* ── Hero header ── */
    .hero-wrap {
        text-align: center;
        padding: 36px 0 28px 0;
        margin-bottom: 10px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .hero-tag {
        display: inline-block;
        background: linear-gradient(90deg, #1565C0, #1E88E5);
        color: #fff !important;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 3px;
        text-transform: uppercase;
        padding: 6px 18px;
        border-radius: 20px;
        margin-bottom: 18px;
    }
    .hero-title {
        font-size: 5.95rem;
        font-weight: 900;
        color: #ffffff;
        line-height: 1.1;
        margin: 0 0 14px 0;
        text-align: center;
        width: 100%;
    }
    .hero-title span { color: #42A5F5; }
    .hero-sub {
        font-size: 1.2rem;
        color: #7a9bb5;
        margin: 0;
        letter-spacing: 1px;
    }
    .hero-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #1E88E5 20%, #42A5F5 50%, #1E88E5 80%, transparent 100%);
        border: none;
        margin: 22px 0 0 0;
        width: 100%;
        display: block;
    }

    /* ── Upload zone ── */
    [data-testid="stFileUploader"] {
        background: #0f1e33;
        border: 1.5px dashed #1E88E5;
        border-radius: 12px;
        padding: 10px 18px;
    }

    /* ── Stat cards ── */
    .stat-card {
        background: linear-gradient(135deg, #112240, #0f1e33);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 14px 20px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .stat-icon {
        font-size: 22px;
        min-width: 32px;
        text-align: center;
    }
    .stat-label {
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #7a9bb5;
        margin-bottom: 3px;
    }
    .stat-value {
        font-size: 24px;
        font-weight: 800;
        color: #E2B93B;
    }

    /* ── Section headings inside main area ── */
    .section-label {
        font-size: 15px;
        font-weight: 800;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #42A5F5;
        margin: 28px 0 10px 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .section-label::after {
        content: '';
        flex: 1;
        height: 1.5px;
        background: linear-gradient(90deg, #1E88E5 0%, transparent 100%);
    }

    /* ── Report box ── */
    .report-box {
        background: #0a1628;
        border: 1px solid #1e3a5f;
        border-left: 3px solid #1E88E5;
        border-radius: 10px;
        padding: 22px 24px;
        font-family: 'Courier New', monospace;
        font-size: 13px;
        line-height: 1.8;
        white-space: pre-wrap;
        color: #c9d8ea;
        max-height: 420px;
        overflow-y: auto;
    }

    /* ── Sidebar section label ── */
    .sidebar-section {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #42A5F5 !important;
        margin: 18px 0 8px 0;
    }
    .sidebar-badge {
        display: inline-block;
        background: #112240;
        border: 1px solid #1e3a5f;
        border-radius: 6px;
        padding: 6px 12px;
        font-size: 14px;
        color: #c9d8ea !important;
        margin-bottom: 6px;
    }

    /* ── Download button ── */
    [data-testid="stDownloadButton"] button {
        background: linear-gradient(90deg, #1565C0, #1E88E5) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
        width: 100%;
    }
    [data-testid="stDownloadButton"] button:hover {
        background: linear-gradient(90deg, #1E88E5, #42A5F5) !important;
        box-shadow: 0 0 12px #1E88E555 !important;
    }

    /* ── Spinner / warning / info tweaks ── */
    [data-testid="stSpinner"] { color: #42A5F5 !important; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-wrap">
    <div class="hero-tag">AI-Powered Cardiology Tool</div>
    <h1 class="hero-title">Coronary Angiogram<br><span>Frame Interpretation</span></h1>
    <p class="hero-sub">YOLO11n stenosis detection &nbsp;·&nbsp; Gemini clinical report</p>
    <hr class="hero-divider"/>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫀 Control Panel")
    st.markdown("---")

    st.markdown('<div class="sidebar-section">API Configuration</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="sidebar-section">System Info</div>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-badge">🤖 Detector &nbsp; YOLO11n v2 (80/10/10)</span>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-badge">✨ LLM &nbsp; gemini-2.0-flash-lite</span>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-badge">📄 Export &nbsp; PDF (ReportLab)</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-section">Accepted Formats</div>', unsafe_allow_html=True)
    st.markdown("🎬 **Video:** MP4, AVI, MOV  \n🖼️ **Image:** BMP, JPG, PNG")

# ── YOLO loader ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_yolo():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "runs", "detect", "stenosis_v2_yolo11n", "weights", "best.pt")
    return YOLO(model_path)

# ── Image processing ───────────────────────────────────────────────────────────
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

        color = (0, 215, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        label = f"Stenosis: {data['percentage']}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (x1, y1 - 26), (x1 + tw + 10, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 5, y1 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return annotated, data

# ── Severity classification ────────────────────────────────────────────────────
def classify_severity(all_detections):
    if not all_detections:
        return (
            "None", 0.0,
            "No hemodynamically significant coronary artery stenosis detected.",
            "Routine clinical follow-up as clinically indicated."
        )
    max_conf = max(d["raw_conf"] for d in all_detections)
    max_conf_pct = max_conf * 100

    if max_conf_pct < 50:
        return (
            "Mild", max_conf_pct,
            (f"Mild coronary artery stenosis identified (peak AI confidence {max_conf_pct:.1f}%). "
             "Likely non-obstructive coronary artery disease."),
            "Medical management and lifestyle modification advised. Repeat imaging in 6-12 months."
        )
    elif max_conf_pct < 70:
        return (
            "Moderate", max_conf_pct,
            (f"Moderate coronary artery stenosis identified (peak AI confidence {max_conf_pct:.1f}%). "
             "Clinical correlation with patient symptoms is strongly advised."),
            ("Cardiology consultation recommended. Consider functional assessment (FFR or iFR) "
             "and optimisation of anti-anginal therapy.")
        )
    elif max_conf_pct < 90:
        return (
            "Severe", max_conf_pct,
            (f"Severe coronary artery stenosis identified (peak AI confidence {max_conf_pct:.1f}%). "
             "Significant lumen compromise is present."),
            ("Urgent cardiology review required. Percutaneous coronary intervention (PCI) or "
             "coronary artery bypass grafting (CABG) should be considered.")
        )
    else:
        return (
            "Critical", max_conf_pct,
            (f"Critical coronary artery stenosis identified (peak AI confidence {max_conf_pct:.1f}%). "
             "Near-total or total occlusion cannot be excluded."),
            ("Immediate cardiology intervention required. Emergency PCI or surgical referral "
             "should be arranged without delay.")
        )

# ── Gemini report generation ───────────────────────────────────────────────────
def generate_report_gemini(api_key, file_name, all_detections, total_frames=None, is_video=False):
    client = genai.Client(
        api_key=api_key,
        http_options={"api_version": "v1"}
    )

    severity_class, max_conf_pct, impression, recommendation = classify_severity(all_detections)

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

    models_to_try = [
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.5-flash",
    ]

    last_error = None
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            return response.text
        except Exception as e:
            last_error = e
            continue

    raise last_error

# ── PDF builder ────────────────────────────────────────────────────────────────
SEVERITY_COLORS = {
    "None":     colors.HexColor("#2e7d32"),
    "Mild":     colors.HexColor("#f9a825"),
    "Moderate": colors.HexColor("#ef6c00"),
    "Severe":   colors.HexColor("#c62828"),
    "Critical": colors.HexColor("#6a0000"),
}

def build_pdf(
    file_name: str,
    is_video: bool,
    data: dict,
    all_detections: list,
    total_frames: int,
    severity_class: str,
    max_conf_pct: float,
    impression: str,
    recommendation: str,
    report_text: str,
    annotated_bgr: np.ndarray,
    top_frames_bgr: list,          # up to 4 best frames (video only)
) -> bytes:
    """Assemble the PDF in memory and return raw bytes."""

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    W = A4[0] - 36 * mm   # usable width
    styles = getSampleStyleSheet()

    # ── Custom paragraph styles ────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=18,
        textColor=colors.HexColor("#1E88E5"),
        spaceAfter=4,
        alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.grey,
        alignment=TA_CENTER,
        spaceAfter=2,
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=colors.HexColor("#1E88E5"),
        spaceBefore=10,
        spaceAfter=4,
        borderPad=2,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10,
        leading=15,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
    caption_style = ParagraphStyle(
        "Caption",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.grey,
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    label_style = ParagraphStyle(
        "Label",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.white,
        alignment=TA_CENTER,
    )

    sev_color = SEVERITY_COLORS.get(severity_class, colors.grey)

    story = []

    # ── Header ─────────────────────────────────────────────────────────────────
    story.append(Paragraph("CORONARY ANGIOGRAM AI ANALYSIS REPORT", title_style))
    story.append(Paragraph("Powered by YOLO11n v2 (80/10/10 split) + Google Gemini", subtitle_style))
    story.append(Paragraph(
        f"Generated: {datetime.datetime.now().strftime('%d %B %Y, %H:%M')} &nbsp;|&nbsp; "
        f"File: <b>{file_name}</b>",
        subtitle_style,
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1E88E5"),
                             spaceAfter=10))

    # ── Severity badge ─────────────────────────────────────────────────────────
    badge_data = [[
        Paragraph(f"<b>SEVERITY: {severity_class.upper()}</b>", label_style),
        Paragraph(f"<b>PEAK CONFIDENCE: {max_conf_pct:.1f}%</b>", label_style),
        Paragraph(f"<b>STENOSIS DETECTED: {data['stenosis_yn']}</b>", label_style),
    ]]
    badge_table = Table(badge_data, colWidths=[W / 3] * 3)
    badge_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), sev_color),
        ("TEXTCOLOR",  (0, 0), (-1, -1), colors.white),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("ROWHEIGHT",  (0, 0), (-1, -1), 22),
        ("ROUNDEDCORNERS", [4, 4, 4, 4]),
    ]))
    story.append(badge_table)
    story.append(Spacer(1, 8 * mm))

    # ── Detection stats table ──────────────────────────────────────────────────
    story.append(Paragraph("DETECTION SUMMARY", section_style))

    if is_video and total_frames:
        pct_frames = (len(all_detections) / total_frames * 100) if total_frames else 0
        media_info = f"Video — {total_frames} frames analysed; stenosis in {len(all_detections)} frames ({pct_frames:.1f}%)"
    else:
        media_info = "Single image"

    stats_rows = [
        ["Field", "Value"],
        ["Media type", media_info],
        ["Stenosis detected", data["stenosis_yn"]],
        ["Best detection coordinates", data["coords"]],
        ["Lumen diameter (best frame)", data["lumen_dia"]],
        ["AI confidence (best frame)", data["percentage"]],
        ["Severity class", severity_class],
    ]
    stats_table = Table(stats_rows, colWidths=[60 * mm, W - 60 * mm])
    stats_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#1E88E5")),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("BACKGROUND",   (0, 1), (-1, -1), colors.HexColor("#f5f5f5")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f5f5f5"), colors.white]),
        ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("ROWHEIGHT",    (0, 0), (-1, -1), 16),
    ]))
    story.append(stats_table)
    story.append(Spacer(1, 8 * mm))

    # ── Annotated image ────────────────────────────────────────────────────────
    story.append(Paragraph("ANNOTATED ANGIOGRAM", section_style))

    def bgr_to_rl_image(bgr_arr, max_width, max_height=None):
        rgb = cv2.cvtColor(bgr_arr, cv2.COLOR_BGR2RGB)
        pil_buf = io.BytesIO()
        from PIL import Image as PILImage
        PILImage.fromarray(rgb).save(pil_buf, format="PNG")
        pil_buf.seek(0)
        img = RLImage(pil_buf)
        scale = max_width / img.drawWidth
        if max_height:
            scale = min(scale, max_height / img.drawHeight)
        img.drawWidth  *= scale
        img.drawHeight *= scale
        return img

    story.append(bgr_to_rl_image(annotated_bgr, W, max_height=90 * mm))
    story.append(Paragraph(
        "Figure 1: Best-confidence frame with AI-detected stenosis bounding box (golden yellow).",
        caption_style,
    ))

    # ── Additional frames for video ────────────────────────────────────────────
    if is_video and top_frames_bgr:
        story.append(Paragraph("KEY DETECTION FRAMES", section_style))
        n = len(top_frames_bgr)
        cell_w = (W - (n - 1) * 4 * mm) / n
        img_row = [[bgr_to_rl_image(f, cell_w, max_height=55 * mm) for f in top_frames_bgr]]
        cap_row = [[Paragraph(f"Frame {i+1}", caption_style) for i in range(n)]]
        frame_table = Table(img_row + cap_row, colWidths=[cell_w] * n)
        frame_table.setStyle(TableStyle([
            ("ALIGN",  (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(frame_table)
        story.append(Spacer(1, 4 * mm))

    # ── Impression & recommendation quick-read ─────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc"),
                             spaceAfter=6))
    story.append(Paragraph("IMPRESSION (PRE-COMPUTED)", section_style))
    story.append(Paragraph(impression, body_style))
    story.append(Paragraph("RECOMMENDATION (PRE-COMPUTED)", section_style))
    story.append(Paragraph(recommendation, body_style))

    # ── Full Gemini report ─────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1E88E5"),
                             spaceAfter=6))
    story.append(Paragraph("AI CLINICAL REPORT (GEMINI)", section_style))

    for raw_line in report_text.split("\n"):
        line = raw_line.strip()
        if not line:
            story.append(Spacer(1, 3 * mm))
            continue
        # Section headings inside the report (ALL CAPS lines)
        if line.isupper() or (len(line) < 60 and line.endswith(":")):
            story.append(Paragraph(f"<b>{line}</b>", body_style))
        else:
            story.append(Paragraph(line, body_style))

    # ── Footer disclaimer ──────────────────────────────────────────────────────
    story.append(Spacer(1, 8 * mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey, spaceAfter=4))
    story.append(Paragraph(
        "<i>This report is generated by an AI-assisted system and is intended for research and "
        "educational purposes only. It does not constitute a medical diagnosis. All findings must "
        "be reviewed and validated by a qualified medical professional before any clinical decision "
        "is made.</i>",
        ParagraphStyle("Disclaimer", parent=styles["Normal"], fontSize=8,
                        textColor=colors.grey, alignment=TA_CENTER),
    ))

    doc.build(story)
    return buf.getvalue()

# ── Main App ───────────────────────────────────────────────────────────────────
with st.spinner("Loading YOLO model..."):
    yolo_model = load_yolo()

uploaded_file = st.file_uploader(
    "Upload Angiogram Video or Image",
    type=["mp4", "avi", "mov", "bmp", "jpg", "png"]
)

if uploaded_file is not None:
    if not api_key:
        st.warning("⚠️ Please enter your Gemini API key in the sidebar to generate the report and PDF.")

    is_video = uploaded_file.name.lower().endswith(('.mp4', '.avi', '.mov'))

    with st.spinner("Analysing Angiogram with YOLO..."):
        if not is_video:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
            annotated_img, data = process_single_frame(frame, yolo_model)

            all_detections = [data] if data["stenosis_yn"] == "Y" else []
            total_frames = 1
            top_frames_bgr = []
            final_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            annotated_bgr = annotated_img

        else:
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
            detection_frames = []   # (conf, annotated_bgr)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                annotated_img, data = process_single_frame(frame, yolo_model)
                out.write(annotated_img)
                total_frames += 1

                if data["stenosis_yn"] == "Y":
                    all_detections.append(data)
                    detection_frames.append((data["raw_conf"], annotated_img.copy()))

                if data["raw_conf"] > best_data["raw_conf"]:
                    best_data = data
                    best_annotated_img = annotated_img

            cap.release()
            out.release()

            if best_annotated_img is None:
                best_annotated_img = annotated_img

            # Pick up to 4 most-confident frames for the PDF (excluding the best already shown)
            detection_frames.sort(key=lambda x: x[0], reverse=True)
            top_frames_bgr = [f for _, f in detection_frames[1:5]]

            final_img = cv2.cvtColor(best_annotated_img, cv2.COLOR_BGR2RGB)
            data = best_data
            annotated_bgr = best_annotated_img

    # --- Display Layout ---
    col1, col2 = st.columns([0.5, 0.5])

    # ── Generate report & PDF bytes first (needed by both columns) ─────────────
    report_text = None
    severity_class, max_conf_pct, impression, recommendation = classify_severity(all_detections)

    if api_key:
        with st.spinner("Generating detailed report..."):
            try:
                report_text = generate_report_gemini(
                    api_key,
                    uploaded_file.name,
                    all_detections,
                    total_frames=total_frames,
                    is_video=is_video,
                )
            except Exception as e:
                st.error(f"❌ Gemini API error: {e}")
                st.info("Check your API key in the sidebar or visit https://aistudio.google.com/app/apikey")

    pdf_bytes = None
    if report_text:
        with st.spinner("Building PDF..."):
            pdf_bytes = build_pdf(
                file_name=uploaded_file.name,
                is_video=is_video,
                data=data,
                all_detections=all_detections,
                total_frames=total_frames,
                severity_class=severity_class,
                max_conf_pct=max_conf_pct,
                impression=impression,
                recommendation=recommendation,
                report_text=report_text,
                annotated_bgr=annotated_bgr,
                top_frames_bgr=top_frames_bgr if is_video else [],
            )

    with col1:
        st.markdown('<div class="section-label">Annotated Frame</div>', unsafe_allow_html=True)
        st.image(final_img, caption="Best-confidence frame · Bounding box = detected stenosis region", use_container_width=True)

        # ── Download buttons below the image ──────────────────────────────────
        st.markdown('<div class="section-label">Downloads</div>', unsafe_allow_html=True)

        if pdf_bytes:
            pdf_name = f"Stenosis_Report_{os.path.splitext(uploaded_file.name)[0]}.pdf"
            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name=pdf_name,
                mime="application/pdf",
            )
        elif not api_key:
            st.info("Enter your Gemini API key to enable PDF download.")
        else:
            st.warning("PDF will be available once the Gemini report is generated.")

        if is_video:
            with open(out_video_file.name, "rb") as vf:
                st.download_button(
                    label="⬇️ Download Processed Video",
                    data=vf,
                    file_name=f"AI_Analyzed_{uploaded_file.name}",
                    mime="video/mp4",
                )

    with col2:
        st.markdown('<div class="section-label">Detection Results</div>', unsafe_allow_html=True)

        sev_icon = {"None": "✅", "Mild": "🟡", "Moderate": "🟠", "Severe": "🔴", "Critical": "🚨"}.get(
            classify_severity(all_detections)[0], "⬜")

        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon">🩺</div>
            <div><div class="stat-label">Stenosis Detected</div>
            <div class="stat-value">{data['stenosis_yn']}</div></div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">📍</div>
            <div><div class="stat-label">Coordinates</div>
            <div class="stat-value">{data['coords']}</div></div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">📏</div>
            <div><div class="stat-label">Lumen Diameter</div>
            <div class="stat-value">{data['lumen_dia']}</div></div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">{sev_icon}</div>
            <div><div class="stat-label">AI Confidence</div>
            <div class="stat-value">{data['percentage']}</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">AI Clinical Report</div>', unsafe_allow_html=True)

        if report_text:
            st.markdown(
                f"<div class='report-box'>{report_text}</div>",
                unsafe_allow_html=True
            )
        elif not api_key:
            st.info("Enter your Gemini API key in the sidebar to generate the report.")
