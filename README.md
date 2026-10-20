# Coronary Angiogram Frame Interpretation using Vision-Based Model

> Final Year Project — Universiti Teknologi MARA (UiTM)
> 
> **Author:** Harith Ikhwan bin Suhaimi | 2023226656
> **Supervisor:** Farah Syazwani Mohamed Rashid
> **Email:** harithikhwan050604@gmail.com

---

## Overview

An AI-powered prototype system for automated detection and reporting of coronary artery stenosis from angiogram videos and images. The system integrates **YOLOv11n** (vision-based stenosis detection) with **Google Gemini 2.0 Flash** (clinical report generation) into a single end-to-end pipeline, accessible through a **Streamlit** web interface.

This project directly addresses the limitations of manual angiogram interpretation — inter-observer variability, diagnostic fatigue, and lack of explainability — by providing cardiologists with a fast, automated, and clinician-readable AI-assisted decision support tool.

---

## Demo

| Step | Description |
|---|---|
| 1 | Upload a coronary angiogram video (MP4/AVI/MOV) or image (BMP/JPG/PNG) |
| 2 | System preprocesses each frame (Grayscale → CLAHE → Resize 512×512) |
| 3 | YOLOv11n detects stenotic regions with bounding box and confidence score |
| 4 | Severity is automatically classified (None / Mild / Moderate / Severe / Critical) |
| 5 | Gemini LLM generates a structured 6-section clinical report |
| 6 | Download the PDF report and annotated video |

---

## Features

- **End-to-end automation** — raw angiogram input to structured clinical PDF report
- **Real-time video processing** — frame-by-frame detection with best frame selection
- **Severity classification** — 5 levels based on AI confidence score
- **Explainable AI** — 6-section clinical report addressing the "black box" problem
- **PDF export** — professional report with annotated frame, detection stats, and Gemini report
- **Annotated video export** — processed MP4 with bounding boxes on all frames
- **Streamlit web interface** — no installation required for end users

---

## Model Performance

| Model | Precision | Recall | F1-Score | mAP@0.5 |
|---|---|---|---|---|
| YOLOv8n | 98.08% | 97.01% | 97.54% | 98.52% |
| YOLOv10n | 97.35% | 96.63% | 96.99% | 98.74% |
| **YOLOv11n** ✅ | **98.18%** | **97.70%** | **97.94%** | **99.07%** |

> YOLOv11n was selected as the best model based on highest performance across all metrics, particularly **97.70% recall** — critical for medical screening to minimise missed stenosis cases.

---

## Severity Classification

| Level | Confidence Threshold | Clinical Implication |
|---|---|---|
| None | No detection | No stenosis detected |
| Mild | < 50% | Non-obstructive CAD — monitor |
| Moderate | 50 – 70% | Cardiology consultation advised |
| Severe | 70 – 90% | Urgent cardiology review required |
| Critical | ≥ 90% | Immediate intervention required |

---

## Project Structure

```
├── app_gemini_pdf.py          # Main application (YOLO + Gemini + PDF)
├── app.py                     # Base app version
├── app_gemini.py              # App with Gemini only
├── app_gemini_vision.py       # App with vision input to Gemini
├── train_yolo.py              # YOLOv8n training script
├── train_yolo11.py            # YOLOv11n training script
├── train_yolov10.py           # YOLOv10n training script
├── train_all_models.py        # Train all models sequentially
├── evaluate_all_models.py     # Evaluate and compare all models
├── preprocess_dataset.py      # Grayscale + CLAHE + resize pipeline
├── offline_augmentation.py    # Horizontal/vertical flip + jitter
├── split_dataset.py           # 80/10/10 train/val/test split
├── convert_to_yolo.py         # Convert dataset to YOLO format
├── process_video.py           # Frame extraction and video rebuild
├── post_process.py            # Post-processing utilities
├── generate_report.py         # Standalone report generation
└── runs/                      # Training outputs (excluded from Git)
    └── detect/
        └── stenosis_v2_yolo11n/
            └── weights/
                └── best.pt    # Best model weights
```

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/harithikhwan050604-ai/Coronary-Angiogram-Frame-Interpretation.git
cd Coronary-Angiogram-Frame-Interpretation
```

### 2. Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download model weights
Download `best.pt` from the [Releases](https://github.com/harithikhwan050604-ai/Coronary-Angiogram-Frame-Interpretation/releases) page and place it at:
```
runs/detect/stenosis_v2_yolo11n/weights/best.pt
```

### 5. Get a Gemini API key
Get a free API key at [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

---

## Running the App

```bash
streamlit run app_gemini_pdf.py
```

Then open your browser at `http://localhost:8501`, paste your Gemini API key in the sidebar, and upload an angiogram video or image.

---

## Dataset

- **Source:** [Angiographic Dataset for Stenosis Detection — Mendeley Data](https://data.mendeley.com/datasets/ydrm75xywg/1)
- **Size:** 8,325 grayscale angiographic images from 100 patients
- **Split:** 80% Train / 10% Validation / 10% Test
- **Augmentation:** Horizontal flip, vertical flip, brightness/contrast jitter

> The dataset is not included in this repository. Download it directly from Mendeley Data.

---

## Dependencies

Key libraries used:

```
streamlit
ultralytics
opencv-python
google-genai
reportlab
Pillow
numpy
torch
```

Install all via:
```bash
pip install -r requirements.txt
```

---

## Clinical Report Structure

The Gemini-generated report follows standard radiology reporting format:

1. **Examination** — procedure type and file reference
2. **Clinical Indication** — reason for the study
3. **Technique** — imaging method and AI preprocessing steps
4. **Findings** — detailed description of detected stenosis
5. **Impression** — severity summary and clinical significance
6. **Recommendation** — suggested clinical action

> ⚠️ **Disclaimer:** This system is an AI-assisted research prototype intended for educational and research purposes only. All findings must be reviewed and validated by a qualified medical professional before any clinical decision is made.

---

## Research Significance

- **SDG 3 — Good Health and Well-Being:** Improves efficiency and accuracy of CAD diagnosis
- **SDG 4 — Quality Education:** Assists medical students in learning angiographic interpretation through structured AI-generated reports

---

## Future Works

- Multi-vessel localisation (LAD, RCA, LCX identification)
- Quantitative Coronary Angiography (QCA) lumen diameter measurement
- HL7 FHIR integration with hospital EHR systems
- Multi-centre dataset training for improved generalisation

---

## License

This project is developed for academic purposes as part of a Final Year Project at Universiti Teknologi MARA (UiTM). All rights reserved.

---

## Acknowledgements

- Dataset: Mendeley Data — *Angiographic Dataset for Stenosis Detection*
- Detection model: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- Report generation: [Google Gemini API](https://aistudio.google.com)
- UI framework: [Streamlit](https://streamlit.io)
