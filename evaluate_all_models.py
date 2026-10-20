from ultralytics import YOLO
from pathlib import Path

DATA_YAML = "stenosis_yolo_dataset/data.yaml"

MODELS = {
    "YOLOv8n":  "runs/detect/stenosis_detector/weights/best.pt",
    "YOLO11n":  "runs/detect/stenosis_detector_yolo11-2/weights/best.pt",
    "YOLOv10n": "runs/detect/stenosis_detector_yolov10/weights/best.pt",
}

results_summary = []

for model_name, weights_path in MODELS.items():
    if not Path(weights_path).exists():
        print(f"[SKIP] {model_name} — weights not found: {weights_path}")
        continue

    print(f"\n{'='*55}")
    print(f"  Evaluating {model_name} on TEST SET")
    print(f"{'='*55}")

    model = YOLO(weights_path)
    metrics = model.val(data=DATA_YAML, split="test", imgsz=512, batch=16, device=0)

    precision = metrics.box.mp
    recall    = metrics.box.mr
    map50     = metrics.box.map50
    map50_95  = metrics.box.map

    results_summary.append({
        "Model":        model_name,
        "Precision":    precision,
        "Recall":       recall,
        "mAP@0.5":      map50,
        "mAP@0.5:0.95": map50_95,
    })

    print(f"\n  {model_name} Test Results:")
    print(f"    Precision    : {precision*100:.2f}%")
    print(f"    Recall       : {recall*100:.2f}%")
    print(f"    mAP@0.5      : {map50*100:.2f}%")
    print(f"    mAP@0.5:0.95 : {map50_95*100:.2f}%")

# Final summary table
print(f"\n{'='*55}")
print(f"  FINAL COMPARISON — TEST SET (80/10/10 Split)")
print(f"{'='*55}")
print(f"{'Model':<12} {'Precision':>10} {'Recall':>10} {'mAP@0.5':>10} {'mAP@0.5:0.95':>14}")
print("-" * 58)
for r in results_summary:
    print(f"{r['Model']:<12} {r['Precision']*100:>9.2f}% {r['Recall']*100:>9.2f}% {r['mAP@0.5']*100:>9.2f}% {r['mAP@0.5:0.95']*100:>13.2f}%")
print(f"{'='*55}")
