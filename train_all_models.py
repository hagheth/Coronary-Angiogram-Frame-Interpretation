from ultralytics import YOLO
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(BASE_DIR, 'stenosis_yolo_dataset', 'data.yaml')

MODELS = [
    {"weights": "yolov8n.pt",  "name": "stenosis_v2_yolov8n"},
    {"weights": "yolo11n.pt",  "name": "stenosis_v2_yolo11n"},
    {"weights": "yolov10n.pt", "name": "stenosis_v2_yolov10n"},
]

TRAIN_PARAMS = dict(
    data=yaml_path,
    epochs=50,
    imgsz=512,
    batch=16,
    device=0,
)

if __name__ == '__main__':
    print(f"Dataset: {yaml_path}")
    print(f"Split: 80% train / 10% val / 10% test")
    print(f"Training {len(MODELS)} models sequentially...\n")

    for m in MODELS:
        print(f"\n{'='*55}")
        print(f"  Training {m['name']} ...")
        print(f"{'='*55}")
        model = YOLO(m["weights"])
        model.train(**TRAIN_PARAMS, name=m["name"])
        print(f"\n  Done: runs/detect/{m['name']}/weights/best.pt")

    print(f"\n{'='*55}")
    print("  ALL MODELS TRAINED SUCCESSFULLY")
    print(f"  Results saved in runs/detect/")
    print(f"{'='*55}")
