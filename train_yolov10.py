from ultralytics import YOLO
import os

if __name__ == '__main__':
    # 1. Load a pre-trained YOLOv10 PyTorch model
    # 'yolov10n.pt' is the 'nano' model. It is very fast and efficient.
    # You can also upgrade to 'yolov10m.pt' (medium) or 'yolov10x.pt' for accuracy.
    model = YOLO('yolov10n.pt') 
    
    # 2. Grab the path to the YAML configuration file
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(BASE_DIR, 'stenosis_yolo_dataset', 'data.yaml')
    
    print(f"Starting Training (YOLOv10) using {yaml_path}")
    
    # 3. Start the PyTorch Training Loop!
    # Using the same parameters as YOLOv8:
    # - epochs=50
    # - imgsz=512
    # - batch=16
    # - name: Output folder name.
    results = model.train(
        data=yaml_path, 
        epochs=50, 
        imgsz=512, 
        batch=16, 
        name='stenosis_v2_yolov10n',
        device=0 # Uses the primary GPU. Change to 'cpu' if you don't have a GPU.
    )
    
    print("\n--- Training Finished! ---")
    print("Check the 'runs/detect/stenosis_detector_yolov10' folder for your model weights and validation charts.")
