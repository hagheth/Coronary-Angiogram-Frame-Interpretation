from ultralytics import YOLO
import os

if __name__ == '__main__':
    # 1. Load a pre-trained YOLO11 PyTorch model
    # 'yolo11n.pt' is the 'nano' model. It is the fastest, lightest version to train.
    # Once it works, you can upgrade to 'yolo11m.pt' (medium) or 'yolo11x.pt' for higher accuracy.
    model = YOLO('yolo11n.pt') 
    
    # 2. Grab the path to the YAML configuration file
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(BASE_DIR, 'stenosis_yolo_dataset', 'data.yaml')
    
    print(f"Starting Training (YOLO11) using {yaml_path}")
    
    # 3. Start the PyTorch Training Loop!
    # - epochs=50: A good starting point to see loss convergence.
    # - imgsz=512: The size we preprocessed our images to.
    # - batch=16: Standard batch size. Lower it if you get Out Of Memory errors.
    # - name: Output folder name.
    results = model.train(
        data=yaml_path, 
        epochs=50, 
        imgsz=512, 
        batch=16, 
        name='stenosis_v2_yolo11n',
        device=0 # Uses the primary GPU. Change to 'cpu' if you don't have a GPU.
    )
    
    print("\n--- Training Finished! ---")
    print("Check the 'runs/detect/stenosis_detector_yolo11' folder for your model weights and validation charts.")
