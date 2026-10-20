from ultralytics import YOLO
import os

if __name__ == '__main__':
    # 1. Load a pre-trained YOLOv8 PyTorch model
    # 'yolov8n.pt' is the 'nano' model. It is the fastest, lightest version to train 
    # to see if your data works. Once it works, you can upgrade to 'yolov8m.pt' (medium) for accuracy.
    model = YOLO('yolov8n.pt') 
    
    # 2. Grab the path to the YAML configuration file we just created
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(BASE_DIR, 'stenosis_yolo_dataset', 'data.yaml')
    
    print(f"Starting Training using {yaml_path}")
    
    # 3. Start the PyTorch Training Loop!
    # - epochs=50: A good starting point to see loss convergence.
    # - imgsz=512: The size we preprocessed our images to.
    # - batch=16: Standard batch size for 8-12GB GPU VRAM. Lower it if you get Out Of Memory errors.
    # - name: This is what the output folder will be named.
    results = model.train(
        data=yaml_path, 
        epochs=50, 
        imgsz=512, 
        batch=16, 
        name='stenosis_v2_yolov8n',
        device=0 # Uses the primary GPU. Change to 'cpu' if you don't have a GPU.
    )
    
    print("\n--- Training Finished! ---")
    print("Check the 'runs/detect/stenosis_detector' folder for your model weights and validation charts.")
