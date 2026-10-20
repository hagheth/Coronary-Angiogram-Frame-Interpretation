import os
import cv2
import pandas as pd
from pathlib import Path
import numpy as np

def apply_clahe(image):
    """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def preprocess_images_and_labels(
    input_img_dir, 
    output_img_dir, 
    label_csv_paths, 
    target_size=(512, 512)
):
    """
    Preprocesses the dataset by converting to grayscale, applying CLAHE, resizing, 
    and also scales the bounding box annotations to match the new image sizes.
    """
    os.makedirs(output_img_dir, exist_ok=True)
    
    # Keep track of the scaling factors for each image to adjust the bounding boxes
    scale_factors = {}
    
    print(f"--- Starting Image Preprocessing ---")
    image_files = [f for f in os.listdir(input_img_dir) if f.endswith(('.bmp', '.png', '.jpg'))]
    processed_count = 0
    
    for filename in image_files:
        img_path = os.path.join(input_img_dir, filename)
        
        # 1. Read the image in Grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Failed to read {filename}. Skipping...")
            continue
            
        original_height, original_width = img.shape
        
        # Calculate scale factors for bounding box adjustment
        scale_x = target_size[0] / original_width
        scale_y = target_size[1] / original_height
        scale_factors[filename] = (scale_x, scale_y)
        
        # 2. Apply CLAHE (Contrast Enhancement)
        img_clahe = apply_clahe(img)
        
        # 3. Optional Noise Reduction (Mild Gaussian Blur)
        # img_blur = cv2.GaussianBlur(img_clahe, (3, 3), 0)
        
        # 4. Resize the image
        img_resized = cv2.resize(img_clahe, target_size, interpolation=cv2.INTER_AREA)
        
        # Save the preprocessed image
        out_path = os.path.join(output_img_dir, filename)
        cv2.imwrite(out_path, img_resized)
        processed_count += 1
        
        if processed_count % 100 == 0:
            print(f"Processed {processed_count}/{len(image_files)} images...")
            
    print(f"Successfully preprocessed {processed_count} images.")
    
    print(f"\n--- Adjusting Label Coordinates ---")
    for csv_path in label_csv_paths:
        if not os.path.exists(csv_path):
            print(f"Warning: Cannot find {csv_path}. Skipping...")
            continue
            
        df = pd.read_csv(csv_path)
        
        # Assuming format: filename, width, height, class, xmin, ymin, xmax, ymax
        new_df = df.copy()
        
        for index, row in new_df.iterrows():
            filename = row['filename']
            if filename in scale_factors:
                scale_x, scale_y = scale_factors[filename]
                
                # Update dimensions
                new_df.at[index, 'width'] = target_size[0]
                new_df.at[index, 'height'] = target_size[1]
                
                # Scale bounding boxes
                # Make sure to keep them as integers
                new_df.at[index, 'xmin'] = int(row['xmin'] * scale_x)
                new_df.at[index, 'ymin'] = int(row['ymin'] * scale_y)
                new_df.at[index, 'xmax'] = int(row['xmax'] * scale_x)
                new_df.at[index, 'ymax'] = int(row['ymax'] * scale_y)
                
        # Save the new CSV file
        path_obj = Path(csv_path)
        new_csv_filename = f"{path_obj.stem}_preprocessed{path_obj.suffix}"
        new_csv_path = os.path.join(path_obj.parent, new_csv_filename)
        new_df.to_csv(new_csv_path, index=False)
        print(f"Saved preprocessed labels to {new_csv_filename}")

if __name__ == "__main__":
    # Define your paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_DATASET_DIR = os.path.join(BASE_DIR, "dataset")
    OUTPUT_DATASET_DIR = os.path.join(BASE_DIR, "dataset_preprocessed")
    
    LABEL_FILES = [
        os.path.join(BASE_DIR, "train_labels.csv"),
        os.path.join(BASE_DIR, "test_labels.csv")
    ]
    
    # Run the preprocessing
    preprocess_images_and_labels(
        input_img_dir=INPUT_DATASET_DIR,
        output_img_dir=OUTPUT_DATASET_DIR,
        label_csv_paths=LABEL_FILES,
        target_size=(512, 512) # Uniform size expected by PyTorch CNNs
    )
    print("\nPreprocessing complete! Data is ready for PyTorch.")
