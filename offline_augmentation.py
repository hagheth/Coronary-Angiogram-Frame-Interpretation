import os
import cv2
import pandas as pd
from pathlib import Path
import numpy as np
import random

def augment_dataset(
    input_img_dir, 
    output_img_dir, 
    label_csv_paths
):
    """
    Reads preprocessed images and labels, applies offline augmentation with base cv2,
    and returns a new dataset with updated bounding boxes.
    """
    os.makedirs(output_img_dir, exist_ok=True)
    print(f"--- Starting Native CV2 Offline Data Augmentation ---")
    
    for csv_path in label_csv_paths:
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found. Skipping...")
            continue
            
        df = pd.read_csv(csv_path)
        is_test_set = "test" in os.path.basename(csv_path).lower()
        
        if is_test_set:
            print(f"\nINFO: Detected a TEST set ({os.path.basename(csv_path)}).")
            print("      Test data should NEVER be augmented so evaluation is accurate.")
            print("      We are just transferring the original images into the new folder.")
        else:
            print(f"\nAugmenting training set: {os.path.basename(csv_path)}...")
            
        new_records = []
        processed = 0
        
        for index, row in df.iterrows():
            filename = row['filename']
            img_path = os.path.join(input_img_dir, filename)
            
            if not os.path.exists(img_path):
                continue
                
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            
            xmin, ymin = row['xmin'], row['ymin']
            xmax, ymax = row['xmax'], row['ymax']
            width, height = row['width'], row['height']
            
            # --- 1. ALWAYS ADD ORIGINAL ---
            orig_filename = f"orig_{filename}"
            cv2.imwrite(os.path.join(output_img_dir, orig_filename), img)
            
            orig_row = row.copy()
            orig_row['filename'] = orig_filename
            new_records.append(orig_row)
            
            if not is_test_set:
                # --- 2. AUGMENT 1: Horizontal Flip ---
                aug1_img = cv2.flip(img, 1) # 1 = horizontal
                aug1_xmin = width - xmax
                aug1_xmax = width - xmin
                
                aug1_name = f"aug1_hflip_{filename}"
                cv2.imwrite(os.path.join(output_img_dir, aug1_name), aug1_img)
                aug1_row = row.copy()
                aug1_row['filename'] = aug1_name
                aug1_row['xmin'], aug1_row['xmax'] = aug1_xmin, aug1_xmax
                new_records.append(aug1_row)
                
                # --- 3. AUGMENT 2: Vertical Flip ---
                aug2_img = cv2.flip(img, 0) # 0 = vertical
                aug2_ymin = height - ymax
                aug2_ymax = height - ymin
                
                aug2_name = f"aug2_vflip_{filename}"
                cv2.imwrite(os.path.join(output_img_dir, aug2_name), aug2_img)
                aug2_row = row.copy()
                aug2_row['filename'] = aug2_name
                aug2_row['ymin'], aug2_row['ymax'] = aug2_ymin, aug2_ymax
                new_records.append(aug2_row)
                
                # --- 4. AUGMENT 3: Brightness / Contrast Jitter (BBox unchanged) ---
                alpha = random.uniform(0.8, 1.2) # Contrast control
                beta = random.randint(-15, 15)  # Brightness control
                aug3_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                
                aug3_name = f"aug3_brt_{filename}"
                cv2.imwrite(os.path.join(output_img_dir, aug3_name), aug3_img)
                aug3_row = row.copy()
                aug3_row['filename'] = aug3_name
                new_records.append(aug3_row)

            processed += 1
            if processed % 100 == 0:
                print(f"Processed {processed} original images -> Created {len(new_records)} augmented images...")
                
        # Save new augmented CSV
        new_df = pd.DataFrame(new_records)
        path_obj = Path(csv_path)
        new_csv_filename = f"{path_obj.stem}_augmented{path_obj.suffix}"
        new_csv_path = os.path.join(BASE_DIR, new_csv_filename)
        new_df.to_csv(new_csv_path, index=False)
        print(f"Finished {csv_path}: Original records={len(df)} -> New records={len(new_df)}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_DATASET_DIR = os.path.join(BASE_DIR, "dataset_preprocessed") 
    OUTPUT_DATASET_DIR = os.path.join(BASE_DIR, "dataset_augmented")   
    
    LABEL_FILES = [
        os.path.join(BASE_DIR, "train_labels_preprocessed.csv"),
        os.path.join(BASE_DIR, "test_labels_preprocessed.csv")
    ]
    
    # Run simple CV2 based offline augmentation to avoid Windows C++ compiler pip errors
    augment_dataset(
        input_img_dir=INPUT_DATASET_DIR,
        output_img_dir=OUTPUT_DATASET_DIR,
        label_csv_paths=LABEL_FILES
    )
    print("\nProcess Complete! Results are inside 'dataset_augmented'.")
