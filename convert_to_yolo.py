import os
import shutil
import pandas as pd
from pathlib import Path

def convert_bbox_to_yolo(xmin, ymin, xmax, ymax, img_width, img_height):
    """
    Converts standard pixel bounds to YOLOv8 normalized format:
    x_center, y_center, width, height (all normalized 0.0 to 1.0)
    """
    x_center = ((xmin + xmax) / 2.0) / img_width
    y_center = ((ymin + ymax) / 2.0) / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    
    # Clip limits to 0.0-1.0 to prevent bounding boxes jumping slightly out of frame
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    
    return x_center, y_center, width, height

def process_split(csv_path, source_img_dir, split_name, output_root):
    img_out_dir = os.path.join(output_root, 'images', split_name)
    label_out_dir = os.path.join(output_root, 'labels', split_name)
    
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(label_out_dir, exist_ok=True)
    
    if not os.path.exists(csv_path):
        print(f"Cannot find {csv_path}. Skipping {split_name}.")
        return
        
    df = pd.read_csv(csv_path)
    # Using groupby allows us to handle images that might have MULTIPLE stenoses
    grouped = df.groupby('filename')
    
    print(f"Processing {split_name} split. Total unique images: {len(grouped)}")
    
    processed_count = 0
    for filename, group in grouped:
        src_img_path = os.path.join(source_img_dir, filename)
        
        if not os.path.exists(src_img_path):
            continue
            
        # 1. Copy Image to the strict YOLO directory
        dst_img_path = os.path.join(img_out_dir, filename)
        if not os.path.exists(dst_img_path): 
            shutil.copy2(src_img_path, dst_img_path)
        
        # 2. Write YOLO .txt label file
        base_name = os.path.splitext(filename)[0]
        txt_path = os.path.join(label_out_dir, f"{base_name}.txt")
        
        with open(txt_path, 'w') as f:
            for _, row in group.iterrows():
                # We only have one class: Stenosis -> ID: 0
                class_id = 0 
                
                img_width = float(row['width'])
                img_height = float(row['height'])
                
                x_center, y_center, w, h = convert_bbox_to_yolo(
                    row['xmin'], row['ymin'], row['xmax'], row['ymax'],
                    img_width, img_height
                )
                
                # Write standard YOLO format line
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
                
        processed_count += 1
        if processed_count % 1000 == 0:
            print(f"Converted {processed_count} files for {split_name}...")

    print(f"Finished {split_name} split! Successfully formatted {processed_count} images.")

def generate_yaml(output_root):
    yaml_path = os.path.join(output_root, 'data.yaml')
    
    # Absolute paths eliminate pathing errors in YOLO
    abs_root = os.path.abspath(output_root)
    
    # We map backslashes to forward slashes strictly for python/yaml compatibility
    abs_root = abs_root.replace('\\', '/')
    
    yaml_content = f"""path: {abs_root}
train: images/train
val: images/val

# Classes
nc: 1
names: ['Stenosis']
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"\nCreated YOLOv8 configuration file at: {yaml_path}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SOURCE_IMG_DIR = os.path.join(BASE_DIR, "dataset_augmented")
    OUTPUT_ROOT = os.path.join(BASE_DIR, "stenosis_yolo_dataset")
    
    TRAIN_CSV = os.path.join(BASE_DIR, "train_labels_preprocessed_augmented.csv")
    TEST_CSV = os.path.join(BASE_DIR, "test_labels_preprocessed_augmented.csv")
    
    print("--- Starting YOLOv8 Format Conversion ---")
    
    # Convert and format Train Data
    process_split(TRAIN_CSV, SOURCE_IMG_DIR, "train", OUTPUT_ROOT)
    
    # Convert and format Validation (Test) Data
    # Note: YOLO calls evaluation tests "val"
    process_split(TEST_CSV, SOURCE_IMG_DIR, "val", OUTPUT_ROOT)
    
    # Create required training yaml
    generate_yaml(OUTPUT_ROOT)
    print("--- YOLOv8 Dataset Preparation Complete ---")
