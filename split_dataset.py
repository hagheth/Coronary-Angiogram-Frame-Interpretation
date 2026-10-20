import os
import shutil
import random
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
DATASET_DIR = Path(r"C:\Users\Usera\Desktop\FYP\Stenosis detection\stenosis_yolo_dataset")
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10
SEED        = 42
# ────────────────────────────────────────────────────────────────────────────

random.seed(SEED)

img_dir = DATASET_DIR / "images"
lbl_dir = DATASET_DIR / "labels"

# ── Backup original dataset ──────────────────────────────────────────────────
BACKUP_DIR = DATASET_DIR.parent / "stenosis_yolo_dataset_BACKUP"
if BACKUP_DIR.exists():
    print(f"Backup already exists at: {BACKUP_DIR}")
    print("Skipping backup to avoid overwriting previous backup.")
else:
    print(f"Creating backup at: {BACKUP_DIR} ...")
    shutil.copytree(str(DATASET_DIR), str(BACKUP_DIR))
    print("Backup complete.")
# ────────────────────────────────────────────────────────────────────────────

# Collect all images from train and val folders
all_images = []
for split in ["train", "val"]:
    folder = img_dir / split
    if folder.exists():
        all_images.extend(list(folder.glob("*.*")))

print(f"Total images found: {len(all_images)}")

# Shuffle
random.shuffle(all_images)

# Calculate split sizes
n_total = len(all_images)
n_train = int(n_total * TRAIN_RATIO)
n_val   = int(n_total * VAL_RATIO)
n_test  = n_total - n_train - n_val

train_imgs = all_images[:n_train]
val_imgs   = all_images[n_train:n_train + n_val]
test_imgs  = all_images[n_train + n_val:]

print(f"Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")

# Create new split folders (images + labels)
splits = {"train": train_imgs, "val": val_imgs, "test": test_imgs}

for split_name, img_list in splits.items():
    (img_dir / split_name).mkdir(parents=True, exist_ok=True)
    (lbl_dir / split_name).mkdir(parents=True, exist_ok=True)

# Copy files to new temp folders
TEMP_DIR = DATASET_DIR / "_temp_split"
for split_name, img_list in splits.items():
    temp_img = TEMP_DIR / "images" / split_name
    temp_lbl = TEMP_DIR / "labels" / split_name
    temp_img.mkdir(parents=True, exist_ok=True)
    temp_lbl.mkdir(parents=True, exist_ok=True)

    for img_path in img_list:
        # Copy image
        shutil.copy2(img_path, temp_img / img_path.name)

        # Copy corresponding label
        lbl_path = lbl_dir / img_path.parent.name / (img_path.stem + ".txt")
        if lbl_path.exists():
            shutil.copy2(lbl_path, temp_lbl / lbl_path.name)
        else:
            print(f"  [WARN] No label found for {img_path.name}")

print("Temp split created. Replacing original folders...")

# Remove old train/val folders
for split_name in ["train", "val"]:
    shutil.rmtree(img_dir / split_name, ignore_errors=True)
    shutil.rmtree(lbl_dir / split_name, ignore_errors=True)

# Move temp folders into place
for split_name in ["train", "val", "test"]:
    shutil.move(str(TEMP_DIR / "images" / split_name), str(img_dir / split_name))
    shutil.move(str(TEMP_DIR / "labels" / split_name), str(lbl_dir / split_name))

# Cleanup temp
shutil.rmtree(TEMP_DIR, ignore_errors=True)

# Remove old cache files
for cache in DATASET_DIR.glob("*.cache"):
    cache.unlink()
    print(f"Removed cache: {cache.name}")

# Update data.yaml
yaml_path = DATASET_DIR / "data.yaml"
yaml_content = f"""path: {DATASET_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

# Classes
nc: 1
names: ['Stenosis']
"""
yaml_path.write_text(yaml_content)
print("data.yaml updated with test split.")

# Final count
print("\n=== Final Dataset Split ===")
for split_name in ["train", "val", "test"]:
    count = len(list((img_dir / split_name).glob("*.*")))
    print(f"  {split_name:6s}: {count} images")
print("Done.")
