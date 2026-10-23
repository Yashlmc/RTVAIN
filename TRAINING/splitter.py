import os, random, shutil
from pathlib import Path

base = Path("dataset")
images_dir = base / "images"
labels_dir = base / "labels"

# Create the split directories
for split in ["train", "val"]:
    os.makedirs(base / "images" / split, exist_ok=True)
    os.makedirs(base / "labels" / split, exist_ok=True)

# Collect all images
images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(images)

split_index = int(0.90 * len(images))  # 90/10 split
train_images = images[:split_index]
val_images = images[split_index:]

def move_files(file_list, split_name):
    for img_file in file_list:
        label_file = os.path.splitext(img_file)[0] + ".txt"
        if os.path.exists(labels_dir / label_file):
            shutil.move(str(images_dir / img_file), str(base / "images" / split_name / img_file))
            shutil.move(str(labels_dir / label_file), str(base / "labels" / split_name / label_file))

move_files(train_images, "train")
move_files(val_images, "val")

print("✅ Dataset split into train and val.")
