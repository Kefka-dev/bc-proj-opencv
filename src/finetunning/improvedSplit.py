import os
import shutil
import random

import re # Import regex module for checking directory names


# --- Configuration ---
# Define the PARENT directory containing all ch{n} folders
base_dataset_path = "../../dataset_v3" # CHANGE THIS to your main directory

# Define train-validation-test split ratios
train_ratio = 0.7
val_ratio = 0.20 # 20% validation
test_ratio = 0.10 # 10% test (Ensure ratios sum to 1.0)
# --- End Configuration ---

# Function to move image-label pairs

def move_files(files, src_img_folder, src_lbl_folder, dest_img_folder, dest_lbl_folder):

    for file in files:
        # Move image
        shutil.move(os.path.join(src_img_folder, file), os.path.join(dest_img_folder, file))
        # Find corresponding label file (replace extension)
        label_file = os.path.splitext(file)[0] + ".txt"
        if os.path.exists(os.path.join(src_lbl_folder, label_file)):
            shutil.move(os.path.join(src_lbl_folder, label_file), os.path.join(dest_lbl_folder, label_file))

#Splits the channel into 3 categories by train and val ratio,and the rest will be used for test data
def split_channel(channel_path, train_ratio, val_ratio):

    print("Splitting channel ", channel_path)
    images_path = os.path.join(channel_path, "images")
    labels_path = os.path.join(channel_path, "labels")
    split_dirs= {
        "train_images": os.path.join(images_path, "train"),
        "val_images": os.path.join(images_path, "val"),
        "test_images": os.path.join(images_path, "test"),
        "train_labels": os.path.join(labels_path, "train"),
        "val_labels": os.path.join(labels_path, "val"),
        "test_labels": os.path.join(labels_path, "test"),
    }


    # Create train/val/test folders if they don't exist
    for path in split_dirs.values():
        os.makedirs(path, exist_ok=True)

    # List all images (assuming they are .jpg, .png, or .jpeg)
    image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(image_files) # Shuffle for randomness


    train_size = int(len(image_files) * train_ratio)
    val_size = int(len(image_files) * val_ratio)


    # Split dataset
    train_files = image_files[:train_size]
    val_files = image_files[train_size:train_size + val_size]
    test_files = image_files[train_size + val_size:] # Remaining files go to test set

    move_files(train_files, images_path, labels_path, split_dirs["train_images"], split_dirs["train_labels"])
    move_files(val_files, images_path, labels_path, split_dirs["val_images"], split_dirs["val_labels"])
    move_files(test_files, images_path, labels_path, split_dirs["test_images"], split_dirs["test_labels"])



with os.scandir(base_dataset_path) as entries:
    for entry in entries:
        if entry.is_dir() and entry.name.startswith("ch"):
            channel_path = entry.path
            split_channel(channel_path, train_ratio, val_ratio)
            print("Channel ", entry.name, "Split complete.")