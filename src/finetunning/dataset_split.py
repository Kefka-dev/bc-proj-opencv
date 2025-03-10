import os
import shutil
import random

# Define dataset paths
dataset_path = "E:\\Skola\\FEI\\bakalarka\\bc-proj-opencv\\dataset2"
images_path = os.path.join(dataset_path, "images")
labels_path = os.path.join(dataset_path, "labels")

# Define output directories
output_dirs = {
    "train_images": os.path.join(images_path, "train"),
    "val_images": os.path.join(images_path, "val"),
    "test_images": os.path.join(images_path, "test"),
    "train_labels": os.path.join(labels_path, "train"),
    "val_labels": os.path.join(labels_path, "val"),
    "test_labels": os.path.join(labels_path, "test"),
}

# Create train/val/test folders if they don't exist
for path in output_dirs.values():
    os.makedirs(path, exist_ok=True)

# List all images (assuming they are .jpg, .png, or .jpeg)
image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
random.shuffle(image_files)  # Shuffle for randomness

# Define train-validation-test split ratios
train_ratio = 0.7
val_ratio = 0.20  # 20% validation
test_ratio = 0.10  # 10% test

train_size = int(len(image_files) * train_ratio)
val_size = int(len(image_files) * val_ratio)

# Split dataset
train_files = image_files[:train_size]
val_files = image_files[train_size:train_size + val_size]
test_files = image_files[train_size + val_size:]  # Remaining files go to test set

# Function to move image-label pairs
def move_files(files, src_img_folder, src_lbl_folder, dest_img_folder, dest_lbl_folder):
    for file in files:
        # Move image
        shutil.move(os.path.join(src_img_folder, file), os.path.join(dest_img_folder, file))

        # Find corresponding label file (replace extension)
        label_file = os.path.splitext(file)[0] + ".txt"
        if os.path.exists(os.path.join(src_lbl_folder, label_file)):
            shutil.move(os.path.join(src_lbl_folder, label_file), os.path.join(dest_lbl_folder, label_file))

# Move datasets
move_files(train_files, images_path, labels_path, output_dirs["train_images"], output_dirs["train_labels"])
move_files(val_files, images_path, labels_path, output_dirs["val_images"], output_dirs["val_labels"])
move_files(test_files, images_path, labels_path, output_dirs["test_images"], output_dirs["test_labels"])

print(f"Dataset split completed!")
print(f"Train: {len(train_files)} images")
print(f"Validation: {len(val_files)} images")
print(f"Test: {len(test_files)} images")
