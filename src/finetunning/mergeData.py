import os
import shutil

# --- Configuration ---
# Define the PARENT directory containing all ch{n} folders (same as before)
base_dataset_path = "../../dataset2"  # CHANGE THIS to your main directory

# Define the directory where the merged dataset will be created
merged_dataset_path = "../../merged_dataset"  # CHANGE THIS if you want a different name

# Define the names of the train, val, and test subdirectories
split_names = ["train", "val", "test"]
image_folder_name = "images"
label_folder_name = "labels"
# --- End Configuration ---

# Create the main merged dataset directory if it doesn't exist
os.makedirs(merged_dataset_path, exist_ok=True)

for split in split_names:
    # Create train/val/test subdirectories in the merged dataset
    os.makedirs(os.path.join(merged_dataset_path, split, image_folder_name), exist_ok=True)
    os.makedirs(os.path.join(merged_dataset_path, split, label_folder_name), exist_ok=True)

with os.scandir(base_dataset_path) as entries:
    for entry in entries:
        if entry.is_dir() and entry.name.startswith("ch"):
            channel_path = entry.path
            print(f"Processing channel: {entry.name}")

            for split in split_names:
                src_img_path = os.path.join(channel_path, image_folder_name, split)
                src_lbl_path = os.path.join(channel_path, label_folder_name, split)
                dest_img_path = os.path.join(merged_dataset_path, split, image_folder_name)
                dest_lbl_path = os.path.join(merged_dataset_path, split, label_folder_name)

                if os.path.exists(src_img_path):
                    for filename in os.listdir(src_img_path):
                        src_file = os.path.join(src_img_path, filename)
                        dest_file = os.path.join(dest_img_path, filename)
                        shutil.move(src_file, dest_file)
                        print(f"Moved image: {filename} from {entry.name}/{split} to merged/{split}")

                if os.path.exists(src_lbl_path):
                    for filename in os.listdir(src_lbl_path):
                        src_file = os.path.join(src_lbl_path, filename)
                        dest_file = os.path.join(dest_lbl_path, filename)
                        shutil.move(src_file, dest_file)
                        print(f"Moved label: {filename} from {entry.name}/{split} to merged/{split}")

print("Merging of channels complete.")