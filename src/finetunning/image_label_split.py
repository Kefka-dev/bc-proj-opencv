import os
import shutil

mixed_folder = "F:\\PC Backup\\ListerineLinux\\Documents\\bc-proj-opencv\\video_frames (Copy)"
labels = "E:\\Skola\\FEI\\bakalarka\\bc-proj-opencv\\dataset\\labels"
images = "E:\\Skola\\FEI\\bakalarka\\bc-proj-opencv\\dataset\\images"


def split_data_to_label_and_images(mixed_folder_path, labels_path, images_path):
    # Ensure destination folders exist
    os.makedirs(labels, exist_ok=True)
    os.makedirs(images, exist_ok=True)

    # Iterate through files in the source folder
    for filename in os.listdir(mixed_folder):
        file_path = os.path.join(mixed_folder, filename)

        # Ensure it's a file (not a subdirectory)
        if os.path.isfile(file_path):
            # Check file extension
            if filename.lower().endswith(".txt"):
                print("copied from ", file_path, " to ", labels)
                shutil.copy(file_path, os.path.join(labels, filename))
            else:
                print("copied from ", file_path, " to ", images)
                shutil.copy(file_path, os.path.join(images, filename))

    print("Files copied successfully!")

