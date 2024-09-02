import random
import os
import shutil
import sys

def copy_and_rename_files(root_dir, destination_dir, set_dir, num_samples=10000):
    # Define source directories for images and labels
    image_source_dir = os.path.join(root_dir, 'images', set_dir)
    label_source_dir = os.path.join(root_dir, 'labels', set_dir)

    # Create destination directories for images and labels
    dest_image_dir = os.path.join(destination_dir, 'images', set_dir)
    dest_label_dir = os.path.join(destination_dir, 'labels', set_dir)
    os.makedirs(dest_image_dir, exist_ok=True)
    os.makedirs(dest_label_dir, exist_ok=True)

    # Get list of image files (assuming they end with .jpg or .png)
    image_files = [f for f in os.listdir(image_source_dir)]
    print(f"Found {len(image_files)} image files.")

    # Limit to num_samples if specified (random)
    image_files = random.sample(image_files, num_samples) if num_samples < len(image_files) else image_files
    print(f"Copying {len(image_files)} image files.")

    # Copy images and corresponding label files
    for i, image_file in enumerate(image_files):
        # Define new file name
        new_file_name = f"{i}"

        # Copy and rename image file
        image_extension = image_file.rsplit('.', 1)[-1]
        new_image_name = f"{new_file_name}.{image_extension}"
        shutil.copy(os.path.join(image_source_dir, image_file), os.path.join(dest_image_dir, new_image_name))

        # Corresponding label file (assuming the same name with .txt extension)
        label_file = image_file.rsplit('.', 1)[0] + '.txt'
        
        # Check if label file exists and copy it
        if os.path.exists(os.path.join(label_source_dir, label_file)):
            new_label_name = f"{new_file_name}.txt"
            shutil.copy(os.path.join(label_source_dir, label_file), os.path.join(dest_label_dir, new_label_name))
        else:
            print(f"Label file not found for {image_file}")

    print("Copy and rename process completed.")

if __name__ == "__main__":
    # Check if the right number of arguments is provided
    if len(sys.argv) != 5:
        print("Usage: python script.py <root_dir> <destination_dir> <set> <num_samples>")
        sys.exit(1)

    # Get root and destination directories from arguments
    root_dir = sys.argv[1]
    destination_dir = sys.argv[2]
    set_dir = sys.argv[3]
    num_samples = int(sys.argv[4])

    # Execute the function
    copy_and_rename_files(root_dir, destination_dir, set_dir, num_samples)
