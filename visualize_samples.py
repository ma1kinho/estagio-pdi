import string
import os
import cv2
import matplotlib.pyplot as plt


# Mapping of class IDs to class names (names are letters and numbers)
ID_TO_CLASS = {id_: class_ for id_, class_ in enumerate(string.ascii_uppercase + string.digits)}
COLOR_MAP = plt.get_cmap("tab20", len(ID_TO_CLASS))


def visualize_bboxes(images_dir, labels_dir, num_samples=5):
    """
    Visualizes bounding boxes for images and labels provided in YOLO format.

    Args:
    images_dir (str): Directory containing images.
    labels_dir (str): Directory containing YOLO format label files.
    """
    # Get list of image files
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files[:num_samples]:
        # Load image
        image_path = os.path.join(images_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image {image_path}")
            continue

        # Get corresponding label file
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_file)

        if not os.path.exists(label_path):
            print(f"No label file found for {image_file}")
            continue

        # Read label file
        with open(label_path, 'r') as f:
            labels = f.readlines()

        # Get image dimensions
        h, w, _ = image.shape

        # Draw bounding boxes
        label_str = ""
        for label in labels:
            label = label.strip().split()
            class_id = int(label[0])
            class_name = ID_TO_CLASS.get(class_id, class_id)
            label_str += f"{class_name}"
            x_center = float(label[1]) * w
            y_center = float(label[2]) * h
            width = float(label[3]) * w
            height = float(label[4]) * h
            color = [float(i) * 255 for i in COLOR_MAP(class_id)]

            # Calculate bounding box coordinates
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            # Draw rectangle and label on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
            cv2.putText(image, f"{class_name}", (x1, y2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)

        # Convert BGR image (OpenCV default) to RGB for displaying with matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display image with bounding boxes
        plt.figure(figsize=(5, 5))
        plt.imshow(image_rgb)
        plt.title(f"Bboxes for {image_file} with label {label_str}")
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize bounding boxes from YOLO format labels.")
    parser.add_argument("images_dir", type=str, help="Directory containing images.")
    parser.add_argument("labels_dir", type=str, help="Directory containing YOLO format label files.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize.")
    args = parser.parse_args()

    visualize_bboxes(args.images_dir, args.labels_dir, args.num_samples)

