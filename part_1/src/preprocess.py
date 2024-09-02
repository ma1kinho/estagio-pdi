import os
import pickle
import random

import cv2
import numpy as np
from skimage import filters
from tqdm import tqdm

from utils import TQDM_FORMAT


DISPLAY = True

data_dir = "data/images/train"
output_file = "outputs/preprocessed.pkl"
blur_kernel = 7
threshold_method = ''
morph_kernel = 3


def load_image(image_path):
    """Load an image from a file path and convert it to grayscale."""
    image = cv2.imread(image_path)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

def reduce_noise(image, kernel_size=5):
    """Apply Gaussian blur to reduce noise in the image."""
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred_image

def apply_threshold(image, method='otsu'):
    """Apply thresholding to binarize the image."""
    if method == 'otsu':
        threshold_value = filters.threshold_otsu(image)
        binary_image = image > threshold_value
        binary_image = binary_image.astype(np.uint8) * 255
    elif method == 'adaptive':
        binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 2)
    else:
        # raise ValueError("Invalid thresholding method. Choose from 'otsu' or 'adaptive'.")
        return image

    return binary_image

def morphological_operations(image, kernel=3):
    """Apply morphological operations to separate characters."""
    kernel = np.ones((kernel, kernel), np.uint8)
    morph_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return morph_image

def preprocess_image(image_path, blur_kernel=5, threshold_method='otsu', morph_kernel=3):
    """Complete preprocessing pipeline for OCR."""
    image = load_image(image_path)
    image = reduce_noise(image, kernel_size=blur_kernel)
    image = apply_threshold(image, method=threshold_method)
    image = morphological_operations(image, kernel=morph_kernel)
    return image

if __name__ == "__main__":
    # Preprocess images and save the results as npy file
    images = []
    for image in tqdm(os.listdir(data_dir), bar_format=TQDM_FORMAT, desc="Preprocessing images"):
        image_path = os.path.join(data_dir, image)
        processed_image = preprocess_image(image_path,
                                           blur_kernel=blur_kernel,
                                           threshold_method=threshold_method,
                                           morph_kernel=morph_kernel)
        images.append(processed_image)


    # Make output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Write pkl file with preprocessed images
    print(f"Saving preprocessed images to {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(images, f)

    if DISPLAY:
        # Display a sample of preprocessed images
        samples = random.sample(images, 5)
        for i in range(5):
            cv2.imshow("Preprocessed Image", samples[i])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

