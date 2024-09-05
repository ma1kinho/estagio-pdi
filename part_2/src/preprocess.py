import argparse
from pathlib import Path

import cv2
import numpy as np
from skimage import filters
from skimage.transform import rotate
from tqdm import tqdm

from utils import TQDM_FORMAT

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the preprocessing script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Preprocess images for OCR.")
    parser.add_argument("--data_dir", type=str, default="data/images/train", help="Directory of images to preprocess.")
    parser.add_argument("--output_dir", type=str, default="outputs/preprocessed", help="Output dir to save preprocessed images.")
    # Add CLAHHE parameters
    parser.add_argument("--clip_limit", type=float, default=2.0, help="Clip limit for CLAHE.")
    parser.add_argument("--tile_grid_size", type=tuple, default=(8, 8), help="Tile grid size for CLAHE.")
    parser.add_argument("--blur_kernel", type=int, default=7, help="Kernel size for Gaussian blur.")
    # Add thresholding parameters
    parser.add_argument("--threshold_method", type=str, default="otsu", choices=['otsu', 'adaptive'], help="Thresholding method.")
    parser.add_argument("--block_size", type=int, default=8, help="Block size for adaptive thresholding.")
    parser.add_argument("--c", type=int, default=2, help="Constant for adaptive thresholding.")
    parser.add_argument("--morph_kernel", type=int, default=3, help="Kernel size for morphological operations.")
    return parser.parse_args()

def load_image(image_path: Path) -> np.ndarray:
    """
    Load an image from a file path and convert it to grayscale.

    Args:
        image_path (Path): Path to the image file.

    Returns:
        np.ndarray: Grayscale image.
    """
    image = cv2.imread(str(image_path))
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

def reduce_noise(image: np.ndarray, kernel_size: int = 3, **kwargs) -> np.ndarray:
    """
    Apply Gaussian blur to reduce noise in the image.

    Args:
        image (np.ndarray): Grayscale image.
        kernel_size (int): Size of the Gaussian kernel. Default is 5.

    Returns:
        np.ndarray: Blurred image.
    """
    # Return if the kernel size is less than 3
    if kernel_size < 3:
        return image
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred_image

def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8), **kwargs) -> np.ndarray:
    """
    Enhance the contrast of the image with CLAHE.

    Args:
        image (np.ndarray): Grayscale image.

    Returns:
        np.ndarray: Image with enhanced contrast.
    """
    # Return the original image if the clip limit is less than or equal to 1
    if clip_limit <= 0 or tile_grid_size[0] <= 0 or tile_grid_size[1] <= 0:
        return image
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_image = clahe.apply(image)
    return enhanced_image


def detect_and_zoom_plate(image: np.ndarray) -> np.ndarray:

    # Step 2: Apply Edge Detection
    # Use Canny edge detection to find edges in the image
    edges = cv2.Canny(image, 100, 200)

    # Step 3: Find Contours
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Step 4: Filter Contours by Aspect Ratio and Size
    plate_contour = None
    for contour in contours:
        # Approximate the contour to a polygon
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        
        # Check if the approximated contour has 4 points (likely a rectangle)
        if len(approx) == 4:
            # Compute the bounding box of the contour
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            
            # Check if the bounding box matches the expected aspect ratio of a license plate
            if 2 < aspect_ratio < 5 and 1000 < cv2.contourArea(contour) < 15000:  # Adjust values based on image size
                plate_contour = approx
                break

    # Step 5: Crop the License Plate Region
    if plate_contour is not None:
        # Draw the contour on the original image (optional for visualization)
        cv2.drawContours(image, [plate_contour], -1, (0, 255, 0), 3)
        
        # Extract the bounding box coordinates and crop the image
        x, y, w, h = cv2.boundingRect(plate_contour)
        cropped_plate = image[y:y+h, x:x+w]
        
        return cropped_plate
    else:
        return image


def _invert_image_by_thresholding(image: np.ndarray) -> np.ndarray:
    """
    Invert the image.

    Args:
        image (np.ndarray): Grayscale image.

    Returns:
        np.ndarray: Inverted image.
    """
    # Apply Otsu's thresholding
    _, thresh_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Determine the number of pixels in each class
    foreground_pixels = np.sum(thresh_image == 0)  # Pixels below the threshold
    background_pixels = np.sum(thresh_image == 255)  # Pixels above the threshold

    # Invert if the background is dark
    if foreground_pixels > background_pixels:
        image = cv2.bitwise_not(image)
    return image


def _invert_image_by_histogram(image: np.ndarray) -> np.ndarray:
    """
    Invert the image based on the histogram of pixel intensities.

    Args:
        image (np.ndarray): Grayscale image.

    Returns:
        np.ndarray: Inverted image.
    """
    # Compute histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    # Determine if there are more pixels in the dark range
    dark_peak = np.argmax(hist[:128])
    light_peak = np.argmax(hist[128:]) + 128

    if dark_peak > light_peak:
        image = cv2.bitwise_not(image)

    return image


def _invert_image(image: np.ndarray, method: str) -> np.ndarray:
    """
    Invert the image based on the specified method.

    Args:
        image (np.ndarray): Grayscale image.
        method (str): Method to use for inverting the image.

    Returns:
        np.ndarray: Inverted image.
    """
    if method == 'thresholding':
        return _invert_image_by_thresholding(image)
    elif method == 'histogram':
        return _invert_image_by_histogram(image)
    else:
        return image

def apply_threshold(image: np.ndarray,  method: str = 'otsu', block_size: int = 8, c=2, **kwargs) -> np.ndarray:
    """
    Apply thresholding to binarize the image.

    Args:
        image (np.ndarray): Grayscale image.
        method (str): Thresholding method, either 'otsu' or 'adaptive'. Default is 'otsu'.

    Returns:
        np.ndarray: Binarized image.
    """

    if block_size < 3:
        return image

    if method == 'otsu':
        threshold_value = filters.threshold_otsu(image)
        binary_image = image > threshold_value
        binary_image = binary_image.astype(np.uint8) * 255
    elif method == 'adaptive':
        if block_size % 2 == 0:
            block_size += 1
        binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 
                                             blockSize=block_size, 
                                             C=c)
    else:
        return image  # Return the original image if the method is not recognized.

    return binary_image

def deskew_image(image: np.ndarray) -> np.ndarray:
    """
    Deskew the image by rotating it to straighten the text.
    """
    # Use Hough Transform to detect lines
    lines = cv2.HoughLinesP(image, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is None:
        return image
    
    # Compute the angle of the detected lines
    angles = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            angles.append(angle)
    
    # Compute the median angle
    median_angle = np.median(angles)
    
    # Rotate the image to deskew
    deskewed_image = rotate(image, median_angle, resize=True, mode='edge')
    
    return (deskewed_image * 255).astype(np.uint8)  # Convert back to uint8

def morphological_operations(image: np.ndarray, kernel: int = 3, **kwargs) -> np.ndarray:
    """
    Apply morphological operations to separate characters in the image.

    Args:
        image (np.ndarray): Binarized image.
        kernel (int): Size of the kernel for morphological operations. Default is 3.

    Returns:
        np.ndarray: Image after morphological operations.
    """
    image = _invert_image(image, "histogram")
    image = deskew_image(image)
    kernel_matrix = np.ones((kernel, kernel), np.uint8)
    morph_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_matrix)
    return morph_image


def preprocess_image(image_path: Path, **kwargs) -> np.ndarray:
    """
    Complete preprocessing pipeline for OCR, including noise reduction, thresholding, and morphological operations.

    Args:
        image_path (Path): Path to the image file.
        blur_kernel (int): Kernel size for Gaussian blur. Default is 5.
        threshold_method (str): Thresholding method ('otsu' or 'adaptive'). Default is 'otsu'.
        morph_kernel (int): Kernel size for morphological operations. Default is 3.

    Returns:
        np.ndarray: Preprocessed image ready for OCR.
    """
    image = load_image(image_path)
    image = enhance_contrast(image, **kwargs)
    image = detect_and_zoom_plate(image)
    image = reduce_noise(image, **kwargs)
    image = apply_threshold(image, **kwargs)
    image = morphological_operations(image, **kwargs)
    return image

def main(args: argparse.Namespace) -> None:
    """
    Main function to preprocess images and save the results to a pickle file.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    # Convert the data directory and output file path to Path objects
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Iterate through each image in the specified directory and preprocess it
    for image_path in tqdm(data_dir.iterdir(), bar_format=TQDM_FORMAT, desc="Preprocessing images"):
        if image_path.is_file():
            processed_image = preprocess_image(
                image_path,
                **vars(args)
            )
            # Write image to output directory
            output_path = output_dir / image_path.name
            cv2.imwrite(str(output_path), processed_image)

    
if __name__ == "__main__":
    args = parse_arguments()
    main(args)
