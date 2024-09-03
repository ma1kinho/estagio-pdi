import argparse
from pathlib import Path
from typing import Dict
import numpy as np
import cv2
import pickle
from skimage.feature import hog, local_binary_pattern

def parse_arguments() -> argparse.Namespace:
    """
    Parses command line arguments for the feature extraction script.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Extract features from segmented images and save results.")
    parser.add_argument(
        "--segmentation_file",
        type=str,
        default="outputs/segmentation_results.pkl",
        help="Path to segmentation results file."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/images/train",
        help="Directory of images to process."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="outputs/feature_extraction_results.pkl",
        help="Output file to save extracted features."
    )
    parser.add_argument(
        "--hog_orientations",
        type=int,
        default=9,
        help="Number of orientation bins for HOG feature extraction."
    )
    parser.add_argument(
        "--hog_pixels_per_cell",
        type=int,
        default=8,
        help="Size (in pixels) of a cell for HOG feature extraction."
    )
    parser.add_argument(
        "--hog_cells_per_block",
        type=int,
        default=2,
        help="Number of cells in each block for HOG feature extraction."
    )
    parser.add_argument(
        "--lbp_radius",
        type=int,
        default=1,
        help="Radius for LBP feature extraction."
    )
    parser.add_argument(
        "--lbp_points",
        type=int,
        default=8,
        help="Number of points for LBP feature extraction."
    )
    return parser.parse_args()

def extract_features(image: np.ndarray, hog_orientations: int, hog_pixels_per_cell: int,
                     hog_cells_per_block: int, lbp_radius: int, lbp_points: int) -> np.ndarray:
    """
    Extracts HOG and LBP features from the image.

    Args:
        image (np.ndarray): Input image.
        hog_orientations (int): Number of orientation bins for HOG.
        hog_pixels_per_cell (int): Size of each cell for HOG.
        hog_cells_per_block (int): Number of cells in each block for HOG.
        lbp_radius (int): Radius for LBP.
        lbp_points (int): Number of points for LBP.

    Returns:
        np.ndarray: Concatenated HOG and LBP features.
    """
    # Convert image to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # HOG feature extraction
    hog_features = hog(
        gray,
        orientations=hog_orientations,
        pixels_per_cell=(hog_pixels_per_cell, hog_pixels_per_cell),
        cells_per_block=(hog_cells_per_block, hog_cells_per_block),
        block_norm='L2-Hys',
        visualize=False
    )

    # LBP feature extraction
    lbp = local_binary_pattern(gray, P=lbp_points, R=lbp_radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(257), density=True)

    # Combine features into a single array
    combined_features = np.hstack((hog_features, lbp_hist))
    
    return combined_features

def main():
    """
    Main function to perform feature extraction on segmented images and save the results.
    """
    args = parse_arguments()
    segmentation_file = Path(args.segmentation_file)
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)

    # Load segmentation results
    with segmentation_file.open('rb') as f:
        segmentation_results = pickle.load(f)

    feature_extraction_results = {}

    # Extract features for each image based on segmentation results
    for image_name, bounding_boxes in segmentation_results.items():
        image_path = input_dir / image_name
        image = cv2.imread(str(image_path))

        # Extract features for each bounding box
        for box in bounding_boxes:
            x1, y1, x2, y2 = box
            roi = image[y1:y2, x1:x2]  # Extract region of interest
            features = extract_features(
                roi,
                hog_orientations=args.hog_orientations,
                hog_pixels_per_cell=args.hog_pixels_per_cell,
                hog_cells_per_block=args.hog_cells_per_block,
                lbp_radius=args.lbp_radius,
                lbp_points=args.lbp_points
            )
            feature_extraction_results[image_name] = features

    # Save feature extraction results to a file
    with output_file.open('wb') as f:
        pickle.dump(feature_extraction_results, f)

    print(f"Feature extraction results saved to {output_file}")

if __name__ == "__main__":
    main()
