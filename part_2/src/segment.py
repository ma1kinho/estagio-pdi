import argparse
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
import pickle

def parse_arguments() -> argparse.Namespace:
    """
    Parses command line arguments for the segmentation script.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Segment images and save bounding boxes.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/images/train",
        help="Directory of images to segment."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="outputs/segmentation_results.pkl",
        help="Output file to save segmentation results."
    )
    parser.add_argument(
        "--min_area",
        type=int,
        default=50,
        help="Minimum area of connected components to consider in segmentation."
    )
    parser.add_argument(
        "--max_area",
        type=int,
        default=500,
        help="Maximum area of connected components to consider in segmentation."
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        default=8,
        help="Connectivity for connected component analysis."
    )
    return parser.parse_args()

def cca(image: np.ndarray, min_area: int, max_area: int, connectivity: int) -> List[Tuple[int, int, int, int]]:
    """
    Performs connected component analysis on an image to find bounding boxes around characters.

    Args:
        image (np.ndarray): Input binary image.
        min_area (int): Minimum area of connected components to consider.
        connectivity (int): Connectivity for connected component analysis.

    Returns:
        list[tuple[int, int, int, int]]: List of bounding boxes for each connected component.
    """
    # Convert image to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(gray, connectivity=connectivity)
    
    # Extract bounding boxes for each component
    bounding_boxes = []
    for i in range(1, num_labels):  # Skip background label 0
        x, y, w, h, area = stats[i]
        if min_area < area < max_area:  # Filter out small areas
            bounding_boxes.append((x, y, x + w, y + h))

    return bounding_boxes

def segment_image(image: np.ndarray, min_area: int = 50, max_area: int = 500, connectivity: int = 8) -> List[Tuple[int, int, int, int]]:
    """
    Segment an image into characters using connected component analysis.

    Args:
        image (np.ndarray): Input image.
        min_area (int): Minimum area of connected components to consider.
        connectivity (int): Connectivity for connected component

    Returns:
        list[tuple[int, int, int, int]]: List of bounding boxes for each character.
    """
    # TODO: improve this method to enhance the segmentation results somehow
    return cca(image, min_area, max_area, connectivity)

def main():
    """
    Main function to perform image segmentation and save bounding boxes to a file.
    """
    args = parse_arguments()
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)

    segmentation_results = {}

    # Perform segmentation for each image in the directory
    for image_path in input_dir.glob("*.png"):  # Assuming .png images; change if needed
        image = cv2.imread(str(image_path))
        bounding_boxes = segment_image(image, args.min_area, args.connectivity)
        segmentation_results[image_path.name] = bounding_boxes

    # Save segmentation results to a file
    with output_file.open('wb') as f:
        pickle.dump(segmentation_results, f)

    print(f"Segmentation results saved to {output_file}")

if __name__ == "__main__":
    main()
