import argparse
import pickle
import json
from pathlib import Path
from typing import Dict, Tuple

from preprocess import preprocess_image
from segment import segment_image
from extract import extract_features
from utils import load_labels

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the evaluation script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate trained OCR model on a set of images and labels.")
    
    # General arguments
    parser.add_argument("--model_file", type=str, required=True, help="Path to the trained model file.")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing images to evaluate.")
    parser.add_argument("--labels_dir", type=str, required=True, help="Directory containing ground truth labels.")
    parser.add_argument("--metrics_file", type=str, default="outputs/metrics.json", help="Output file to save evaluation metrics.")
    parser.add_argument("--predictions_file", type=str, default="outputs/predictions.csv", help="Output file to save evaluation metrics.")
    
    # Preprocessing arguments
    parser.add_argument("--blur_kernel", type=int, default=7, help="Kernel size for Gaussian blur in preprocessing.")
    parser.add_argument("--threshold_method", type=str, default="otsu", choices=['otsu', 'adaptive'], help="Thresholding method for binarization ('otsu' or 'adaptive').")
    parser.add_argument("--morph_kernel", type=int, default=3, help="Kernel size for morphological operations in preprocessing.")

    # Segmentation arguments
    parser.add_argument("--min_area", type=int, default=50, help="Minimum area of connected components to consider in segmentation.")

    # Feature extraction arguments
    parser.add_argument("--hog_orientations", type=int, default=9, help="Number of orientation bins for HOG feature extraction.")
    parser.add_argument("--hog_pixels_per_cell", type=int, default=8, help="Size (in pixels) of a cell for HOG feature extraction.")
    parser.add_argument("--hog_cells_per_block", type=int, default=2, help="Number of cells in each block for HOG feature extraction.")
    parser.add_argument("--lbp_radius", type=int, default=1, help="Radius for LBP feature extraction.")
    parser.add_argument("--lbp_points", type=int, default=8, help="Number of points for LBP feature extraction.")

    return parser.parse_args()

def load_model(model_file: Path) -> object:
    """
    Load the trained OCR model from a file.

    Args:
        model_file (Path): Path to the trained model file.

    Returns:
        object: Loaded model object.
    """
    with model_file.open('rb') as f:
        model = pickle.load(f)
    return model

def evaluate_model(model: object, images_dir: Path, labels: Dict[str, str], args: argparse.Namespace) -> Tuple[float, float, dict]:
    """
    Evaluate the model on the given set of images and labels.

    Args:
        model (object): Trained OCR model.
        images_dir (Path): Directory containing images to evaluate.
        labels (Dict[str, str]): Dictionary of ground truth labels.
        args (argparse.Namespace): Command-line arguments with parameters for the pipeline.

    Returns:
        Tuple[float, float]: Character-level accuracy and sample-level accuracy.
    """
    char_correct = 0
    char_total = 0
    sample_correct = 0
    sample_total = 0
    predictions = {}

    for image_path in images_dir.glob("*.png"):  # Adjust if your images are not in PNG format
        sample_name = image_path.stem
        if sample_name not in labels:
            continue

        # Preprocess the image
        processed_image = preprocess_image(
            image_path,
            blur_kernel=args.blur_kernel,
            threshold_method=args.threshold_method,
            morph_kernel=args.morph_kernel
        )

        # Segment characters in the image
        bounding_boxes = segment_image(processed_image, kernel=args.morph_kernel)

        # Extract features for each character
        predicted_text = ""
        for box in bounding_boxes:
            x1, y1, x2, y2 = box
            roi = processed_image[y1:y2, x1:x2]
            features = extract_features(
                roi,
                hog_orientations=args.hog_orientations,
                hog_pixels_per_cell=args.hog_pixels_per_cell,
                hog_cells_per_block=args.hog_cells_per_block,
                lbp_radius=args.lbp_radius,
                lbp_points=args.lbp_points
            )

            # Predict character using the trained model
            predicted_char = model.predict([features])[0]
            predicted_text += predicted_char

        # Compare the predicted text with the ground truth label
        true_text = labels[sample_name]
        char_total += len(true_text)
        char_correct += sum(1 for p, t in zip(predicted_text, true_text) if p == t)

        # Add the prediction to the dictionary
        predictions[sample_name] = predicted_text
        
        sample_total += 1
        if predicted_text == true_text:
            sample_correct += 1

    char_accuracy = char_correct / char_total if char_total > 0 else 0
    sample_accuracy = sample_correct / sample_total if sample_total > 0 else 0

    return char_accuracy, sample_accuracy, predictions

def save_metrics(char_accuracy: float, sample_accuracy: float, metric_file: Path) -> None:
    """
    Save evaluation metrics to a JSON file.

    Args:
        char_accuracy (float): Character-level accuracy.
        sample_accuracy (float): Sample-level accuracy.
        output_file (Path): Path to the output JSON file.
    """
    metrics = {
        "character_accuracy": char_accuracy,
        "sample_accuracy": sample_accuracy
    }
    
    with metric_file.open('w') as f:
        json.dump(metrics, f, indent=4)

def save_predictions(predictions: dict, predictions_file: Path) -> None:
    """
    Save the predictions to a CSV file.

    Args:
        predictions (dict): Dictionary of sample names and corresponding predictions.
        predictions_file (Path): Path to the output CSV file.
    """
    with predictions_file.open('w') as f:
        f.write("sample,prediction\n")
        for sample, prediction in predictions.items():
            f.write(f"{sample},{prediction}\n")

def main() -> None:
    """
    Main function to evaluate the OCR model.
    """
    args = parse_arguments()

    # Load the trained model
    model = load_model(Path(args.model_file))

    # Load the ground truth labels
    labels = load_labels(Path(args.labels_dir))

    # Evaluate the model
    char_accuracy, sample_accuracy, predictions = evaluate_model(model, Path(args.images_dir), labels, args)

    # Save the metrics to a file
    save_metrics(char_accuracy, sample_accuracy, Path(args.metrics_file))
    # Save the predictions to a file
    save_predictions(predictions, Path(args.predictions_file))

    print(f"Evaluation completed. Metrics saved to {args.metrics_file}. Predictions saved to {args.predictions_file}.")

if __name__ == "__main__":
    main()
