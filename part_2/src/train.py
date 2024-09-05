import argparse
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from utils import TQDM_FORMAT, load_labels

def parse_arguments() -> argparse.Namespace:
    """
    Parses command line arguments for the training script.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Train OCR model.")
    parser.add_argument(
        "--feature_file",
        type=str,
        default="outputs/feature_extraction_results.pkl",
        help="Path to feature extraction results file."
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        default="data/labels/train",
        help="Directory of labels for training images."
    )
    parser.add_argument(
        "--output_model",
        type=str,
        default="outputs/ocr_model.pkl",
        help="Output path for saving the OCR model."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="SVM",
        choices=['SVM', 'RandomForest'],
        help="Model type to train."
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for reproducibility."
    )
    return parser.parse_args()

def train_classifier(features: np.ndarray, labels: np.ndarray, model_name: str, random_state: int) -> object:
    """
    Trains a classifier on the provided features and labels.

    Args:
        features (np.ndarray): Feature matrix.
        labels (np.ndarray): Label vector.
        model_name (str): Type of model to train ('SVM' or 'RandomForest').
        random_state (int): Random state for reproducibility.

    Returns:
        object: Trained model.
    """
    # Train the selected model
    if model_name == 'SVM':
        model = SVC(kernel='linear', probability=True, random_state=random_state)
    elif model_name == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    else:
        raise ValueError("Unsupported model type. Choose either 'SVM' or 'RandomForest'.")

    # Standardize features
    scaler = StandardScaler()
    try:
        features = scaler.fit_transform(features)
        # Fit the model
        model.fit(features, labels)
    except ValueError as e:
        print(f"Error training model: {e}")

    return model

def main():
    """
    Main method to execute the OCR training pipeline based on command-line arguments.
    """
    args = parse_arguments()

    feature_file = Path(args.feature_file)
    labels_dir = Path(args.labels_dir)
    output_model_path = Path(args.output_model)

    # Load the extracted features
    with feature_file.open('rb') as f:
        feature_extraction_results = pickle.load(f)

    # Load labels
    labels = load_labels(labels_dir)
    
    # Prepare data for training
    all_features = []
    all_labels = []

    print("Preparing data for training...")
    for image_name, features in feature_extraction_results.items():
        all_features.append(features)
        image_id = Path(image_name).stem
        label = labels.get(image_id)
        all_labels.append(label)
        print(f"Image ID: {image_id}, Label: {label}")

    all_features = np.array(all_features)
    all_labels = np.array(all_labels)

    # Train the classifier
    print(f"Training {args.model_name} model...")
    model = train_classifier(all_features, all_labels, model_name=args.model_name, random_state=args.random_state)

    # Save the trained OCR model
    print(f"Saving OCR model to {output_model_path}")
    with output_model_path.open('wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    main()
