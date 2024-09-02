from collections import defaultdict
import os
import pickle
import random

import cv2
import numpy as np
from tqdm import tqdm
from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from utils import TQDM_FORMAT, load_labels


data_dir = "data/images/train"
labels_dir = "data/labels/train"
processed_images = "outputs/preprocessed.pkl"
output_model = "outputs/ocr_model.pkl"


def connected_component_analysis(image):
    """Perform connected component analysis for character segmentation."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)
    characters = []
    for stat in stats:
        x, y, w, h, area = stat
        if area > 100:  # Example filter condition
            characters.append((x, y, w, h))
    return characters


def extract_features(character_image):
    """Extract features using LBP for better character recognition."""
    
    # Local Binary Patterns (LBP)
    lbp_features = local_binary_pattern(character_image, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp_features, bins=np.arange(257), density=True)
    
    # Combine features
    combined_features = np.hstack(([], lbp_hist))

    # Normalize features
    scaler = StandardScaler()
    combined_features = scaler.fit_transform(combined_features.reshape(1, -1)).flatten()
    
    return combined_features

def train_classifier(features, labels):
    """Train a classifier using extracted features and labels."""
    clf = SVC(kernel='linear', C=1, probability=True)
    clf.fit(features, labels)
    return clf

def train_random_forest(features, labels):
    """Train a Random Forest classifier using extracted features and labels."""
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(features, labels)
    return clf


def train_model(features, labels, model_name):
    """Train a classifier using the extracted features and labels."""
    if model_name == "SVM":
        model = train_classifier(features, labels)
    elif model_name == "RandomForest":
        model = train_random_forest(features, labels)
    else:
        raise ValueError("Invalid model name. Choose from 'SVM' or 'RandomForest'.")
    return model

def extract_features_and_labels(data_dir, images, labels):
    """Extract features and corresponding character labels from images."""
    features = []
    char_labels = []

    for image_path, image in tqdm(zip(os.listdir(data_dir), images), bar_format=TQDM_FORMAT, desc="Extracting features...", total=len(images)):
        key = os.path.basename(image_path).split('.')[0]
        label = labels[key]  # image: label
        
        characters = connected_component_analysis(image)
        
        for idx, (x, y, w, h) in enumerate(characters):
            char_image = image[y:y+h, x:x+w]
            features_i = extract_features(char_image)
            features.append(features_i)
            char_labels.append(label[idx])

    return np.array(features), np.array(char_labels)

def ocr_pipeline(data_dir, images, labels, model_name="SVM"):
    """Complete OCR pipeline for character recognition."""
    # Extract features and labels
    features, char_labels = extract_features_and_labels(data_dir, images, labels)

    model = train_model(features, char_labels, model_name=model_name)

    return model


if __name__ == "__main__":
    # Load the preprocessed images
    with open(processed_images, 'rb') as f:
        images = pickle.load(f)

    labels = load_labels(labels_dir)
    model = ocr_pipeline(data_dir, images, labels)

    # Save model
    print(f"Saving OCR model to {output_model}")
    with open(output_model, 'wb') as f:
        pickle.dump(model, f)

