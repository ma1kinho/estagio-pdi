import os
import string
import pickle
import random
import cv2

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from utils import  load_labels, load_predictions, TQDM_FORMAT
from ocr import connected_component_analysis, extract_features


DISPLAY = True

data_dir = "data/images/train"
labels_dir = "data/labels/train"
processed_images = "outputs/preprocessed.pkl"
predictions_file = "outputs/predictions.csv"
model_path = "outputs/ocr_model.pkl"

def evaluate_ocr_model(model, images):
    """Evaluate the trained model on a set of images."""
    predictions = {}

    for image_path, image in tqdm(zip(os.listdir(data_dir), images), bar_format=TQDM_FORMAT, desc="Evaluating...", total=len(images)):
        characters = connected_component_analysis(image)
        pred_text = ""

        for (x, y, w, h) in characters:
            char_image = image[y:y+h, x:x+w]
            features_i = extract_features(char_image)
            char_pred = model.predict([features_i])
            pred_text += char_pred[0]
        
        # Image ID
        image_id = os.path.basename(image_path).split('.')[0]
        predictions[image_id] = pred_text

    return predictions


def calculate_accuracy(predicted_text, true_text):
    """Calculate the accuracy of the OCR by comparing with the ground truth."""
    correct_chars = sum(p == t for p, t in zip(predicted_text, true_text))
    accuracy = correct_chars / len(true_text) if true_text else 0
    return accuracy

def display_image(data_dir, samples):
    """Display the image with the predicted text."""
    for image_id, prediction in samples:
        image_path = os.path.join(data_dir, image_id + ".jpg")
        image = cv2.imread(image_path)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f"Predicted Text: {prediction}")
        plt.show()


if __name__ == "__main__":
    # Load the preprocessed images
    with open(processed_images, 'rb') as f:
        images = pickle.load(f)

    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    labels = load_labels(labels_dir)
    predictions = evaluate_ocr_model(model, images)
    # labels and predictions are dictionaries with sample ID as key and text as value
    accuracies = []
    for sample in labels.keys():
        true_text = labels[sample]
        predicted_text = predictions[sample]
        accuracy = calculate_accuracy(predicted_text, true_text)
        accuracies.append(accuracy)

    print(f"Avg. Accuracy: {np.mean(accuracies):.4f}")
    print(f"Std. Deviation: {np.std(accuracies):.4f}")

    # Write predictions to a CSV file (output_file)
    with open(predictions_file, 'w') as f:
        f.write("image,predictio\n")
        for image, prediction in predictions.items():
            f.write(f"{image},{prediction}\n")

    if DISPLAY:
        # Select 5 random images for display
        samples = random.sample(list(predictions.items()), 5)
        display_image(data_dir, samples)
