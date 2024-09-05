import os
import string
from tqdm import tqdm


TQDM_FORMAT = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"


def load_labels(labels_dir):
    """Load the ground truth labels (YOLO format) for the images."""
    labels = {}
    ID_TO_CLASS = {id_: class_ for id_, class_ in enumerate(string.ascii_uppercase + string.digits)}
    for label in tqdm(os.listdir(labels_dir), bar_format=TQDM_FORMAT, desc="Loading labels"):
        with open(os.path.join(labels_dir, label), 'r') as f:
            lines = f.readlines()
            sample = label.split('.')[0]
            # Convert id 2 class name
            label_str = ""
            for line in lines:
                id_, *_ = line.split()
                class_name = ID_TO_CLASS[int(id_)]
                label_str += class_name
            labels[sample] = label_str
    return labels

def load_predictions(predictions_file):
    """Load the OCR predictions from the output file."""
    predictions = {}
    with open(predictions_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            sample, prediction = line.split(',')
            # Sample is full filepath. Get filename without extension.
            sample = os.path.basename(sample).split('.')[0]
            predictions[sample] = prediction
    return predictions

