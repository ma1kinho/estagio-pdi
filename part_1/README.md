### Day 1 Practice

#### Setting up the Environment

Begin by setting up the project environment and verifying that all necessary packages are installed.

#### Download data

Because of Google Drive limitations regarding DVC, you will need to download `data.zip` from [here](https://drive.google.com/file/d/1VvDZKi9SMeSeIvfOJqnouJzh79QYO8lf/view?usp=sharing). Keep it alongside `src`.

#### Project Structure and DVC Setup

This project has the following key files

- `preprocess.py`: For image preprocessing (e.g., grayscale conversion, noise reduction, thresholding).
- `segment.py`: For character segmentation.
- `extract.py`: For feature extraction.
- `train.py`: For character recognition.
- `evaluate.py`: For calculating accuracy and visualizing results.
- `utils.py`: For shared functions.

#### Running and Tracking Experiments with DVC

Define and run experiments using DvC by setting up `dvc.yaml` and running different configurations.

#### Homework

- Create a notebook with 
  - A new image enhancement technique (e.g. CLAHE)
  - A new thresholding method 
  - Combine thresholding with inversion detection
  - Improve character segmentation technique (e.g., improving morphological operations, projection profiles etc.)
  - Add a new feature extraction method (e.g. ORB)
  - Add a new character recognition technique (e.g. template matching, MLPs etc.)
- Experiment with different parameters
- Track all experiments using DvC
