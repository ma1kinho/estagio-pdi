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

- Experiment with different preprocessing techniques (e.g., edge detection, different thresholding methods)
- Implement and test a template matching or other machnine learning approaches for character recognition
- Track all experiments using DvC
