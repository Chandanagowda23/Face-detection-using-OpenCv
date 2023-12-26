# Face Detection using OpenCV

This repository implements a face detection project using OpenCV in Python. The project utilizes the Haar Cascade Classifier for detecting faces in images, and the results are stored in JSON format.

## Project Overview

The face detection algorithm is implemented in the `face_detection.py` script. It takes input images from a specified directory, detects faces using the Haar Cascade Classifier, and saves the results in a JSON file. The goal is to provide a simple and effective face detection solution.

## Usage

To use the project, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Chandanagowda23/Face-detection-using-OpenCv.git
   cd Face-detection-using-OpenCv


# Face detection on validation data
python face_detection.py --input_path data/validation_folder/images --output ./result_task1_val.json

# Validation
python ComputeFBeta/ComputeFBeta.py --preds result_task1_val.json --groundtruth data/validation_folder/ground-truth.json

# Face detection on test data
python face_detection.py --input_path data/test_folder/images --output ./result_task1.json


## Directory Structure

data/: Contains sample input images for testing.
test_folder/: Additional test images.
validation_folder/: Images for validation.
face_detection.py: Main script for face detection.
result_task1.json: Output file containing detected faces for the provided test images.
result_task1_val.json: Output file containing detected faces for the validation images.
utils.py: Utility functions for displaying images.

## Results

The detected faces and their bounding boxes are stored in JSON format in the result files (result_task1.json and result_task1_val.json). You can visualize the results using your preferred JSON viewer.

## Dependencies
OpenCV
NumPy
