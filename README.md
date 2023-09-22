# Tile-Crack-Detection

This project is designed to detect cracks in tiles using computer vision techniques. It includes image preprocessing, feature extraction, and a convolutional neural network (CNN) based model for crack detection. The project utilizes the OpenCV, NumPy, Matplotlib, and TensorFlow libraries. Below, you'll find a step-by-step explanation of the key components of this project.

## Prerequisites

Before running the code, make sure you have the following installed:

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- TensorFlow
- Google Colab (for the provided code example)

## Dataset

The dataset used in this project has this structure:

- `Dataset.zip`: Contains the tile images.
- `Patterns.zip`: Contains pattern images used for comparison.

Make sure to replace these zip file paths with your actual dataset paths.

## Image Preprocessing

### Grayscale Conversion

The input tile image is converted to grayscale using the `cv2.cvtColor` function, simplifying further processing.

### Image Blurring

To reduce noise, a bilateral filter is applied using `cv2.bilateralFilter`. This filter smoothens the image while preserving edges.

### Edge Detection

The Canny edge detection method (`cv2.Canny`) is applied to detect edges in the blurred image.

### Finding Tile Vertices

Contours in the edge-detected image are found using `cv2.findContours`. The largest contour is identified as the tile's outline. Four vertices of the tile are extracted.

### Blank Page Removal

The function `blank_page` removes any text or unwanted details from the tile image. It uses morphological operations and the GrabCut algorithm (`cv2.grabCut`) to segment the tile.

## Histogram Matching

A pattern image is selected based on the JSON data, and its histogram is matched to the segmented tile using the `skimage.exposure.match_histograms` function. This helps to adjust the pattern image's lighting and contrast to match the tile.

## Tile Warping

Tile warping is performed to align the matched pattern image with the tile. Two methods are employed:

1. Pattern Rotation: The pattern is rotated by 90 degrees at a time to find the best alignment based on the Dice coefficient (`dice_coef`).
2. Warp Perspective: Perspective transformation (`cv2.getPerspectiveTransform`) is applied to achieve the best alignment.

## Siamese U-Net Architecture

A Siamese U-Net architecture is used for change detection. This architecture consists of two identical U-Net networks, one for each "pattern" and "tile" image. The key components are as follows:

- U-Net Encoder (contracting path): Downsampling of input images.
- U-Net Decoder (expanding path): Upsampling and feature map concatenation.
- Feature Extraction: The encoder extracts features from both images.
- Contrastive Loss: A contrastive loss function is used to measure the similarity between feature vectors from both images. It encourages the model to differentiate between unchanged and changed regions.

## Loss Function

The model is trained using a custom loss function that combines binary focal loss and Dice loss. It encourages the model to focus on challenging regions (cracks) and improve segmentation accuracy.

## Training

The model is trained on a dataset consisting of tile images and their corresponding crack masks. The dataset is split into a training set and a validation set. The training progress is monitored using various metrics such as accuracy, Dice coefficient, IoU, F1-score, precision, and recall.

## Results

The trained model can be used to predict crack masks for new tile images. The predictions can be visualized to identify cracked regions.


This project can be further extended and customized for specific crack detection applications.
