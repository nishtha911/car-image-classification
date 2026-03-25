# Car Object Detection using ResNet50

## Overview

This project implements car object detection using a Convolutional Neural Network based on ResNet50. The model predicts bounding box coordinates for cars in images and compares performance across multiple optimization algorithms.

## Dataset

Source: Kaggle Car Object Detection Dataset
Images: ~1000 training images
Annotations: Bounding box coordinates (xmin, ymin, xmax, ymax)

## Project Structure

```
├── data
│   ├── train
│   ├── test
│   └── train_solution_bounding_boxes.csv
├── notebooks
├── training
├── models
├── evaluation
└── app
```

## Data Pipeline

* Loaded bounding box annotations
* Created image path mapping
* Performed train-validation split
* Resized images to 224x224
* Normalized pixel values
* Normalized bounding box coordinates
* Visualized bounding boxes for verification

## Model (Planned)

* ResNet50 (Transfer Learning)
* Bounding box regression head
* Optimizer comparison:

  * SGD
  * Adam
  * RMSprop
  * Adagrad

## Evaluation Metrics (Planned)

* Loss
* Mean Absolute Error
* Intersection over Union (IoU)
* Visual bounding box comparison

## Status

Data pipeline completed. Model training in progress.
