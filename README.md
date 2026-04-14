# Car Object Detection Pipeline

A deep learning project focused on detecting cars in images using a Convolutional Neural Network (ResNet50). The project implements a custom bounded-box regression head and compares multiple stochastic optimization algorithms (Adam, SGD, RMSprop, Adagrad) to maximize validation accuracy.

## Tech Stack

| Component | Technology |
| --- | --- |
| **Language** | Python 3.11 |
| **Deep Learning Framework** | TensorFlow & Keras |
| **CNN Backbone** | ResNet50 (ImageNet weights) |
| **Data Processing** | Pandas, NumPy |
| **Computer Vision** | OpenCV (cv2) |
| **Environment** | Jupyter Notebooks |

## Key Implementations

This project resolves two primary challenges with standard bounding box regression applied to raw computer vision datasets:

1. **Multi-Target Resolution:** The original Kaggle dataset contained multiple bounding box coordinates per image (e.g., an image with three cars had three rows in the CSV). Training a basic continuous-value regression model on conflicting targets causes geometric decay. This was resolved by building a Pandas pipeline that isolates and extracts only the largest bounding box area per image as the primary target.
2. **GIoU Loss Function:** Standard Mean Squared Error (MSE) treats bounding box coordinates (`xmin, ymin, xmax, ymax`) as independent variables rather than spatial boundaries. This project implements a custom **Generalized Intersection over Union (GIoU)** loss metric, which forces the neural network to mathematically overlap the geometric area of the ground truth.

## Setup Instructions

TensorFlow strictly requires Python 3.11 limits on Windows environments.

1. Clone the repository natively and create a 3.11 virtual environment:
```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install notebook ipykernel
```

## How to Run

To present or modify the pipeline, open Jupyter Notebook (`jupyter notebook`) and execute the files sequentially from the `/notebooks` directory.

### 1. `01_Data_Preparation.ipynb`
This notebook demonstrates the conflicting multi-target issue. It calculates bounding box areas, extracts the largest prominent car per image, visualizes the result, and exports the clean `cleaned_bounding_boxes.csv` file needed for training.

### 2. `02_Model_Training.ipynb`
This handles core pipeline execution. It builds the ResNet50 backbone and the custom Sigmoid density head. It then actively trails and benchmarks the GIoU loss logic spanning four different optimization algorithms (Adam, SGD, RMSprop, Adagrad), outputting the trained models alongside a comparative `matplotlib` chart.

### 3. `03_Inference.ipynb`
This script loads the most accurate model file determined from step 2 (generally Adam). It denormalizes the predicted `[0,1]` scale tensors back into exact pixel dimensions and draws the target versus predicted bounding boxes over raw unseen images.
