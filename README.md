<div align="center">

# 🚗 Car Object Detection Pipeline
**Featuring ResNet50 Backbone & GIoU Bounding Box Regression**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-FF6F00?style=flat&logo=tensorflow)]()
[![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=flat&logo=Keras&logoColor=white)]()
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?style=flat&logo=Jupyter&logoColor=white)]()

A comprehensive deep learning project focusing on bridging theoretical classification networks (ResNet50) into fully functional spatial object detection architectures. 

</div>

---

## 📌 Problem Statement Overview
The project tackles the detection of cars across varying multi-view datasets sourced from Kaggle. By utilizing a **ResNet50** Convolutional Neural Network, the primary objectives are to:
1. Accurately draw regressive bounding boxes around automobiles in unseen data.
2. Formulate and construct comparative baseline outputs for various stochastic optimization algorithms (**Adam, SGD, RMSprop, Adagrad**) to determine optimal gradient convergence and predictive accuracy.

---

## 🔬 Core Technical Contributions & Solutions

When implementing object detection using simple continuous-value regression, several major mathematical and data-based roadblocks inherently manifest. This project actively mitigates them through the following features:

### 1. The Multi-Target Anomaly (Data Generation)
**The Problem:** The original CSV contains multiple bounding boxes for the same image when there are multiple cars. Standard regression models cannot handle conflicting outputs mapped to the exact same input tensor; doing so produces a corrupted "average" tracking box.
**The Solution:** An ETL pipeline was built in Pandas to mathematically calculate bounding box areas, forcing the model to strictly isolate and target the single **largest bounded area** (the most prominent vehicle) per layout.

### 2. Geometric Mapping via Generalized Intersection over Union (GIoU) 
**The Problem:** Standard regression loss metrics, like Mean Squared Error (MSE), interpret `[xmin, ymin, xmax, ymax]` coordinates independently as scalar numbers, failing entirely to comprehend the spatial alignment and physical geometric boundaries of target objects.
**The Solution:** A custom **GIoU (Generalized Intersection over Union)** constraint was natively scripted into the model fit loop. GIoU actively evaluates the intersecting area ratios overlaid mathematically against the theoretical enclosing box, maximizing spatial precision.

---

## 🛠️ Environment Setup & Installation

*Note: TensorFlow strictly supports up to Python 3.11 on Windows architectures. Please ensure you do not use Python 3.12+.*

**1. Create & Activate a virtual environment (Python 3.11):**
```powershell
# Create venv
py -3.11 -m venv venv

# Activate venv (Windows)
.\venv\Scripts\Activate.ps1
```

**2. Install dependencies & notebook kernels:**
```bash
pip install -r requirements.txt
pip install notebook ipykernel
```

---

## 🚀 Execution Guide (Jupyter Notebook Sequence)

The entire project runs natively within a progressive, 3-part Jupyter Notebook pipeline. Please execute them sequentially from the `notebooks/` directory.

### 📂 `notebooks/01_Data_Preparation.ipynb`
* **Purpose:** Handles the data cleaning. Visualizes the conflicting multi-target boxes in red, and outputs the isolated "largest target" box in solid green.
* **Output:** Generates `data/cleaned_bounding_boxes.csv` required for training.

### 📂 `notebooks/02_Model_Training.ipynb`
* **Purpose:** Freezes a ResNet50 architectural backbone and connects a custom 4-node continuous-value Dense regression head. Compiles iterations of the network using the custom GIoU loss variable across exactly 4 optimizers (Adam, RMSprop, SGD, Adagrad). 
* **Output:** Saves the trained networks to `/models` and plots `optimizer_comparison.png` benchmarking the validation losses and mean absolute mapping errors.

### 📂 `notebooks/03_Inference.ipynb`
* **Purpose:** Loads the most optimal trained network architecture (Adam generally converges most accurately). Denormalizes the `[0,1]` float layer arrays into physical image pixel coordinates.
* **Output:** Visual overlays comparing the **Green Ground Truth Rectangles** versus the **Red Predicted Overlay Rectangles** side-by-side using `matplotlib` and `cv2`.

---
<div align="center">
  <i>Developed and optimized for Computer Vision Portfolio implementation.</i>
</div>
