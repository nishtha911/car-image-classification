# Car Object Detection — ResNet-50 Bounding Box Regression

**Assignment:** Car Object Detection using ConvNet / CNN (ResNet-50)  
**Dataset:** [Kaggle — Car Object Detection](https://www.kaggle.com/datasets/sshikamaru/car-object-detection)  
**Stack:** TensorFlow · Keras · ResNet-50 · GIoU Loss

---

## Project Structure

```
car-object-detection/
├── data/
│   ├── train/                         # 1000 training images
│   ├── test/                          # test images
│   └── train_solution_bounding_boxes.csv
│
├── training/
│   ├── data_generator.py              # ✅ Fixed - groups multi-box images
│   ├── giou_loss.py                   # ✅ New  - GIoU + IoU metric
│   └── resnet50_train.py              # ✅ New  - trains 4 optimisers, plots results
│
├── models/                            # saved .keras / .h5 files
├── evaluation/
│   ├── evaluate.py                    # ✅ New  - IoU report + prediction grids
│   ├── iou_distribution.png
│   ├── predict_grid_best.png
│   ├── predict_grid_worst.png
│   ├── optimizer_comparison.png
│   └── evaluation_report.json
│
├── app/
│   └── api.py                         # ✅ Fixed - Flask REST API with denormalization
│
├── logs/                              # per-optimiser CSV training logs
├── requirements.txt
└── README.md
```

---

## Why Were the Bounding Boxes Inaccurate?

Five root causes were identified and fixed:

| # | Root Cause | Fix |
|---|-----------|-----|
| 1 | **Multi-box CSV mishandled** — same image appeared N times with N different targets, giving the model contradictory training signals | `DataGenerator` now groups by image and picks the **largest bounding box** per image |
| 2 | **MSE loss** doesn't penalise spatial misalignment | Replaced with **GIoU loss** that directly maximises intersection-over-union |
| 3 | **No denormalisation** during inference | `api.py` scales `[0,1]` predictions back to pixel coordinates using original image dimensions |
| 4 | **No output clamping** | `np.clip(pred, 0, 1)` before denormalisation prevents off-screen boxes |
| 5 | **No inference pipeline** (`api.py` was empty) | Full Flask REST API with CLI demo mode |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train all optimisers and generate comparison charts
```bash
python training/resnet50_train.py
```
This trains with **SGD, RMSprop, Adagrad, Adam** and saves:
- `models/resnet50_<optimizer>.keras`
- `evaluation/optimizer_comparison.png`

### 3. Evaluate the best model
```bash
python evaluation/evaluate.py
```
Outputs: IoU distribution chart, best/worst prediction grids, JSON report.

### 4. Run the API server
```bash
python app/api.py
```

### 5. Predict on one image (CLI)
```bash
python app/api.py data/test/vid_5_25100.jpg
```

### 6. Predict via HTTP
```bash
curl -X POST http://localhost:5000/predict -F "image=@data/test/vid_5_25100.jpg"
```

---

## Model Architecture

```
Input (224×224×3)
  ↓
ResNet-50 backbone (ImageNet weights)   ← frozen in Phase 1
  ↓                                       ← conv5 block unfrozen in Phase 2
GlobalAveragePooling2D
  ↓
Dense(256) → BatchNorm → ReLU → Dropout(0.3)
  ↓
Dense(128) → BatchNorm → ReLU → Dropout(0.2)
  ↓
Dense(4, sigmoid)    →  [xmin, ymin, xmax, ymax]  in [0, 1]
```

### Training Strategy
- **Phase 1** — backbone frozen, train regression head only (15 epochs, early stopping)
- **Phase 2** — unfreeze `conv5_block*` layers, fine-tune at 10× lower LR

### Loss Function — GIoU
$$\mathcal{L}_{GIoU} = 1 - \text{GIoU} = 1 - \left(\text{IoU} - \frac{|\mathcal{C} \setminus (A \cup B)|}{|\mathcal{C}|}\right)$$

where $\mathcal{C}$ is the smallest enclosing box of the predicted and ground-truth boxes.

---

## API Reference

### `GET /health`
```json
{"status": "ok", "model": "resnet50_adam.keras"}
```

### `POST /predict`
Body: `multipart/form-data` with field `image` (JPEG/PNG).  
Optional: `draw=1` to receive annotated image as base64.

```json
{
  "bbox_normalized": [0.21, 0.28, 0.64, 0.87],
  "bbox_pixels":     [143, 100, 435, 313],
  "image_size":      {"width": 676, "height": 360},
  "model_used":      "resnet50_adam.keras"
}
```

### `GET /predict/demo`
Returns a JPEG image with the predicted bounding box drawn on a random test image.
