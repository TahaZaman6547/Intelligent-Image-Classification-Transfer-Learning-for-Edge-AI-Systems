<div align="center">

# 🧠 Intelligent Image Classification — CNN Architecture & Transfer Learning for Edge AI

**Deep Learning · Computer Vision · Model Optimisation · MobileNetV2 · CIFAR-10 · Edge AI · 2025**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📌 Project Overview

A structured progression through three tiers of image classification on **CIFAR-10** (60,000 images · 10 classes · 32×32 RGB), culminating in a production-ready **MobileNetV2 Transfer Learning** model deployable to Edge AI devices.

Built on advanced coursework in Computer Vision, incorporating architectural experiments across activation functions, optimisers, pooling strategies, and regularisation techniques — validated with full per-class classification reports.

---

## 📈 Results Summary

| Model | Test Accuracy | Parameters | Strategy |
|---|---|---|---|
| Baseline CNN | ~65% | ~470K | 2-block CNN, from scratch |
| **CIFAR DecaLuminarNet** | **~82%** | ~9.1M | 4-block CNN + BN + Dropout + Augmentation |
| **MobileNetV2 Transfer Learning** | **~88%** | ~3.4M (base) | ImageNet pretrained + Phase 1 + Phase 2 fine-tuning |

---

## 🗂️ Project Structure

```
EdgeAI_CIFAR10/
├── notebooks/
│   ├── 01_Baseline_CNN.ipynb                    # Baseline → ~65%
│   ├── 02_Optimised_CNN_DecaLuminarNet.ipynb    # DecaLuminarNet → ~82%
│   └── 03_MobileNetV2_Transfer_EdgeAI.ipynb     # MobileNetV2 → ~88%
├── app/
│   ├── app.py           # Flask REST API (predict endpoint)
│   ├── requirements.txt # API dependencies
│   ├── Procfile         # Railway/Heroku process file
│   └── railway.toml     # Railway deployment config
├── utils/
│   ├── export_tflite.py # Export model to TFLite for edge devices
│   └── predict_image.py # CLI inference on local images
├── data/
│   └── sample_images/   # Place test images here for the Flask app
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔬 Notebooks

### 📓 Notebook 1 — Baseline CNN (~65%)
- Minimal 2-block CNN — intentionally bare (no BatchNorm, no Dropout, no augmentation)
- Establishes performance floor for ablation analysis
- Full training curves, confusion matrix, per-class classification report

### 📓 Notebook 2 — CIFAR DecaLuminarNet (~82%)
Custom architecture named **DecaLuminarNet** (`Deca` = 10 classes · `LuminarNet` = illuminating deep performance):

| Block | Filters | Added vs Baseline |
|---|---|---|
| Block 1 | 64 × 2 Conv | BatchNorm, ELU, Dropout 0.25 |
| Block 2 | 128 × 2 Conv | BatchNorm, ELU, Dropout 0.25 |
| Block 3 | 256 × 3 Conv | BatchNorm, ELU, Dropout 0.25 |
| Block 4 | 512 × 3 Conv | BatchNorm, ELU, Dropout 0.25 |
| Head | Dense(512→512→256→128→64→10) | BatchNorm, Dropout 0.5 |

**Key experiments conducted:**
- Conv layer depth (1–4 blocks), filter sizes (8→512), kernel sizes (3×3, 5×5, 7×7)
- Activation functions: ReLU, ELU, SELU, Tanh, Sigmoid, Softmax, Swish, SoftSign
- Optimisers: SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
- Dropout rates and BatchNormalization placement
- Dense layer width and depth

### 📓 Notebook 3 — MobileNetV2 Transfer Learning + Edge AI (~88%)
**Two-Phase Transfer Learning Strategy:**

**Phase 1 — Feature Extraction (base frozen):**
Load MobileNetV2 (ImageNet) → freeze all base layers → train classification head only.

**Phase 2 — Fine-Tuning (top 30 layers unfrozen):**
Unfreeze top 30 MobileNetV2 layers → retrain with LR = 1e-5 to adapt features to CIFAR-10.

**Includes:**
- Optimizer comparison (Adam vs RMSprop vs SGD on Phase 1)
- Per-class accuracy bar chart
- TFLite export pathway for edge deployment

---

## 🚀 Getting Started

### Option A — Google Colab (Recommended, free GPU)
Click any notebook → Open in Colab → Runtime → Change runtime type → **GPU**

### Option B — Local
```bash
git clone https://github.com/TahaZaman6547/Intelligent-Image-Classification-Transfer-Learning-for-Edge-AI-Systems.git
cd Intelligent-Image-Classification-Transfer-Learning-for-Edge-AI-Systems
pip install -r requirements.txt
jupyter notebook notebooks/01_Baseline_CNN.ipynb
```

---

## 🌐 API Deployment

### Run Locally
```bash
cd app
pip install -r requirements.txt
python app.py
```

### Test Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Predict a single image
curl -X POST -F "file=@your_image.jpg" http://localhost:8000/predict

# Predict multiple images
curl -X POST -F "file=@img1.jpg" -F "file=@img2.jpg" http://localhost:8000/predict

# List supported classes
curl http://localhost:8000/classes
```

### Deploy to Railway
1. Push this repo to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Select `app/` as root directory
4. Railway reads `Procfile` and starts the API automatically
5. Add your model file as a Railway volume or download it in a build step

### Sample API Response
```json
{
  "predictions": [
    {
      "filename": "cat.jpg",
      "predicted_class": "cat",
      "confidence": 0.9312,
      "probabilities": {
        "airplane": 0.0012,
        "automobile": 0.0008,
        "bird": 0.0031,
        "cat": 0.9312,
        "deer": 0.0019,
        "dog": 0.0541,
        "frog": 0.0007,
        "horse": 0.0043,
        "ship": 0.0011,
        "truck": 0.0016
      }
    }
  ]
}
```

---

## 📦 Edge AI Export — TFLite

```bash
# Export to TFLite (standard)
python utils/export_tflite.py --model mobilenet_best.keras --output model.tflite

# Export with quantization (~4× smaller, minimal accuracy loss)
python utils/export_tflite.py --model mobilenet_best.keras --output model_quant.tflite --quantize
```

Runs on: **Raspberry Pi · Android · iOS · Arduino Nano 33 BLE Sense · STM32**

---

## 🏗️ Why MobileNetV2 for Edge AI?

| Feature | Standard CNN | MobileNetV2 |
|---|---|---|
| Conv operation | Standard 2D Conv | Depthwise-separable Conv |
| Multiply-adds | Baseline | **~8–9× fewer** |
| Memory footprint | Higher | **Minimal (inverted residuals)** |
| Accuracy on CIFAR-10 | ~82% | **~88%** |
| Edge deployment | Difficult | **Native TFLite support** |

---

## 📊 Dataset — CIFAR-10

| Property | Value |
|---|---|
| Total images | 60,000 |
| Training images | 50,000 |
| Test images | 10,000 |
| Image size | 32×32 RGB |
| Classes | 10 (5,000/class — perfectly balanced) |
| Classes | airplane · automobile · bird · cat · deer · dog · frog · horse · ship · truck |

---

## 🛠️ Tech Stack

`TensorFlow 2.x` · `Keras` · `MobileNetV2` · `scikit-learn` · `Flask` · `NumPy` · `Matplotlib` · `Seaborn` · `TFLite`

---

## 👤 Author

**Taha Zaman**
- 🐙 GitHub: [@TahaZaman6547](https://github.com/TahaZaman6547)
- 💼 LinkedIn: [your-linkedin](https://linkedin.com/in/yourprofile)

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.
