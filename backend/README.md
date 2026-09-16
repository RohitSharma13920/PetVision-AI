# 🐾 PetVision AI: Deep CNN Pet Classifier

An end-to-end computer vision web application that classifies images of domestic cats and dogs in real-time. Powered by a custom **4-Stage Deep Convolutional Neural Network (CNN)** with **Adaptive Global Average Pooling** built in PyTorch and deployed via a lightweight asynchronous **FastAPI** backend with a modern dark-mode UI.

---

## 📌 Architecture Highlights

Unlike basic tutorial networks that suffer from spatial overfitting and background bias, PetVision AI incorporates modern architectural design:

```text
Input (128x128x3 RGB)
       │
   [Conv2D (32) + BatchNorm + ReLU + MaxPool]
       │
   [Conv2D (64) + BatchNorm + ReLU + MaxPool]
       │
   [Conv2D (128) + BatchNorm + ReLU + MaxPool]
       │
   [Conv2D (128) + BatchNorm + ReLU]
       │
   [AdaptiveAvgPool2d (2x2)]  <-- Mitigates background correlation
       │
   [Flatten + Dropout(0.3)]
       │
   [Dense(64) + ReLU + Dropout(0.2)]
       │
   [Dense(2) -> Softmax Probabilities]
   Resolution: 128 × 128 RGB Image Tensor

Data Augmentation: Random Horizontal Flips & Image Normalization

Inference Latency: ~25ms – 45ms per image (CPU inference)

Deployment: Zero external runtime image libraries (torchvision free deployment pipeline)

🚀 Key Features
Real-time Forward Pass: Instant inference with confidence probability breakdown.

Interactive UI: Drag-and-drop file uploader with live preview and inference latency telemetry.

Robust Backend: Asynchronous REST API with structured response schemas via Pydantic.

Resilient to Spurious Correlation: Trained on balanced datasets to eliminate classification bias.

🛠️ Project Structure
├── backend/
│   └── main.py          # FastAPI application & model inference pipeline
├── frontend/
│   ├── index.html       # Tailwind CSS UI
│   └── app.js           # Client-side file handling & async fetch logic
├── models/
│   └── cat_dog_cnn.pth  # Serialized PyTorch state dictionary
├── train.py             # Custom Dataset loader & balanced CNN training script
├── requirements.txt     # Production dependencies
└── .gitignore