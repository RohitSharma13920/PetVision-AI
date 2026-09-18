# 🐾 PetVision AI — Neural Vision Core & HUD

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C.svg)](https://pytorch.org/)
[![TailwindCSS](https://img.shields.io/badge/UI-TailwindCSS-38B2AC.svg)](https://tailwindcss.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PetVision AI is an interactive computer vision application featuring real-time image recognition, live webcam scanning, a Sci-Fi cyberpunk HUD, and synthesized voice feedback. 

Initially developed as a custom CNN classifier, the core engine has been upgraded to a **1000-class ImageNet-trained MobileNetV2 architecture**, enabling rapid inference across animals, birds, vehicles, humans, and everyday objects.

---

## ⚡ Key Features

* **1,000-Class Universal Detection:** Identifies a wide variety of wildlife (lions, elephants, tigers), domestic pets (dogs, cats), vehicles, humans, and items via PyTorch `MobileNetV2`.
* **Cyber HUD & Laser Scanner:** Interactive UI built with Tailwind CSS featuring dynamic reticle overlays, animated laser scanlines, and glow aesthetics.
* **Live Webcam & File Upload Modes:** Toggle between local image upload and real-time webcam frame extraction.
* **Top-3 Neural Breakdown:** Displays top 3 class predictions with calculated probability distribution bars and inference latency metrics (in milliseconds).
* **AI Voice Synthesis:** Built-in audio readout utilizing the Web Speech API and procedural audio feedback via Web Audio API.
* **Asynchronous Backend:** High-throughput API built with FastAPI and Uvicorn.

---

## 🛠️ Tech Stack

* **Machine Learning Engine:** PyTorch, Torchvision (`MobileNet_V2`)
* **Backend:** FastAPI, Uvicorn, Pydantic, Python Multipart
* **Frontend:** Vanilla JavaScript (ES6+), Tailwind CSS, Web Speech API, Web Audio API
* **Image Processing:** Pillow (PIL)

---

## 📁 Project Structure

```text
cats_and_dogs_data/
├── backend/
│   └── main.py              # FastAPI inference pipeline & static file mounting
├── frontend/
│   ├── index.html           # Sci-Fi HUD interface
│   └── app.js               # Webcam capture, API calls, and voice logic
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
🚀 Quick Start
1. Clone the Repository
Bash
git clone [https://github.com/RohitSharma13920/PetVision-AI.git](https://github.com/RohitSharma13920/PetVision-AI.git)
cd PetVision-AI
2. Install Dependencies
Bash
py -m pip install -r requirements.txt
3. Run the Application
Bash
py -m uvicorn main:app --app-dir backend --reload --port 8000
Open your browser and navigate to:

Plaintext
[http://127.0.0.1:8000](http://127.0.0.1:8000)
🔌 API Reference
Inference Endpoint
URL: /predict

Method: POST

Payload: multipart/form-data (file: image)

Response Sample:

JSON
{
  "primary_class": "African Lion",
  "confidence": 91.4,
  "latency_ms": 38.2,
  "top_3": [
    { "label": "African Lion", "confidence": 91.4 },
    { "label": "Cheetah", "confidence": 5.1 },
    { "label": "Leopard", "confidence": 2.3 }
  ]
}
