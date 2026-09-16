import io
import os
import time
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

app = FastAPI(title="PetVision AI - Advanced PyTorch CNN", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

# 1. 4-Stage Architecture matching training
class PetVisionCNN(nn.Module):
    def __init__(self):
        super(PetVisionCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * 2 * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PetVisionCNN().to(device)

weights_path = os.path.join(PROJECT_ROOT, "models", "cat_dog_cnn.pth")
if not os.path.exists(weights_path):
    weights_path = "models/cat_dog_cnn.pth"

model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    resized_img = pil_image.resize((128, 128))
    img_arr = np.array(resized_img, dtype=np.float32) / 255.0
    img_arr = np.transpose(img_arr, (2, 0, 1))

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    normalized_arr = (img_arr - mean) / std

    return torch.tensor(normalized_arr, dtype=torch.float32).unsqueeze(0).to(device)

class ClassificationResponse(BaseModel):
    label: str
    confidence: float
    cat_probability: float
    dog_probability: float
    verdict: str
    latency_ms: float

@app.post("/api/predict", response_model=ClassificationResponse)
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image (PNG/JPG).")

    try:
        start_time = time.perf_counter()
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor_image = preprocess_image(pil_image)

        with torch.no_grad():
            outputs = model(tensor_image)
            probs = torch.softmax(outputs, dim=1)[0]
            cat_prob = round(float(probs[0].item()) * 100, 2)
            dog_prob = round(float(probs[1].item()) * 100, 2)

        latency = round((time.perf_counter() - start_time) * 1000, 1)

        if cat_prob > dog_prob:
            label = "Cat"
            confidence = cat_prob
            verdict = f"Feline Detected ({cat_prob}%)"
        else:
            label = "Dog"
            confidence = dog_prob
            verdict = f"Canine Detected ({dog_prob}%)"

        return ClassificationResponse(
            label=label,
            confidence=confidence,
            cat_probability=cat_prob,
            dog_probability=dog_prob,
            verdict=verdict,
            latency_ms=latency
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
def serve_home():
    with open(os.path.join(FRONTEND_DIR, "index.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.get("/app.js")
def serve_js():
    with open(os.path.join(FRONTEND_DIR, "app.js"), "r", encoding="utf-8") as f:
        return Response(f.read(), media_type="application/javascript")