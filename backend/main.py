import io
import time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

app = FastAPI(title="PetVision Neural HUD", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load lightweight MobileNetV2
weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights)
model.eval()
preprocess = weights.transforms()
categories = weights.meta["categories"]

class PredictionItem(BaseModel):
    label: str
    confidence: float

class PredictionResponse(BaseModel):
    primary_class: str
    confidence: float
    latency_ms: float
    top_3: List[PredictionItem]

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        start_time = time.perf_counter()

        input_tensor = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output[0], dim=0)

        # Get Top 3 Predictions
        top_probs, top_catids = torch.topk(probabilities, 3)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        top_3_results = []
        for prob, catid in zip(top_probs, top_catids):
            raw_label = categories[catid.item()]
            clean_name = raw_label.split(",")[0].replace("_", " ").title()
            top_3_results.append(PredictionItem(
                label=clean_name,
                confidence=round(prob.item() * 100, 1)
            ))

        primary = top_3_results[0]
        return PredictionResponse(
            primary_class=primary.label,
            confidence=primary.confidence,
            latency_ms=latency_ms,
            top_3=top_3_results
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")