import io
import time
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# 1. Initialize Application
app = FastAPI(title="PetVision Cyber HUD Core", version="4.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load Lightweight MobileNetV2 Model (ImageNet 1000 classes)
print("Loading Neural Vision Weights...")
weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights)
model.eval()
preprocess = weights.transforms()
categories = weights.meta["categories"]
print("Neural Engine Loaded Successfully!")

# Extended list of human-associated clothing, gear, and wearables in ImageNet
HUMAN_INDICATORS = {
    "seat belt", "lab coat", "suit", "jersey", "academic gown",
    "trench coat", "military uniform", "sweatshirt", "cardigan",
    "cloak", "abaya", "kimono", "bulletproof vest", "bow tie",
    "sunglasses", "wig", "stethoscope", "neck brace", "apron",
    "jean", "pajamas", "mortarboard"
}

# 3. Pydantic Response Schemas
class PredictionItem(BaseModel):
    label: str
    confidence: float

class PredictionResponse(BaseModel):
    primary_class: str
    confidence: float
    latency_ms: float
    top_3: List[PredictionItem]

def format_entity_label(raw_name: str) -> str:
    clean_name = raw_name.split(",")[0].replace("_", " ").title()
    name_lower = clean_name.lower()

    # Intercept human attire/accessories and classify as Human
    if any(indicator in name_lower for indicator in HUMAN_INDICATORS):
        return "👤 Human (Person)"
    
    # Animals & Specific Entity Emoji Badging
    if any(w in name_lower for w in ["lion", "tiger", "cheetah", "leopard", "jaguar"]):
        return f"🦁 {clean_name}"
    if any(w in name_lower for w in ["dog", "puppy", "terrier", "retriever", "hound", "husky"]):
        return f"🐶 {clean_name}"
    if any(w in name_lower for w in ["cat", "kitten", "tabby", "siamese"]):
        return f"🐱 {clean_name}"
    if any(w in name_lower for w in ["python", "snake", "boa", "viper", "cobra"]):
        return f"🐍 {clean_name}"
    if any(w in name_lower for w in ["elephant"]):
        return f"🐘 {clean_name}"
    if any(w in name_lower for w in ["car", "cab", "convertible", "jeep", "sports car"]):
        return f"🚗 {clean_name}"
    
    return f"🏷️ {clean_name}"

# 4. Inference Route
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file provided.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        start_time = time.perf_counter()

        # Preprocess and forward pass
        input_tensor = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output[0], dim=0)

        # Retrieve top 3 predictions
        top_probs, top_catids = torch.topk(probabilities, 3)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        top_3_results: List[PredictionItem] = []
        for prob, catid in zip(top_probs, top_catids):
            raw_label = categories[catid.item()]
            formatted_label = format_entity_label(raw_label)
            top_3_results.append(PredictionItem(
                label=formatted_label,
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

# 5. Serve Frontend Static Assets
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)