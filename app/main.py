import base64
import cv2
import numpy as np
import os
import subprocess
import sys
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from inference.preprocess import preprocess
from inference.predict import Predictor
from utils.config import MODEL_PATH

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "data/self_data"

predictor = None
model_loaded = False

if os.path.exists(MODEL_PATH):
    predictor = Predictor()
    model_loaded = True
else:
    print("Warning: Model not found at startup. Train first.")

class PredictRequest(BaseModel):
    image: str

def decode_image(b64_str: str) -> np.ndarray:
    encoded_data = b64_str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

@app.post("/predict")
async def predict(req: PredictRequest):
    if not model_loaded or predictor is None:
        return {"error": "Model not loaded"}
    
    img = decode_image(req.image)
    processed = preprocess(img)
    pred, conf = predictor.predict_with_confidence(processed)
    return {"prediction": pred, "confidence": conf}

class SaveRequest(BaseModel):
    image: str
    label: str

@app.post("/save")
async def save(req: SaveRequest):
    label = req.label.upper()
    if len(label) != 1 or not label.isalpha():
        return {"error": "Invalid label"}
        
    save_dir = os.path.join(DATA_DIR, label)
    os.makedirs(save_dir, exist_ok=True)
    
    img = decode_image(req.image)
    processed = preprocess(img)
    idx = len(os.listdir(save_dir))
    np.save(os.path.join(save_dir, f"{idx}.npy"), processed)
    return {"message": f"Saved sample to {label}/{idx}.npy"}

@app.post("/retrain")
async def retrain():
    try:
        subprocess.run([sys.executable, "-m", "model.train"], cwd=os.getcwd(), check=True)
        global predictor, model_loaded
        predictor = Predictor()
        model_loaded = True
        return {"message": "Model retrained and loaded successfully"}
    except Exception as e:
        return {"error": str(e)}

# Serve static files last to not override APIs
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
