import base64
import io
import numpy as np
from PIL import Image

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

from keras.models import load_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

model1 = None
model2 = None
model3 = None

def load_models():
    global model1, model2, model3

    if model1 is None:

        print("Loading models...")

        model1 = load_model("models/snake_vs_nonsnake_best.keras", compile=False, safe_mode=False)
        model2 = load_model("models/valid_vs_invalid_best.keras", compile=False, safe_mode=False)
        model3 = load_model( "models/invalid_reason_best.keras", compile=False, safe_mode=False)

        print("✅ Models loaded")

def preprocess(img):
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/classify-image")
async def classify(request: Request):

    try:
        load_models()

        data = await request.json()
        image_base64 = data.get("image_base64")

        if not image_base64:
            return {"error": "No image provided"}

        # Handle base64 prefix
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        image_bytes = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        x = preprocess(img)

        # Stage 1
        p1 = model1.predict(x, verbose=0)
        idx1 = int(np.argmax(p1))

        if idx1 == 0:
            return {"final": "Non snake image"}

        # Stage 2
        p2 = model2.predict(x, verbose=0)
        idx2 = int(np.argmax(p2))

        if idx2 == 1:
            return {"final": "Valid snake image"}

        # Stage 3
        p3 = model3.predict(x, verbose=0)
        idx3 = int(np.argmax(p3))

        labels = ["Blur", "Dark", "Edited", "Noisy"]

        return {"final": f"Invalid snake image ({labels[idx3]})"}

    except Exception as e:
        print("🔥 ERROR:", str(e))
        return {"error": str(e)}