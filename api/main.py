from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import torch
import torch.nn as nn
from model import LeNet
from ..api.src.utils.lenet_preprocess import preprocess_image

# -------------------------
# Khmer digit mapping
# -------------------------
KHMER_DIGITS = {
    0: "០", 1: "១", 2: "២", 3: "៣", 4: "៤",
    5: "៥", 6: "៦", 7: "៧", 8: "៨", 9: "៩"
}

# -------------------------
# FastAPI setup
# -------------------------
app = FastAPI(title="Khmer Digit Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LeNet().to(device)
model.load_state_dict(torch.load("models/lenet/lenet.pth", map_location=device))
model.eval()

# -------------------------
# API endpoint
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)

    img_tensor = preprocess_image(img).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    pred = int(probs.argmax())

    return {
        "digit": pred,
        "khmer": KHMER_DIGITS[pred],
        "confidence": float(probs[pred]),
        "probabilities": probs.tolist()
    }
