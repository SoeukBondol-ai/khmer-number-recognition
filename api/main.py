import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

API_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = API_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.src.common.model_lenet import LeNet
from api.src.common.model_resnet import resnet18 as ResNetModel
from api.src.utils.lenet_preprocess import preprocess_image
from api.src.utils.resnet_preprocess import preprocess_digit

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

WEIGHTS_DIR = API_DIR / "model"
LENET_WEIGHTS = WEIGHTS_DIR / "lenet" / "lenet.pth"
RESNET_WEIGHTS = WEIGHTS_DIR / "resnet" / "resnet_digits.pth"

if not LENET_WEIGHTS.exists():
    raise FileNotFoundError(f"Missing LeNet weights at {LENET_WEIGHTS}")

if not RESNET_WEIGHTS.exists():
    raise FileNotFoundError(f"Missing ResNet weights at {RESNET_WEIGHTS}")

model_lenet = LeNet().to(device)
model_lenet.load_state_dict(torch.load(LENET_WEIGHTS, map_location=device))
model_lenet.eval()

model_resnet = ResNetModel(num_classes=10).to(device)
model_resnet.load_state_dict(torch.load(RESNET_WEIGHTS, map_location=device))
model_resnet.eval()

# -------------------------
# API endpoint
# -------------------------
@app.post("/predict/lenet")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)

    img_tensor = preprocess_image(img).to(device)

    with torch.no_grad():
        output = model_lenet(img_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    pred = int(probs.argmax())

    return {
        "digit": pred,
        "khmer": KHMER_DIGITS[pred],
        "confidence": float(probs[pred]),
        "probabilities": probs.tolist()
    }

@app.post("/predict/resnet")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    img = preprocess_digit(content).to(device)

    with torch.no_grad():
        output = model_resnet(img)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())

    return {
        "digit": pred,
        "khmer": KHMER_DIGITS[pred],
        "confidence": float(probs[pred]),
        "probabilities": probs.tolist()
    }