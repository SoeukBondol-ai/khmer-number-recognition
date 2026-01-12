from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Khmer digit mapping
# -------------------------
KHMER_DIGITS = {
    0: "០", 1: "១", 2: "២", 3: "៣", 4: "៤",
    5: "៥", 6: "៦", 7: "៧", 8: "៨", 9: "៩"
}

# -------------------------
# LeNet 
# -------------------------
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# -------------------------
# Image preprocessing 
# -------------------------
def preprocess_image(img: np.ndarray) -> torch.Tensor:
    # Ensure grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binary threshold: white digit on black bg
    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

    # Find digit bounding box
    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y+h, x:x+w]

    # Pad to square
    h, w = img.shape
    size = max(h, w)
    padded = np.zeros((size, size), dtype=np.uint8)
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    padded[y_offset:y_offset+h, x_offset:x_offset+w] = img

    # Resize to 28x28
    img = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalize
    img = img.astype(np.float32) / 255.0

    # Shape: (1, 1, 28, 28)
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0)

    return img

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
