import cv2
import numpy as np
import torch

def preprocess_image(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float32) / 255.0

    bw = (img > 0.1).astype(np.uint8) * 255
    coords = cv2.findNonZero(bw)

    if coords is None:
        return torch.zeros((1, 1, 28, 28))

    x, y, w, h = cv2.boundingRect(coords)

    pad = int(0.2 * max(w, h))
    x0 = max(x - pad, 0)
    y0 = max(y - pad, 0)
    x1 = min(x + w + pad, img.shape[1])
    y1 = min(y + h + pad, img.shape[0])

    digit = img[y0:y1, x0:x1]
    digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)

    
    return torch.tensor(digit).unsqueeze(0).unsqueeze(0)
