import cv2
import numpy as np
import torch

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