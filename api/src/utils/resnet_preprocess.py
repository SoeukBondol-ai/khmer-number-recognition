import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

resnet_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

def preprocess_digit(img_path):

    nparr = np.frombuffer(img_path, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, binary = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        digit = binary[y:y+h, x:x+w]
    else:
        digit = binary
    digit = cv2.resize(digit, (180,180))
    digit = cv2.copyMakeBorder(
        digit, 22,22,22,22,
        cv2.BORDER_CONSTANT, value=0
    )
    digit = cv2.cvtColor(digit, cv2.COLOR_GRAY2RGB)


    digit = Image.fromarray(digit)
    digit = resnet_transform(digit).unsqueeze(0)

    return digit