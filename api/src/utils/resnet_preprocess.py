import cv2

def preprocess_digit(img_path):
    img = cv2.imread(img_path)
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

    return digit