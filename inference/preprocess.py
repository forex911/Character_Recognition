import cv2
import numpy as np
from utils.config import IMG_SIZE

def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(thresh)
    x, y, w, h = cv2.boundingRect(coords)

    cropped = thresh[y:y+h, x:x+w]

    padded = cv2.copyMakeBorder(
        cropped, 20, 20, 20, 20,
        cv2.BORDER_CONSTANT, value=0
    )

    resized = cv2.resize(padded, (28, 28))
    resized = cv2.bitwise_not(resized)
    resized = resized / 255.0

    return resized.reshape(1, 1, 28, 28).astype("float32")

