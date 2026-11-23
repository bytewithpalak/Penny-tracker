"""OpenCV preprocessing: grayscale, resize, denoise, deskew, optional binarize."""
import os

import cv2
import numpy as np

MIN_SIDE = 1000   # upscale images whose longest side is below this
MAX_SIDE = int(os.getenv('OCR_MAX_SIDE', '1800'))   # downscale big photos: OCR time grows with pixel count (set OCR_MAX_SIDE=2500 for the old behaviour)


def preprocess(img_bgr, binarize=False):
    """Return a cleaned grayscale image (numpy uint8)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    longest = max(gray.shape)
    if longest < MIN_SIDE:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    elif longest > MAX_SIDE:
        scale = MAX_SIDE / longest
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 5, 11)    # lighter settings: ~3x faster, same OCR result in tests
    gray = deskew(gray)
    if binarize:
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 15)
    return gray


def deskew(gray):
    """Rotate the image so text lines are horizontal. Ignores tiny or suspicious angles."""
    inv = cv2.bitwise_not(gray)
    _, th = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(th > 0))
    if len(coords) < 100:
        return gray
    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.5 or abs(angle) > 15:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)
