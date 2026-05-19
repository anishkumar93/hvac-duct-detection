"""Stage 2 — Image Preprocessing.

Converts image to grayscale, applies binary thresholding, and cleans noise.
"""
import cv2
import numpy as np


def preprocess(image_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load image and produce cleaned binary for line extraction.
    Returns (original_bgr, grayscale, binary_inv).
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # OTSU binary threshold (lines=white on black background)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological open to remove tiny noise (dots, speckles)
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open, iterations=1)

    # Close small gaps in lines
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close, iterations=1)

    return img, gray, binary
