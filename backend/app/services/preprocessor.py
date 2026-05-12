import cv2
import numpy as np


def preprocess(image_path: str, max_dimension: int = None) -> tuple[np.ndarray, np.ndarray]:
    """Load and preprocess image for detection. Returns (original, processed binary).
    Set max_dimension=None to process at full resolution.
    """
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # Downscale only if max_dimension is set
    scale = 1.0
    if max_dimension and max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        print(f"[Preprocess] Downscaled from {w}x{h} to {img.shape[1]}x{img.shape[0]}")
    else:
        print(f"[Preprocess] Processing at full resolution: {w}x{h}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # For mechanical drawings: use binary threshold (lines are dark on white)
    # OTSU works well for clean engineering drawings
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove very thin noise (text, hatching) with morphological opening
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # Close small gaps in duct lines
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    return img, binary


def deskew(image: np.ndarray) -> np.ndarray:
    """Correct slight rotation in scanned drawings."""
    coords = np.column_stack(np.where(image > 0))
    if len(coords) < 5:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.5:
        return image
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
