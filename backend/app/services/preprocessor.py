import cv2
import numpy as np


def preprocess(image_path: str, max_dimension: int = None) -> tuple[np.ndarray, np.ndarray]:
    """Load image at full resolution and apply adaptive binarization.

    max_dimension is accepted for API compatibility but ignored — the pipeline
    always processes at the original DPI so pixel thresholds stay meaningful.

    Returns (original_bgr, binary) where binary: lines=white(255), bg=black(0).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]
    print(f"[Preprocess] Full resolution: {w}×{h}")
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = _binarize(gray)
    return img, binary


def _binarize(gray: np.ndarray) -> np.ndarray:
    """Adaptive Gaussian thresholding with OTSU fallback.

    Adaptive handles scanned drawings with uneven illumination.
    OTSU is preferred when adaptive picks up significantly more noise
    (detected via fill-ratio comparison), which happens on clean vector exports.

    BINARY_INV: dark ink on white paper → white(255) in output (lines=white).
    """
    h, w = gray.shape[:2]

    # block_size must be odd and scale with image resolution
    raw        = max(31, min(h, w) // 100)
    block_size = raw if raw % 2 == 1 else raw + 1

    adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size, 7
    )
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    a_fill = np.count_nonzero(adaptive) / adaptive.size
    o_fill = np.count_nonzero(otsu)     / otsu.size

    if o_fill > 0 and a_fill > o_fill * 2.0:
        binary = otsu
        print(f"[Preprocess] OTSU selected  (adaptive {a_fill:.1%} vs OTSU {o_fill:.1%})")
    else:
        binary = adaptive
        print(f"[Preprocess] Adaptive Gaussian (block={block_size}, fill={a_fill:.1%})")

    # Remove isolated noise pixels (1-2px specks from compression artefacts)
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    )
    # Close tiny breaks inside line strokes so CC extraction sees whole segments
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )
    return binary
