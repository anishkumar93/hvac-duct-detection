"""PaddleOCR tiled processing with parallel workers.

Splits large images into tiles, processes each with PaddleOCR in separate processes,
deduplicates overlapping results.

Usage:
    from app.services.paddle_ocr import paddle_ocr_tiled
    results = paddle_ocr_tiled(image, roi=(x1, y1, x2, y2))
"""
import os
import re
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import set_start_method

# Configuration
TILE_W = int(os.environ.get('PADDLE_TILE_W', 1000))
TILE_H = int(os.environ.get('PADDLE_TILE_H', 600))
OVERLAP = int(os.environ.get('PADDLE_OVERLAP', 150))
WORKERS = int(os.environ.get('PADDLE_WORKERS', 2))

# Ensure spawn method for macOS compatibility
try:
    set_start_method('spawn', force=True)
except RuntimeError:
    pass


class PaddleOCRResult:
    def __init__(self, text: str, bbox: list, confidence: float):
        self.text = text
        self.bbox = bbox
        self.confidence = confidence
        self.center_x = sum(p[0] for p in bbox) / len(bbox)
        self.center_y = sum(p[1] for p in bbox) / len(bbox)


def _process_batch(batch):
    """Process a batch of tiles in a worker process."""
    import paddle
    paddle.set_flags({
        'FLAGS_allocator_strategy': 'auto_growth',
        'FLAGS_use_mkldnn': True,
    })
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang='en',
    )

    results = []
    for tile_data, offset_x, offset_y in batch:
        page_results = list(ocr.predict(tile_data))
        for page in page_results:
            texts = page.get('rec_texts', [])
            scores = page.get('rec_scores', [])
            polys = page.get('dt_polys', [])
            for text, score, poly in zip(texts, scores, polys):
                if score < 0.3 or not text.strip():
                    continue
                # Convert tile coords to image coords
                bbox = [
                    [int(poly[0][0]) + offset_x, int(poly[0][1]) + offset_y],
                    [int(poly[1][0]) + offset_x, int(poly[1][1]) + offset_y],
                    [int(poly[2][0]) + offset_x, int(poly[2][1]) + offset_y],
                    [int(poly[3][0]) + offset_x, int(poly[3][1]) + offset_y],
                ]
                results.append((text.strip(), float(score), bbox))

    return results


def paddle_ocr_tiled(image: np.ndarray, roi: tuple = None) -> list[PaddleOCRResult]:
    """Run PaddleOCR on an image using tiled parallel processing.

    Args:
        image: BGR image (full resolution)
        roi: (x1, y1, x2, y2) region of interest, or None for full image

    Returns:
        List of PaddleOCRResult with text, bbox, confidence
    """
    if roi:
        x1, y1, x2, y2 = roi
        crop = image[y1:y2, x1:x2]
        offset_x, offset_y = x1, y1
    else:
        crop = image
        offset_x, offset_y = 0, 0

    crop_h, crop_w = crop.shape[:2]

    # Generate tiles
    tiles = []
    for ty in range(0, crop_h - OVERLAP, TILE_H - OVERLAP):
        for tx in range(0, crop_w - OVERLAP, TILE_W - OVERLAP):
            tx2 = min(tx + TILE_W, crop_w)
            ty2 = min(ty + TILE_H, crop_h)
            tiles.append((tx, ty, tx2, ty2))

    if not tiles:
        return []

    print(f"[PaddleOCR] {len(tiles)} tiles ({TILE_W}x{TILE_H}), {WORKERS} workers")

    # Split into batches per worker
    batches = [[] for _ in range(WORKERS)]
    for i, (tx1, ty1, tx2, ty2) in enumerate(tiles):
        tile_data = crop[ty1:ty2, tx1:tx2].copy()
        batches[i % WORKERS].append((tile_data, tx1 + offset_x, ty1 + offset_y))

    # Process in parallel
    all_results = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(_process_batch, batch) for batch in batches]
        for future in futures:
            all_results.extend(future.result())

    # Deduplicate overlapping results
    deduped = _deduplicate(all_results)

    print(f"[PaddleOCR] {len(all_results)} raw → {len(deduped)} after dedup")

    # Convert to PaddleOCRResult objects
    results = []
    for text, score, bbox in deduped:
        results.append(PaddleOCRResult(text=text, bbox=bbox, confidence=score))

    return results


def _deduplicate(results: list, dist_thresh: int = 30) -> list:
    """Remove duplicate detections from overlapping tiles.
    Keeps higher confidence version when two detections are at the same position.
    """
    # Sort by confidence descending
    sorted_results = sorted(results, key=lambda r: -r[1])
    kept = []

    for text, score, bbox in sorted_results:
        cx = sum(p[0] for p in bbox) / 4
        cy = sum(p[1] for p in bbox) / 4
        is_dup = False
        for _, _, kbbox in kept:
            kcx = sum(p[0] for p in kbbox) / 4
            kcy = sum(p[1] for p in kbbox) / 4
            if abs(cx - kcx) < dist_thresh and abs(cy - kcy) < dist_thresh:
                is_dup = True
                break
        if not is_dup:
            kept.append((text, score, bbox))

    return kept
