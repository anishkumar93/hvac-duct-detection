"""
HVAC Duct Detection — Geometry-first pipeline.

Stage 1  Scale calibration   title-block OCR → PPI
Stage 2  ROI detection       exclude title block and notes section
Stage 3  Binarization        adaptive Gaussian, full resolution (in preprocessor)
Stage 4-6 Geometry engine    CC segments + hollowness validation + stitching
Stage 7  Global OCR          one Tesseract pass for dimension labels
Stage 8  Optimal association Hungarian algorithm (scipy) with greedy fallback
Stage 9  Classification + cleanup + annotation
"""

import os
import re
import cv2
import numpy as np

from app.models.schemas import DetectionResult, DuctSegment, DuctType, BoundingBox
from app.services.preprocessor import preprocess
from app.services.geometry import detect_ducts_geometry, detect_drawing_roi
from app.services.ocr import extract_text, filter_dimensions, OCRResult
from app.services.associator import associate_labels_optimal
from app.services.classifier import classify_pressure
from app.services.annotator import annotate_image
from app.services.scale_extractor import (
    extract_scale, compute_pixels_per_inch, validate_duct_dimension,
)
from app.services.post_filters import (
    filter_by_context,
    validate_connectivity,
    filter_boundary_detections,
    validate_scale_unlabelled,
    compute_confidence_scores,
    filter_closed_short_corridors,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")


# ═════════════════════════════════════════════════════════════════════════════
# Public entry point
# ═════════════════════════════════════════════════════════════════════════════

def run_detection_pipeline(
    image_path: str,
    file_id: str,
    scale: str = None,
    pdf_path: str = None,
    resolution: int = None,           # accepted for API compat, ignored
) -> DetectionResult:
    """Run geometry-first HVAC duct detection at full image resolution."""
    print(f"\n[Pipeline] ── {image_path}")

    # Load full-resolution image (reference copy kept for annotation + OCR)
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Cannot read image: {image_path}")
    full_h, full_w = original.shape[:2]
    print(f"[Pipeline] Image: {full_w}×{full_h}")

    # ── Stage 1: Scale calibration ────────────────────────────────────────────
    drawing_scale, ppi = _calibrate_scale(original, full_w, full_h)

    # Caller-supplied scale string overrides OCR when title block is unreadable
    if scale and not drawing_scale:
        fake          = OCRResult(text=scale,
                                  bbox=[[0,0],[1,0],[1,1],[0,1]], confidence=0.95)
        drawing_scale = extract_scale([fake])
        if drawing_scale:
            ppi = compute_pixels_per_inch(drawing_scale, full_w, None)
            print(f"[Pipeline] Scale (caller): {drawing_scale['text']} → {ppi:.1f} px/in")

    # ── Stage 2 + 3: ROI detection and binarization ───────────────────────────
    _, binary   = preprocess(image_path)
    # Pass original (BGR) so Tesseract-based ROI detection can run at full res
    drawing_roi = detect_drawing_roi(binary, original=original)
    roi_x1, roi_y1, roi_x2, roi_y2 = drawing_roi

    # ── Stages 4-6: Geometry detection ───────────────────────────────────────
    duct_boxes = detect_ducts_geometry(binary, drawing_roi, ppi=ppi)
    print(f"[Pipeline] Geometry candidates: {len(duct_boxes)}")

    # ── Stage 7: Global OCR on drawing area ───────────────────────────────────
    # Runs once on the colour original — Tesseract works better on BGR/grey
    # than on an inverted binary.  The ROI excludes the title block and notes.
    ocr_results      = extract_text(original, roi=drawing_roi)
    dimension_labels = filter_dimensions(ocr_results)
    print(f"[Pipeline] Dimension labels: {len(dimension_labels)}")

    # ── Stage 8: Optimal label-to-duct association ────────────────────────────
    max_dist     = max(full_w, full_h) * 0.08
    associations = associate_labels_optimal(
        duct_boxes, dimension_labels, max_distance=max_dist
    )

    # ── Scale validation (soft gate — only rejects labelled ducts) ────────────
    if ppi:
        duct_boxes, associations = _scale_validate(duct_boxes, associations, ppi)
    print(f"[Pipeline] After scale validation: {len(duct_boxes)} ducts")

    # ── Post-detection false-positive filters ─────────────────────────────────
    pre_filter_count = len(duct_boxes)

    # Filter equipment boxes (nearly-square with internal structure)
    duct_boxes, associations = filter_closed_short_corridors(
        duct_boxes, associations, binary[roi_y1:roi_y2, roi_x1:roi_x2],
        roi_offset=(roi_x1, roi_y1),
    )

    # Remove detections near non-duct text (room labels, equipment tags)
    duct_boxes, associations = filter_by_context(
        duct_boxes, associations, ocr_results,
    )

    # Remove unlabelled ducts whose size doesn't match confirmed ducts
    duct_boxes, associations = validate_scale_unlabelled(
        duct_boxes, associations, ppi,
    )

    # Remove isolated detections not connected to any other duct
    duct_boxes, associations = validate_connectivity(
        duct_boxes, associations,
    )

    # Remove detections at ROI boundary (partial walls)
    duct_boxes, associations = filter_boundary_detections(
        duct_boxes, associations, drawing_roi,
    )

    post_filter_count = len(duct_boxes)
    if pre_filter_count != post_filter_count:
        print(f"[Pipeline] Post-filters: {pre_filter_count} → {post_filter_count} ducts")

    # ── Confidence scoring ────────────────────────────────────────────────────
    confidence_scores = compute_confidence_scores(
        duct_boxes, associations, ocr_results, ppi, drawing_roi,
    )

    # ── Stage 9: Build DuctSegment objects ────────────────────────────────────
    ducts = []
    for i, bbox in enumerate(duct_boxes):
        label    = associations.get(i)
        dim_text = label.text       if label else None
        conf     = confidence_scores[i] if i < len(confidence_scores) else 0.50
        ducts.append(DuctSegment(
            id=i + 1,
            duct_type=DuctType.UNKNOWN,
            dimension=dim_text,
            length=None,
            pressure_class=classify_pressure(dim_text),
            bbox=bbox,
            confidence=conf,
        ))

    # Bounds cleanup — remove any bbox that somehow escaped the image area
    ducts = [
        d for d in ducts
        if 0 < d.bbox.x < full_w
        and 0 < d.bbox.y < full_h
        and d.bbox.width  < full_w * 0.5
        and d.bbox.height < full_h * 0.5
    ]
    for i, d in enumerate(ducts):
        d.id = i + 1

    print(f"[Pipeline] Final duct count: {len(ducts)}")

    # Annotate and save
    out_dir = os.path.join(OUTPUT_DIR, file_id)
    os.makedirs(out_dir, exist_ok=True)
    annotate_image(original, ducts, os.path.join(out_dir, "annotated.png"))

    return DetectionResult(
        image_width=full_w,
        image_height=full_h,
        scale=drawing_scale["text"] if drawing_scale else scale,
        ducts=ducts,
        annotated_image_path=f"/outputs/{file_id}/annotated.png",
    )


# ═════════════════════════════════════════════════════════════════════════════
# Stage 1 helper — scale calibration
# ═════════════════════════════════════════════════════════════════════════════

def _calibrate_scale(
    image: np.ndarray, full_w: int, full_h: int
) -> tuple[dict | None, float | None]:
    """Try title-block crop regions to extract the drawing scale.

    Crop priority (ANSI/ISO title blocks are typically bottom-right):
      1. Bottom-right quarter
      2. Top-right quarter
      3. Full bottom strip

    Returns (scale_dict, ppi) or (None, None) if no scale string is found.
    """
    crops = [
        (int(full_w * 0.55), int(full_h * 0.55), full_w,            full_h),
        (int(full_w * 0.55), 0,                   full_w,            int(full_h * 0.45)),
        (0,                  int(full_h * 0.80),   full_w,            full_h),
    ]
    for crop in crops:
        results       = extract_text(image, roi=crop)
        drawing_scale = extract_scale(results)
        if drawing_scale:
            ppi = compute_pixels_per_inch(drawing_scale, full_w, None)
            print(f"[Pipeline] Scale: '{drawing_scale['text']}' → {ppi:.1f} px/in")
            return drawing_scale, ppi

    print("[Pipeline] No drawing scale found — geometry uses pixel defaults")
    return None, None


# ═════════════════════════════════════════════════════════════════════════════
# Scale validation helper
# ═════════════════════════════════════════════════════════════════════════════

def _scale_validate(
    boxes: list[BoundingBox],
    associations: dict,
    ppi: float,
    tolerance: float = 0.5,
) -> tuple[list[BoundingBox], dict]:
    """Remove ducts whose pixel thickness contradicts their stated dimension.

    Unlabelled ducts (geometry-only) are always kept — we cannot validate
    what OCR didn't read, and silently dropping geometry detections would
    reduce recall without any quality benefit.
    """
    valid_boxes: list[BoundingBox] = []
    valid_assoc: dict = {}

    for i, box in enumerate(boxes):
        label    = associations.get(i)
        dim_text = label.text if label else None

        if dim_text and '⌀' not in dim_text:  # skip round ducts (⌀) — geometry engine detects rectangles only
            thickness_px = min(box.width, box.height)
            if not validate_duct_dimension(dim_text, thickness_px, ppi):
                m = re.search(r'(\d+)', dim_text)
                if m:
                    expected = int(m.group(1)) * ppi
                    print(f"[Pipeline] Scale reject: '{dim_text}' "
                          f"thickness={thickness_px:.0f}px "
                          f"expected≈{expected:.0f}px")
                continue   # drop this duct

        new_i = len(valid_boxes)
        valid_boxes.append(box)
        if label:
            valid_assoc[new_i] = label

    return valid_boxes, valid_assoc
