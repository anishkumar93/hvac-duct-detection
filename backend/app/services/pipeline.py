"""Pipeline Orchestrator — Ties all 9 stages together.

PDF → Image → Preprocess → Scale OCR → Line Extraction → Remove Text →
Detect Candidates → Filter → Merge → Measure → JSON/Overlay
"""
import os
import math
import cv2
import numpy as np
from app.models.schemas import DetectionResult, DuctSegment, DuctType, BoundingBox, PressureClass
from app.services.preprocessor import preprocess
from app.services.ocr import extract_scale, extract_dimensions, ScaleInfo
from app.services.detector import extract_lines, remove_text_noise, detect_candidates, filter_unwanted, merge_segments
from app.services.annotator import annotate_image
from app.services.classifier import classify_pressure

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")


def run_detection_pipeline(image_path: str, file_id: str, scale: str = None,
                           pdf_path: str = None, resolution: int = None) -> DetectionResult:
    """Full multi-stage detection pipeline."""
    print(f"[Pipeline] Processing: {image_path}")

    # ─── Stage 2: Preprocess ─────────────────────────────────────────────
    original, gray, binary = preprocess(image_path)
    h, w = original.shape[:2]
    print(f"[Pipeline] Image size: {w}x{h}")

    # ─── Stage 3: OCR Scale Extraction + Dimension Labels ────────────────
    # Extract scale (provides pixel → feet/inches conversion)
    page_w_pts = _get_pdf_page_width(pdf_path) if pdf_path else None
    scale_info = extract_scale(original, page_width_pts=page_w_pts)

    # Extract dimension labels from drawing
    dimension_labels = extract_dimensions(original)

    # ─── Stage 4: Structural Line Extraction ─────────────────────────────
    h_mask, v_mask = extract_lines(binary)
    print(f"[Pipeline] Lines extracted (H + V masks)")

    # ─── Stage 4b: Remove text/noise ─────────────────────────────────────
    line_mask = remove_text_noise(binary, h_mask, v_mask)

    # ─── Stage 5: Candidate Rectangle Detection ─────────────────────────
    candidates = detect_candidates(line_mask, h_mask, v_mask)

    # ─── Stage 6: Filter Grid Lines / Walls / Equipment ──────────────────
    filtered = filter_unwanted(candidates, h_mask, v_mask, binary)

    # ─── Stage 7: Merge Duct Segments ────────────────────────────────────
    merged = merge_segments(filtered)

    # ─── Stage 8: Measurement Calculation ────────────────────────────────
    # Associate dimension labels to nearest duct
    label_assignments = _associate_labels(merged, dimension_labels)

    ducts = []
    for i, box in enumerate(merged):
        dimension = None
        length_str = None

        # If OCR found a dimension label near this duct, use it
        if i in label_assignments:
            dimension = label_assignments[i].text

        # If no OCR label but we have scale, convert pixels → feet/inches
        # Formula: real_size = (pixel_length / pixels_per_scale_unit) × actual_scale
        if not dimension and scale_info:
            dimension = scale_info.pixels_to_dimension(min(box.width, box.height))

        # Compute length using scale's pixel → feet/inches conversion
        if scale_info:
            length_str = scale_info.pixels_to_feet_inches(max(box.width, box.height))

        pressure = classify_pressure(dimension)

        ducts.append(DuctSegment(
            id=i + 1,
            duct_type=DuctType.UNKNOWN,
            dimension=dimension,
            length=length_str,
            pressure_class=pressure,
            bbox=box,
            confidence=0.75,
        ))

    print(f"[Pipeline] Final ducts: {len(ducts)}")

    # ─── Stage 9: Annotated Output ───────────────────────────────────────
    out_dir = os.path.join(OUTPUT_DIR, file_id)
    os.makedirs(out_dir, exist_ok=True)
    annotated_path = os.path.join(out_dir, "annotated.png")
    annotate_image(original, ducts, annotated_path)

    return DetectionResult(
        image_width=w,
        image_height=h,
        scale=scale_info.text if scale_info else scale,
        ducts=ducts,
        annotated_image_path=f"/outputs/{file_id}/annotated.png",
    )


def _associate_labels(ducts: list[BoundingBox], labels: list, max_distance: float = None) -> dict:
    """Associate each dimension label to its nearest duct.
    Returns dict: duct_index → OCRResult.
    """
    if not labels or not ducts:
        return {}

    if max_distance is None:
        if ducts:
            max_dim = max(max(d.width, d.height) for d in ducts)
            max_distance = max_dim * 2
        else:
            max_distance = 500

    assignments = {}

    for label in labels:
        best_idx = -1
        best_dist = float('inf')

        for i, duct in enumerate(ducts):
            dist = math.hypot(label.center_x - duct.x, label.center_y - duct.y)
            if dist > max_distance:
                continue

            # Bonus if label is inside duct bbox
            inside = (abs(label.center_x - duct.x) < duct.width / 2 and
                      abs(label.center_y - duct.y) < duct.height / 2)
            score = dist * (0.3 if inside else 1.0)

            if score < best_dist:
                best_dist = score
                best_idx = i

        if best_idx >= 0:
            if best_idx not in assignments or label.confidence > assignments[best_idx].confidence:
                assignments[best_idx] = label

    return assignments


def _get_pdf_page_width(pdf_path: str) -> float | None:
    """Get PDF page width in points for DPI calculation."""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        width = doc[0].rect.width
        doc.close()
        return width
    except Exception:
        return None
