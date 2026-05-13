import os
import re
import cv2
import numpy as np
from app.models.schemas import DetectionResult, DuctSegment, DuctType, BoundingBox
from app.services.preprocessor import preprocess
from app.services.detector import detect_ducts, detect_ducts_from_text
from app.services.ocr import extract_text, extract_text_near_ducts, filter_dimensions, OCRResult
from app.services.associator import associate_labels
from app.services.classifier import classify_pressure
from app.services.annotator import annotate_image
from app.services.pdf_analyzer import analyze_pdf
from app.services.scale_extractor import extract_scale, compute_pixels_per_inch, validate_duct_dimension

# Use PaddleOCR tiled if available, else EasyOCR
try:
    from app.services.paddle_ocr import paddle_ocr_tiled
    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")


def run_detection_pipeline(image_path: str, file_id: str, scale: str = None, pdf_path: str = None, resolution: int = None) -> DetectionResult:
    """Full detection pipeline.
    Architecture: OCR-first → Text-first detection → Line-pair fallback → LLM refinement.
    resolution: max dimension for processing (5000=fast, 7000=balanced, None=full-res)
    """
    print(f"[Pipeline] Processing: {image_path}")

    # Load original full-res for annotation
    original_full = cv2.imread(image_path)
    full_h, full_w = original_full.shape[:2]
    print(f"[Pipeline] Full image size: {full_w}x{full_h}")

    # 0. Extract vector data from PDF if available
    vector_data = None
    if pdf_path and pdf_path.lower().endswith('.pdf'):
        vector_data = analyze_pdf(pdf_path)

    # 1. Preprocess
    processed_img, binary = preprocess(image_path, max_dimension=resolution)
    proc_h, proc_w = processed_img.shape[:2]
    scale_factor = full_w / proc_w

    # 2. OCR FIRST — find all dimension labels and their positions
    from app.services.detector import auto_detect_roi
    det_roi = auto_detect_roi(binary, vector_data=vector_data)
    ocr_roi = (
        int(det_roi[0] * scale_factor),
        int(det_roi[1] * scale_factor),
        int(det_roi[2] * scale_factor),
        int(det_roi[3] * scale_factor),
    )
    print(f"[Pipeline] OCR ROI (full-res): {ocr_roi[0]},{ocr_roi[1]} to {ocr_roi[2]},{ocr_roi[3]}")

    # Global pass with Tesseract (fast, broad coverage)
    ocr_results = extract_text(original_full, roi=ocr_roi)
    dimension_labels = filter_dimensions(ocr_results)

    # Use line-pair detection to find candidate positions for targeted EasyOCR
    boxes = detect_ducts(image_path, binary, processed_img, vector_data=vector_data)
    from app.models.schemas import BoundingBox as BB
    candidate_boxes = [BB(x=b.x * scale_factor, y=b.y * scale_factor,
                          width=b.width * scale_factor, height=b.height * scale_factor,
                          angle=b.angle) for b in boxes]

    # Run targeted OCR on candidate duct regions (catches small text Tesseract misses)
    # PaddleOCR (tiled, parallel) if available, else EasyOCR
    if HAS_PADDLE:
        from app.services.paddle_ocr import PaddleOCRResult
        paddle_results = paddle_ocr_tiled(original_full, roi=ocr_roi)
        for pr in paddle_results:
            near_results.append(OCRResult(text=pr.text, bbox=pr.bbox, confidence=pr.confidence))
        extra_dims = filter_dimensions([OCRResult(text=pr.text, bbox=pr.bbox, confidence=pr.confidence) for pr in paddle_results])
    else:
        near_results = extract_text_near_ducts(original_full, candidate_boxes, padding=200)
        extra_dims = filter_dimensions(near_results)

    if extra_dims:
        dimension_labels.extend(extra_dims)
        engine = "PaddleOCR" if HAS_PADDLE else "EasyOCR"
        print(f"[Pipeline] {engine} found {len(extra_dims)} additional dimensions")

    print(f"[Pipeline] OCR found {len(dimension_labels)} dimension labels total")

    # Supplement with vector PDF dimensions if available
    if vector_data and vector_data.dimensions:
        pdf_scale_x = full_w / vector_data.page_width if vector_data.page_width else 1
        pdf_scale_y = full_h / vector_data.page_height if vector_data.page_height else 1
        for dim_text, px, py in vector_data.dimensions:
            ix = px * pdf_scale_x
            iy = py * pdf_scale_y
            bbox = [[int(ix), int(iy)], [int(ix + 50), int(iy)],
                    [int(ix + 50), int(iy + 20)], [int(ix), int(iy + 20)]]
            dimension_labels.append(OCRResult(text=dim_text, bbox=bbox, confidence=0.95))

    # 3. TEXT-FIRST DETECTION — for all dimension labels
    #    Diameter labels (⌀/Ø/∅/DIA) get wider search range
    text_first_boxes, learned_thickness = detect_ducts_from_text(binary, dimension_labels, scale_factor)
    text_first_full = []
    for b in text_first_boxes:
        fx, fy = b.x * scale_factor, b.y * scale_factor
        fw, fh = b.width * scale_factor, b.height * scale_factor
        if fx < 0 or fx > full_w or fy < 0 or fy > full_h:
            continue
        if max(fw, fh) / (min(fw, fh) + 1) >= 20:
            continue
        if min(fw, fh) / scale_factor > 80:
            continue
        text_first_full.append(BoundingBox(x=fx, y=fy, width=fw, height=fh, angle=b.angle))
    print(f"[Pipeline] Text-first detected: {len(text_first_full)} ducts")

    # 4. For diameter labels where text-first couldn't find lines,
    #    create a duct at the text position (OCR guarantees a duct exists there)
    from app.services.detector import merge_overlapping
    text_first_positions = set()
    for b in text_first_full:
        text_first_positions.add(f"{int(b.x/200)}_{int(b.y/200)}")

    for label in dimension_labels:
        pos_key = f"{int(label.center_x/200)}_{int(label.center_y/200)}"
        if pos_key not in text_first_positions:
            # No lines found but dimension confirms duct — create bbox from text position
            text_first_full.append(BoundingBox(
                x=label.center_x, y=label.center_y,
                width=300.0, height=60.0,  # Default size estimate
                angle=0.0
            ))

    full_boxes = merge_overlapping(text_first_full, iou_threshold=0.3)
    print(f"[Pipeline] Total ducts (OCR + text-first): {len(full_boxes)}")

    # 4. ASSOCIATE labels to ducts
    max_assoc_dist = max(full_w, full_h) * 0.08
    associations = associate_labels(full_boxes, dimension_labels, max_distance=max_assoc_dist)

    # 5. Build duct segments
    ducts = []
    for i, full_bbox in enumerate(full_boxes):
        dim_text = associations[i].text if i in associations else None
        pressure = classify_pressure(dim_text)
        conf = associations[i].confidence if i in associations else 0.7

        ducts.append(DuctSegment(
            id=i + 1,
            duct_type=DuctType.UNKNOWN,
            dimension=dim_text,
            length=None,
            pressure_class=pressure,
            bbox=full_bbox,
            confidence=conf,
        ))

    # 6. Scale validation — reject ducts whose pixel size doesn't match stated dimension
    # Look for scale in title block area (outside main ROI)
    title_block_roi = (int(full_w * 0.6), int(full_h * 0.6), full_w, full_h)
    title_results = extract_text(original_full, roi=title_block_roi)
    drawing_scale = extract_scale(title_results + ocr_results)
    if drawing_scale:
        page_w_pts = vector_data.page_width if vector_data else None
        ppi = compute_pixels_per_inch(drawing_scale, full_w, page_w_pts)
        print(f"[Pipeline] Scale: {drawing_scale['text']} → {ppi:.1f} px/inch")

        valid_ducts = []
        for d in ducts:
            if d.dimension:
                thickness_px = min(d.bbox.width, d.bbox.height)
                if validate_duct_dimension(d.dimension, thickness_px, ppi):
                    valid_ducts.append(d)
                else:
                    dim_val = int(re.search(r"(\d+)", d.dimension).group(1))
                    print(f"[Pipeline] Scale rejected duct #{d.id}: {d.dimension} thickness={thickness_px:.0f}px expected={dim_val * ppi:.0f}px")
            else:
                valid_ducts.append(d)
        ducts = valid_ducts

    # 7. Final cleanup
    ducts = [d for d in ducts
             if 0 < d.bbox.x < full_w and 0 < d.bbox.y < full_h
             and d.bbox.width < full_w * 0.5 and d.bbox.height < full_h * 0.5]
    for i, d in enumerate(ducts):
        d.id = i + 1

    # 11. Annotate
    out_dir = os.path.join(OUTPUT_DIR, file_id)
    os.makedirs(out_dir, exist_ok=True)
    annotated_path = os.path.join(out_dir, "annotated.png")
    annotate_image(original_full, ducts, annotated_path)

    return DetectionResult(
        image_width=full_w,
        image_height=full_h,
        scale=scale,
        ducts=ducts,
        annotated_image_path=f"/outputs/{file_id}/annotated.png",
    )


def _has_nearby_label(bbox: BoundingBox, labels: list, img_w: int) -> bool:
    """Check if any dimension label is near this duct."""
    import math
    threshold = img_w * 0.05
    for label in labels:
        dist = math.hypot(label.center_x - bbox.x, label.center_y - bbox.y)
        if dist < threshold:
            return True
    return False

