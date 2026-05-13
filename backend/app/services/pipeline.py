import os
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
from app.services.llm_processor import llm_identify_ducts

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")


def _merge_llm_results(cv_ducts: list[DuctSegment], llm_ducts: list[dict],
                       img_w: int, img_h: int) -> list[DuctSegment]:
    """Merge LLM-identified ducts with CV-detected ducts.
    LLM can: add missing dimensions, classify supply/return, boost confidence.
    """
    # Map LLM positions (PDF points) to image pixels
    # PDF page is typically 2592x1728 pts for this drawing
    # We need to match LLM duct positions to CV duct positions

    type_map = {
        'supply': DuctType.SUPPLY,
        'return': DuctType.RETURN,
        'unknown': DuctType.UNKNOWN,
    }

    for llm_duct in llm_ducts:
        lx = llm_duct.get('center_x', 0)
        ly = llm_duct.get('center_y', 0)
        ldim = llm_duct.get('dimension', '')
        ltype = llm_duct.get('type', 'unknown')
        lconf = llm_duct.get('confidence', 0.5)

        # Scale LLM coords (PDF points) to image pixels
        # Assume standard PDF: 2592x1728 pts → img_w x img_h
        scale_x = img_w / 2592 if lx > 0 else 1
        scale_y = img_h / 1728 if ly > 0 else 1
        px = lx * scale_x
        py = ly * scale_y

        # Find nearest CV duct
        best_idx = -1
        best_dist = img_w * 0.05  # Max 5% of image
        for i, duct in enumerate(cv_ducts):
            dist = ((duct.bbox.x - px)**2 + (duct.bbox.y - py)**2)**0.5
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx >= 0:
            # Update existing duct with LLM info
            duct = cv_ducts[best_idx]
            if ldim and not duct.dimension:
                duct.dimension = ldim
                duct.pressure_class = classify_pressure(ldim)
            if ltype in type_map and ltype != 'unknown':
                duct.duct_type = type_map[ltype]
            duct.confidence = max(duct.confidence, lconf)
        else:
            # LLM found a duct CV missed — add it
            if ldim and px > 0 and py > 0:
                lw = llm_duct.get('width', 200)
                lh = llm_duct.get('height', 50)
                new_duct = DuctSegment(
                    id=len(cv_ducts) + 1,
                    duct_type=type_map.get(ltype, DuctType.UNKNOWN),
                    dimension=ldim,
                    length=None,
                    pressure_class=classify_pressure(ldim),
                    bbox=BoundingBox(x=px, y=py, width=float(lw * scale_x),
                                     height=float(lh * scale_y), angle=0.0),
                    confidence=lconf,
                )
                cv_ducts.append(new_duct)

    # Re-number
    for i, d in enumerate(cv_ducts):
        d.id = i + 1

    return cv_ducts


def run_detection_pipeline(image_path: str, file_id: str, scale: str = None, pdf_path: str = None) -> DetectionResult:
    """Full detection pipeline: preprocess → detect → OCR → classify → annotate."""
    print(f"[Pipeline] Processing: {image_path}")

    # Load original full-res for annotation
    original_full = cv2.imread(image_path)
    full_h, full_w = original_full.shape[:2]
    print(f"[Pipeline] Full image size: {full_w}x{full_h}")

    # 0. Extract vector data from PDF if available (hybrid approach)
    vector_data = None
    if pdf_path and pdf_path.lower().endswith('.pdf'):
        vector_data = analyze_pdf(pdf_path)
        if vector_data.dimensions:
            print(f"[Pipeline] Vector PDF dimensions found: {[d[0] for d in vector_data.dimensions]}")

    # 1. Preprocess (downscale for better accuracy - avoids wall false positives)
    processed_img, binary = preprocess(image_path, max_dimension=5000)
    proc_h, proc_w = processed_img.shape[:2]
    scale_factor = full_w / proc_w  # To map coords back to full-res

    # 2. Detect ducts on processed image
    boxes = detect_ducts(image_path, binary, processed_img)
    print(f"[Pipeline] Ducts detected: {len(boxes)}")

    # 3. OCR — use same ROI as detector (drawing area only, excludes notes/title block)
    from app.services.detector import auto_detect_roi
    det_roi = auto_detect_roi(binary)  # ROI at processed scale
    # Scale ROI to full-res coordinates
    ocr_roi = (
        int(det_roi[0] * scale_factor),
        int(det_roi[1] * scale_factor),
        int(det_roi[2] * scale_factor),
        int(det_roi[3] * scale_factor),
    )
    print(f"[Pipeline] OCR ROI (full-res): {ocr_roi[0]},{ocr_roi[1]} to {ocr_roi[2]},{ocr_roi[3]}")
    ocr_results = extract_text(original_full, roi=ocr_roi)

    # Scale boxes to full-res for targeted OCR
    from app.models.schemas import BoundingBox as BB
    full_boxes = [BB(x=b.x * scale_factor, y=b.y * scale_factor,
                     width=b.width * scale_factor, height=b.height * scale_factor,
                     angle=b.angle) for b in boxes]

    # Targeted OCR near each duct (catches labels global pass may miss)
    near_results = extract_text_near_ducts(original_full, full_boxes, padding=200)
    # Merge, dedup by proximity
    seen = set()
    for r in ocr_results:
        seen.add(f"{int(r.center_x/30)}_{int(r.center_y/30)}")
    for r in near_results:
        key = f"{int(r.center_x/30)}_{int(r.center_y/30)}"
        if key not in seen:
            ocr_results.append(r)
            seen.add(key)

    dimension_labels = filter_dimensions(ocr_results)
    print(f"[Pipeline] OCR results: {len(ocr_results)} total, {len(dimension_labels)} dimensions")

    # Supplement with vector PDF dimensions (higher confidence than OCR)
    if vector_data and vector_data.dimensions:
        pdf_scale_x = full_w / vector_data.page_width if vector_data.page_width else 1
        pdf_scale_y = full_h / vector_data.page_height if vector_data.page_height else 1
        for dim_text, px, py in vector_data.dimensions:
            ix = px * pdf_scale_x
            iy = py * pdf_scale_y
            bbox = [[int(ix), int(iy)], [int(ix + 50), int(iy)],
                    [int(ix + 50), int(iy + 20)], [int(ix), int(iy + 20)]]
            vec_result = OCRResult(text=dim_text, bbox=bbox, confidence=0.95)
            dimension_labels.append(vec_result)
        print(f"[Pipeline] Added {len(vector_data.dimensions)} vector dimensions, total: {len(dimension_labels)}")

    # 3b. Text-first detection: use dimension labels to find ducts around them
    text_first_boxes, learned_thickness = detect_ducts_from_text(binary, dimension_labels, scale_factor)
    from app.services.detector import merge_overlapping

    # Scale text-first boxes to full-res
    text_first_full = [BoundingBox(
        x=b.x * scale_factor, y=b.y * scale_factor,
        width=b.width * scale_factor, height=b.height * scale_factor,
        angle=b.angle) for b in text_first_boxes]
    # Only reject extreme aspect ratios
    text_first_full = [b for b in text_first_full
                       if max(b.width, b.height) / (min(b.width, b.height) + 1) < 20]

    all_boxes = full_boxes + text_first_full
    full_boxes = merge_overlapping(all_boxes, iou_threshold=0.3)
    print(f"[Pipeline] After text-first merge: {len(full_boxes)} ducts")

    # 4. Associate labels to ducts (use full-res coordinates)
    # Scale max_distance with image size
    max_assoc_dist = max(full_w, full_h) * 0.08  # 8% of image size
    associations = associate_labels(full_boxes, dimension_labels, max_distance=max_assoc_dist)

    # 5. Build duct segments
    ducts = []
    for i, full_bbox in enumerate(full_boxes):
        dim_text = associations[i].text if i in associations else None
        pressure = classify_pressure(dim_text)

        ducts.append(DuctSegment(
            id=i + 1,
            duct_type=DuctType.UNKNOWN,
            dimension=dim_text,
            length=None,
            pressure_class=pressure,
            bbox=full_bbox,
            confidence=associations[i].confidence if i in associations else 0.0,
        ))

    # 5b. LLM post-processing: refine results using Claude (if API key set)
    if pdf_path:
        llm_ducts = llm_identify_ducts(pdf_path, ocr_dimensions=dimension_labels, detected_ducts=ducts)
        if llm_ducts:
            ducts = _merge_llm_results(ducts, llm_ducts, full_w, full_h)

    # 6. Annotate full-res image
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
