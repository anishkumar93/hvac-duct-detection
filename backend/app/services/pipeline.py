import os
import cv2
from app.models.schemas import DetectionResult, DuctSegment, DuctType, BoundingBox
from app.services.preprocessor import preprocess
from app.services.detector import detect_ducts
from app.services.ocr import extract_text, extract_text_near_ducts, filter_dimensions
from app.services.associator import associate_labels
from app.services.classifier import classify_pressure
from app.services.annotator import annotate_image

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")


def run_detection_pipeline(image_path: str, file_id: str, scale: str = None) -> DetectionResult:
    """Full detection pipeline: preprocess → detect → OCR → classify → annotate."""
    print(f"[Pipeline] Processing: {image_path}")

    # Load original full-res for annotation
    original_full = cv2.imread(image_path)
    full_h, full_w = original_full.shape[:2]
    print(f"[Pipeline] Full image size: {full_w}x{full_h}")

    # 1. Preprocess (downscale for better accuracy - avoids wall false positives)
    processed_img, binary = preprocess(image_path, max_dimension=5000)
    proc_h, proc_w = processed_img.shape[:2]
    scale_factor = full_w / proc_w  # To map coords back to full-res

    # 2. Detect ducts on processed image
    boxes = detect_ducts(image_path, binary, processed_img)
    print(f"[Pipeline] Ducts detected: {len(boxes)}")

    # 3. OCR — wider ROI + targeted OCR near detected ducts
    full_h, full_w = original_full.shape[:2]
    ocr_roi = (int(full_w * 0.02), int(full_h * 0.02), int(full_w * 0.85), int(full_h * 0.78))
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
