import re
import numpy as np
import cv2
import os
import subprocess
import csv

# Dimension patterns - comprehensive for HVAC drawings
DIMENSION_PATTERNS = [
    re.compile(r'(\d+)\s*["\u2033\u201d\']+\s*[×xX]\s*(\d+)\s*["\u2033\u201d\']*'),  # WxH: 22"x14"
    re.compile(r'(\d+)\s*[×xX]\s*(\d+)'),                                              # WxH no quotes: 22x14
    re.compile(r'(\d+)\s*["\u2033\u201d\']+\s*[⌀∅Øø@6oO0°)]+\s*$'),                   # 18"⌀, 18"@
    re.compile(r'(\d+)\s*[°⌀∅Øø]\s*[6%oO0]*\s*$'),                                    # 14⌀, 14°, 12°6
    re.compile(r'(\d+)\s*["\u2033\u201d\']+\s*DIA', re.IGNORECASE),                     # DIA
    re.compile(r'[⌀∅Øø]\s*(\d+)'),                                                     # ⌀14
    re.compile(r'^(\d+)\s*["\u2033\u201d\']+\s*$'),                                     # bare: 12"
    re.compile(r'(\d+)\s*["\u2033\u201d\']+\s*[xX×]'),                                  # leading dim: 22"x
    re.compile(r'(\d+)\s*["\u2033\u201d\']+\s*[@]'),                                    # 18"@ (misread of ⌀)
]

CFM_PATTERNS = [
    re.compile(r'(\d[\d,]*)\s*CFM', re.IGNORECASE),
]

LENGTH_PATTERNS = [
    re.compile(r"(\d+)\s*[']\s*-\s*(\d+)\s*[\"]"),       # 12'-6"
    re.compile(r"(\d+)\s*[']\s*-\s*(\d+)\s*['\"]"),      # 10' - 0"
]


class OCRResult:
    def __init__(self, text: str, bbox: list, confidence: float):
        self.text = text
        self.bbox = bbox
        self.confidence = confidence
        self.center_x = sum(p[0] for p in bbox) / len(bbox)
        self.center_y = sum(p[1] for p in bbox) / len(bbox)


def extract_text(image: np.ndarray, roi: tuple = None) -> list[OCRResult]:
    """Run enhanced Tesseract OCR on image, optionally within ROI."""
    if roi:
        x1, y1, x2, y2 = roi
        ocr_image = image[y1:y2, x1:x2].copy()
        offset_x, offset_y = x1, y1
    else:
        ocr_image = image.copy()
        offset_x, offset_y = 0, 0

    # Multi-pass OCR with different preprocessing for better coverage
    results = _run_tesseract_multipass(ocr_image)

    # Offset coordinates back to full image space
    for r in results:
        r.bbox = [[p[0] + offset_x, p[1] + offset_y] for p in r.bbox]
        r.center_x += offset_x
        r.center_y += offset_y

    print(f"[OCR] Found {len(results)} text regions")
    print(f"[OCR] Sample: {[r.text for r in results[:20]]}")
    return results


def _run_tesseract_multipass(image: np.ndarray) -> list[OCRResult]:
    """Run Tesseract with geometry lines masked out for cleaner text detection."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape[:2]

    # Downscale if very large
    scale = 1.0
    max_side = 7000
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        print(f"[OCR] Downscaled for Tesseract: {w}x{h} -> {gray.shape[1]}x{gray.shape[0]}")

    # Create text-only image by masking out geometry lines
    _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    text_mask = _mask_geometry_lines(binary_inv)
    # Convert back to white-bg for Tesseract (text=black, bg=white)
    text_image = cv2.bitwise_not(text_mask)

    all_results = []

    # Pass 1: Text-only (geometry masked out) — primary pass
    r1 = _run_tesseract_single(text_image, psm=11, tag="pass1_clean")
    all_results.extend(r1)

    # Pass 2: Standard binary (fallback for text touching lines)
    _, binary_std = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    r2 = _run_tesseract_single(binary_std, psm=11, tag="pass2_std")
    all_results.extend(r2)

    # Pass 3: Upscaled text-only (small dimension labels)
    upscaled = cv2.resize(text_mask, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    r3 = _run_tesseract_single(cv2.bitwise_not(upscaled), psm=6, tag="pass3_up", scale_factor=1.5)
    all_results.extend(r3)

    # Scale results back to original coordinates
    if scale != 1.0:
        for r in all_results:
            r.bbox = [[int(p[0] / scale), int(p[1] / scale)] for p in r.bbox]
            r.center_x /= scale
            r.center_y /= scale

    deduped = _deduplicate_results(all_results)
    return merge_nearby_text(deduped)


def _mask_geometry_lines(binary: np.ndarray) -> np.ndarray:
    """Remove geometry lines using connected component analysis.
    Keeps text-sized components, removes long/thin line components.
    Operates on downscaled image for speed, then upscales mask.
    """
    h, w = binary.shape[:2]

    # Downscale for faster CCA (2000px is enough to classify components)
    cca_scale = 1.0
    if max(h, w) > 2000:
        cca_scale = 2000 / max(h, w)
        small = cv2.resize(binary, None, fx=cca_scale, fy=cca_scale, interpolation=cv2.INTER_AREA)
        _, small = cv2.threshold(small, 127, 255, cv2.THRESH_BINARY)
    else:
        small = binary

    sh, sw = small.shape[:2]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(small, connectivity=8)

    # Build mask of lines to remove
    line_mask_small = np.zeros_like(small)

    for i in range(1, num_labels):
        comp_w = stats[i, cv2.CC_STAT_WIDTH]
        comp_h = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = max(comp_w, comp_h) / (min(comp_w, comp_h) + 1)
        length = max(comp_w, comp_h)

        is_line = aspect > 12 and length > max(sw, sh) * 0.02
        is_thick_line = aspect > 6 and length > max(sw, sh) * 0.06
        is_border = comp_w > sw * 0.8 or comp_h > sh * 0.8

        if is_line or is_thick_line or is_border:
            line_mask_small[labels == i] = 255

    # Upscale line mask back to original size and dilate to cover edges
    if cca_scale != 1.0:
        line_mask = cv2.resize(line_mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        line_mask = line_mask_small

    # Dilate to ensure full line coverage
    dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    line_mask = cv2.dilate(line_mask, dilate_k, iterations=1)

    # Subtract lines from original
    text_only = cv2.subtract(binary, line_mask)
    return text_only


def _remove_lines(binary: np.ndarray) -> np.ndarray:
    """Remove long horizontal and vertical lines from binary image, leaving text.
    Uses large kernels to only remove lines significantly longer than text characters.
    """
    h, w = binary.shape[:2]
    # Kernel length: ~2% of image width — removes only lines longer than any text
    k_len = max(80, w // 50)

    # Extract horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_len, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    # Extract vertical lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_len))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    # Dilate lines slightly to cover anti-aliased edges
    dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    lines_mask = cv2.bitwise_or(h_lines, v_lines)
    lines_mask = cv2.dilate(lines_mask, dilate_k, iterations=1)

    # Subtract from original
    text_only = cv2.subtract(binary, lines_mask)

    return text_only


def _run_tesseract_single(image: np.ndarray, psm: int = 11, tag: str = "",
                          scale_factor: float = 1.0) -> list[OCRResult]:
    """Run a single Tesseract pass."""
    from PIL import Image

    tmp_dir = os.path.join(os.path.dirname(__file__), "..", "..", "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    input_path = os.path.join(tmp_dir, f"ocr_{tag}.png")
    output_base = os.path.join(tmp_dir, f"ocr_{tag}")

    Image.fromarray(image).save(input_path)
    subprocess.run(
        ["tesseract", input_path, output_base, "--psm", str(psm), "tsv"],
        capture_output=True
    )

    tsv_path = output_base + ".tsv"
    results = []
    if os.path.exists(tsv_path):
        with open(tsv_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                text = row.get("text", "").strip()
                conf = int(float(row.get("conf", 0)))
                if not text or conf < 20:
                    continue
                x = int(row["left"]) / scale_factor
                y = int(row["top"]) / scale_factor
                w = int(row["width"]) / scale_factor
                h = int(row["height"]) / scale_factor
                bbox = [[int(x), int(y)], [int(x + w), int(y)],
                        [int(x + w), int(y + h)], [int(x), int(y + h)]]
                results.append(OCRResult(text=text, bbox=bbox, confidence=conf / 100.0))

    return results


def _deduplicate_results(results: list[OCRResult], dist_thresh: int = 30) -> list[OCRResult]:
    """Remove duplicate detections from multiple passes."""
    if not results:
        return results

    # Sort by confidence descending — keep higher confidence version
    results.sort(key=lambda r: r.confidence, reverse=True)
    kept = []
    for r in results:
        is_dup = False
        for k in kept:
            if (abs(r.center_x - k.center_x) < dist_thresh and
                    abs(r.center_y - k.center_y) < dist_thresh):
                is_dup = True
                break
        if not is_dup:
            kept.append(r)
    return kept


def extract_text_near_ducts(image: np.ndarray, ducts, padding: int = 150) -> list[OCRResult]:
    """Run OCR specifically in regions near detected ducts for better label capture."""
    h, w = image.shape[:2]
    all_results = []
    seen_texts = set()

    for duct in ducts:
        bbox = duct if hasattr(duct, 'x') else duct.bbox
        x1 = max(0, int(bbox.x - bbox.width / 2 - padding))
        y1 = max(0, int(bbox.y - bbox.height / 2 - padding))
        x2 = min(w, int(bbox.x + bbox.width / 2 + padding))
        y2 = min(h, int(bbox.y + bbox.height / 2 + padding))

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Single focused pass on duct region with PSM 6 (block of text)
        if len(crop.shape) == 3:
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray_crop = crop

        # Upscale small crops for better OCR
        ch, cw = gray_crop.shape[:2]
        up = 1.0
        if max(ch, cw) < 500:
            up = 500 / max(ch, cw)
            gray_crop = cv2.resize(gray_crop, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)

        _, binary = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results = _run_tesseract_single(binary, psm=11, tag=f"duct_{id(duct)}", scale_factor=up)

        for r in results:
            # Offset to full image coords
            r.bbox = [[p[0] + x1, p[1] + y1] for p in r.bbox]
            r.center_x += x1
            r.center_y += y1

            key = f"{r.text}_{int(r.center_x / 50)}_{int(r.center_y / 50)}"
            if key not in seen_texts:
                seen_texts.add(key)
                all_results.append(r)

    return all_results


def merge_nearby_text(results: list[OCRResult], max_gap_x: int = 40, max_gap_y: int = 15) -> list[OCRResult]:
    """Merge text fragments that are close together horizontally."""
    if not results:
        return results

    results.sort(key=lambda r: (r.center_y, r.center_x))
    merged = []
    used = set()

    for i, r in enumerate(results):
        if i in used:
            continue

        group_text = r.text
        group_bbox = [list(p) for p in r.bbox]
        group_conf = r.confidence

        for j in range(i + 1, len(results)):
            if j in used:
                continue
            r2 = results[j]
            if abs(r2.center_y - r.center_y) > max_gap_y:
                break

            right_edge = max(group_bbox[1][0], group_bbox[2][0])
            left_edge = min(r2.bbox[0][0], r2.bbox[3][0])
            x_gap = left_edge - right_edge

            if 0 <= x_gap <= max_gap_x:
                group_text += r2.text
                group_bbox[1][0] = max(group_bbox[1][0], r2.bbox[1][0])
                group_bbox[2][0] = max(group_bbox[2][0], r2.bbox[2][0])
                group_bbox[2][1] = max(group_bbox[2][1], r2.bbox[2][1])
                group_bbox[3][1] = max(group_bbox[3][1], r2.bbox[3][1])
                group_conf = min(group_conf, r2.confidence)
                used.add(j)

        merged.append(OCRResult(text=group_text, bbox=group_bbox, confidence=group_conf))

    return merged


def normalize_dimension(text: str) -> str:
    """Normalize OCR misreads to proper dimension format."""
    t = text.strip()
    t = t.replace('\u201c', '"').replace('\u201d', '"').replace('\u2033', '"')
    t = t.replace('\u2018', "'").replace('\u2019', "'")
    # 18"@ or 12"@ -> diameter (@ is common misread of ⌀)
    t = re.sub(r'(\d+)\s*["]+\s*[@]+', r'\1"⌀', t)
    # "18"6" or "12"0" or "14"O" -> diameter
    t = re.sub(r'(\d+)\s*["\u2033\']+\s*[6oO0)]+\s*$', r'\1"⌀', t)
    # 12°6 or 12°% or 14°6 -> diameter (° is misread of ", trailing is misread of ⌀)
    t = re.sub(r'(\d+)\s*°\s*[6%oO0)]+\s*$', r'\1"⌀', t)
    # "14°" -> 14"⌀
    t = re.sub(r'(\d+)\s*°\s*$', r'\1"⌀', t)
    # "8'o" -> 8"⌀
    t = re.sub(r"(\d+)\s*[']\s*[oO0]\s*$", r'\1"⌀', t)
    # Normalize "x" separator
    t = re.sub(r'(\d+)\s*["\u2033\']*\s*[xX×]\s*(\d+)\s*["\u2033\']*', r'\1"×\2"', t)
    # "10' Q"" -> 10'-0"
    t = re.sub(r"(\d+)\s*[']\s*[QO]\s*[\"']", r"\1'-0\"", t)
    # Bare number followed by quote-like -> add "
    t = re.sub(r'(\d+)\s*[`´]+', r'\1"', t)
    return t


def filter_dimensions(ocr_results: list[OCRResult]) -> list[OCRResult]:
    """Filter OCR results to dimension-like text and normalize."""
    dims = []
    for r in ocr_results:
        r.text = normalize_dimension(r.text)

        for pattern in DIMENSION_PATTERNS + CFM_PATTERNS + LENGTH_PATTERNS:
            if pattern.search(r.text):
                # Filter out unrealistic duct dimensions (< 4" or > 100")
                match = re.search(r'(\d+)', r.text)
                if match:
                    val = int(match.group(1))
                    if val < 4 or val > 100:
                        break
                dims.append(r)
                break

    print(f"[OCR] Dimension matches: {[r.text for r in dims]}")
    return dims
