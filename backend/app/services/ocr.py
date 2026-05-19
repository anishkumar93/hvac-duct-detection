"""Stage 3 — OCR Scale Extraction + Dimension Label Detection.

1. Locates text regions in the image
2. Runs Tesseract OCR
3. Regex parses for drawing scale (e.g., 1/4"=1'-0")
4. Regex parses for duct dimension labels (e.g., 18"⌀, 22"x14")
5. Provides pixel → feet/inches conversion using extracted scale
"""
import re
import os
import csv
import subprocess
import cv2
import numpy as np


# Scale patterns
SCALE_PATTERNS = [
    re.compile(r'(\d+)/(\d+)\s*["\u2033]\s*=\s*(\d+)\s*[\'\u2032]\s*-?\s*(\d+)\s*["\u2033]?'),
    re.compile(r'(\d+)\s*["\u2033]\s*=\s*(\d+)\s*[\'\u2032]\s*-?\s*(\d+)\s*["\u2033]?'),
]

# Dimension patterns for duct sizes
DIMENSION_PATTERNS = [
    re.compile(r'(\d+)\s*["\u2033\u201d\']+\s*[×xX]\s*(\d+)\s*["\u2033\u201d\']*'),  # 22"x14"
    re.compile(r'(\d+)\s*[×xX]\s*(\d+)'),                                              # 22x14
    re.compile(r'(\d+)\s*["\u2033\u201d\']+\s*[⌀∅Øø@]+'),                              # 18"⌀
    re.compile(r'(\d+)\s*[°⌀∅Øø]'),                                                    # 14⌀
    re.compile(r'(\d+)\s*["\u2033\u201d\']+\s*DIA', re.IGNORECASE),                     # 18" DIA
    re.compile(r'[⌀∅Øø]\s*(\d+)'),                                                     # ⌀14
    re.compile(r'^(\d+)\s*["\u2033\u201d\']+\s*$'),                                     # bare: 12"
]


class OCRResult:
    """Holds a single OCR detection with text, position, and confidence."""
    def __init__(self, text: str, x: int, y: int, w: int, h: int, confidence: float):
        self.text = text
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.center_x = x + w / 2
        self.center_y = y + h / 2
        self.confidence = confidence


class ScaleInfo:
    """Holds extracted scale and provides pixel → real-world conversion."""
    def __init__(self, text: str, paper_inches: float, real_inches: float, pixels_per_inch: float):
        self.text = text
        self.paper_inches = paper_inches
        self.real_inches = real_inches
        self.scale_ratio = real_inches / paper_inches
        self.pixels_per_inch = pixels_per_inch

    def pixels_to_inches(self, pixels: float) -> float:
        """Convert pixel length to real-world inches."""
        return pixels / self.pixels_per_inch

    def pixels_to_feet_inches(self, pixels: float) -> str:
        """Convert pixel length to feet-inches string (e.g., 3'-6\")."""
        total_inches = self.pixels_to_inches(pixels)
        feet = int(total_inches // 12)
        inches = int(total_inches % 12)
        if feet > 0:
            return f"{feet}'-{inches}\""
        return f'{int(total_inches)}"'

    def pixels_to_dimension(self, pixels: float) -> str | None:
        """Convert pixel thickness to duct dimension string.
        Returns None if result is unrealistic (<4" or >100").
        """
        inches = self.pixels_to_inches(pixels)
        if inches < 4 or inches > 100:
            return None
        return f'{int(round(inches))}"'


def extract_scale(image: np.ndarray, page_width_pts: float = None) -> ScaleInfo | None:
    """Run OCR on title block region and extract drawing scale.
    Returns ScaleInfo with pixel→real conversion methods, or None.
    """
    h, w = image.shape[:2]

    # Title block is typically bottom-right quadrant
    roi = image[int(h * 0.6):h, int(w * 0.5):w]
    text = _run_tesseract_text(roi)
    scale_dict = _parse_scale(text)

    if not scale_dict:
        # Try full bottom strip
        roi2 = image[int(h * 0.75):h, :]
        text2 = _run_tesseract_text(roi2)
        scale_dict = _parse_scale(text2)

    if not scale_dict:
        return None

    # Compute pixels per real-world inch
    # DPI: if we know PDF page width in points, compute actual DPI
    # Otherwise assume 300 DPI (standard engineering print scan)
    dpi = 300.0
    if page_width_pts:
        page_inches = page_width_pts / 72.0
        dpi = w / page_inches

    # pixels_per_inch = (DPI × paper_inches) / real_inches
    ppi = (dpi * scale_dict['paper_inches']) / scale_dict['real_inches']

    info = ScaleInfo(
        text=scale_dict['text'],
        paper_inches=scale_dict['paper_inches'],
        real_inches=scale_dict['real_inches'],
        pixels_per_inch=ppi,
    )

    print(f"[Scale] Found: {info.text} → {info.pixels_per_inch:.2f} px/inch (DPI={dpi:.0f})")
    return info


def extract_dimensions(image: np.ndarray) -> list[OCRResult]:
    """Run OCR on the full image and extract duct dimension labels.

    Flow: Locate text regions → OCR → Regex parsing → Filter valid dimensions.
    """
    # Run Tesseract with TSV output to get word positions
    all_words = _run_tesseract_tsv(image)

    # Filter to dimension-like text
    dimensions = []
    for word in all_words:
        normalized = _normalize_dimension(word.text)
        if _is_dimension(normalized):
            # Validate size range (4-100 inches)
            match = re.search(r'(\d+)', normalized)
            if match:
                val = int(match.group(1))
                if 4 <= val <= 100:
                    word.text = normalized
                    dimensions.append(word)

    print(f"[OCR] Dimension labels found: {len(dimensions)}")
    for d in dimensions[:10]:
        print(f"  '{d.text}' at ({d.center_x:.0f}, {d.center_y:.0f})")

    return dimensions


def _run_tesseract_text(image: np.ndarray) -> str:
    """Run Tesseract and return raw text string."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    tmp_dir = os.path.join(os.path.dirname(__file__), "..", "..", "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    input_path = os.path.join(tmp_dir, "scale_ocr.png")
    output_base = os.path.join(tmp_dir, "scale_ocr")

    cv2.imwrite(input_path, gray)
    subprocess.run(
        ["tesseract", input_path, output_base, "--psm", "6"],
        capture_output=True
    )

    txt_path = output_base + ".txt"
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return ""


def _run_tesseract_tsv(image: np.ndarray) -> list[OCRResult]:
    """Run Tesseract with TSV output to get word-level bounding boxes."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape[:2]

    # Downscale if very large for speed
    scale = 1.0
    max_side = 7000
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    tmp_dir = os.path.join(os.path.dirname(__file__), "..", "..", "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    input_path = os.path.join(tmp_dir, "dim_ocr.png")
    output_base = os.path.join(tmp_dir, "dim_ocr")

    cv2.imwrite(input_path, gray)
    subprocess.run(
        ["tesseract", input_path, output_base, "--psm", "11", "tsv"],
        capture_output=True
    )

    results = []
    tsv_path = output_base + ".tsv"
    if os.path.exists(tsv_path):
        with open(tsv_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                text = row.get("text", "").strip()
                conf = int(float(row.get("conf", 0)))
                if not text or conf < 20:
                    continue
                x = int(int(row["left"]) / scale)
                y = int(int(row["top"]) / scale)
                rw = int(int(row["width"]) / scale)
                rh = int(int(row["height"]) / scale)
                results.append(OCRResult(text=text, x=x, y=y, w=rw, h=rh,
                                         confidence=conf / 100.0))

    return results


def _parse_scale(text: str) -> dict | None:
    """Parse scale from OCR text using regex."""
    for pattern in SCALE_PATTERNS:
        m = pattern.search(text)
        if m:
            groups = m.groups()
            if len(groups) == 4:
                paper_inches = int(groups[0]) / int(groups[1])
                real_inches = int(groups[2]) * 12 + int(groups[3])
            elif len(groups) == 3:
                paper_inches = int(groups[0])
                real_inches = int(groups[1]) * 12 + int(groups[2])
            else:
                continue

            if real_inches == 0:
                continue

            return {
                'text': m.group(0),
                'paper_inches': paper_inches,
                'real_inches': real_inches,
            }
    return None


def _normalize_dimension(text: str) -> str:
    """Normalize common OCR misreads to proper dimension format."""
    t = text.strip()
    t = t.replace('\u201c', '"').replace('\u201d', '"').replace('\u2033', '"')
    # 66" → 6"⌀ (leading 6 = misread ⌀)
    t = re.sub(r'^6(\d+)\s*["]+\s*$', r'\1"⌀', t)
    # 18"@ → 18"⌀
    t = re.sub(r'(\d+)\s*["]+\s*[@]+', r'\1"⌀', t)
    # 12°6 → 12"⌀
    t = re.sub(r'(\d+)\s*°\s*[6%oO0)]+\s*$', r'\1"⌀', t)
    # 14° → 14"⌀
    t = re.sub(r'(\d+)\s*°\s*$', r'\1"⌀', t)
    # Normalize x separator
    t = re.sub(r'(\d+)\s*["\u2033\']*\s*[xX×]\s*(\d+)\s*["\u2033\']*', r'\1"×\2"', t)
    return t


def _is_dimension(text: str) -> bool:
    """Check if text matches any duct dimension pattern."""
    for pattern in DIMENSION_PATTERNS:
        if pattern.search(text):
            return True
    return False
