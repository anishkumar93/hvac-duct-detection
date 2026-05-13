"""PDF vector data extraction for hybrid detection pipeline.

Detects whether a PDF is vector, raster, or hybrid, and extracts
structured data (text + line geometry) when available.
Also validates whether the PDF contains a mechanical/engineering drawing.
"""
import re
import math

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


# Keywords that indicate a mechanical/HVAC drawing
MECHANICAL_KEYWORDS = [
    'mechanical', 'hvac', 'duct', 'ductwork', 'cfm', 'supply', 'return',
    'exhaust', 'diffuser', 'rtu', 'ahu', 'vav', 'floor plan',
    'air handling', 'thermostat', 'grease duct', 'kitchen hood',
    'pressure', 'damper', 'plenum', 'register',
]

# Keywords for any engineering/architectural drawing
DRAWING_KEYWORDS = [
    'plan', 'elevation', 'section', 'detail', 'schedule',
    'scale', 'drawn', 'checked', 'revision', 'sheet',
    'north', 'architect', 'engineer', 'contractor',
]


class PDFVectorData:
    """Structured data extracted from a vector PDF."""
    def __init__(self):
        self.is_vector = False
        self.is_hybrid = False
        self.texts = []        # [(text, x, y, width, height)]
        self.dimensions = []   # [(dimension_text, x, y)] - duct sizes
        self.lines = []        # [(x1, y1, x2, y2, length, orientation)]
        self.page_width = 0
        self.page_height = 0


# Patterns for duct dimensions in vector text
VECTOR_DIM_PATTERNS = [
    re.compile(r"(\d+)\s*[\"']\s*[×xX]\s*(\d+)\s*[\"']?"),   # 22"x14"
    re.compile(r"(\d+)\s*[\"']\s*[⌀∅Øø]"),                    # 18"⌀
    re.compile(r"[⌀∅Øø]\s*(\d+)"),                             # ⌀18
    re.compile(r"(\d+)\s*[\"']\s*DIA", re.IGNORECASE),         # 18" DIA
]

# Patterns for ceiling heights / lengths (NOT duct dimensions)
LENGTH_PATTERNS = [
    re.compile(r"(\d+)'\s*-\s*(\d+)\""),   # 10' - 0"
]


def analyze_pdf(pdf_path: str) -> PDFVectorData:
    """Analyze PDF and extract vector data if available."""
    data = PDFVectorData()

    if not HAS_PYMUPDF:
        return data

    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return data

    page = doc[0]
    data.page_width = page.rect.width
    data.page_height = page.rect.height

    # Determine PDF type
    num_images = len(page.get_images())
    text_blocks = page.get_text('blocks')
    num_text = len([b for b in text_blocks if len(b) > 4 and b[4].strip()])
    paths = page.get_drawings()

    # Heuristic: vector PDFs have many paths and text, few/no images
    # Hybrid: has both images and vector text/paths
    if num_images == 0 and len(paths) > 100:
        data.is_vector = True
    elif num_images > 0 and num_text > 10:
        data.is_hybrid = True
    else:
        # Mostly raster — not much to extract
        doc.close()
        return data

    # Extract text with positions
    blocks = page.get_text('dict')['blocks']
    for block in blocks:
        if 'lines' not in block:
            continue
        for line in block['lines']:
            for span in line['spans']:
                text = span['text'].strip()
                if not text:
                    continue
                bbox = span['bbox']
                data.texts.append((text, bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))

                # Check if it's a duct dimension (not a ceiling height/length)
                is_length = any(lp.search(text) for lp in LENGTH_PATTERNS)
                if is_length:
                    continue
                for pattern in VECTOR_DIM_PATTERNS:
                    if pattern.search(text):
                        match = re.search(r'(\d+)', text)
                        if match and 4 <= int(match.group(1)) <= 100:
                            data.dimensions.append((text, bbox[0], bbox[1]))
                        break

    # Extract significant lines (potential duct walls)
    for p in paths:
        for item in p['items']:
            if item[0] == 'l':  # line segment
                x1, y1 = item[1].x, item[1].y
                x2, y2 = item[2].x, item[2].y
                length = math.hypot(x2 - x1, y2 - y1)

                if length < 20:
                    continue

                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                if dx > dy * 3:
                    orient = 'H'
                elif dy > dx * 3:
                    orient = 'V'
                else:
                    orient = 'D'

                data.lines.append((x1, y1, x2, y2, length, orient))

    doc.close()

    print(f"[PDF] Type: {'vector' if data.is_vector else 'hybrid'}")
    print(f"[PDF] Text elements: {len(data.texts)}, Dimensions: {len(data.dimensions)}, Lines: {len(data.lines)}")

    return data


def is_vector_pdf(pdf_path: str) -> bool:
    """Quick check if PDF has extractable vector data."""
    if not HAS_PYMUPDF:
        return False
    try:
        doc = fitz.open(pdf_path)
        page = doc[0]
        num_images = len(page.get_images())
        text_blocks = page.get_text('blocks')
        num_text = len([b for b in text_blocks if len(b) > 4 and b[4].strip()])
        doc.close()
        return num_text > 10 or num_images == 0
    except Exception:
        return False


def validate_mechanical_drawing(pdf_path: str) -> tuple[bool, str]:
    """Check if PDF likely contains a mechanical/engineering drawing.
    Returns (is_valid, reason).
    """
    if not HAS_PYMUPDF:
        # Can't validate without PyMuPDF, allow it through
        return True, "ok"

    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return False, "Cannot open PDF file"

    if doc.page_count == 0:
        doc.close()
        return False, "PDF has no pages"

    page = doc[0]
    page_w = page.rect.width
    page_h = page.rect.height

    # Extract all text
    full_text = page.get_text().lower()
    text_blocks = page.get_text('blocks')
    num_text_blocks = len([b for b in text_blocks if len(b) > 4 and b[4].strip()])

    # Count vector paths
    try:
        paths = page.get_drawings()
        num_paths = len(paths)
    except Exception:
        num_paths = 0

    num_images = len(page.get_images())
    doc.close()

    # Score-based validation
    score = 0
    reasons = []

    # Check for mechanical keywords
    mech_hits = sum(1 for kw in MECHANICAL_KEYWORDS if kw in full_text)
    if mech_hits >= 2:
        score += 40
        reasons.append(f"{mech_hits} mechanical keywords")

    # Check for general drawing keywords
    draw_hits = sum(1 for kw in DRAWING_KEYWORDS if kw in full_text)
    if draw_hits >= 2:
        score += 20
        reasons.append(f"{draw_hits} drawing keywords")

    # Page size: engineering drawings are typically large format
    # Letter=612x792, Tabloid=792x1224, ARCH D=1728x2592
    page_area = page_w * page_h
    if page_area > 1000000:  # Larger than tabloid
        score += 15
        reasons.append("large format sheet")

    # Landscape orientation (common for floor plans)
    if page_w > page_h:
        score += 5
        reasons.append("landscape")

    # High line density = engineering drawing
    if num_paths > 500:
        score += 20
        reasons.append(f"{num_paths} vector paths")
    elif num_paths > 100:
        score += 10

    # Has embedded images (common in hybrid CAD exports)
    if num_images > 5:
        score += 10
        reasons.append(f"{num_images} embedded images")

    # Minimal text blocks = not a document/report
    if num_text_blocks < 5 and num_paths < 50 and num_images == 0:
        return False, "Appears to be an empty or minimal PDF"

    # Threshold
    if score >= 30:
        return True, f"Mechanical drawing (score={score}: {', '.join(reasons)})"
    elif score >= 15:
        return True, f"Likely engineering drawing (score={score}: {', '.join(reasons)})"
    else:
        return False, f"Does not appear to be a mechanical drawing (score={score})"
