"""PDF vector data extraction for hybrid detection pipeline.

Detects whether a PDF is vector, raster, or hybrid, and extracts
structured data (text + line geometry) when available.
"""
import re
import math

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


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
