"""LLM-assisted post-processing for duct detection.

Uses Claude to reason about structured vector data extracted from PDFs,
identifying ducts, classifying supply/return, and rejecting false positives.
"""
import os
import json
import math

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import fitz
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

ANTHROPIC_API_KEY = None


def _get_api_key():
    global ANTHROPIC_API_KEY
    if ANTHROPIC_API_KEY is None:
        from dotenv import load_dotenv
        load_dotenv()
        ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
    return ANTHROPIC_API_KEY

SYSTEM_PROMPT = """You are an HVAC mechanical drawing analyst. You receive structured data extracted from engineering drawings (text labels with positions, and nearby line geometry).

Your task: Identify HVAC duct segments from the data.

Rules:
- Duct dimensions use formats: 18"⌀ (round), 22"x14" (rectangular), or just 18" with a diameter symbol (⌀, Ø, ∅, DIA)
- Ceiling heights use format: 10'-0", 8'-6" (feet-inches with apostrophe)
- Room numbers are plain numbers: 101, 102, 103
- Grid references are letters+numbers at edges: A.1, B.1, C.1
- CFM values are numbers followed by "CFM": 400 CFM
- Equipment tags have prefixes: RTU-1, DOAS-1, EF-1
- A duct is confirmed when dimension text sits between or on top of two parallel lines
- Supply ducts are labeled "S", "SA", "SUPPLY"
- Return ducts are labeled "R", "RA", "RETURN"
- Exhaust ducts are labeled "E", "EA", "EXHAUST"

Respond ONLY with valid JSON."""

USER_PROMPT_TEMPLATE = """Analyze this HVAC drawing data. Page size: {page_width}x{page_height} points.

Text elements with nearby geometry:
{text_data}

Identified parallel line pairs (potential ducts):
{line_pairs}

For each text element, determine:
1. Is it a duct dimension? (not a room number, ceiling height, or grid reference)
2. If yes, what is the duct size?
3. What type? (supply/return/exhaust/unknown)
4. Which parallel line pair does it belong to?

Respond with JSON:
{{
  "ducts": [
    {{
      "dimension": "18\\"⌀",
      "type": "supply|return|exhaust|unknown",
      "center_x": <x position>,
      "center_y": <y position>,
      "width": <duct length in points>,
      "height": <duct cross-section in points>,
      "orientation": "horizontal|vertical",
      "confidence": 0.0-1.0,
      "reasoning": "brief explanation"
    }}
  ],
  "rejected_texts": [
    {{"text": "10'-0\\"", "reason": "ceiling height (C02 tag)"}}
  ]
}}"""


def extract_structured_data(pdf_path: str) -> dict | None:
    """Extract structured spatial data from vector PDF for LLM analysis."""
    if not HAS_PYMUPDF:
        return None

    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return None

    page = doc[0]
    pw, ph = page.rect.width, page.rect.height

    # Extract text with positions
    texts = []
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
                texts.append({
                    'text': text,
                    'x': round(bbox[0], 1),
                    'y': round(bbox[1], 1),
                })

    # Extract lines
    all_lines = []
    paths = page.get_drawings()
    for p in paths:
        for item in p['items']:
            if item[0] == 'l':
                x1, y1 = item[1].x, item[1].y
                x2, y2 = item[2].x, item[2].y
                length = math.hypot(x2 - x1, y2 - y1)
                if length < 20:
                    continue
                all_lines.append((x1, y1, x2, y2, length))

    doc.close()

    # Build spatial context: for each text, find nearby lines
    text_with_context = []
    for t in texts:
        tx, ty = t['x'], t['y']
        nearby = []
        for x1, y1, x2, y2, length in all_lines:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            dist = math.hypot(tx - mx, ty - my)
            if dist < 100:
                dx, dy = abs(x2 - x1), abs(y2 - y1)
                orient = 'H' if dx > dy * 3 else ('V' if dy > dx * 3 else 'D')
                nearby.append({
                    'type': orient,
                    'length': round(length),
                    'x1': round(x1), 'y1': round(y1),
                    'x2': round(x2), 'y2': round(y2),
                })

        if nearby:
            text_with_context.append({
                'text': t['text'],
                'x': t['x'],
                'y': t['y'],
                'nearby_lines': nearby[:8],  # Limit to avoid token bloat
            })

    # Find parallel line pairs (potential duct walls)
    h_lines = [(x1, y1, x2, y2, l) for x1, y1, x2, y2, l in all_lines
               if abs(x2 - x1) > abs(y2 - y1) * 3 and l > 50]
    h_lines.sort(key=lambda l: l[1])

    line_pairs = []
    used = set()
    for i, li in enumerate(h_lines):
        if i in used:
            continue
        for j in range(i + 1, len(h_lines)):
            if j in used:
                continue
            lj = h_lines[j]
            gap = abs(lj[1] - li[1])
            if gap < 5:
                continue
            if gap > 80:
                break
            # Check overlap
            overlap_start = max(li[0], lj[0])
            overlap_end = min(li[2], lj[2])
            overlap = overlap_end - overlap_start
            min_len = min(li[4], lj[4])
            if overlap > min_len * 0.3:
                line_pairs.append({
                    'line1_y': round(li[1]),
                    'line2_y': round(lj[1]),
                    'x_start': round(overlap_start),
                    'x_end': round(overlap_end),
                    'gap': round(gap),
                    'length': round(overlap),
                })
                used.add(i)
                used.add(j)
                break

    return {
        'page_width': round(pw),
        'page_height': round(ph),
        'text_with_context': text_with_context,
        'line_pairs': line_pairs[:30],  # Limit pairs
    }


def llm_identify_ducts(pdf_path: str, ocr_dimensions: list = None, detected_ducts: list = None) -> list[dict] | None:
    """Use Claude to identify ducts from structured data.
    For vector PDFs: uses vector text + geometry.
    For hybrid/raster: uses OCR results + detected duct positions.
    """
    if not HAS_ANTHROPIC or not _get_api_key():
        print("[LLM] Skipped: Anthropic API key not set (export ANTHROPIC_API_KEY=...)")
        return None

    structured = extract_structured_data(pdf_path) if HAS_PYMUPDF else None

    # If we have OCR dimensions + detected ducts, use those (hybrid/raster PDF)
    if ocr_dimensions or detected_ducts:
        return _llm_from_ocr(ocr_dimensions or [], detected_ducts or [], structured)

    # Otherwise try vector-only
    if structured and structured['text_with_context']:
        return _llm_from_vector(structured)

    print("[LLM] Skipped: No data to analyze")
    return None


def _llm_from_vector(structured: dict) -> list[dict] | None:
    """Send vector PDF data to Claude."""
    text_data = json.dumps(structured['text_with_context'][:60], indent=2)
    line_pairs = json.dumps(structured['line_pairs'], indent=2)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        page_width=structured['page_width'],
        page_height=structured['page_height'],
        text_data=text_data,
        line_pairs=line_pairs,
    )
    return _call_claude(user_prompt)


def _llm_from_ocr(ocr_dims: list, ducts: list, vector_data: dict = None) -> list[dict] | None:
    """Send OCR results + detected duct positions to Claude for validation."""
    ocr_info = []
    for dim in ocr_dims:
        ocr_info.append({
            'text': dim.text if hasattr(dim, 'text') else str(dim),
            'x': round(dim.center_x) if hasattr(dim, 'center_x') else 0,
            'y': round(dim.center_y) if hasattr(dim, 'center_y') else 0,
            'confidence': round(dim.confidence, 2) if hasattr(dim, 'confidence') else 0.5,
        })

    duct_info = []
    for d in ducts:
        bbox = d.bbox if hasattr(d, 'bbox') else d
        duct_info.append({
            'id': d.id if hasattr(d, 'id') else 0,
            'dimension': d.dimension if hasattr(d, 'dimension') else None,
            'center_x': round(bbox.x),
            'center_y': round(bbox.y),
            'width': round(bbox.width),
            'height': round(bbox.height),
            'orientation': 'horizontal' if bbox.width > bbox.height else 'vertical',
        })

    vector_texts = []
    if vector_data and vector_data.get('text_with_context'):
        vector_texts = [{'text': t['text'], 'x': round(t['x']), 'y': round(t['y'])}
                        for t in vector_data['text_with_context'][:50]]

    user_prompt = f"""Analyze this HVAC mechanical drawing data.

OCR-detected dimension labels (may have misreads):
{json.dumps(ocr_info, indent=2)}

Detected duct segments (from image analysis):
{json.dumps(duct_info, indent=2)}

Vector text from PDF (room names, notes, equipment tags - these are accurate):
{json.dumps(vector_texts, indent=2)}

Tasks:
1. For each detected duct, validate/correct the OCR dimension.
2. Classify each duct type (supply/return/exhaust/unknown) using nearby vector text labels.
3. Identify any false positives that should be rejected.
4. Identify any ducts that OCR found dimensions for but image analysis missed.

Respond with JSON:
{{
  "ducts": [
    {{
      "id": 0,
      "dimension": "corrected dimension",
      "type": "supply|return|exhaust|unknown",
      "center_x": 0,
      "center_y": 0,
      "width": 0,
      "height": 0,
      "orientation": "horizontal|vertical",
      "confidence": 0.9,
      "reasoning": "brief explanation"
    }}
  ],
  "rejected_ducts": [
    {{"id": 1, "reason": "why not a duct"}}
  ]
}}"""

    return _call_claude(user_prompt)


def _call_claude(user_prompt: str) -> list[dict] | None:
    """Make the API call to Claude."""
    print(f"[LLM] Sending {len(user_prompt)} chars to Claude")

    try:
        client = anthropic.Anthropic(api_key=_get_api_key())
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        content = response.content[0].text
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        result = json.loads(content)
        ducts = result.get('ducts', [])
        rejected = result.get('rejected_ducts', result.get('rejected_texts', []))

        print(f"[LLM] Identified {len(ducts)} ducts, rejected {len(rejected)}")
        for d in ducts:
            print(f"[LLM]   {d.get('dimension', '?')} ({d.get('type', '?')}) - {d.get('reasoning', '')}")

        return ducts

    except Exception as e:
        print(f"[LLM] Error: {e}")
        return None
