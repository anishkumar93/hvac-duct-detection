import re
from app.models.schemas import PressureClass


def classify_pressure(dimension_text: str | None, ocr_texts: list[str] = None) -> PressureClass:
    """Classify duct pressure based on dimension and context clues.

    Heuristics:
    - Explicit labels: LP, MP, HP in nearby text
    - Size-based fallback:
        - ≤ 12" or ≤ 12x12 → could be HP (small, high velocity)
        - 12-24" → Medium
        - > 24" → Low (large, low velocity)
    """
    # Check explicit pressure labels in nearby OCR text
    if ocr_texts:
        combined = " ".join(ocr_texts).upper()
        if "HP" in combined or "HIGH PRESS" in combined:
            return PressureClass.HIGH
        if "MP" in combined or "MED PRESS" in combined or "MEDIUM PRESS" in combined:
            return PressureClass.MEDIUM
        if "LP" in combined or "LOW PRESS" in combined:
            return PressureClass.LOW

    if not dimension_text:
        return PressureClass.UNKNOWN

    # Extract numeric dimension
    match = re.search(r"(\d+)", dimension_text)
    if not match:
        return PressureClass.UNKNOWN

    size = int(match.group(1))
    if size <= 12:
        return PressureClass.HIGH
    elif size <= 24:
        return PressureClass.MEDIUM
    else:
        return PressureClass.LOW
