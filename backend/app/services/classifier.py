"""Pressure classification based on duct dimension."""
import re
from app.models.schemas import PressureClass


def classify_pressure(dimension_text: str | None) -> PressureClass:
    """Classify duct pressure based on dimension size.
    ≤12" → High, 13-24" → Medium, >24" → Low.
    """
    if not dimension_text:
        return PressureClass.UNKNOWN

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
