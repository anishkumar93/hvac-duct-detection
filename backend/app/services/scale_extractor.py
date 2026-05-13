"""Drawing scale extraction and duct size validation.

Extracts scale from title block (e.g., 1/4"=1'-0") and uses it to
validate detected duct dimensions against their pixel sizes.
"""
import re


# Common scale patterns in HVAC drawings
SCALE_PATTERNS = [
    # 1/4"=1'-0", 3/8"=1'-0", 1/2"=1'-0"
    re.compile(r'(\d+)/(\d+)\s*["\u2033]\s*=\s*(\d+)\s*[\'\u2032]\s*-?\s*(\d+)\s*["\u2033]?'),
    # 1"=1'-0"
    re.compile(r'(\d+)\s*["\u2033]\s*=\s*(\d+)\s*[\'\u2032]\s*-?\s*(\d+)\s*["\u2033]?'),
]


def extract_scale(ocr_results: list) -> dict | None:
    """Find and parse drawing scale from OCR results.
    Returns dict with pixels_per_inch or None if not found.
    """
    for r in ocr_results:
        text = r.text if hasattr(r, 'text') else str(r)

        for pattern in SCALE_PATTERNS:
            m = pattern.search(text)
            if m:
                groups = m.groups()
                if len(groups) == 4:
                    # Fraction: 1/4"=1'-0"
                    numerator = int(groups[0])
                    denominator = int(groups[1])
                    feet = int(groups[2])
                    inches = int(groups[3])
                    paper_inches = numerator / denominator
                    real_inches = feet * 12 + inches
                elif len(groups) == 3:
                    # Whole: 1"=1'-0"
                    paper_inches = int(groups[0])
                    feet = int(groups[1])
                    inches = int(groups[2])
                    real_inches = feet * 12 + inches
                else:
                    continue

                if real_inches == 0:
                    continue

                # Scale ratio: how many real inches per paper inch
                scale_ratio = real_inches / paper_inches

                return {
                    'text': text,
                    'paper_inches': paper_inches,
                    'real_inches': real_inches,
                    'scale_ratio': scale_ratio,
                }

    return None


def compute_pixels_per_inch(scale: dict, image_width: int, page_width_pts: float = None) -> float:
    """Compute pixels per real-world inch given the scale and image dimensions.
    
    For a 300 DPI scan: 1 paper inch = 300 pixels
    At scale 1/4"=1'-0": 1/4 paper inch = 12 real inches
    So 1 real inch = (300 * 0.25) / 12 = 6.25 pixels
    """
    # Assume 300 DPI if we don't know the actual scan resolution
    # Can be refined if page_width_pts is known (PDF page width)
    dpi = 300
    if page_width_pts:
        # PDF points to pixels: page_width_pts / 72 = page inches
        page_inches = page_width_pts / 72
        dpi = image_width / page_inches

    pixels_per_paper_inch = dpi
    pixels_per_real_inch = (pixels_per_paper_inch * scale['paper_inches']) / scale['real_inches']

    return pixels_per_real_inch


def validate_duct_dimension(dimension_text: str, duct_thickness_px: float,
                            pixels_per_inch: float, tolerance: float = 0.5) -> bool:
    """Check if a duct's pixel thickness matches its stated dimension.
    
    Args:
        dimension_text: e.g., "18\"⌀" or "22\"×14\""
        duct_thickness_px: measured pixel gap between duct walls
        pixels_per_inch: from compute_pixels_per_inch()
        tolerance: allowed error ratio (0.5 = 50%)
    
    Returns True if the dimension is plausible at this scale.
    """
    # Extract the dimension value (first number for diameter, second for height in WxH)
    dim_match = re.search(r'(\d+)', dimension_text)
    if not dim_match:
        return True  # Can't validate, assume OK

    dim_inches = int(dim_match.group(1))
    expected_px = dim_inches * pixels_per_inch

    if expected_px == 0:
        return True

    error = abs(duct_thickness_px - expected_px) / expected_px

    return error <= tolerance
