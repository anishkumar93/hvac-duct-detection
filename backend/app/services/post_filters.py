"""
Post-detection filters for reducing false positives.

Applies context-aware filtering, connectivity validation, confidence scoring,
boundary exclusion, and scale-based validation for unlabelled ducts.
"""

import math
import re
import numpy as np
from app.models.schemas import BoundingBox
from app.services.ocr import OCRResult


# Keywords near a detection that indicate it's NOT a duct
NON_DUCT_KEYWORDS = [
    'ROOM', 'RM', 'AHU', 'RTU', 'VAV', 'FCU', 'MAU',
    'WALL', 'COLUMN', 'COL', 'BEAM', 'SLAB',
    'DOOR', 'WINDOW', 'STAIR', 'ELEV', 'SHAFT',
    'TOILET', 'KITCHEN', 'OFFICE', 'CORRIDOR',
    'EF', 'SF', 'RF',  # fan tags without dimension context
]

# Keywords that CONFIRM a duct (boost confidence)
DUCT_CONFIRM_KEYWORDS = [
    'CFM', 'SUPPLY', 'RETURN', 'EXHAUST', 'OUTSIDE AIR',
    'SA', 'RA', 'EA', 'OA', 'DUCT', 'FLEX',
]


def filter_by_context(
    boxes: list[BoundingBox],
    associations: dict[int, OCRResult],
    all_ocr: list[OCRResult],
    search_radius: float = 200.0,
) -> tuple[list[BoundingBox], dict]:
    """Remove detections near non-duct text (room labels, equipment tags).

    Only penalizes unlabelled ducts — if a duct already has a dimension label,
    it's confirmed and kept regardless of nearby text.
    """
    if not all_ocr or not boxes:
        return boxes, associations

    valid_boxes = []
    valid_assoc = {}

    for i, box in enumerate(boxes):
        # Labelled ducts are confirmed — always keep
        if i in associations:
            new_i = len(valid_boxes)
            valid_boxes.append(box)
            valid_assoc[new_i] = associations[i]
            continue

        # Check nearby OCR text for non-duct keywords
        is_suspect = False
        for ocr in all_ocr:
            dist = math.hypot(ocr.center_x - box.x, ocr.center_y - box.y)
            if dist > search_radius:
                continue
            text_upper = ocr.text.upper().strip()
            # Skip dimension-like text (numbers with quotes)
            if re.search(r'\d+\s*["\u2033]', text_upper):
                continue
            for kw in NON_DUCT_KEYWORDS:
                if kw in text_upper:
                    is_suspect = True
                    break
            if is_suspect:
                break

        if not is_suspect:
            new_i = len(valid_boxes)
            valid_boxes.append(box)

    removed = len(boxes) - len(valid_boxes)
    if removed:
        print(f"[PostFilter] Context filter removed {removed} detections near non-duct text")
    return valid_boxes, valid_assoc


def validate_connectivity(
    boxes: list[BoundingBox],
    associations: dict[int, OCRResult],
    max_endpoint_dist: float = None,
) -> tuple[list[BoundingBox], dict]:
    """Remove isolated detections that don't connect to any other duct.

    A real duct network is connected — ducts meet at T-junctions, elbows, or
    equipment. An isolated detection with no neighbors is likely a wall segment.

    Labelled ducts are never removed (they're confirmed by OCR).
    """
    if len(boxes) < 2:
        return boxes, associations

    # Auto-compute max endpoint distance from median duct gap
    if max_endpoint_dist is None:
        gaps = [min(b.width, b.height) for b in boxes]
        median_gap = sorted(gaps)[len(gaps) // 2] if gaps else 50
        max_endpoint_dist = median_gap * 4.0

    # Build connectivity: for each box, check if any endpoint is near another duct
    connected = set()
    for i, bi in enumerate(boxes):
        eps_i = _get_endpoints(bi)
        for j, bj in enumerate(boxes):
            if i == j:
                continue
            # Check if endpoint of i is near body or endpoint of j
            for ep in eps_i:
                if _point_near_duct(ep, bj, max_endpoint_dist):
                    connected.add(i)
                    connected.add(j)
                    break
            if i in connected:
                break

    # Keep connected ducts + all labelled ducts
    valid_boxes = []
    valid_assoc = {}
    removed = 0

    for i, box in enumerate(boxes):
        if i in connected or i in associations:
            new_i = len(valid_boxes)
            valid_boxes.append(box)
            if i in associations:
                valid_assoc[new_i] = associations[i]
        else:
            removed += 1

    if removed:
        print(f"[PostFilter] Connectivity filter removed {removed} isolated detections")
    return valid_boxes, valid_assoc


def filter_boundary_detections(
    boxes: list[BoundingBox],
    associations: dict[int, OCRResult],
    roi: tuple[int, int, int, int],
    margin: int = 20,
) -> tuple[list[BoundingBox], dict]:
    """Remove detections that touch the ROI boundary (likely partial walls)."""
    roi_x1, roi_y1, roi_x2, roi_y2 = roi

    valid_boxes = []
    valid_assoc = {}

    for i, box in enumerate(boxes):
        x1 = box.x - box.width / 2
        y1 = box.y - box.height / 2
        x2 = box.x + box.width / 2
        y2 = box.y + box.height / 2

        at_boundary = (
            x1 < roi_x1 + margin or
            y1 < roi_y1 + margin or
            x2 > roi_x2 - margin or
            y2 > roi_y2 - margin
        )

        # Labelled ducts at boundary are kept (confirmed by OCR)
        if not at_boundary or i in associations:
            new_i = len(valid_boxes)
            valid_boxes.append(box)
            if i in associations:
                valid_assoc[new_i] = associations[i]

    removed = len(boxes) - len(valid_boxes)
    if removed:
        print(f"[PostFilter] Boundary filter removed {removed} edge detections")
    return valid_boxes, valid_assoc


def validate_scale_unlabelled(
    boxes: list[BoundingBox],
    associations: dict[int, OCRResult],
    ppi: float | None,
) -> tuple[list[BoundingBox], dict]:
    """Remove unlabelled ducts whose gap doesn't match known duct sizes on this drawing.

    If we have PPI and confirmed labelled ducts, unlabelled detections whose
    gap falls far outside the range of confirmed sizes are likely walls.
    """
    if not ppi or not associations:
        return boxes, associations

    # Collect confirmed duct gaps (in real inches)
    confirmed_inches = []
    for i, label in associations.items():
        if i < len(boxes):
            gap_px = min(boxes[i].width, boxes[i].height)
            confirmed_inches.append(gap_px / ppi)

    if not confirmed_inches:
        return boxes, associations

    min_real = min(confirmed_inches)
    max_real = max(confirmed_inches)

    # Allow range: 50% below smallest to 200% above largest
    low_bound = min_real * 0.5
    high_bound = max_real * 2.0

    valid_boxes = []
    valid_assoc = {}
    removed = 0

    for i, box in enumerate(boxes):
        if i in associations:
            new_i = len(valid_boxes)
            valid_boxes.append(box)
            valid_assoc[new_i] = associations[i]
            continue

        gap_inches = min(box.width, box.height) / ppi
        if low_bound <= gap_inches <= high_bound:
            valid_boxes.append(box)
        else:
            removed += 1

    if removed:
        print(f"[PostFilter] Scale validation removed {removed} unlabelled ducts "
              f"(expected {low_bound:.1f}\"-{high_bound:.1f}\")")
    return valid_boxes, valid_assoc


def compute_confidence_scores(
    boxes: list[BoundingBox],
    associations: dict[int, OCRResult],
    all_ocr: list[OCRResult],
    ppi: float | None,
    roi: tuple[int, int, int, int],
) -> list[float]:
    """Compute multi-factor confidence score for each detection.

    Factors:
    - Has dimension label: +0.30
    - Has duct-confirming keyword nearby: +0.10
    - Has connected neighbor: +0.10
    - Scale-validated (if PPI available): +0.10
    - Near non-duct text: -0.15
    - Isolated (no neighbors): -0.10
    - At ROI boundary: -0.05

    Base confidence: 0.50
    """
    if not boxes:
        return []

    n = len(boxes)
    scores = [0.50] * n
    roi_x1, roi_y1, roi_x2, roi_y2 = roi

    # Pre-compute connectivity
    gaps = [min(b.width, b.height) for b in boxes]
    median_gap = sorted(gaps)[len(gaps) // 2] if gaps else 50
    max_ep_dist = median_gap * 4.0
    connected = _compute_connected_set(boxes, max_ep_dist)

    # Pre-compute confirmed duct size range
    confirmed_inches = []
    if ppi and associations:
        for i, label in associations.items():
            if i < n:
                confirmed_inches.append(min(boxes[i].width, boxes[i].height) / ppi)

    for i, box in enumerate(boxes):
        # Has dimension label
        if i in associations:
            scores[i] += 0.30

        # Connected to network
        if i in connected:
            scores[i] += 0.10
        else:
            scores[i] -= 0.10

        # Scale validated (unlabelled only)
        if ppi and confirmed_inches and i not in associations:
            gap_inches = min(box.width, box.height) / ppi
            min_r, max_r = min(confirmed_inches) * 0.5, max(confirmed_inches) * 2.0
            if min_r <= gap_inches <= max_r:
                scores[i] += 0.10

        # Boundary proximity
        margin = 20
        x1 = box.x - box.width / 2
        y1 = box.y - box.height / 2
        x2 = box.x + box.width / 2
        y2 = box.y + box.height / 2
        if x1 < roi_x1 + margin or y1 < roi_y1 + margin or x2 > roi_x2 - margin or y2 > roi_y2 - margin:
            scores[i] -= 0.05

        # Context from nearby OCR
        if all_ocr:
            for ocr in all_ocr:
                dist = math.hypot(ocr.center_x - box.x, ocr.center_y - box.y)
                if dist > 200:
                    continue
                text_upper = ocr.text.upper().strip()
                if re.search(r'\d+\s*["\u2033]', text_upper):
                    continue
                for kw in NON_DUCT_KEYWORDS:
                    if kw in text_upper:
                        scores[i] -= 0.15
                        break
                for kw in DUCT_CONFIRM_KEYWORDS:
                    if kw in text_upper:
                        scores[i] += 0.10
                        break

        # Clamp
        scores[i] = max(0.0, min(1.0, scores[i]))

    return scores


def compute_adaptive_gap(
    confirmed_boxes: list[BoundingBox],
    current_min: int,
    current_max: int,
) -> tuple[int, int]:
    """Refine gap window based on confirmed duct sizes.

    After the first detection pass with labels, narrow the gap window
    for a potential second pass or for filtering.
    """
    if not confirmed_boxes:
        return current_min, current_max

    gaps = [min(b.width, b.height) for b in confirmed_boxes]
    if not gaps:
        return current_min, current_max

    median_gap = sorted(gaps)[len(gaps) // 2]
    refined_min = max(current_min, int(median_gap * 0.4))
    refined_max = min(current_max, int(median_gap * 2.5))

    # Safety: don't make window too narrow
    if refined_max - refined_min < 20:
        return current_min, current_max

    print(f"[PostFilter] Adaptive gap: [{refined_min}-{refined_max}]px "
          f"(median confirmed={median_gap:.0f}px)")
    return refined_min, refined_max


def filter_closed_short_corridors(
    boxes: list[BoundingBox],
    associations: dict[int, OCRResult],
    roi_binary: np.ndarray,
    roi_offset: tuple[int, int],
) -> tuple[list[BoundingBox], dict]:
    """Remove nearly-square detections with internal structure (equipment boxes).

    Equipment outlines (AHU, VAV, diffuser frames) are roughly square and
    contain internal subdivisions. Real ducts are elongated and hollow.
    """
    offset_x, offset_y = roi_offset
    img_h, img_w = roi_binary.shape[:2]

    valid_boxes = []
    valid_assoc = {}
    removed = 0

    for i, box in enumerate(boxes):
        # Labelled ducts are always kept
        if i in associations:
            new_i = len(valid_boxes)
            valid_boxes.append(box)
            valid_assoc[new_i] = associations[i]
            continue

        length = max(box.width, box.height)
        gap = min(box.width, box.height)

        # Only check nearly-square detections (aspect < 3.0)
        if gap > 0 and length / gap < 3.0:
            # Sample the corridor interior for internal structure
            lx = int(box.x - box.width / 2 - offset_x)
            ly = int(box.y - box.height / 2 - offset_y)
            rx = int(box.x + box.width / 2 - offset_x)
            ry = int(box.y + box.height / 2 - offset_y)

            lx, ly = max(0, lx), max(0, ly)
            rx, ry = min(img_w, rx), min(img_h, ry)

            if rx > lx and ry > ly:
                corridor = roi_binary[ly:ry, lx:rx]
                # Check for internal horizontal or vertical lines
                k_len = max(10, int(gap * 0.4))
                h_internal = cv2.morphologyEx(
                    corridor, cv2.MORPH_OPEN,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (k_len, 1))
                )
                v_internal = cv2.morphologyEx(
                    corridor, cv2.MORPH_OPEN,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_len))
                )
                internal_fill = (np.count_nonzero(h_internal) + np.count_nonzero(v_internal)) / max(corridor.size, 1)

                if internal_fill > 0.08:
                    removed += 1
                    continue

        new_i = len(valid_boxes)
        valid_boxes.append(box)
        if i in associations:
            valid_assoc[new_i] = associations[i]

    if removed:
        print(f"[PostFilter] Equipment box filter removed {removed} short/square detections")
    return valid_boxes, valid_assoc


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _get_endpoints(box: BoundingBox) -> list[tuple[float, float]]:
    """Get the two endpoints of a duct (ends of the longer axis)."""
    is_h = box.width > box.height
    if is_h:
        return [
            (box.x - box.width / 2, box.y),
            (box.x + box.width / 2, box.y),
        ]
    else:
        return [
            (box.x, box.y - box.height / 2),
            (box.x, box.y + box.height / 2),
        ]


def _point_near_duct(point: tuple[float, float], duct: BoundingBox, max_dist: float) -> bool:
    """Check if a point is near a duct's body or endpoints."""
    px, py = point

    # Near endpoint
    eps = _get_endpoints(duct)
    for ep in eps:
        if math.hypot(px - ep[0], py - ep[1]) < max_dist:
            return True

    # Near body (perpendicular distance to duct centerline)
    is_h = duct.width > duct.height
    if is_h:
        # Point must be within x-range of duct
        if duct.x - duct.width / 2 - max_dist < px < duct.x + duct.width / 2 + max_dist:
            if abs(py - duct.y) < max_dist:
                return True
    else:
        if duct.y - duct.height / 2 - max_dist < py < duct.y + duct.height / 2 + max_dist:
            if abs(px - duct.x) < max_dist:
                return True

    return False


def _compute_connected_set(boxes: list[BoundingBox], max_dist: float) -> set[int]:
    """Return indices of boxes that are connected to at least one other box."""
    connected = set()
    for i, bi in enumerate(boxes):
        if i in connected:
            continue
        eps_i = _get_endpoints(bi)
        for j, bj in enumerate(boxes):
            if i == j:
                continue
            for ep in eps_i:
                if _point_near_duct(ep, bj, max_dist):
                    connected.add(i)
                    connected.add(j)
                    break
            if i in connected:
                break
    return connected


# Import cv2 at module level (needed for equipment box filter)
import cv2
