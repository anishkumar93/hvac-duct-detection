"""
Grid-line removal — thickness-based filtering.

Grid lines are drawn with a thinner stroke than duct walls. This module:
1. Isolates all long H/V lines via morphological OPEN.
2. Measures each line CC's thickness (area / length).
3. Finds the natural gap between thin lines (grid) and thick lines (ducts).
4. Erases only the thin cluster.

The key constraint: a duct wall maintains consistent thickness along its
length. A grid line is uniformly thin. When thickness alone can't separate
them (both are 2px), the geometry engine's own MIN_THICKNESS floor handles
the rejection — this filter focuses on cases where there IS a clear
thickness gap.
"""

import os
import cv2
import numpy as np


def strip_grid_lines(
    binary: np.ndarray,
    roi: tuple[int, int, int, int],
    ppi: float | None = None,
    **kwargs,
) -> np.ndarray:
    """Erase thin grid lines that are clearly separable from duct walls.

    Args:
        binary : full-image binary (lines=white 255, bg=black 0)
        roi    : (x1, y1, x2, y2) of the drawing area
        ppi    : pixels per real inch

    Returns:
        A new binary with grid-line CCs removed.
    """
    rx1, ry1, rx2, ry2 = roi
    roi_binary = binary[ry1:ry2, rx1:rx2]
    roi_h, roi_w = roi_binary.shape[:2]
    if roi_h <= 0 or roi_w <= 0:
        return binary

    # Kernel length: long enough to isolate real lines, short enough to keep ducts.
    # Grid lines span large portions of the drawing; duct walls are shorter but
    # still long. Use a moderate kernel that captures both.
    if ppi and ppi > 0:
        min_len = max(50, int(ppi * 18))
    else:
        min_len = max(50, int(max(roi_w, roi_h) * 0.06))

    # ── Step 1: Isolate long H and V lines ────────────────────────────────────
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1))
    h_lines = cv2.morphologyEx(roi_binary, cv2.MORPH_OPEN, h_kernel)

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_len))
    v_lines = cv2.morphologyEx(roi_binary, cv2.MORPH_OPEN, v_kernel)

    lines_mask = cv2.bitwise_or(h_lines, v_lines)

    # ── Step 2: Measure thickness of each line CC ─────────────────────────────
    n, labels, stats, _ = cv2.connectedComponentsWithStats(lines_mask, connectivity=8)
    if n <= 2:
        return binary

    line_info = []  # (cc_index, thickness, length)
    for i in range(1, n):
        cc_w = stats[i, cv2.CC_STAT_WIDTH]
        cc_h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # Skip frame lines
        if cc_w > roi_w * 0.80 or cc_h > roi_h * 0.80:
            continue

        length = max(cc_w, cc_h)
        thickness = area / max(length, 1)
        line_info.append((i, thickness, length))

    if len(line_info) < 3:
        return binary

    # ── Step 3: Find threshold via first significant gap ──────────────────────
    sorted_by_thick = sorted(line_info, key=lambda x: x[1])
    thicknesses = [t for _, t, _ in sorted_by_thick]

    threshold = _find_first_gap(thicknesses)
    if threshold is None:
        return binary

    thin_ccs = [(idx, t) for idx, t, _ in line_info if t <= threshold]
    thick_ccs = [(idx, t) for idx, t, _ in line_info if t > threshold]

    if not thin_ccs or not thick_ccs:
        return binary

    # Validate: thick must be meaningfully heavier than thin
    thin_max = max(t for _, t in thin_ccs)
    thick_min = min(t for _, t in thick_ccs)
    if thick_min < thin_max * 1.5:
        return binary

    print(f"[Grid] Thickness split: thin≤{thin_max:.1f}px ({len(thin_ccs)} CCs) | "
          f"thick≥{thick_min:.1f}px ({len(thick_ccs)} CCs)")

    # ── Step 4: Erase thin CCs ────────────────────────────────────────────────
    grid_mask = np.zeros_like(roi_binary)
    for idx, _ in thin_ccs:
        grid_mask[labels == idx] = 255

    removed = int(np.count_nonzero(grid_mask))
    if removed == 0:
        return binary

    print(f"[Grid] Removed {removed:,} grid-line pixels ({len(thin_ccs)} lines)")
    _dbg(grid_mask, "10_grid_mask.png")

    cleaned = binary.copy()
    cleaned[ry1:ry2, rx1:rx2] = cv2.subtract(roi_binary, grid_mask)
    return cleaned


def _find_first_gap(sorted_values: list[float], min_ratio: float = 0.4) -> float | None:
    """Find the first significant relative gap scanning from thinnest upward.

    Returns midpoint of the gap, or None if no gap ≥ min_ratio found.
    """
    if len(sorted_values) < 2:
        return None

    for i in range(len(sorted_values) - 1):
        lo = sorted_values[i]
        hi = sorted_values[i + 1]
        if lo <= 0:
            continue
        if (hi - lo) / lo >= min_ratio:
            return (lo + hi) / 2.0

    return None


def _dbg(img: np.ndarray, filename: str) -> None:
    dbg_dir = os.path.join(os.path.dirname(__file__), "..", "..", "debug")
    os.makedirs(dbg_dir, exist_ok=True)
    cv2.imwrite(os.path.join(dbg_dir, filename), img)
