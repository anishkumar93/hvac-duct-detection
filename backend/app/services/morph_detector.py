"""
Morphological + Contour-based duct detector.
Detects ducts purely from image geometry — no OCR, no vector data.

Pipeline:
1. Binarize & invert (adaptive threshold)
2. Mask out title block and page borders (ROI extraction)
3. Double-pass morphological filtering:
   a. Small kernel to preserve all H/V elements
   b. Connected components to measure each segment's dimensions
   c. Filter by thickness (reject single grid lines) and length (reject borders)
4. Contour-based detection with aspect ratio filtering
5. Strict parallel-line pairing by duct width constraint
"""

import cv2
import numpy as np
from app.models.schemas import BoundingBox


# Duct width range as fraction of image short side.
DUCT_WIDTH_FRAC_MIN = 0.0025
DUCT_WIDTH_FRAC_MAX = 0.015
# Absolute fallback bounds
DUCT_WIDTH_ABS_MIN = 12
DUCT_WIDTH_ABS_MAX = 120
# Minimum duct length as fraction of image dimension
MIN_LENGTH_FRAC = 0.015
MIN_ASPECT_RATIO = 4.0
# Maximum length: reject lines longer than this fraction (likely walls/borders)
MAX_LENGTH_FRAC = 0.35


def detect_ducts_morphological(image_path: str, debug_dir: str = None) -> tuple[list[BoundingBox], int, int]:
    """Detect ducts using morphology + contour analysis + parallel line pairing.
    Returns (list of BoundingBox, image_width, image_height).
    """
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # 1. Binarize & invert
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean)

    # 2. Mask out title block and page borders
    roi_mask = _create_roi_mask(binary, w, h)
    binary = cv2.bitwise_and(binary, roi_mask)

    # Compute scale-dependent thresholds
    short_side = min(w, h)
    duct_w_min = max(DUCT_WIDTH_ABS_MIN, int(short_side * DUCT_WIDTH_FRAC_MIN))
    duct_w_max = min(DUCT_WIDTH_ABS_MAX, int(short_side * DUCT_WIDTH_FRAC_MAX))
    min_length = max(60, int(max(w, h) * MIN_LENGTH_FRAC))
    max_length = int(max(w, h) * MAX_LENGTH_FRAC)
    print(f"[MorphDetector] Thresholds: duct_width={duct_w_min}-{duct_w_max}px, length={min_length}-{max_length}px")

    # 3. Drop thin lines from binary
    #    Engineering drawings use line weight hierarchy:
    #    - Thinnest (1-3px): dimension lines, grid lines, center lines, hatching
    #    - Medium (4-7px): room walls, partition lines
    #    - Heavy (8+px): duct walls, section cuts, equipment outlines
    #    We drop everything below the threshold to keep only heavy lines.
    #    Scale-dependent: for large images (10800px), threshold ~5px;
    #    for smaller images (3000px), threshold ~3px.
    min_line_weight = max(3, min(10, int(short_side * 0.0007)))
    binary = _remove_thin_lines(binary, min_thickness=min_line_weight)

    # 4. Double-pass morphological filtering + connected components
    h_segments = _extract_line_segments(binary, axis='h', img_w=w, img_h=h,
                                        min_length=min_length, max_length=max_length,
                                        max_thickness=duct_w_max // 2)
    v_segments = _extract_line_segments(binary, axis='v', img_w=w, img_h=h,
                                        min_length=min_length, max_length=max_length,
                                        max_thickness=duct_w_max // 2)

    print(f"[MorphDetector] Filtered segments: {len(h_segments)}H, {len(v_segments)}V")

    # 5. Strict parallel-line pairing by duct width
    boxes = []
    h_boxes = _pair_by_duct_width(h_segments, axis='h', duct_w_min=duct_w_min,
                                   duct_w_max=duct_w_max, min_length=min_length)
    v_boxes = _pair_by_duct_width(v_segments, axis='v', duct_w_min=duct_w_min,
                                   duct_w_max=duct_w_max, min_length=min_length)
    boxes.extend(h_boxes)
    boxes.extend(v_boxes)

    # Filter: final aspect ratio check on paired results
    boxes = [b for b in boxes
             if max(b.width, b.height) / (min(b.width, b.height) + 1) >= MIN_ASPECT_RATIO]

    # Merge overlapping
    boxes = _merge_overlapping(boxes)

    if debug_dir:
        _save_debug(img, binary, h_segments, v_segments, boxes, debug_dir)

    print(f"[MorphDetector] Final: {len(boxes)} ducts ({len(h_boxes)}H + {len(v_boxes)}V)")
    return boxes, w, h


def _remove_thin_lines(binary: np.ndarray, min_thickness: int = 10) -> np.ndarray:
    """Remove line structures thinner than min_thickness from the binary image.

    Logic: Uses connected components on morphologically-isolated H/V lines.
    Any connected component whose thickness (height for H lines, width for V lines)
    is below min_thickness gets erased. This drops:
    - Dimension lines (1-3px)
    - Grid lines (1-2px)
    - Center lines (1-2px dashed)
    - Thin partition walls (2-5px)

    While preserving:
    - Duct walls (typically 7-20px in engineering drawings)
    - Equipment outlines
    - Section cut lines
    """
    result = binary.copy()

    # Process horizontal thin lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    h_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(h_mask, connectivity=8)
    for i in range(1, num_labels):
        thickness = stats[i, cv2.CC_STAT_HEIGHT]
        if thickness < min_thickness:
            result[labels == i] = 0

    # Process vertical thin lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    v_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(v_mask, connectivity=8)
    for i in range(1, num_labels):
        thickness = stats[i, cv2.CC_STAT_WIDTH]
        if thickness < min_thickness:
            result[labels == i] = 0

    removed = cv2.countNonZero(binary) - cv2.countNonZero(result)
    total = cv2.countNonZero(binary)
    print(f"[MorphDetector] Thin line removal (<{min_thickness}px): "
          f"dropped {removed} pixels ({removed * 100 // (total + 1)}% of content)")

    return result


def _create_roi_mask(binary: np.ndarray, w: int, h: int) -> np.ndarray:
    """Dynamically detect the drawing area by:
    1. Finding page border lines (long lines spanning most of the page)
    2. Identifying the inner drawing rectangle from border intersections
    3. Detecting title block / notes zones using internal dividers + density
       (works regardless of whether title block is left, right, top, or bottom)
    """
    mask = np.ones((h, w), dtype=np.uint8) * 255

    # --- Step 1: Find page border frame ---
    left_x, top_y, right_x, bottom_y = _detect_inner_frame(binary, w, h)

    # --- Step 2: Detect title block zone ---
    # Priority 1: internal divider lines (most reliable)
    # Priority 2: density-based strip analysis (fallback)
    title_bounds = _find_internal_divider(binary, left_x, top_y, right_x, bottom_y)
    if not title_bounds:
        title_bounds = _detect_title_block_by_density(binary, left_x, top_y, right_x, bottom_y)

    if title_bounds:
        t_left, t_top, t_right, t_bottom = title_bounds
        # Shrink drawing area to exclude title block
        if t_left <= left_x and t_right < right_x:
            left_x = t_right  # title block on left
        elif t_right >= right_x and t_left > left_x:
            right_x = t_left  # title block on right
        elif t_top <= top_y and t_bottom < bottom_y:
            top_y = t_bottom  # title block on top
        elif t_bottom >= bottom_y and t_top > top_y:
            bottom_y = t_top  # title block on bottom

    # Apply mask
    mask[:, :left_x] = 0
    mask[:, right_x:] = 0
    mask[:top_y, :] = 0
    mask[bottom_y:, :] = 0

    print(f"[MorphDetector] ROI: x={left_x}-{right_x}, y={top_y}-{bottom_y} "
          f"({(right_x-left_x)/w*100:.0f}%W x {(bottom_y-top_y)/h*100:.0f}%H)")

    return mask


def _detect_inner_frame(binary: np.ndarray, w: int, h: int) -> tuple:
    """Find the inner drawing frame by detecting page border lines.
    Only considers lines spanning >70% of the page dimension (true borders,
    not internal section dividers). Only looks in the outer 10% margins.
    Returns (left_x, top_y, right_x, bottom_y).
    """
    # Detect very long vertical lines (>70% of page height)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h * 0.7)))
    v_borders = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    col_sums = np.sum(v_borders, axis=0) / 255
    v_border_cols = np.where(col_sums > h * 0.5)[0]

    # Detect very long horizontal lines (>70% of page width)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.7), 1))
    h_borders = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    row_sums = np.sum(h_borders, axis=1) / 255
    h_border_rows = np.where(row_sums > w * 0.5)[0]

    v_clusters = _cluster_positions(v_border_cols)
    h_clusters = _cluster_positions(h_border_rows)

    # Left: innermost border in left 10%
    left_x = 0
    for c in v_clusters:
        mid = (c[0] + c[-1]) // 2
        if mid < w * 0.1:
            left_x = max(left_x, c[-1] + 5)

    # Right: innermost border in right 10%
    right_x = w
    for c in reversed(v_clusters):
        mid = (c[0] + c[-1]) // 2
        if mid > w * 0.9:
            right_x = min(right_x, c[0] - 5)

    # Top: innermost border in top 10%
    top_y = 0
    for c in h_clusters:
        mid = (c[0] + c[-1]) // 2
        if mid < h * 0.1:
            top_y = max(top_y, c[-1] + 5)

    # Bottom: innermost border in bottom 10%
    bottom_y = h
    for c in reversed(h_clusters):
        mid = (c[0] + c[-1]) // 2
        if mid > h * 0.9:
            bottom_y = min(bottom_y, c[0] - 5)

    # Fallback: use small margin if no borders detected
    if left_x == 0:
        left_x = int(w * 0.02)
    if right_x == w:
        right_x = int(w * 0.98)
    if top_y == 0:
        top_y = int(h * 0.02)
    if bottom_y == h:
        bottom_y = int(h * 0.98)

    return left_x, top_y, right_x, bottom_y


def _find_internal_divider(binary: np.ndarray, left_x: int, top_y: int,
                            right_x: int, bottom_y: int) -> tuple | None:
    """Find internal divider lines that separate drawing from title block.
    These are lines that span a significant portion (>30%) of one dimension
    and sit between 60-95% along the perpendicular axis.

    Validates by comparing pixel density on each side of the divider:
    the title block side will have higher density (more text/small boxes).

    Works for title blocks on any edge (left, right, top, bottom).
    """
    frame_w = right_x - left_x
    frame_h = bottom_y - top_y
    if frame_w < 100 or frame_h < 100:
        return None

    roi = binary[top_y:bottom_y, left_x:right_x]

    # --- Vertical dividers (title block on left or right) ---
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(frame_h * 0.3)))
    v_lines = cv2.morphologyEx(roi, cv2.MORPH_OPEN, v_kernel)
    col_sums = np.sum(v_lines, axis=0) / 255
    v_positions = np.where(col_sums > frame_h * 0.25)[0]
    v_clusters = _cluster_positions(v_positions)

    # Collect all valid right-side and left-side divider candidates
    right_candidates = []  # (frac, mid) — closer to edge = more likely title block
    left_candidates = []

    for c in v_clusters:
        mid = (c[0] + c[-1]) // 2
        frac = mid / frame_w

        if 0.6 < frac < 0.95:
            left_area = frame_h * mid
            right_area = frame_h * (frame_w - mid)
            if left_area == 0 or right_area == 0:
                continue
            left_density = cv2.countNonZero(roi[:, :mid]) / left_area
            right_density = cv2.countNonZero(roi[:, mid:]) / right_area
            if right_density > left_density * 1.2:
                right_candidates.append((frac, mid))

        elif 0.05 < frac < 0.4:
            left_area = frame_h * mid
            right_area = frame_h * (frame_w - mid)
            if left_area == 0 or right_area == 0:
                continue
            left_density = cv2.countNonZero(roi[:, :mid]) / left_area
            right_density = cv2.countNonZero(roi[:, mid:]) / right_area
            if left_density > right_density * 1.2:
                left_candidates.append((frac, mid))

    # Pick the divider closest to the edge (excludes least drawing area)
    if right_candidates:
        # Highest frac = closest to right edge
        _, mid = max(right_candidates, key=lambda x: x[0])
        return (left_x + mid, top_y, right_x, bottom_y)
    if left_candidates:
        # Lowest frac = closest to left edge
        _, mid = min(left_candidates, key=lambda x: x[0])
        return (left_x, top_y, left_x + mid, bottom_y)

    # --- Horizontal dividers (title block on top or bottom) ---
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(frame_w * 0.3), 1))
    h_lines = cv2.morphologyEx(roi, cv2.MORPH_OPEN, h_kernel)
    row_sums = np.sum(h_lines, axis=1) / 255
    h_positions = np.where(row_sums > frame_w * 0.25)[0]
    h_clusters = _cluster_positions(h_positions)

    bottom_candidates = []
    top_candidates = []

    for c in h_clusters:
        mid = (c[0] + c[-1]) // 2
        frac = mid / frame_h

        if 0.6 < frac < 0.95:
            top_area = mid * frame_w
            bot_area = (frame_h - mid) * frame_w
            if top_area == 0 or bot_area == 0:
                continue
            top_density = cv2.countNonZero(roi[:mid, :]) / top_area
            bot_density = cv2.countNonZero(roi[mid:, :]) / bot_area
            if bot_density > top_density * 1.2:
                bottom_candidates.append((frac, mid))

        elif 0.05 < frac < 0.4:
            top_area = mid * frame_w
            bot_area = (frame_h - mid) * frame_w
            if top_area == 0 or bot_area == 0:
                continue
            top_density = cv2.countNonZero(roi[:mid, :]) / top_area
            bot_density = cv2.countNonZero(roi[mid:, :]) / bot_area
            if top_density > bot_density * 1.2:
                top_candidates.append((frac, mid))

    if bottom_candidates:
        _, mid = max(bottom_candidates, key=lambda x: x[0])
        return (left_x, top_y + mid, right_x, bottom_y)
    if top_candidates:
        _, mid = min(top_candidates, key=lambda x: x[0])
        return (left_x, top_y, right_x, top_y + mid)

    return None


def _detect_title_block_by_density(binary: np.ndarray, left_x: int, top_y: int,
                                    right_x: int, bottom_y: int) -> tuple | None:
    """Fallback: detect title block by comparing text density vs drawing-line density
    in edge strips. Only triggers with very high confidence (score > 8).
    """
    frame_w = right_x - left_x
    frame_h = bottom_y - top_y
    if frame_w < 100 or frame_h < 100:
        return None

    roi = binary[top_y:bottom_y, left_x:right_x]

    # Drawing lines mask
    min_draw_line = max(50, min(frame_w, frame_h) // 20)
    h_k = cv2.getStructuringElement(cv2.MORPH_RECT, (min_draw_line, 1))
    v_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_draw_line))
    drawing_lines = cv2.bitwise_or(
        cv2.morphologyEx(roi, cv2.MORPH_OPEN, h_k),
        cv2.morphologyEx(roi, cv2.MORPH_OPEN, v_k)
    )

    # Check 4 edge strips (15% of dimension)
    candidates = []
    strip_w = int(frame_w * 0.15)
    strip_h = int(frame_h * 0.15)

    strips = [
        ('right', roi[:, frame_w - strip_w:], drawing_lines[:, frame_w - strip_w:],
         (right_x - strip_w, top_y, right_x, bottom_y)),
        ('left', roi[:, :strip_w], drawing_lines[:, :strip_w],
         (left_x, top_y, left_x + strip_w, bottom_y)),
        ('bottom', roi[frame_h - strip_h:, :], drawing_lines[frame_h - strip_h:, :],
         (left_x, bottom_y - strip_h, right_x, bottom_y)),
        ('top', roi[:strip_h, :], drawing_lines[:strip_h, :],
         (left_x, top_y, right_x, top_y + strip_h)),
    ]

    for side, strip, draw_strip, bounds in strips:
        area = strip.shape[0] * strip.shape[1] + 1
        td = cv2.countNonZero(strip) / area
        ld = cv2.countNonZero(draw_strip) / area
        if td < 0.01:
            continue
        score = td / (ld + 0.003)
        candidates.append((score, bounds))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    best_score, best_bounds = candidates[0]

    # High threshold — only mask if very clearly non-drawing
    if best_score > 8.0:
        return best_bounds

    return None


def _cluster_positions(positions: np.ndarray, gap: int = 20) -> list:
    """Group nearby positions into clusters."""
    if len(positions) == 0:
        return []
    clusters = []
    current = [positions[0]]
    for p in positions[1:]:
        if p - current[-1] < gap:
            current.append(p)
        else:
            clusters.append(current)
            current = [p]
    clusters.append(current)
    return clusters


def _extract_line_segments(binary: np.ndarray, axis: str, img_w: int, img_h: int,
                           min_length: int = 80, max_length: int = 3000,
                           max_thickness: int = 30) -> list:
    """Double-pass kernel strategy:
    1. Small kernel to preserve all H/V elements
    2. Connected components to measure each segment
    3. Filter by thickness and length

    Returns list of (center_pos, start, end, length, thickness) for each valid segment.
    For horizontal: (y_center, x_start, x_end, length, thickness)
    For vertical: (x_center, y_start, y_end, length, thickness)
    """
    # Pass 1: small kernel to preserve line elements
    if axis == 'h':
        small_k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    else:
        small_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))

    line_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, small_k)

    # Pass 2: connected components with stats
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(line_mask, connectivity=8)

    segments = []
    for i in range(1, num_labels):  # skip background (0)
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]

        if axis == 'h':
            length = cw
            thickness = ch
            if length < min_length or length > max_length:
                continue
            if thickness < 1 or thickness > max_thickness:
                continue
            if length / (thickness + 1) < MIN_ASPECT_RATIO:
                continue
            y_center = y + ch / 2.0
            segments.append((y_center, float(x), float(x + cw), float(length), float(thickness)))
        else:
            length = ch
            thickness = cw
            if length < min_length or length > max_length:
                continue
            if thickness < 1 or thickness > max_thickness:
                continue
            if length / (thickness + 1) < MIN_ASPECT_RATIO:
                continue
            x_center = x + cw / 2.0
            segments.append((x_center, float(y), float(y + ch), float(length), float(thickness)))

    return segments


def _pair_by_duct_width(segments: list, axis: str, duct_w_min: int, duct_w_max: int,
                        min_length: int = 80) -> list[BoundingBox]:
    """Strict parallel-line pairing: only pairs lines whose perpendicular distance
    falls within [duct_w_min, duct_w_max]. Lines without a matching partner
    at the correct distance are discarded.

    segments: list of (center_pos, start, end, length, thickness)
    """
    if not segments:
        return []

    # Sort by perpendicular position (y for horizontal, x for vertical)
    segments.sort(key=lambda s: s[0])

    used = set()
    boxes = []

    for i in range(len(segments)):
        if i in used:
            continue

        best_j = -1
        best_score = float('inf')

        for j in range(i + 1, len(segments)):
            if j in used:
                continue

            # Perpendicular distance between the two lines
            gap = abs(segments[j][0] - segments[i][0])

            # STRICT duct width constraint
            if gap < duct_w_min:
                continue
            if gap > duct_w_max:
                break  # sorted, no more valid pairs

            # Check overlap along the line direction
            overlap_start = max(segments[i][1], segments[j][1])
            overlap_end = min(segments[i][2], segments[j][2])
            overlap = overlap_end - overlap_start

            if overlap < min_length:
                continue

            min_len = min(segments[i][3], segments[j][3])
            overlap_ratio = overlap / min_len
            if overlap_ratio < 0.4:
                continue

            # Similar length (reject if one is >3x longer)
            len_ratio = min(segments[i][3], segments[j][3]) / max(segments[i][3], segments[j][3])
            if len_ratio < 0.3:
                continue

            # Similar thickness (reject if one is >3x thicker)
            t_i, t_j = segments[i][4], segments[j][4]
            if max(t_i, t_j) > 3 * min(t_i, t_j) + 2:
                continue

            # Score: prefer close gap, high overlap, similar length
            score = gap - overlap_ratio * 40 - len_ratio * 30

            if score < best_score:
                best_score = score
                best_j = j

        if best_j >= 0:
            used.add(i)
            used.add(best_j)

            pos1 = segments[i][0]
            pos2 = segments[best_j][0]
            start = max(segments[i][1], segments[best_j][1])
            end = min(segments[i][2], segments[best_j][2])
            duct_thickness = abs(pos2 - pos1)
            duct_length = end - start

            if axis == 'h':
                cx = (start + end) / 2
                cy = (pos1 + pos2) / 2
                boxes.append(BoundingBox(x=cx, y=cy, width=float(duct_length),
                                         height=float(duct_thickness), angle=0.0))
            else:
                cx = (pos1 + pos2) / 2
                cy = (start + end) / 2
                boxes.append(BoundingBox(x=cx, y=cy, width=float(duct_thickness),
                                         height=float(duct_length), angle=0.0))

    return boxes


def _merge_overlapping(boxes: list[BoundingBox], iou_threshold=0.3) -> list[BoundingBox]:
    """Merge overlapping bounding boxes."""
    if not boxes:
        return boxes
    boxes.sort(key=lambda b: b.width * b.height, reverse=True)
    keep = []
    for box in boxes:
        is_dup = False
        for kept in keep:
            if _compute_iou(box, kept) > iou_threshold:
                is_dup = True
                break
        if not is_dup:
            keep.append(box)
    return keep


def _compute_iou(a: BoundingBox, b: BoundingBox) -> float:
    ax1, ay1 = a.x - a.width / 2, a.y - a.height / 2
    ax2, ay2 = a.x + a.width / 2, a.y + a.height / 2
    bx1, by1 = b.x - b.width / 2, b.y - b.height / 2
    bx2, by2 = b.x + b.width / 2, b.y + b.height / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    union = a.width * a.height + b.width * b.height - intersection
    return intersection / union if union > 0 else 0.0


def _save_debug(img, binary, h_segs, v_segs, boxes, debug_dir):
    """Save debug images for inspection."""
    import os
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, "morph_binary_roi.png"), binary)

    # Draw all candidate segments
    seg_vis = img.copy()
    for (y_c, x1, x2, l, t) in h_segs:
        cv2.line(seg_vis, (int(x1), int(y_c)), (int(x2), int(y_c)), (255, 0, 0), 1)
    for (x_c, y1, y2, l, t) in v_segs:
        cv2.line(seg_vis, (int(x_c), int(y1)), (int(x_c), int(y2)), (0, 0, 255), 1)
    cv2.imwrite(os.path.join(debug_dir, "morph_segments.png"), seg_vis)

    # Draw final detections
    det_vis = img.copy()
    for box in boxes:
        x1 = int(box.x - box.width / 2)
        y1 = int(box.y - box.height / 2)
        x2 = int(box.x + box.width / 2)
        y2 = int(box.y + box.height / 2)
        cv2.rectangle(det_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if box.width > box.height:
            cv2.line(det_vis, (x1, int(box.y)), (x2, int(box.y)), (0, 255, 0), 1)
        else:
            cv2.line(det_vis, (int(box.x), y1), (int(box.x), y2), (0, 255, 0), 1)
    cv2.imwrite(os.path.join(debug_dir, "morph_detections.png"), det_vis)
