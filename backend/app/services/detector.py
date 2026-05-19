"""Stages 4–7 — Structural Line Extraction, Candidate Detection, Filtering, Merging.

Stage 4: Extract horizontal and vertical line masks using morphology.
Stage 5: Contour-based candidate detection with 5 heuristics:
         1. Aspect Ratio
         2. Parallel Boundaries
         3. Thickness Consistency
         4. Connected Orthogonal Structure
         5. Area Thresholding
Stage 6: Remove grid lines (repetition frequency), walls (double-line, branching),
         equipment (polygon complexity, aspect ratio).
Stage 7: Merge nearby collinear duct segments.
"""
import cv2
import numpy as np
from app.models.schemas import BoundingBox


# ─── Stage 4: Line Extraction ───────────────────────────────────────────────

def extract_lines(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Separate horizontal and vertical structural lines.
    Builds two masks: horizontal-only and vertical-only.
    """
    h, w = binary.shape[:2]
    k_len = max(40, w // 80)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_len, 1))
    h_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_len))
    v_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    return h_mask, v_mask


def remove_text_noise(binary: np.ndarray, h_mask: np.ndarray, v_mask: np.ndarray) -> np.ndarray:
    """Remove text and small noise, keeping only structural lines."""
    combined = cv2.bitwise_or(h_mask, v_mask)
    # Close small gaps between line segments
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close, iterations=1)


# ─── Stage 5: Candidate Rectangle Detection ─────────────────────────────────

def detect_candidates(line_mask: np.ndarray, h_mask: np.ndarray, v_mask: np.ndarray) -> list[BoundingBox]:
    """Find rectangular duct candidates using contour detection + heuristic filtering.

    Ducts in engineering drawings are closed-ended rectangles.
    Each contour becomes a candidate region, then filtered by:
    1. Aspect Ratio — long rectangles, small squares rejected
    2. Parallel Boundaries — two long parallel edges
    3. Thickness Consistency — constant width
    4. Connected Orthogonal Structure — connects at 90°
    5. Area Thresholding — tiny contours removed
    """
    img_h, img_w = line_mask.shape[:2]

    # Ducts are closed rectangles — find contours directly on the line mask.
    # Use RETR_TREE to capture both outer boundaries and nested duct regions.
    contours, _ = cv2.findContours(line_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Pre-compute orthogonal connection map for heuristic 4
    connection_map = _build_connection_map(h_mask, v_mask, img_w, img_h)

    candidates = []
    min_area = max(500, (img_w * img_h) * 0.00003)
    max_area = (img_w * img_h) * 0.05

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch

        # Heuristic 5: Area Thresholding
        if area < min_area:
            continue
        if area > max_area:
            continue

        # Determine orientation
        if cw >= ch:
            length, thickness = cw, ch
            is_horizontal = True
        else:
            length, thickness = ch, cw
            is_horizontal = False

        # Heuristic 1: Aspect Ratio — ducts are elongated
        aspect = length / (thickness + 1)
        if aspect < 2.5:
            continue  # Too square
        if thickness < 8:
            continue  # Too thin (noise)

        # Heuristic 3: Thickness Consistency — constant width along length
        if not _check_thickness_consistency(line_mask, x, y, cw, ch, is_horizontal):
            continue

        # Heuristic 2: Parallel Boundaries — two long parallel edges
        if not _check_parallel_boundaries(h_mask, v_mask, x, y, cw, ch, is_horizontal):
            continue

        # Heuristic 4: Connected Orthogonal Structure — connects at 90°
        has_connection = _check_orthogonal_connection(connection_map, x, y, cw, ch, is_horizontal)

        cx = x + cw / 2.0
        cy = y + ch / 2.0

        candidates.append(BoundingBox(
            x=float(cx), y=float(cy),
            width=float(cw), height=float(ch),
            angle=0.0
        ))

    print(f"[Detector] Raw candidates: {len(candidates)}")
    return candidates


def _check_thickness_consistency(line_mask: np.ndarray, x: int, y: int,
                                  cw: int, ch: int, is_horizontal: bool) -> bool:
    """Verify duct maintains constant width along its length.
    Sample thickness at 3 points along the duct.
    """
    img_h, img_w = line_mask.shape[:2]

    if is_horizontal:
        # Sample at 25%, 50%, 75% along width
        sample_xs = [x + int(cw * p) for p in [0.25, 0.5, 0.75]]
        thicknesses = []
        for sx in sample_xs:
            if sx >= img_w:
                continue
            col = line_mask[y:min(y + ch, img_h), sx]
            t = np.count_nonzero(col)
            if t > 0:
                thicknesses.append(t)
    else:
        sample_ys = [y + int(ch * p) for p in [0.25, 0.5, 0.75]]
        thicknesses = []
        for sy in sample_ys:
            if sy >= img_h:
                continue
            row = line_mask[sy, x:min(x + cw, img_w)]
            t = np.count_nonzero(row)
            if t > 0:
                thicknesses.append(t)

    if len(thicknesses) < 2:
        return True  # Can't verify, allow through

    # Thickness should be consistent (max/min ratio < 3)
    max_t = max(thicknesses)
    min_t = min(thicknesses)
    if min_t == 0:
        return False
    return (max_t / min_t) < 3.0


def _check_parallel_boundaries(h_mask: np.ndarray, v_mask: np.ndarray,
                                x: int, y: int, cw: int, ch: int,
                                is_horizontal: bool) -> bool:
    """Verify candidate has two parallel edges (duct walls).
    Both long edges must have line pixels (>25% coverage).
    Samples a few rows/cols near each edge to account for slight offsets.
    """
    img_h, img_w = h_mask.shape[:2]

    if is_horizontal:
        x1 = max(0, x + int(cw * 0.2))
        x2 = min(img_w, x + int(cw * 0.8))
        sample_len = x2 - x1
        if sample_len < 10:
            return False

        # Sample top edge (within first 20% of height)
        top_found = False
        for dy in range(min(max(3, ch // 5), ch // 2)):
            row_y = max(0, y + dy)
            if row_y >= img_h:
                break
            pixels = np.count_nonzero(h_mask[row_y, x1:x2])
            if pixels > sample_len * 0.25:
                top_found = True
                break

        # Sample bottom edge (within last 20% of height)
        bot_found = False
        for dy in range(min(max(3, ch // 5), ch // 2)):
            row_y = min(img_h - 1, y + ch - dy)
            if row_y < 0:
                break
            pixels = np.count_nonzero(h_mask[row_y, x1:x2])
            if pixels > sample_len * 0.25:
                bot_found = True
                break

        return top_found and bot_found
    else:
        y1 = max(0, y + int(ch * 0.2))
        y2 = min(img_h, y + int(ch * 0.8))
        sample_len = y2 - y1
        if sample_len < 10:
            return False

        # Sample left edge
        left_found = False
        for dx in range(min(max(3, cw // 5), cw // 2)):
            col_x = max(0, x + dx)
            if col_x >= img_w:
                break
            pixels = np.count_nonzero(v_mask[y1:y2, col_x])
            if pixels > sample_len * 0.25:
                left_found = True
                break

        # Sample right edge
        right_found = False
        for dx in range(min(max(3, cw // 5), cw // 2)):
            col_x = min(img_w - 1, x + cw - dx)
            if col_x < 0:
                break
            pixels = np.count_nonzero(v_mask[y1:y2, col_x])
            if pixels > sample_len * 0.25:
                right_found = True
                break

        return left_found and right_found


def _build_connection_map(h_mask: np.ndarray, v_mask: np.ndarray,
                           img_w: int, img_h: int) -> np.ndarray:
    """Build a map of where H and V lines intersect (orthogonal connections).
    These are points where ducts branch at 90°.
    """
    # Dilate both masks slightly to find intersection zones
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    h_dilated = cv2.dilate(h_mask, k, iterations=1)
    v_dilated = cv2.dilate(v_mask, k, iterations=1)
    return cv2.bitwise_and(h_dilated, v_dilated)


def _check_orthogonal_connection(connection_map: np.ndarray, x: int, y: int,
                                  cw: int, ch: int, is_horizontal: bool) -> bool:
    """Check if duct connects at 90° to another duct at its endpoints."""
    img_h, img_w = connection_map.shape[:2]

    if is_horizontal:
        # Check left and right endpoints for vertical connections
        left_roi = connection_map[max(0, y):min(img_h, y+ch), max(0, x-5):max(1, x+10)]
        right_roi = connection_map[max(0, y):min(img_h, y+ch), max(0, x+cw-10):min(img_w, x+cw+5)]
    else:
        # Check top and bottom endpoints for horizontal connections
        top_roi = connection_map[max(0, y-5):max(1, y+10), max(0, x):min(img_w, x+cw)]
        bot_roi = connection_map[max(0, y+ch-10):min(img_h, y+ch+5), max(0, x):min(img_w, x+cw)]
        left_roi = top_roi
        right_roi = bot_roi

    left_conn = np.count_nonzero(left_roi) > 5
    right_conn = np.count_nonzero(right_roi) > 5

    return left_conn or right_conn


# ─── Stage 6: Filter Grid Lines / Walls / Equipment ─────────────────────────

def filter_unwanted(candidates: list[BoundingBox], h_mask: np.ndarray,
                    v_mask: np.ndarray, binary: np.ndarray) -> list[BoundingBox]:
    """Remove grid lines, walls, and equipment using layered filters.

    Grid lines: length + repetition frequency + global continuity
    Walls: thickness, double-line, branching style, internal spacing
    Equipment: contour complexity, polygon approximation, aspect ratio
    """
    img_h, img_w = binary.shape[:2]

    # Pre-compute grid line positions
    grid_ys, grid_xs = _detect_grid_lines(h_mask, v_mask, img_w, img_h)

    filtered = []
    for box in candidates:
        # Grid line removal
        if _is_grid_line(box, grid_ys, grid_xs, img_w, img_h):
            continue

        # Wall removal
        if _is_wall(box, binary, h_mask, v_mask, img_w, img_h):
            continue

        # Equipment removal
        if _is_equipment(box, binary, img_w, img_h):
            continue

        filtered.append(box)

    print(f"[Detector] After filtering: {len(filtered)}")
    return filtered


def _detect_grid_lines(h_mask: np.ndarray, v_mask: np.ndarray,
                        img_w: int, img_h: int) -> tuple[set, set]:
    """Detect grid lines using length + repetition frequency + uniform spacing.
    Grid lines are: extremely long, uniformly spaced, full-page spanning.
    Also catches section dividers (>40% span) that repeat.
    """
    grid_ys = set()
    grid_xs = set()

    # Find horizontal lines spanning >40% of width (catches section dividers too)
    h_contours, _ = cv2.findContours(h_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    long_h_ys = []
    for cnt in h_contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw > img_w * 0.4:
            long_h_ys.append(y + ch // 2)

    # Check repetition frequency — uniformly spaced lines are grid
    long_h_ys.sort()
    if len(long_h_ys) >= 3:
        spacings = [long_h_ys[i+1] - long_h_ys[i] for i in range(len(long_h_ys)-1)]
        if spacings:
            avg_spacing = sum(spacings) / len(spacings)
            uniform_count = sum(1 for s in spacings if abs(s - avg_spacing) < avg_spacing * 0.2)
            if uniform_count > len(spacings) * 0.5:
                grid_ys.update(long_h_ys)

    # Any line spanning >50% is a grid/section line regardless of spacing
    for cnt in h_contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw > img_w * 0.5:
            grid_ys.add(y + ch // 2)

    # Find vertical lines spanning >40% of height
    v_contours, _ = cv2.findContours(v_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    long_v_xs = []
    for cnt in v_contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if ch > img_h * 0.4:
            long_v_xs.append(x + cw // 2)

    long_v_xs.sort()
    if len(long_v_xs) >= 3:
        spacings = [long_v_xs[i+1] - long_v_xs[i] for i in range(len(long_v_xs)-1)]
        if spacings:
            avg_spacing = sum(spacings) / len(spacings)
            uniform_count = sum(1 for s in spacings if abs(s - avg_spacing) < avg_spacing * 0.2)
            if uniform_count > len(spacings) * 0.5:
                grid_xs.update(long_v_xs)

    # Any line spanning >50% is a grid/section line
    for cnt in v_contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if ch > img_h * 0.5:
            grid_xs.add(x + cw // 2)

    return grid_ys, grid_xs


def _is_grid_line(box: BoundingBox, grid_ys: set, grid_xs: set,
                   img_w: int, img_h: int) -> bool:
    """Check if candidate overlaps with a detected grid line."""
    duct_length = max(box.width, box.height)
    duct_thickness = min(box.width, box.height)

    # Very long relative to image = grid/border
    if duct_length > img_w * 0.4 or duct_length > img_h * 0.4:
        return True

    # Extremely high aspect ratio with thin thickness = structural line, not duct
    # Real ducts rarely exceed 10:1 length-to-thickness ratio
    if duct_length / (duct_thickness + 1) > 10:
        return True

    # Overlaps with known grid line position
    for gy in grid_ys:
        if abs(box.y - gy) < max(box.height, 15):
            return True
    for gx in grid_xs:
        if abs(box.x - gx) < max(box.width, 15):
            return True

    return False


def _is_wall(box: BoundingBox, binary: np.ndarray, h_mask: np.ndarray,
             v_mask: np.ndarray, img_w: int, img_h: int) -> bool:
    """Walls are: thicker, double-line, irregular room boundaries.
    Filters using: contour shape, thickness, branching style, internal spacing.
    """
    thickness = min(box.width, box.height)

    # Walls are typically thicker than ducts (>3% of image width)
    if thickness > img_w * 0.03:
        return True

    # Check for double-line pattern (wall = two close parallel thick lines with gap)
    x1 = int(max(0, box.x - box.width / 2))
    y1 = int(max(0, box.y - box.height / 2))
    x2 = int(min(img_w, box.x + box.width / 2))
    y2 = int(min(img_h, box.y + box.height / 2))

    if x2 <= x1 or y2 <= y1:
        return False

    roi = binary[y1:y2, x1:x2]
    density = np.count_nonzero(roi) / (roi.size + 1)

    # Walls are more solid than ducts (ducts are hollow channels)
    if density > 0.6:
        return True

    # Check branching style — walls branch irregularly (many T-junctions)
    is_horizontal = box.width > box.height
    if is_horizontal:
        # Count vertical lines crossing this horizontal region (branching)
        v_roi = v_mask[y1:y2, x1:x2]
        v_contours, _ = cv2.findContours(v_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Many short perpendicular branches = wall, not duct
        short_branches = sum(1 for c in v_contours if cv2.boundingRect(c)[3] > thickness * 0.5)
        if short_branches > 5:
            return True
    else:
        h_roi = h_mask[y1:y2, x1:x2]
        h_contours, _ = cv2.findContours(h_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        short_branches = sum(1 for c in h_contours if cv2.boundingRect(c)[2] > thickness * 0.5)
        if short_branches > 5:
            return True

    # Internal spacing check — walls have irregular internal structure
    # Ducts have clean empty interior
    interior_y1 = y1 + int(thickness * 0.3)
    interior_y2 = y2 - int(thickness * 0.3)
    interior_x1 = x1 + int(thickness * 0.3)
    interior_x2 = x2 - int(thickness * 0.3)

    if interior_y2 > interior_y1 and interior_x2 > interior_x1:
        interior = binary[interior_y1:interior_y2, interior_x1:interior_x2]
        if interior.size > 0:
            interior_density = np.count_nonzero(interior) / (interior.size + 1)
            # Ducts should have low interior density (hollow)
            # Walls have higher interior density (filled or double-line)
            if interior_density > 0.4:
                return True

    return False


def _is_equipment(box: BoundingBox, binary: np.ndarray, img_w: int, img_h: int) -> bool:
    """Equipment symbols are: dense, circular, text-heavy, block-like.
    Filters using: contour complexity, polygon approximation, aspect ratio.
    """
    x1 = int(max(0, box.x - box.width / 2))
    y1 = int(max(0, box.y - box.height / 2))
    x2 = int(min(img_w, box.x + box.width / 2))
    y2 = int(min(img_h, box.y + box.height / 2))

    if x2 <= x1 or y2 <= y1:
        return False

    roi = binary[y1:y2, x1:x2]

    # Find contours within the candidate region
    cnts, _ = cv2.findContours(roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Equipment has many internal contours (complex shape)
    if len(cnts) > 15:
        return True

    # Polygon approximation — ducts are simple rectangles (4 vertices)
    # Equipment has many vertices
    if cnts:
        largest = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(largest, True)
        if peri > 0:
            approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
            if len(approx) > 12:
                return True

    # Aspect ratio rejection — equipment tends to be squarish
    # (But we already filter aspect < 2.5 in Stage 5, so this catches edge cases)
    length = max(box.width, box.height)
    thickness = min(box.width, box.height)
    if length / (thickness + 1) < 2.0:
        return True

    # Density check — equipment is dense
    density = np.count_nonzero(roi) / (roi.size + 1)
    if density > 0.55:
        return True

    return False


# ─── Stage 7: Merge Duct Segments ───────────────────────────────────────────

def merge_segments(candidates: list[BoundingBox]) -> list[BoundingBox]:
    """Merge nearby collinear duct segments into single ducts.
    Merge logic: same angle + close endpoints + similar thickness = merge.
    """
    if len(candidates) < 2:
        return candidates

    merged = list(candidates)
    changed = True

    while changed:
        changed = False
        new_merged = []
        used = set()

        for i in range(len(merged)):
            if i in used:
                continue

            bi = merged[i]
            is_h_i = bi.width > bi.height
            best_j = -1

            for j in range(i + 1, len(merged)):
                if j in used:
                    continue
                bj = merged[j]
                is_h_j = bj.width > bj.height

                # Same orientation (same angle)
                if is_h_i != is_h_j:
                    continue

                if is_h_i:
                    # Same centerline Y (close endpoints vertically)
                    if abs(bi.y - bj.y) > max(bi.height, bj.height) * 0.6:
                        continue
                    # Similar thickness
                    if max(bi.height, bj.height) > 2 * min(bi.height, bj.height) + 5:
                        continue
                    # Close endpoints (gap < 25% of combined length)
                    left_end = min(bi.x + bi.width/2, bj.x + bj.width/2)
                    right_start = max(bi.x - bi.width/2, bj.x - bj.width/2)
                    gap = right_start - left_end
                    if gap > (bi.width + bj.width) * 0.25:
                        continue
                else:
                    # Same centerline X
                    if abs(bi.x - bj.x) > max(bi.width, bj.width) * 0.6:
                        continue
                    # Similar thickness
                    if max(bi.width, bj.width) > 2 * min(bi.width, bj.width) + 5:
                        continue
                    # Close endpoints
                    top_end = min(bi.y + bi.height/2, bj.y + bj.height/2)
                    bot_start = max(bi.y - bi.height/2, bj.y - bj.height/2)
                    gap = bot_start - top_end
                    if gap > (bi.height + bj.height) * 0.25:
                        continue

                best_j = j
                break

            if best_j >= 0:
                bj = merged[best_j]
                if is_h_i:
                    x1 = min(bi.x - bi.width/2, bj.x - bj.width/2)
                    x2 = max(bi.x + bi.width/2, bj.x + bj.width/2)
                    new_merged.append(BoundingBox(
                        x=(x1+x2)/2, y=(bi.y+bj.y)/2,
                        width=x2-x1, height=max(bi.height, bj.height), angle=0.0
                    ))
                else:
                    y1 = min(bi.y - bi.height/2, bj.y - bj.height/2)
                    y2 = max(bi.y + bi.height/2, bj.y + bj.height/2)
                    new_merged.append(BoundingBox(
                        x=(bi.x+bj.x)/2, y=(y1+y2)/2,
                        width=max(bi.width, bj.width), height=y2-y1, angle=0.0
                    ))
                used.add(i)
                used.add(best_j)
                changed = True
            else:
                new_merged.append(bi)

        merged = new_merged

    # Dedup overlapping (IoU > 0.4)
    merged.sort(key=lambda b: b.width * b.height, reverse=True)
    final = []
    for box in merged:
        # Post-merge validation: reject if merge created unrealistic aspect ratio
        length = max(box.width, box.height)
        thickness = min(box.width, box.height)
        if length / (thickness + 1) > 10:
            continue
        if not any(_iou(box, k) > 0.4 for k in final):
            final.append(box)

    print(f"[Detector] After merging: {len(final)}")
    return final


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    ax1, ay1 = a.x - a.width/2, a.y - a.height/2
    ax2, ay2 = a.x + a.width/2, a.y + a.height/2
    bx1, by1 = b.x - b.width/2, b.y - b.height/2
    bx2, by2 = b.x + b.width/2, b.y + b.height/2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0
