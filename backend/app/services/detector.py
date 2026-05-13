import cv2
import numpy as np
from app.models.schemas import BoundingBox
import os

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "weights", "best.pt")


def detect_ducts(image_path: str, binary_image: np.ndarray, original: np.ndarray, vector_data=None) -> list[BoundingBox]:
    """Run hybrid detection: YOLO if weights exist, else line-pair detection."""
    if os.path.exists(WEIGHTS_PATH):
        return detect_ducts_yolo(image_path)
    return detect_ducts_line_pairs(binary_image, original, vector_data=vector_data)


def detect_ducts_from_text(binary_image: np.ndarray, dimension_labels: list, scale_factor: float) -> list[BoundingBox]:
    """Text-first detection: find parallel lines above/below dimension text.
    Also learns duct line thickness from confirmed detections.
    dimension_labels: list of OCRResult with center_x, center_y in full-res coords.
    Returns BoundingBoxes in processed-image coordinates.
    """
    h, w = binary_image.shape[:2]
    boxes = []
    confirmed_thicknesses = []

    # Extract horizontal line mask
    k_scale = max(1, w // 5000 + 1)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25 * k_scale, 1))
    h_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, h_kernel)

    # Extract vertical line mask
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25 * k_scale))
    v_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, v_kernel)

    for label in dimension_labels:
        # Convert label position to processed-image coords
        cx = int(label.center_x / scale_factor)
        cy = int(label.center_y / scale_factor)

        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            continue

        # ⌀/Ø/∅/DIA or WxH format = confirmed duct, search wider
        is_confirmed_duct = any(s in label.text.lower() for s in ['⌀', '∅', 'ø', 'Ø', 'dia']) or \
                            any(s in label.text for s in ['×', 'x', 'X'])
        search = 150 if is_confirmed_duct else 120

        # Search for horizontal duct (lines above and below text)
        duct_box, thickness = _find_duct_lines_around_text(h_lines, cx, cy, axis='h', img_w=w, img_h=h, search_range=search)
        if duct_box is None:
            # Try vertical duct (lines left and right of text)
            duct_box, thickness = _find_duct_lines_around_text(v_lines, cx, cy, axis='v', img_w=w, img_h=h, search_range=search)

        if duct_box:
            boxes.append(duct_box)
            if thickness:
                confirmed_thicknesses.append(thickness)

    # Learn duct line thickness from confirmed detections
    if confirmed_thicknesses:
        avg_thickness = sum(confirmed_thicknesses) / len(confirmed_thicknesses)
        tolerance = max(avg_thickness * 0.5, 3)
        print(f"[Detector] Learned duct line thickness: {avg_thickness:.0f}px (±{tolerance:.0f})")
    else:
        avg_thickness = None

    print(f"[Detector] Text-first ducts: {len(boxes)}")
    return boxes, avg_thickness


def _find_duct_lines_around_text(line_mask: np.ndarray, cx: int, cy: int,
                                  axis: str, img_w: int, img_h: int,
                                  search_range: int = 120) -> tuple[BoundingBox | None, float | None]:
    """Look for parallel lines around text position.
    First checks if text is INSIDE the duct (lines on both sides within close range).
    If not, checks if text is OUTSIDE (lines on one side, duct is to the left/right/above/below).
    Returns (BoundingBox, line_thickness) or (None, None).
    """
    if axis == 'h':
        x_start = max(0, cx - 200)
        x_end = min(img_w, cx + 200)

        # Find nearest line above and below text center
        line_above, thick_above = _scan_for_line_h(line_mask, cx, cy, -1, x_start, x_end, img_h, search_range)
        line_below, thick_below = _scan_for_line_h(line_mask, cx, cy, +1, x_start, x_end, img_h, search_range)

        if line_above is not None and line_below is not None:
            gap = line_below - line_above
            # Case 1: Text INSIDE duct (both lines close, gap < 150)
            if 10 < gap < 150:
                return _build_h_duct(line_mask, line_above, line_below, thick_above, thick_below, img_w)

        # Case 2: Text OUTSIDE duct — lines are on one side
        # If we found a line above but not below (or very far below), duct is above the text
        if line_above is not None and (line_below is None or (line_below - line_above) > 150):
            # Search for second line ABOVE the first line (duct is above text)
            second_line, thick_second = _scan_for_line_h(line_mask, cx, line_above - 5, -1, x_start, x_end, img_h, 100)
            if second_line is not None:
                gap = line_above - second_line
                if 10 < gap < 150:
                    return _build_h_duct(line_mask, second_line, line_above, thick_second, thick_above, img_w)

        # If we found a line below but not above, duct is below the text
        if line_below is not None and (line_above is None or (line_below - line_above) > 150):
            second_line, thick_second = _scan_for_line_h(line_mask, cx, line_below + 5, +1, x_start, x_end, img_h, 100)
            if second_line is not None:
                gap = second_line - line_below
                if 10 < gap < 150:
                    return _build_h_duct(line_mask, line_below, second_line, thick_below, thick_second, img_w)

    else:  # vertical
        y_start = max(0, cy - 200)
        y_end = min(img_h, cy + 200)

        line_left, thick_left = _scan_for_line_v(line_mask, cx, cy, -1, y_start, y_end, img_w, search_range)
        line_right, thick_right = _scan_for_line_v(line_mask, cx, cy, +1, y_start, y_end, img_w, search_range)

        if line_left is not None and line_right is not None:
            gap = line_right - line_left
            if 10 < gap < 150:
                return _build_v_duct(line_mask, line_left, line_right, thick_left, thick_right, img_h)

        # Text outside — duct is to the left
        if line_left is not None and (line_right is None or (line_right - line_left) > 150):
            second_line, thick_second = _scan_for_line_v(line_mask, line_left - 5, cy, -1, y_start, y_end, img_w, 100)
            if second_line is not None:
                gap = line_left - second_line
                if 10 < gap < 150:
                    return _build_v_duct(line_mask, second_line, line_left, thick_second, thick_left, img_h)

        # Text outside — duct is to the right
        if line_right is not None and (line_left is None or (line_right - line_left) > 150):
            second_line, thick_second = _scan_for_line_v(line_mask, line_right + 5, cy, +1, y_start, y_end, img_w, 100)
            if second_line is not None:
                gap = second_line - line_right
                if 10 < gap < 150:
                    return _build_v_duct(line_mask, line_right, second_line, thick_right, thick_second, img_h)

    return None, None


def _scan_for_line_h(line_mask, cx, start_y, direction, x_start, x_end, img_h, max_dist):
    """Scan vertically for a horizontal line. Returns (y_position, thickness) or (None, None)."""
    for dy in range(0, max_dist):
        y = start_y + direction * dy
        if y < 0 or y >= img_h:
            break
        row = line_mask[y, x_start:x_end]
        if np.count_nonzero(row) > 50:
            # Measure thickness
            thickness = 1
            for t in range(1, 30):
                ny = y + direction * t
                if ny < 0 or ny >= img_h or np.count_nonzero(line_mask[ny, x_start:x_end]) < 50:
                    thickness = t
                    break
            return y, thickness
    return None, None


def _scan_for_line_v(line_mask, start_x, cy, direction, y_start, y_end, img_w, max_dist):
    """Scan horizontally for a vertical line. Returns (x_position, thickness) or (None, None)."""
    for dx in range(0, max_dist):
        x = start_x + direction * dx
        if x < 0 or x >= img_w:
            break
        col = line_mask[y_start:y_end, x]
        if np.count_nonzero(col) > 50:
            thickness = 1
            for t in range(1, 30):
                nx = x + direction * t
                if nx < 0 or nx >= img_w or np.count_nonzero(line_mask[y_start:y_end, nx]) < 50:
                    thickness = t
                    break
            return x, thickness
    return None, None


def _build_h_duct(line_mask, line_above, line_below, thick_above, thick_below, img_w):
    """Build a horizontal duct BoundingBox from two horizontal lines."""
    gap = line_below - line_above
    # Symmetry check
    if thick_above and thick_below:
        ratio = max(thick_above, thick_below) / (min(thick_above, thick_below) + 1)
        if ratio > 3.0:
            return None, None

    row = line_mask[line_above, :]
    cols = np.where(row > 0)[0]
    if len(cols) > 50:
        x1, x2 = int(cols[0]), int(cols[-1])
        duct_w = x2 - x1
        if 80 < duct_w < img_w * 0.4:
            mid_y = (line_above + line_below) // 2
            avg_thick = (thick_above + thick_below) / 2 if thick_below else thick_above
            return BoundingBox(x=float((x1 + x2) / 2), y=float(mid_y),
                               width=float(duct_w), height=float(gap), angle=0.0), avg_thick
    return None, None


def _build_v_duct(line_mask, line_left, line_right, thick_left, thick_right, img_h):
    """Build a vertical duct BoundingBox from two vertical lines."""
    gap = line_right - line_left
    if thick_left and thick_right:
        ratio = max(thick_left, thick_right) / (min(thick_left, thick_right) + 1)
        if ratio > 3.0:
            return None, None

    col = line_mask[:, line_left]
    rows = np.where(col > 0)[0]
    if len(rows) > 50:
        y1, y2 = int(rows[0]), int(rows[-1])
        duct_h = y2 - y1
        if 80 < duct_h < img_h * 0.4:
            mid_x = (line_left + line_right) // 2
            avg_thick = (thick_left + thick_right) / 2 if thick_right else thick_left
            return BoundingBox(x=float(mid_x), y=float((y1 + y2) / 2),
                               width=float(gap), height=float(duct_h), angle=0.0), avg_thick
    return None, None


def auto_detect_roi(binary_image: np.ndarray, vector_data=None) -> tuple[int, int, int, int]:
    """Detect the drawing area by excluding notes/title sections.
    Uses vector text positions if available (precise), falls back to grid analysis.
    """
    h, w = binary_image.shape[:2]

    # If we have vector data, use section headers to find boundaries
    if vector_data and hasattr(vector_data, 'texts') and vector_data.texts:
        roi = _roi_from_vector_text(vector_data, w, h)
        if roi:
            return roi

    # Fallback: grid-based density analysis
    return _roi_from_grid_analysis(binary_image)


def _roi_from_vector_text(vector_data, img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
    """Determine ROI using vector text positions.
    Finds section headers (NOTES, FLOOR PLAN, ISSUE DATE, etc.) and excludes those areas.
    """
    pw = vector_data.page_width
    ph = vector_data.page_height
    if not pw or not ph:
        return None

    scale_x = img_w / pw
    scale_y = img_h / ph

    # Keywords that indicate non-drawing sections
    exclude_keywords = [
        'general notes', 'plan notes', 'notes:', 'notes',
        'issue date', 'project name', 'revision',
        'drawn by', 'checked by', 'scale:',
        'do not scale', 'contractor',
        'floor plan',  # Title block label (large text)
    ]

    # Find the topmost Y position of any exclude keyword
    notes_y = None

    for text, x, y, tw, th in vector_data.texts:
        text_lower = text.lower().strip()
        # Skip if position is beyond page bounds (extended content)
        if y > ph * 1.5 or y < 0:
            continue
        for kw in exclude_keywords:
            if kw in text_lower:
                if notes_y is None or y < notes_y:
                    notes_y = y
                break

    if notes_y is None:
        return None  # No section markers found

    # Add small margin above the first notes text
    notes_y = max(0, notes_y - 20)

    # Convert to image coordinates
    roi_x1 = int(img_w * 0.01)
    roi_y1 = int(img_h * 0.01)
    roi_x2 = int(img_w * 0.95)
    roi_y2 = int(min(notes_y * scale_y, img_h * 0.98))

    # Sanity: ROI must be at least 40% of image height
    if roi_y2 < img_h * 0.4:
        return None
    # If notes are beyond visible area, use most of the image
    if roi_y2 > img_h * 0.95:
        roi_y2 = int(img_h * 0.70)  # Conservative: notes likely at bottom 30%

    return roi_x1, roi_y1, roi_x2, roi_y2


def _roi_from_grid_analysis(binary_image: np.ndarray) -> tuple[int, int, int, int]:
    """Fallback: grid-based density analysis to find drawing area."""
    h, w = binary_image.shape[:2]
    grid_cols = 10
    cell_w = w // grid_cols

    min_draw_line = w // 10
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_draw_line, 1))
    h_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, h_kernel)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_draw_line))
    v_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, v_kernel)

    border_h = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.8), 1))
    border_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h * 0.8)))
    borders = cv2.bitwise_or(
        cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, border_h),
        cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, border_v)
    )
    lines_mask = cv2.subtract(cv2.bitwise_or(h_lines, v_lines), borders)
    text_mask = cv2.subtract(binary_image, cv2.bitwise_or(h_lines, v_lines))

    # Find rightmost drawing column (title block = very high text, no drawing lines)
    roi_x2 = w
    for c in range(grid_cols - 1, grid_cols // 2, -1):
        x1, x2 = c * cell_w, (c + 1) * cell_w
        cell_area = h * cell_w
        line_d = cv2.countNonZero(lines_mask[:, x1:x2]) / cell_area
        text_d = cv2.countNonZero(text_mask[:, x1:x2]) / cell_area
        if text_d > 0.08 and text_d > line_d * 5:
            roi_x2 = c * cell_w
        else:
            break

    roi_x1 = int(w * 0.01)
    roi_y1 = int(h * 0.01)
    roi_y2 = int(h * 0.70)
    roi_x2 = max(roi_x2, int(w * 0.5))

    return roi_x1, roi_y1, roi_x2, roi_y2


def detect_ducts_line_pairs(binary_image: np.ndarray, original: np.ndarray, vector_data=None) -> list[BoundingBox]:
    """Detect ducts by extracting lines and pairing parallel ones."""
    h, w = original.shape[:2]

    roi_x1, roi_y1, roi_x2, roi_y2 = auto_detect_roi(binary_image, vector_data=vector_data)
    print(f"[Detector] ROI: ({roi_x1},{roi_y1}) to ({roi_x2},{roi_y2})")

    roi_binary = binary_image[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_h, roi_w = roi_binary.shape[:2]

    # Save debug
    debug_dir = os.path.join(os.path.dirname(__file__), "..", "..", "debug")
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, "01_roi_binary.png"), roi_binary)

    boxes = []
    k_scale = max(1, roi_w // 5000 + 1)

    # --- Horizontal lines ---
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25 * k_scale, 1))
    h_lines_mask = cv2.morphologyEx(roi_binary, cv2.MORPH_OPEN, h_kernel)
    h_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (15 * k_scale, 1))
    h_lines_mask = cv2.dilate(h_lines_mask, h_connect, iterations=1)
    cv2.imwrite(os.path.join(debug_dir, "02_h_lines.png"), h_lines_mask)

    h_contours, _ = cv2.findContours(h_lines_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_line_len = int(roi_w * 0.025)
    h_segments = []
    max_thickness = int(50 * k_scale)
    for cnt in h_contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw < min_line_len:
            continue
        if cw > roi_w * 0.92:
            continue
        if ch > max_thickness:
            continue
        if cw / (ch + 1) < 3:
            continue
        y_mid = y + ch / 2
        h_segments.append((x, y_mid, x + cw, y_mid, cw, ch))  # Added ch (thickness)

    print(f"[Detector] Horizontal line segments: {len(h_segments)}")

    gap_scale = max(1, roi_w // 5000 + 1)
    h_boxes = pair_parallel_lines(h_segments, roi_x1, roi_y1, axis='h',
                                  min_gap=10 * gap_scale, max_gap=120 * gap_scale, min_overlap_ratio=0.4)
    boxes.extend(h_boxes)
    print(f"[Detector] Horizontal duct pairs: {len(h_boxes)}")

    # --- Vertical lines ---
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25 * k_scale))
    v_lines_mask = cv2.morphologyEx(roi_binary, cv2.MORPH_OPEN, v_kernel)
    v_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15 * k_scale))
    v_lines_mask = cv2.dilate(v_lines_mask, v_connect, iterations=1)
    cv2.imwrite(os.path.join(debug_dir, "03_v_lines.png"), v_lines_mask)

    v_contours, _ = cv2.findContours(v_lines_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_line_len_v = int(roi_h * 0.025)
    v_segments = []
    for cnt in v_contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if ch < min_line_len_v:
            continue
        if ch > roi_h * 0.92:
            continue
        if cw > max_thickness:
            continue
        if ch / (cw + 1) < 3:
            continue
        x_mid = x + cw / 2
        v_segments.append((x_mid, y, x_mid, y + ch, ch, cw))  # Added cw (thickness)

    print(f"[Detector] Vertical line segments: {len(v_segments)}")

    v_boxes = pair_parallel_lines(v_segments, roi_x1, roi_y1, axis='v',
                                  min_gap=10 * gap_scale, max_gap=120 * gap_scale, min_overlap_ratio=0.4)
    boxes.extend(v_boxes)
    print(f"[Detector] Vertical duct pairs: {len(v_boxes)}")

    # Filter out likely false positives
    boxes = filter_false_positives(boxes, roi_w, roi_h)

    # Merge overlapping
    boxes = merge_overlapping(boxes, iou_threshold=0.3)
    print(f"[Detector] Total duct segments: {len(boxes)}")
    return boxes


def filter_false_positives(boxes: list[BoundingBox], roi_w: int, roi_h: int) -> list[BoundingBox]:
    """Remove detections that are too small or have wrong aspect ratio for ducts."""
    filtered = []

    for box in boxes:
        duct_length = max(box.width, box.height)
        duct_thickness = min(box.width, box.height)

        if duct_thickness < 20:
            continue
        if duct_thickness > 80:
            continue
        if duct_length < 100:
            continue

        aspect = duct_length / (duct_thickness + 1)
        if aspect < 2.0:
            continue
        if aspect > 15.0:  # Too elongated — likely a pipe/column, not a duct
            continue

        filtered.append(box)

    return filtered


def _filter_asymmetric_pairs(boxes: list[BoundingBox], roi_binary: np.ndarray,
                             roi_x1: int, roi_y1: int) -> list[BoundingBox]:
    """Remove duct detections where the two walls have very different line coverage.
    A real duct has two solid parallel lines of similar weight.
    """
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    h_lines = cv2.morphologyEx(roi_binary, cv2.MORPH_OPEN, h_kernel)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    v_lines = cv2.morphologyEx(roi_binary, cv2.MORPH_OPEN, v_kernel)

    roi_h, roi_w = roi_binary.shape[:2]
    filtered = []

    for box in boxes:
        cx = int(box.x - roi_x1)
        cy = int(box.y - roi_y1)
        is_horizontal = box.width > box.height

        if is_horizontal:
            top_y = int(cy - box.height / 2)
            bot_y = int(cy + box.height / 2)
            x_start = max(0, cx - int(box.width * 0.3))
            x_end = min(roi_w, cx + int(box.width * 0.3))
            sample_len = x_end - x_start

            if top_y < 0 or bot_y >= roi_h or sample_len < 20:
                filtered.append(box)
                continue

            # Measure coverage: how many pixels in the sample window are line
            top_coverage = np.count_nonzero(h_lines[top_y, x_start:x_end]) / sample_len
            bot_coverage = np.count_nonzero(h_lines[bot_y, x_start:x_end]) / sample_len
        else:
            left_x = int(cx - box.width / 2)
            right_x = int(cx + box.width / 2)
            y_start = max(0, cy - int(box.height * 0.3))
            y_end = min(roi_h, cy + int(box.height * 0.3))
            sample_len = y_end - y_start

            if left_x < 0 or right_x >= roi_w or sample_len < 20:
                filtered.append(box)
                continue

            top_coverage = np.count_nonzero(v_lines[y_start:y_end, left_x]) / sample_len
            bot_coverage = np.count_nonzero(v_lines[y_start:y_end, right_x]) / sample_len

        # Both walls must have decent coverage (> 30%) and be similar
        if top_coverage < 0.2 or bot_coverage < 0.2:
            continue  # One wall barely exists

        ratio = max(top_coverage, bot_coverage) / (min(top_coverage, bot_coverage) + 0.01)
        if ratio <= 2.5:
            filtered.append(box)

    return filtered


def pair_parallel_lines(segments: list, offset_x: int, offset_y: int, axis: str,
                        min_gap: int = 10, max_gap: int = 150, min_overlap_ratio: float = 0.35) -> list[BoundingBox]:
    """Pair parallel line segments that are close together to form duct walls."""
    if axis == 'h':
        segments.sort(key=lambda s: s[1])
    else:
        segments.sort(key=lambda s: s[0])

    boxes = []
    used = set()

    for i in range(len(segments)):
        if i in used:
            continue

        best_j = -1
        best_score = float('inf')

        for j in range(i + 1, len(segments)):
            if j in used:
                continue

            if axis == 'h':
                gap = abs(segments[j][1] - segments[i][1])
                if gap < min_gap:
                    continue
                if gap > max_gap:
                    break

                x_start_i, x_end_i = segments[i][0], segments[i][2]
                x_start_j, x_end_j = segments[j][0], segments[j][2]
                overlap_start = max(x_start_i, x_start_j)
                overlap_end = min(x_end_i, x_end_j)
                overlap = overlap_end - overlap_start

                min_len = min(segments[i][4], segments[j][4])
                if overlap < min_len * min_overlap_ratio:
                    continue

                # Score: prefer close, well-overlapping, similar-length, similar-weight pairs
                len_ratio = min(segments[i][4], segments[j][4]) / max(segments[i][4], segments[j][4])
                # Line weight comparison (index 5 = thickness)
                weight_i = segments[i][5] if len(segments[i]) > 5 else 1
                weight_j = segments[j][5] if len(segments[j]) > 5 else 1
                weight_ratio = min(weight_i, weight_j) / (max(weight_i, weight_j) + 1)
                # Reject if line weights are too different (>3x)
                if max(weight_i, weight_j) > 3 * min(weight_i, weight_j) + 1:
                    continue
                score = gap - (overlap / min_len) * 50 - len_ratio * 80 - weight_ratio * 30

            else:
                gap = abs(segments[j][0] - segments[i][0])
                if gap < min_gap:
                    continue
                if gap > max_gap:
                    break

                y_start_i, y_end_i = segments[i][1], segments[i][3]
                y_start_j, y_end_j = segments[j][1], segments[j][3]
                overlap_start = max(y_start_i, y_start_j)
                overlap_end = min(y_end_i, y_end_j)
                overlap = overlap_end - overlap_start

                min_len = min(segments[i][4], segments[j][4])
                if overlap < min_len * min_overlap_ratio:
                    continue

                len_ratio = min(segments[i][4], segments[j][4]) / max(segments[i][4], segments[j][4])
                weight_i = segments[i][5] if len(segments[i]) > 5 else 1
                weight_j = segments[j][5] if len(segments[j]) > 5 else 1
                weight_ratio = min(weight_i, weight_j) / (max(weight_i, weight_j) + 1)
                if max(weight_i, weight_j) > 3 * min(weight_i, weight_j) + 1:
                    continue
                score = gap - (overlap / min_len) * 50 - len_ratio * 80 - weight_ratio * 30

            if score < best_score:
                best_score = score
                best_j = j

        if best_j >= 0:
            used.add(i)
            used.add(best_j)

            if axis == 'h':
                y1 = segments[i][1]
                y2 = segments[best_j][1]
                x_start = max(segments[i][0], segments[best_j][0])
                x_end = min(segments[i][2], segments[best_j][2])

                cx = offset_x + (x_start + x_end) / 2
                cy = offset_y + (y1 + y2) / 2
                duct_width = x_end - x_start
                duct_height = abs(y2 - y1)
            else:
                x1 = segments[i][0]
                x2 = segments[best_j][0]
                y_start = max(segments[i][1], segments[best_j][1])
                y_end = min(segments[i][3], segments[best_j][3])

                cx = offset_x + (x1 + x2) / 2
                cy = offset_y + (y_start + y_end) / 2
                duct_width = abs(x2 - x1)
                duct_height = y_end - y_start

            boxes.append(BoundingBox(x=cx, y=cy, width=float(duct_width),
                                     height=float(duct_height), angle=0.0))

    return boxes


def merge_overlapping(boxes: list[BoundingBox], iou_threshold: float = 0.3) -> list[BoundingBox]:
    if not boxes:
        return boxes
    # First merge collinear segments (same duct split by breaks)
    boxes = _merge_collinear(boxes)
    # Then merge overlapping
    boxes.sort(key=lambda b: b.width * b.height, reverse=True)
    keep = []
    for box in boxes:
        is_dup = False
        for kept in keep:
            if compute_iou(box, kept) > iou_threshold or compute_containment(box, kept) > 0.6:
                is_dup = True
                break
        if not is_dup:
            keep.append(box)
    return keep


def _merge_collinear(boxes: list[BoundingBox]) -> list[BoundingBox]:
    """Merge duct segments that are collinear (same duct split by a break/symbol).
    Two ducts are collinear if:
    - Same orientation (both H or both V)
    - Same centerline (Y for H, X for V) within tolerance
    - Similar thickness
    - Gap between them is small relative to their length
    """
    if len(boxes) < 2:
        return boxes

    merged = list(boxes)
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

                # Must be same orientation
                if is_h_i != is_h_j:
                    continue

                if is_h_i:  # Both horizontal
                    # Same centerline Y (within thickness tolerance)
                    y_diff = abs(bi.y - bj.y)
                    thickness = max(bi.height, bj.height)
                    if y_diff > thickness * 0.5:
                        continue

                    # Similar thickness
                    thick_ratio = max(bi.height, bj.height) / (min(bi.height, bj.height) + 1)
                    if thick_ratio > 2.0:
                        continue

                    # Check gap between them
                    left_end = min(bi.x + bi.width / 2, bj.x + bj.width / 2)
                    right_start = max(bi.x - bi.width / 2, bj.x - bj.width / 2)
                    gap = right_start - left_end
                    combined_len = bi.width + bj.width

                    # Gap must be small relative to combined length
                    if gap > combined_len * 0.3:
                        continue

                else:  # Both vertical
                    x_diff = abs(bi.x - bj.x)
                    thickness = max(bi.width, bj.width)
                    if x_diff > thickness * 0.5:
                        continue

                    thick_ratio = max(bi.width, bj.width) / (min(bi.width, bj.width) + 1)
                    if thick_ratio > 2.0:
                        continue

                    top_end = min(bi.y + bi.height / 2, bj.y + bj.height / 2)
                    bot_start = max(bi.y - bi.height / 2, bj.y - bj.height / 2)
                    gap = bot_start - top_end
                    combined_len = bi.height + bj.height

                    if gap > combined_len * 0.3:
                        continue

                best_j = j
                break

            if best_j >= 0:
                bj = merged[best_j]
                # Merge: create bounding box spanning both
                if is_h_i:
                    new_x1 = min(bi.x - bi.width / 2, bj.x - bj.width / 2)
                    new_x2 = max(bi.x + bi.width / 2, bj.x + bj.width / 2)
                    new_y = (bi.y + bj.y) / 2
                    new_h = max(bi.height, bj.height)
                    new_merged.append(BoundingBox(x=(new_x1 + new_x2) / 2, y=new_y,
                                                  width=new_x2 - new_x1, height=new_h, angle=0.0))
                else:
                    new_y1 = min(bi.y - bi.height / 2, bj.y - bj.height / 2)
                    new_y2 = max(bi.y + bi.height / 2, bj.y + bj.height / 2)
                    new_x = (bi.x + bj.x) / 2
                    new_w = max(bi.width, bj.width)
                    new_merged.append(BoundingBox(x=new_x, y=(new_y1 + new_y2) / 2,
                                                  width=new_w, height=new_y2 - new_y1, angle=0.0))
                used.add(i)
                used.add(best_j)
                changed = True
            else:
                new_merged.append(bi)

        merged = new_merged

    return merged


def compute_containment(a: BoundingBox, b: BoundingBox) -> float:
    ax1, ay1 = a.x - a.width / 2, a.y - a.height / 2
    ax2, ay2 = a.x + a.width / 2, a.y + a.height / 2
    bx1, by1 = b.x - b.width / 2, b.y - b.height / 2
    bx2, by2 = b.x + b.width / 2, b.y + b.height / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    return intersection / (a.width * a.height) if a.width * a.height > 0 else 0.0


def compute_iou(a: BoundingBox, b: BoundingBox) -> float:
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


def detect_ducts_yolo(image_path: str) -> list[BoundingBox]:
    from ultralytics import YOLO
    model = YOLO(WEIGHTS_PATH)
    results = model(image_path, conf=0.3)
    boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append(BoundingBox(
                x=(x1 + x2) / 2, y=(y1 + y2) / 2,
                width=x2 - x1, height=y2 - y1, angle=0.0
            ))
    return boxes
