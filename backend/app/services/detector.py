import cv2
import numpy as np
from app.models.schemas import BoundingBox
import os

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "weights", "best.pt")


def detect_ducts(image_path: str, binary_image: np.ndarray, original: np.ndarray) -> list[BoundingBox]:
    """Run hybrid detection: YOLO if weights exist, else line-pair detection."""
    if os.path.exists(WEIGHTS_PATH):
        return detect_ducts_yolo(image_path)
    return detect_ducts_line_pairs(binary_image, original)


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

        # Search for horizontal duct (lines above and below text)
        duct_box, thickness = _find_duct_lines_around_text(h_lines, cx, cy, axis='h', img_w=w, img_h=h)
        if duct_box is None:
            # Try vertical duct (lines left and right of text)
            duct_box, thickness = _find_duct_lines_around_text(v_lines, cx, cy, axis='v', img_w=w, img_h=h)

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
    Text is typically INSIDE the duct (overlaid on the duct shape),
    so lines pass through or very close to the text center.
    Both lines must have similar thickness (symmetric duct walls).
    Returns (BoundingBox, line_thickness) or (None, None).
    """
    if axis == 'h':
        line_above = None
        line_above_thickness = None
        line_below = None
        line_below_thickness = None

        x_start = max(0, cx - 200)
        x_end = min(img_w, cx + 200)

        # Search upward from text center
        for dy in range(0, search_range):
            y = cy - dy
            if y < 0:
                break
            row = line_mask[y, x_start:x_end]
            if np.count_nonzero(row) > 50:
                line_above = y
                # Measure thickness: scan further up to find where line ends
                for t in range(1, 30):
                    if y - t < 0 or np.count_nonzero(line_mask[y - t, x_start:x_end]) < 50:
                        line_above_thickness = t
                        break
                break

        # Search downward from text center
        for dy in range(0, search_range):
            y = cy + dy
            if y >= img_h:
                break
            row = line_mask[y, x_start:x_end]
            if np.count_nonzero(row) > 50:
                if line_above is not None and abs(y - line_above) < 10:
                    continue
                line_below = y
                # Measure thickness of bottom line
                for t in range(1, 30):
                    if y + t >= img_h or np.count_nonzero(line_mask[y + t, x_start:x_end]) < 50:
                        line_below_thickness = t
                        break
                break

        if line_above is not None and line_below is not None:
            gap = line_below - line_above
            if 10 < gap < 150:
                # Symmetry check: both walls must have similar thickness
                if line_above_thickness and line_below_thickness:
                    ratio = max(line_above_thickness, line_below_thickness) / (min(line_above_thickness, line_below_thickness) + 1)
                    if ratio > 3.0:  # One wall is 3x thicker than the other = not a duct
                        return None, None

                row = line_mask[line_above, :]
                cols = np.where(row > 0)[0]
                if len(cols) > 50:
                    x1 = int(cols[0])
                    x2 = int(cols[-1])
                    duct_w = x2 - x1
                    mid_y = (line_above + line_below) // 2
                    if 80 < duct_w < img_w * 0.4:
                        avg_thickness = (line_above_thickness + line_below_thickness) / 2 if line_below_thickness else line_above_thickness
                        return BoundingBox(
                            x=float((x1 + x2) / 2),
                            y=float(mid_y),
                            width=float(duct_w),
                            height=float(gap),
                            angle=0.0
                        ), avg_thickness
    else:  # vertical
        line_left = None
        line_left_thickness = None
        line_right = None
        line_right_thickness = None

        y_start = max(0, cy - 200)
        y_end = min(img_h, cy + 200)

        for dx in range(0, search_range):
            x = cx - dx
            if x < 0:
                break
            col = line_mask[y_start:y_end, x]
            if np.count_nonzero(col) > 50:
                line_left = x
                for t in range(1, 30):
                    if x - t < 0 or np.count_nonzero(line_mask[y_start:y_end, x - t]) < 50:
                        line_left_thickness = t
                        break
                break

        for dx in range(0, search_range):
            x = cx + dx
            if x >= img_w:
                break
            col = line_mask[y_start:y_end, x]
            if np.count_nonzero(col) > 50:
                if line_left is not None and abs(x - line_left) < 10:
                    continue
                line_right = x
                for t in range(1, 30):
                    if x + t >= img_w or np.count_nonzero(line_mask[y_start:y_end, x + t]) < 50:
                        line_right_thickness = t
                        break
                break

        if line_left is not None and line_right is not None:
            gap = line_right - line_left
            if 10 < gap < 150:
                # Symmetry check
                if line_left_thickness and line_right_thickness:
                    ratio = max(line_left_thickness, line_right_thickness) / (min(line_left_thickness, line_right_thickness) + 1)
                    if ratio > 3.0:
                        return None, None

                col = line_mask[:, line_left]
                rows = np.where(col > 0)[0]
                if len(rows) > 50:
                    y1 = int(rows[0])
                    y2 = int(rows[-1])
                    duct_h = y2 - y1
                    if 80 < duct_h < img_h * 0.4:
                        avg_thickness = (line_left_thickness + line_right_thickness) / 2 if line_right_thickness else line_left_thickness
                        return BoundingBox(
                            x=float((line_left + line_right) / 2),
                            y=float((y1 + y2) / 2),
                            width=float(gap),
                            height=float(duct_h),
                            angle=0.0
                        ), avg_thickness

    return None, None


def auto_detect_roi(binary_image: np.ndarray) -> tuple[int, int, int, int]:
    """Detect main drawing area excluding title block and notes."""
    h, w = binary_image.shape[:2]
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_rect = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (h * w) * 0.3:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) >= 4 and area > best_area:
            best_area = area
            best_rect = cv2.boundingRect(cnt)

    if best_rect:
        x, y, rw, rh = best_rect
        roi_x1 = x + int(rw * 0.01)
        roi_y1 = y + int(rh * 0.01)
        roi_x2 = x + int(rw * 0.82)
        roi_y2 = y + int(rh * 0.70)
    else:
        roi_x1 = int(w * 0.02)
        roi_y1 = int(h * 0.02)
        roi_x2 = int(w * 0.80)
        roi_y2 = int(h * 0.68)

    return roi_x1, roi_y1, roi_x2, roi_y2


def detect_ducts_line_pairs(binary_image: np.ndarray, original: np.ndarray) -> list[BoundingBox]:
    """Detect ducts by extracting lines and pairing parallel ones."""
    h, w = original.shape[:2]

    roi_x1, roi_y1, roi_x2, roi_y2 = auto_detect_roi(binary_image)
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
        h_segments.append((x, y_mid, x + cw, y_mid, cw))

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
        v_segments.append((x_mid, y, x_mid, y + ch, ch))

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

                # Score: prefer close, well-overlapping, similar-length pairs
                len_ratio = min(segments[i][4], segments[j][4]) / max(segments[i][4], segments[j][4])
                score = gap - (overlap / min_len) * 50 - len_ratio * 20

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
                score = gap - (overlap / min_len) * 50 - len_ratio * 20

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
