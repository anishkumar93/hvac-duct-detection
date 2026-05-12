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

    # Scale kernel sizes relative to image (calibrated at ~5000px width)
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

    # Scale gap thresholds with image size
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
