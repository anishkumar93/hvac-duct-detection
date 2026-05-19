"""
Geometry-first HVAC duct detection engine  (Stages 4-6).

Stage 4 — Dual-pass morphological line isolation + CC-based segment extraction
Stage 5 — Topological pairing with PPI-adaptive gap windows + hollowness validation
Stage 6 — Collinear stitching and IOU-based overlap merge

Public API
----------
detect_ducts_geometry(binary, roi, ppi)  -> list[BoundingBox]
detect_drawing_roi(binary)               -> (x1, y1, x2, y2)
"""

import cv2
import numpy as np
import os
from app.models.schemas import BoundingBox


# ── Segment tuple layout ──────────────────────────────────────────────────────
# idx  0       1         2       3         4       5          6      7      8      9
#  H: (x1,  y_center,  x2,  y_center,  length, thickness, cc_x1, cc_y1, cc_x2, cc_y2)
#  V: (x_c,   y1,     x_c,   y2,      length, thickness, cc_x1, cc_y1, cc_x2, cc_y2)
#
# Indices 0-5 are used by the pairing logic.
# Indices 6-9 (CC bounding box) are used exclusively by the hollowness checker.
# ─────────────────────────────────────────────────────────────────────────────


# ═════════════════════════════════════════════════════════════════════════════
# Public entry points
# ═════════════════════════════════════════════════════════════════════════════

def detect_ducts_geometry(
    binary: np.ndarray,
    roi: tuple[int, int, int, int],
    ppi: float | None = None,
) -> list[BoundingBox]:
    """Run geometry detection pipeline (Stages 4-6).

    Args:
        binary : Full-image binary (lines=white 255, bg=black 0).
        roi    : (x1, y1, x2, y2) drawing area in full-image pixel coords.
        ppi    : Pixels per real-world inch from scale calibration.
                 None → conservative pixel defaults.

    Returns:
        BoundingBoxes in full-image pixel coordinates.
    """
    roi_x1, roi_y1, roi_x2, roi_y2 = roi
    roi_binary = binary[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_h, roi_w = roi_binary.shape[:2]

    # PPI-adaptive gap window.
    #
    # "Double lines" (two closely-spaced parallel lines) are the drawing
    # convention for WALLS, not ducts.  Wall thickness is typically 4–8" in
    # real space.  Setting the minimum gap at 10" safely excludes wall
    # double-lines (≤ 8") while keeping all ducts (10"–48").
    # Without PPI: keep a small pixel floor so uncalibrated drawings still work.
    if ppi and ppi > 0:
        min_gap = max(8,   int(ppi * 6))    # 6" minimum — keeps small branch ducts feeding diffusers
        max_gap = min(500, int(ppi * 48))
        print(f"[Geometry] PPI={ppi:.1f} → gap [{min_gap}–{max_gap}]px "
              f"({min_gap/ppi:.0f}\"–{max_gap/ppi:.0f}\" real)")
    else:
        min_gap = 8
        max_gap = max(200, int(roi_w * 0.04))
        print(f"[Geometry] No PPI → pixel defaults [{min_gap}–{max_gap}]px")

    # ── Stage 4a/4b: Adaptive OPEN kernel lengths ─────────────────────────────
    # The kernel must be long enough to erase structural hatch stripes while
    # short enough to keep real duct walls.
    #
    # Hatch stripes in wall sections are drawn at the scale of the wall
    # thickness — typically 4-8" in real space.  HVAC ducts run continuously
    # for feet or more.  Using PPI: min 18" real ensures hatch stripes
    # (≤ 8" = 50px at 1/4"=1'-0", 300 DPI) are removed while ducts (≥24" =
    # 150px) survive.  Without PPI: 8% of ROI dimension is a safe proxy.

    if ppi and ppi > 0:
        h_klen = max(50, int(ppi * 18))   # 18" min horizontal run
        v_klen = max(50, int(ppi * 18))   # 18" min vertical run
    else:
        h_klen = max(50, int(roi_w * 0.08))
        v_klen = max(50, int(roi_h * 0.08))

    h_mask  = cv2.morphologyEx(
        roi_binary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (h_klen, 1))
    )
    # Small horizontal dilation reconnects gaps caused by text or callouts
    h_mask  = cv2.dilate(
        h_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    )

    v_mask  = cv2.morphologyEx(
        roi_binary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_klen))
    )
    v_mask  = cv2.dilate(
        v_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    )

    _dbg(h_mask, "04_h_mask.png")
    _dbg(v_mask, "05_v_mask.png")

    # Compute CC labels (needed for stroke consistency profiling)
    _, labels_h, _, _ = cv2.connectedComponentsWithStats(h_mask, connectivity=8)
    _, labels_v, _, _ = cv2.connectedComponentsWithStats(v_mask, connectivity=8)

    # ── Stage 4c: CC-based segment extraction ─────────────────────────────────
    h_segs = _extract_cc_segments(h_mask, labels_h, axis='h', roi_w=roi_w, roi_h=roi_h)
    v_segs = _extract_cc_segments(v_mask, labels_v, axis='v', roi_w=roi_w, roi_h=roi_h)

    # ── Stage 4d: Stitch broken segments ──────────────────────────────────────
    # Duct walls broken by overlapping symbols/text appear as multiple short
    # CCs on the same line.  Merge them before pairing so the full wall length
    # is available for the length-equality check.
    h_segs = _stitch_segments(h_segs, axis='h')
    v_segs = _stitch_segments(v_segs, axis='v')
    print(f"[Geometry] CC segments → H:{len(h_segs)}  V:{len(v_segs)}")

    # ── Stage 5: Topological pairing + hollowness gate ────────────────────────
    h_boxes = _pair_and_validate(
        h_segs, roi_binary, h_mask, v_mask,
        axis='h', min_gap=min_gap, max_gap=max_gap,
        offset_x=roi_x1, offset_y=roi_y1,
        roi_h=roi_h, roi_w=roi_w,
    )
    v_boxes = _pair_and_validate(
        v_segs, roi_binary, h_mask, v_mask,
        axis='v', min_gap=min_gap, max_gap=max_gap,
        offset_x=roi_x1, offset_y=roi_y1,
        roi_h=roi_h, roi_w=roi_w,
    )
    print(f"[Geometry] Confirmed pairs → H:{len(h_boxes)}  V:{len(v_boxes)}")

    # ── Stage 6: Stitch collinear + merge overlapping ─────────────────────────
    boxes = h_boxes + v_boxes
    boxes = _stitch_collinear(boxes)
    boxes = _merge_overlapping(boxes, iou_threshold=0.3)

    # ── Stage 6b: Post-pair filters ───────────────────────────────────────────
    before = len(boxes)

    # Minimum duct run length — direction markers and small symbols are short
    if ppi and ppi > 0:
        min_len_px = int(ppi * 24)                  # 24" minimum real duct
    else:
        min_len_px = int(max(roi_w, roi_h) * 0.04)

    # Maximum duct length — real ducts don't span >25% of the ROI.
    # Detections longer than this are room boundaries or structural walls.
    max_len_px = int(max(roi_w, roi_h) * 0.25)

    # Maximum gap — reject unrealistically wide detections.
    # Real HVAC ducts rarely exceed 48" (already capped by max_gap in pairing),
    # but additional cap at 24" (or 150px without PPI) catches room-width pairs
    # that slip through when walls happen to be parallel and hollow.
    if ppi and ppi > 0:
        max_duct_gap = int(ppi * 24)   # 24" max duct cross-section
    else:
        max_duct_gap = 150

    # Minimum aspect ratio — equipment boxes are roughly square; real ducts
    # are elongated.  length / gap >= 2.5 keeps any duct that is at least
    # 2.5× longer than it is wide.
    MIN_ASPECT = 2.5

    filtered = []
    for b in boxes:
        length = max(b.width, b.height)
        gap    = min(b.width, b.height)
        if length < min_len_px:
            continue
        if length > max_len_px:
            continue
        if gap > max_duct_gap:
            continue
        if gap > 0 and length / gap < MIN_ASPECT:
            continue
        filtered.append(b)

    removed = before - len(filtered)
    if removed:
        print(f"[Geometry] Post-pair filter removed {removed} boxes "
              f"(min_len={min_len_px}, max_len={max_len_px}, "
              f"max_gap={max_duct_gap}, min_aspect={MIN_ASPECT})")
    boxes = filtered

    print(f"[Geometry] Final duct candidates: {len(boxes)}")
    return boxes


def detect_drawing_roi(
    binary: np.ndarray,
    original: np.ndarray = None,
) -> tuple[int, int, int, int]:
    """Locate the floor-plan drawing area using Tesseract OCR.

    Runs Tesseract at full resolution on the original image to find section
    header text whose positions mark where the drawing area ends.
    Falls back to pixel-density grid analysis when OCR yields no landmarks.

    Returns (x1, y1, x2, y2) in full-image pixel coordinates.
    """
    h, w = binary.shape[:2]

    if original is not None:
        roi = _roi_from_tesseract(original, w, h)
        if roi:
            return roi
        print("[Geometry] Tesseract ROI found no landmarks — using density fallback")

    return _roi_from_density(binary, w, h)


def _roi_from_tesseract(
    image: np.ndarray, img_w: int, img_h: int
) -> tuple[int, int, int, int] | None:
    """Run full-resolution Tesseract to find ROI boundary landmarks.

    Strategy
    --------
    1. Run pytesseract.image_to_data (PSM 11) on the full grayscale image.
       PSM 11 = sparse text — good for finding isolated labels anywhere.
    2. Group adjacent words into text lines using Tesseract's own block/line IDs.
    3. Match line text against two keyword sets:
       - NOTES_PHRASES  → marks where the notes section begins (→ roi_y2)
       - TITLE_PHRASES  → marks where the title block begins  (→ roi_x2)
    4. Apply safety clamps so the ROI always covers a usable portion of the image.
    """
    try:
        import pytesseract
    except ImportError:
        print("[Geometry] pytesseract not available for ROI detection")
        return None

    # Convert to grayscale — Tesseract works on single-channel images
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

    from PIL import Image as _PIL
    try:
        data = pytesseract.image_to_data(
            _PIL.fromarray(gray),
            config="--psm 11 --oem 1",
            output_type=pytesseract.Output.DICT,
        )
    except Exception as exc:
        print(f"[Geometry] Tesseract error during ROI detection: {exc}")
        return None

    # ── Build word list ──────────────────────────────────────────────────────
    words = []
    n = len(data["text"])
    for i in range(n):
        text = str(data["text"][i]).strip()
        conf = int(data["conf"][i])
        if not text or conf < 20:
            continue
        words.append({
            "text": text.upper(),
            "x":    int(data["left"][i]),
            "y":    int(data["top"][i]),
            "x2":   int(data["left"][i]) + int(data["width"][i]),
            "y2":   int(data["top"][i])  + int(data["height"][i]),
            "key":  (data["page_num"][i], data["block_num"][i], data["line_num"][i]),
        })

    # ── Group words into lines (same page/block/line id) ────────────────────
    line_map: dict = {}
    for w in words:
        line_map.setdefault(w["key"], []).append(w)

    # ── Keywords ─────────────────────────────────────────────────────────────
    # Bottom boundary markers — found in lower portion of image, left of title block
    NOTES_PHRASES = [
        "GENERAL NOTES", "PLAN NOTES", "GENERAL NOTE",
        "KEYNOTES", "LEGEND", "SYMBOLS",
        # "MECHANICAL FLOOR PLAN" appears just above the notes section in many
        # HVAC sheets — use it as an additional bottom boundary indicator when
        # found in the left 80% of the image (not the title block copy).
        "MECHANICAL FLOOR PLAN", "MECHANICAL PLAN",
        "FLOOR PLAN",
    ]
    # Right boundary markers — found only in the right portion of the image
    TITLE_PHRASES = [
        "DO NOT SCALE", "ISSUE DATE", "ISSUED FOR",
        "DRAWN BY", "CHECKED BY", "PROJECT NAME",
        "SHEET TITLE", "SHEET NO", "REVISION",
        "LICENSE", "SEAL",
    ]

    notes_y: int | None = None   # topmost y of a notes / drawing-title header
    title_x: int | None = None   # leftmost x of a title-block marker

    for key, wlist in line_map.items():
        wlist_sorted = sorted(wlist, key=lambda w: w["x"])
        line_text    = " ".join(w["text"] for w in wlist_sorted)
        line_x_min   = wlist_sorted[0]["x"]
        line_y_min   = min(w["y"] for w in wlist_sorted)

        # Bottom boundary: look in bottom 65% of image AND left 85% of image
        # (the title block column also contains "MECHANICAL FLOOR PLAN" — we
        # exclude it by requiring line_x_min < 85% of image width)
        if line_y_min > img_h * 0.30 and line_x_min < img_w * 0.85:
            for phrase in NOTES_PHRASES:
                if phrase in line_text:
                    if notes_y is None or line_y_min < notes_y:
                        notes_y = line_y_min
                        print(f"[Geometry] Bottom marker '{phrase}' at y={line_y_min}")
                    break

        # Right boundary: look in right 45% of image only
        if line_x_min > img_w * 0.50:
            for phrase in TITLE_PHRASES:
                if phrase in line_text:
                    if title_x is None or line_x_min < title_x:
                        title_x = line_x_min
                        print(f"[Geometry] Title marker '{phrase}' at x={line_x_min}")
                    break

    # ── Compute ROI with safety margins ──────────────────────────────────────
    roi_x1 = int(img_w * 0.01)
    roi_y1 = int(img_h * 0.01)
    roi_x2 = (title_x - 30) if title_x is not None else int(img_w * 0.85)
    roi_y2 = (notes_y - 30) if notes_y is not None else int(img_h * 0.72)

    # Never let the ROI shrink below a useful minimum
    roi_x2 = max(roi_x2, int(img_w * 0.45))
    roi_y2 = max(roi_y2, int(img_h * 0.35))
    roi_x2 = min(roi_x2, img_w - 1)
    roi_y2 = min(roi_y2, img_h - 1)

    print(f"[Geometry] Tesseract ROI: ({roi_x1},{roi_y1}) → ({roi_x2},{roi_y2}) "
          f"[title_x={title_x}, notes_y={notes_y}]")
    return roi_x1, roi_y1, roi_x2, roi_y2


def _roi_from_density(binary: np.ndarray, w: int, h: int) -> tuple[int, int, int, int]:
    """Fallback: grid-based pixel-density analysis to find drawing area."""
    min_len = w // 10
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1)))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_len)))
    borders = cv2.bitwise_or(
        cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.8), 1))),
        cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h * 0.8)))),
    )
    lines = cv2.subtract(cv2.bitwise_or(h_lines, v_lines), borders)
    text  = cv2.subtract(binary, cv2.bitwise_or(h_lines, v_lines))

    n_cols = 10
    cell_w = w // n_cols
    roi_x2 = w
    for c in range(n_cols - 1, n_cols // 2, -1):
        x1c, x2c = c * cell_w, (c + 1) * cell_w
        area     = h * (x2c - x1c)
        line_d   = cv2.countNonZero(lines[:, x1c:x2c]) / area
        text_d   = cv2.countNonZero(text[:, x1c:x2c])  / area
        if text_d > 0.08 and text_d > line_d * 5:
            roi_x2 = c * cell_w
        else:
            break
    roi_x2 = max(roi_x2, int(w * 0.50))

    roi_x1 = int(w * 0.01)
    roi_y1 = int(h * 0.01)
    roi_y2 = int(h * 0.72)

    print(f"[Geometry] Density ROI: ({roi_x1},{roi_y1}) → ({roi_x2},{roi_y2})")
    return roi_x1, roi_y1, roi_x2, roi_y2


# ═════════════════════════════════════════════════════════════════════════════
# Stage 4c — CC-based segment extraction
# ═════════════════════════════════════════════════════════════════════════════

def _extract_cc_segments(
    line_mask: np.ndarray,
    labels: np.ndarray,
    axis: str,
    roi_w: int,
    roi_h: int,
) -> list[tuple]:
    """Build structured segment tuples from a directional line mask."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(line_mask, connectivity=8)

    # Minimum thickness: reject very thin lines (1px noise/grid artifacts).
    # Real duct walls are at least 2px at 300 DPI. Lines at exactly 2px are
    # borderline — they could be thin duct walls or grid lines. We keep them
    # and rely on pairing validation to reject false pairs.
    MIN_THICKNESS = max(2.0, min(roi_w, roi_h) * 0.0004)
    MAX_THICKNESS = 80.0
    MIN_ASPECT    = 4.0       # length / thickness

    segs: list[tuple] = []
    for i in range(1, n):    # label 0 = background
        cc_x  = stats[i, cv2.CC_STAT_LEFT]
        cc_y  = stats[i, cv2.CC_STAT_TOP]
        cc_w  = stats[i, cv2.CC_STAT_WIDTH]
        cc_h  = stats[i, cv2.CC_STAT_HEIGHT]
        area  = stats[i, cv2.CC_STAT_AREA]

        if axis == 'h':
            length    = float(cc_w)
            thickness = area / max(cc_w, 1)
            if length    < roi_w * 0.015: continue   # too short
            if cc_w      > roi_w * 0.93:  continue   # full-width border
            if cc_h      > roi_h * 0.08:  continue   # too tall (not a line)
        else:
            length    = float(cc_h)
            thickness = area / max(cc_h, 1)
            if length    < roi_h * 0.015: continue
            if cc_h      > roi_h * 0.93:  continue
            if cc_w      > roi_w * 0.08:  continue

        if length / max(thickness, 0.1) < MIN_ASPECT:           continue
        if not (MIN_THICKNESS <= thickness <= MAX_THICKNESS):    continue

        # ── Stroke consistency check ──────────────────────────────────────────
        # A real duct wall has uniform stroke from end to end.  A CC that is
        # thin at the edges but thick in the middle (grid line merged with a
        # duct wall via dilation) is NOT a single uniform segment.
        # Split such CCs into only the thick (uniform) portion.
        cc_x2, cc_y2 = cc_x + cc_w, cc_y + cc_h

        sub_segs = _split_by_stroke_consistency(
            line_mask, labels, i, cc_x, cc_y, cc_w, cc_h, axis,
            MIN_THICKNESS, roi_w, roi_h,
        )
        if sub_segs:
            segs.extend(sub_segs)
        else:
            # CC has uniform stroke — use as-is
            if axis == 'h':
                y_c = cc_y + cc_h / 2.0
                segs.append((float(cc_x), y_c, float(cc_x2), y_c,
                             length, thickness, cc_x, cc_y, cc_x2, cc_y2))
            else:
                x_c = cc_x + cc_w / 2.0
                segs.append((x_c, float(cc_y), x_c, float(cc_y2),
                             length, thickness, cc_x, cc_y, cc_x2, cc_y2))
    return segs


def _split_by_stroke_consistency(
    line_mask: np.ndarray,
    labels: np.ndarray,
    cc_label: int,
    cc_x: int, cc_y: int, cc_w: int, cc_h: int,
    axis: str,
    min_thickness: float,
    roi_w: int, roi_h: int,
    n_samples: int = 20,
) -> list[tuple] | None:
    """Check if a CC has a merged grid-line + duct-wall structure and trim it.

    A merged CC has a long thin portion (grid line) connected to a thick
    portion (duct wall).  We detect this by checking if a LARGE fraction
    (>30%) of the CC's length is significantly thinner than the thickest
    portion.  Minor endpoint thinning (normal for duct walls) is ignored.

    Returns:
        None if the CC is already uniform (caller uses it as-is).
        List of segment tuples for the thick sub-region if split was needed.
        Empty list if nothing usable remains after trimming.
    """
    # Sample thickness at evenly spaced points
    profile = []
    if axis == 'h':
        for s in range(n_samples):
            sx = cc_x + int(cc_w * (s + 0.5) / n_samples)
            if sx < 0 or sx >= line_mask.shape[1]:
                continue
            col = labels[cc_y:cc_y + cc_h, sx]
            count = int(np.count_nonzero(col == cc_label))
            profile.append((sx, count))
    else:
        for s in range(n_samples):
            sy = cc_y + int(cc_h * (s + 0.5) / n_samples)
            if sy < 0 or sy >= line_mask.shape[0]:
                continue
            row = labels[sy, cc_x:cc_x + cc_w]
            count = int(np.count_nonzero(row == cc_label))
            profile.append((sy, count))

    if len(profile) < 4:
        return None

    thicknesses = [t for _, t in profile]
    t_max = max(thicknesses)
    if t_max <= 0:
        return None

    # A sample is "thin" if it's less than 30% of the max thickness
    thin_threshold = t_max * 0.30
    thin_count = sum(1 for t in thicknesses if t <= thin_threshold)
    thin_fraction = thin_count / len(thicknesses)

    # Only split if >30% of the length is thin (merged grid+duct case).
    # Minor endpoint thinning (1-2 samples) is normal and ignored.
    if thin_fraction < 0.30:
        return None

    # Find the contiguous thick region
    thick_start = None
    thick_end = None
    for pos, t in profile:
        if t > thin_threshold:
            if thick_start is None:
                thick_start = pos
            thick_end = pos

    if thick_start is None or thick_end is None:
        return None

    # Build the trimmed sub-segment
    if axis == 'h':
        sub_length = float(thick_end - thick_start)
        if sub_length < roi_w * 0.015:
            return []  # too short after trimming
        mid_x = (thick_start + thick_end) // 2
        col = labels[cc_y:cc_y + cc_h, mid_x]
        sub_thickness = float(np.count_nonzero(col == cc_label))
        if sub_thickness < min_thickness:
            return []
        y_c = cc_y + cc_h / 2.0
        return [(float(thick_start), y_c, float(thick_end), y_c,
                 sub_length, sub_thickness,
                 thick_start, cc_y, thick_end, cc_y + cc_h)]
    else:
        sub_length = float(thick_end - thick_start)
        if sub_length < roi_h * 0.015:
            return []
        mid_y = (thick_start + thick_end) // 2
        row = labels[mid_y, cc_x:cc_x + cc_w]
        sub_thickness = float(np.count_nonzero(row == cc_label))
        if sub_thickness < min_thickness:
            return []
        x_c = cc_x + cc_w / 2.0
        return [(x_c, float(thick_start), x_c, float(thick_end),
                 sub_length, sub_thickness,
                 cc_x, thick_start, cc_x + cc_w, thick_end)]


# ═════════════════════════════════════════════════════════════════════════════
# Stage 4d — Segment-level stitching
# ═════════════════════════════════════════════════════════════════════════════

def _stitch_segments(segs: list[tuple], axis: str, max_gap_ratio: float = 0.20) -> list[tuple]:
    """Merge collinear broken segments into continuous walls.

    Duct walls broken by overlapping symbols/text appear as multiple short
    CCs on the same centerline.  Merge them if:
      - Same centerline (perpendicular offset < max thickness)
      - Similar thickness (ratio ≤ 2.0)
      - Small gap between endpoints (< 20% of combined length)
    """
    if len(segs) < 2:
        return segs

    # Sort by centerline position then by start coordinate
    if axis == 'h':
        segs = sorted(segs, key=lambda s: (round(s[1] / 5) * 5, s[0]))
    else:
        segs = sorted(segs, key=lambda s: (round(s[0] / 5) * 5, s[1]))

    merged = list(segs)
    changed = True
    while changed:
        changed = False
        out = []
        used = set()

        for i in range(len(merged)):
            if i in used:
                continue
            si = merged[i]
            best_j = -1

            for j in range(i + 1, len(merged)):
                if j in used:
                    continue
                sj = merged[j]

                if axis == 'h':
                    # Same centerline y (within max thickness)
                    if abs(si[1] - sj[1]) > max(si[5], sj[5]) * 1.5:
                        continue
                    # Similar thickness
                    if max(si[5], sj[5]) > min(si[5], sj[5]) * 2.0 + 1:
                        continue
                    # Gap between endpoints
                    gap = sj[0] - si[2]  # left edge of j - right edge of i
                    if gap < 0:
                        gap = si[0] - sj[2]  # maybe j is to the left
                    combined = si[4] + sj[4]
                    if gap > combined * max_gap_ratio:
                        continue
                    if gap > 0 or (min(si[2], sj[2]) - max(si[0], sj[0])) > 0:
                        best_j = j
                        break
                else:
                    if abs(si[0] - sj[0]) > max(si[5], sj[5]) * 1.5:
                        continue
                    if max(si[5], sj[5]) > min(si[5], sj[5]) * 2.0 + 1:
                        continue
                    gap = sj[1] - si[3]
                    if gap < 0:
                        gap = si[1] - sj[3]
                    combined = si[4] + sj[4]
                    if gap > combined * max_gap_ratio:
                        continue
                    if gap > 0 or (min(si[3], sj[3]) - max(si[1], sj[1])) > 0:
                        best_j = j
                        break

            if best_j >= 0:
                sj = merged[best_j]
                if axis == 'h':
                    new_x1 = min(si[0], sj[0])
                    new_x2 = max(si[2], sj[2])
                    new_y = (si[1] * si[4] + sj[1] * sj[4]) / (si[4] + sj[4])
                    new_len = new_x2 - new_x1
                    new_thick = (si[5] * si[4] + sj[5] * sj[4]) / (si[4] + sj[4])
                    new_cc_y1 = min(si[7], sj[7])
                    new_cc_y2 = max(si[9], sj[9])
                    out.append((new_x1, new_y, new_x2, new_y, new_len, new_thick,
                                int(new_x1), new_cc_y1, int(new_x2), new_cc_y2))
                else:
                    new_y1 = min(si[1], sj[1])
                    new_y2 = max(si[3], sj[3])
                    new_x = (si[0] * si[4] + sj[0] * sj[4]) / (si[4] + sj[4])
                    new_len = new_y2 - new_y1
                    new_thick = (si[5] * si[4] + sj[5] * sj[4]) / (si[4] + sj[4])
                    new_cc_x1 = min(si[6], sj[6])
                    new_cc_x2 = max(si[8], sj[8])
                    out.append((new_x, new_y1, new_x, new_y2, new_len, new_thick,
                                new_cc_x1, int(new_y1), new_cc_x2, int(new_y2)))
                used.add(i)
                used.add(best_j)
                changed = True
            else:
                out.append(si)

        merged = out
    return merged


# ═════════════════════════════════════════════════════════════════════════════
# Stage 5 — Topological pairing + hollowness gate
# ═════════════════════════════════════════════════════════════════════════════

def _pair_and_validate(
    segs: list[tuple],
    roi_binary: np.ndarray,
    h_mask: np.ndarray,
    v_mask: np.ndarray,
    axis: str,
    min_gap: int,
    max_gap: int,
    offset_x: int,
    offset_y: int,
    roi_h: int = 0,
    roi_w: int = 0,
    min_overlap: float = 0.40,
) -> list[BoundingBox]:
    """Pair parallel segments and confirm via hollowness check.

    For each segment i:
      1. Collect all geometrically valid candidates j (gap, overlap, weight).
      2. Sort by score (best geometric match first).
      3. Try each candidate; accept the first whose corridor is hollow.

    Trying candidates in order — rather than accepting blindly — handles the
    case where the closest geometric match is a structural wall and the correct
    duct partner is slightly farther away.
    """
    # Sort by position along the pairing axis
    key  = (lambda s: s[1]) if axis == 'h' else (lambda s: s[0])
    segs = sorted(segs, key=key)

    # Build junction mask directly from roi_binary using a long kernel (≥15% of
    # ROI dimension).  This is intentionally separate from the dilated v_mask /
    # h_mask used for segment detection so that dilation artefacts don't inflate
    # short hatch stripes into "long" lines that then get masked as junctions.
    # At 1/4"=1'-0" / 300 DPI: 15% of a 2000px-high ROI ≈ 300px ≈ 9.5 ft of
    # real-world duct — any genuine crossing V-duct will comfortably exceed that.
    perp_dim  = roi_h if axis == 'h' else roi_w
    junc_klen = max(50, int(perp_dim * 0.15))
    if axis == 'h':
        perp = cv2.morphologyEx(
            roi_binary, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, junc_klen))
        )
    else:
        perp = cv2.morphologyEx(
            roi_binary, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (junc_klen, 1))
        )

    used: set[int] = set()
    boxes: list[BoundingBox] = []

    for i in range(len(segs)):
        if i in used:
            continue
        si = segs[i]

        # ── collect and rank candidates ──
        candidates: list[tuple[float, int]] = []
        for j in range(i + 1, len(segs)):
            if j in used:
                continue
            result = _score_pair(si, segs[j], axis, min_gap, max_gap, min_overlap)
            if result is None:
                break           # sorted → all further j are also too far
            if result == 'x':
                continue        # geometrically invalid, keep scanning
            candidates.append((result, j))   # type: ignore[arg-type]

        if not candidates:
            continue

        candidates.sort(key=lambda c: c[0])

        # parallel mask: same-axis lines used for intermediate-line detection
        par = h_mask if axis == 'h' else v_mask

        # ── try best-first; accept first hollow corridor ──
        best_j = -1
        for _, j in candidates:
            sj = segs[j]
            if not _corridor_is_hollow(roi_binary, perp, par, si, sj, axis):
                continue
            if _is_closed_rectangle(roi_binary, si, sj, axis):
                continue   # equipment box — both ends are walled
            if _wall_junction_at_endpoint(perp, si, sj, axis):
                continue   # architectural wall corner
            if _has_t_opening(roi_binary, si, sj, axis):
                continue   # wall with T-opening — one side has a break
            best_j = j
            break

        if best_j < 0:
            continue

        used.add(i)
        used.add(best_j)
        box = _build_box(si, segs[best_j], axis, offset_x, offset_y)
        if box:
            boxes.append(box)

    return boxes


def _score_pair(
    si: tuple, sj: tuple, axis: str,
    min_gap: int, max_gap: int, min_overlap: float,
) -> float | str | None:
    """Compute a compatibility score for a candidate segment pair.

    Return values:
      float   — valid candidate; lower score = better match
      'x'     — geometrically invalid; skip this j but keep scanning
      None    — gap exceeds max_gap; stop the inner loop (segments are sorted)
    """
    gap = (sj[1] - si[1]) if axis == 'h' else (sj[0] - si[0])

    if gap < min_gap:  return 'x'
    if gap > max_gap:  return None

    # "Double-line wall" guard — scale-independent.
    # A duct's interior is substantially wider than the pen stroke used to
    # draw its walls.  If the gap is less than 3× the heavier line weight,
    # the two lines form a "double line" (wall convention), not a duct.
    max_weight = max(si[5], sj[5])
    if gap < max_weight * 3:
        return 'x'

    # Overlap along the shared axis
    if axis == 'h':
        overlap = min(si[2], sj[2]) - max(si[0], sj[0])
    else:
        overlap = min(si[3], sj[3]) - max(si[1], sj[1])

    min_len = min(si[4], sj[4])
    if min_len == 0 or overlap < min_len * min_overlap:
        return 'x'

    # Line weight symmetry — duct walls are drawn with the same pen.
    # Allow up to 2.5× difference (scan variance, anti-aliasing). This rejects
    # 6px+2px pairs (ratio=3.0) but allows 4px+2px (ratio=2.0) and 6px+3px (ratio=2.0).
    # The co-terminus check provides additional protection against grid lines.
    wi, wj = si[5], sj[5]
    if max(wi, wj) > 2.5 * min(wi, wj):
        return 'x'

    # Length equality — both duct walls run the same distance.
    # Ducts are fabricated symmetrically so both walls must be nearly equal.
    # 0.70 means the shorter wall must be at least 70% of the longer one;
    # this kills uneven architectural alignments where a wall segment is
    # nearly twice as long as its partner (0.55 allowed that).
    len_ratio = min(si[4], sj[4]) / max(si[4], sj[4])
    if len_ratio < 0.70:
        return 'x'

    # Co-terminus check — duct walls start and end together.
    # A real duct is a rectangle: both walls share the same start/end x (H)
    # or y (V).  If one line extends far beyond the other on either side,
    # it's a grid/wall line accidentally overlapping a duct wall.
    if axis == 'h':
        overshoot_left  = abs(si[0] - sj[0])
        overshoot_right = abs(si[2] - sj[2])
    else:
        overshoot_left  = abs(si[1] - sj[1])
        overshoot_right = abs(si[3] - sj[3])
    max_overshoot = min(si[4], sj[4]) * 0.30  # allow 30% of shorter line
    if overshoot_left > max_overshoot or overshoot_right > max_overshoot:
        return 'x'

    weight_ratio = min(wi, wj) / (max(wi, wj) + 0.1)
    # Lower score = better match. Heavily penalize weight mismatch so
    # same-weight pairs are strongly preferred over mixed-weight pairs.
    score = gap - (overlap / min_len) * 50 - len_ratio * 80 - weight_ratio * 100
    return score


def _corridor_is_hollow(
    roi_binary: np.ndarray,
    perp_mask: np.ndarray,
    parallel_mask: np.ndarray,
    si: tuple,
    sj: tuple,
    axis: str,
    threshold: float = 0.20,
) -> bool:
    """Return True if the air-gap corridor between two lines is hollow.

    Three checks in order:

    1. Intermediate parallel-line check
       If any row (H-pairs) or column (V-pairs) of the parallel mask inside the
       corridor has > 40% fill, there is a wall line running through the middle of
       the corridor.  This catches cross-pairs where an equipment line pairs with a
       wall line and the actual wall sits in the corridor.

    2. Perpendicular junction masking
       Pixels belonging to long perpendicular lines (T-junction ducts) are zeroed
       before the fill-ratio test so that crossing ducts don't cause valid pairs to
       fail.

    3. Text-density exclusion + fill-ratio threshold
       Text CCs (compact, area < 5000 px², aspect < 5) are zeroed so overlapping
       dimension labels don't inflate the fill count.
       < 20% filled  → hollow air gap  → DUCT  ✓
       ≥ 20% filled  → dense material  → WALL  ✗
    """
    if len(si) < 10 or len(sj) < 10:
        return True

    if axis == 'h':
        y1 = int(si[9]);  y2 = int(sj[7])
        x1 = int(max(si[0], sj[0]));  x2 = int(min(si[2], sj[2]))
    else:
        x1 = int(si[8]);  x2 = int(sj[6])
        y1 = int(max(si[1], sj[1]));  y2 = int(min(si[3], sj[3]))

    if y2 <= y1 or x2 <= x1:
        return True

    img_h, img_w = roi_binary.shape[:2]
    y1, y2 = max(0, y1), min(img_h, y2)
    x1, x2 = max(0, x1), min(img_w,  x2)
    if y2 <= y1 or x2 <= x1:
        return True

    # ── Check 1: intermediate parallel wall line ──────────────────────────────
    # A real duct corridor contains only air.  A wall line crossing the corridor
    # shows up as a row (H) or column (V) with > 40% fill in the parallel mask.
    par_crop = parallel_mask[y1:y2, x1:x2]
    if par_crop.size > 0:
        if axis == 'h':
            span      = max(x2 - x1, 1)
            row_fills = np.count_nonzero(par_crop, axis=1) / span
        else:
            span      = max(y2 - y1, 1)
            row_fills = np.count_nonzero(par_crop, axis=0) / span
        if np.any(row_fills > 0.40):
            return False   # wall line running through the middle of the corridor

    # ── Check 2: perpendicular junction masking ───────────────────────────────
    corridor = roi_binary[y1:y2, x1:x2].copy()
    junction = perp_mask[y1:y2, x1:x2]
    corridor[junction > 0] = 0

    if corridor.size == 0:
        return True

    # ── Text-density exclusion ────────────────────────────────────────────────
    # Dimension labels or room annotations that overlap the corridor inflate
    # the fill ratio and would cause a real duct to be rejected, especially
    # with the tighter 0.20 threshold.  Text CCs are compact (low aspect
    # ratio, bounded area); diagonal hatching lines are elongated (aspect > 5)
    # and survive this filter so they still contribute to the fill count.
    n_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(
        corridor, connectivity=8
    )
    for ci in range(1, n_cc):
        area  = cc_stats[ci, cv2.CC_STAT_AREA]
        cw    = cc_stats[ci, cv2.CC_STAT_WIDTH]
        ch    = cc_stats[ci, cv2.CC_STAT_HEIGHT]
        if area > 5000:
            continue  # too large to be a text character or word
        aspect = max(cw, ch) / max(min(cw, ch), 1)
        if aspect < 5.0:  # compact blob → text / symbol, not a line
            corridor[cc_labels == ci] = 0

    # ── Check 3: fill-ratio threshold ─────────────────────────────────────────
    # Lowered to 0.20 (from 0.30) to catch sparse diagonal hatching that slips
    # under the old threshold.  Text exclusion above prevents false rejections
    # from overlapping dimension labels.
    filled = np.count_nonzero(corridor) / corridor.size
    return filled < threshold


def _is_closed_rectangle(
    roi_binary: np.ndarray,
    si: tuple, sj: tuple,
    axis: str,
    end_margin: int = 4,
    fill_threshold: float = 0.25,
) -> bool:
    """Return True if both ends of the corridor have perpendicular walls.

    An equipment outline (AHU, RTU, VAV box, diffuser frame) is a CLOSED
    rectangle: its top and bottom lines are capped by left and right walls.
    An open duct channel has no perpendicular walls at its ends.

    Samples thin strips from the raw binary at both ends of the corridor
    (rather than the processed masks) so that short equipment-side walls,
    which may be filtered out of the v_mask by the long-kernel MORPH_OPEN,
    are still detected here.

    Two caveats handled by design:
    - T-junction ducts: only ONE end has a perpendicular duct — this function
      only rejects pairs where BOTH ends are walled.
    - Detection tolerance: end_margin covers slight misalignment between line
      CC bbox edges and the actual wall position.
    """
    if len(si) < 10 or len(sj) < 10:
        return False

    img_h, img_w = roi_binary.shape[:2]

    if axis == 'h':
        y1 = int(si[9]);  y2 = int(sj[7])
        x1 = int(max(si[0], sj[0]));  x2 = int(min(si[2], sj[2]))
    else:
        x1 = int(si[8]);  x2 = int(sj[6])
        y1 = int(max(si[1], sj[1]));  y2 = int(min(si[3], sj[3]))

    if y2 <= y1 or x2 <= x1:
        return False

    y1, y2 = max(0, y1), min(img_h, y2)
    x1, x2 = max(0, x1), min(img_w,  x2)

    def _end_fill(ex: int, ey1: int, ey2: int) -> float:
        """Fill ratio of a vertical strip centred on x=ex, y=ey1:ey2."""
        sx1 = max(0, ex - end_margin)
        sx2 = min(img_w, ex + end_margin)
        if sx2 <= sx1:
            return 0.0
        strip = roi_binary[ey1:ey2, sx1:sx2]
        return np.count_nonzero(strip) / max(strip.size, 1)

    def _end_fill_h(ey: int, ex1: int, ex2: int) -> float:
        """Fill ratio of a horizontal strip centred on y=ey, x=ex1:ex2."""
        sy1 = max(0, ey - end_margin)
        sy2 = min(img_h, ey + end_margin)
        if sy2 <= sy1:
            return 0.0
        strip = roi_binary[sy1:sy2, ex1:ex2]
        return np.count_nonzero(strip) / max(strip.size, 1)

    if axis == 'h':
        # Check left and right ends of the H-corridor for vertical walls
        left_fill  = _end_fill(x1, y1, y2)
        right_fill = _end_fill(x2, y1, y2)
        return left_fill > fill_threshold and right_fill > fill_threshold
    else:
        # Check top and bottom ends of the V-corridor for horizontal walls
        top_fill    = _end_fill_h(y1, x1, x2)
        bottom_fill = _end_fill_h(y2, x1, x2)
        return top_fill > fill_threshold and bottom_fill > fill_threshold


def _wall_junction_at_endpoint(
    perp_mask: np.ndarray,
    si: tuple,
    sj: tuple,
    axis: str,
    margin: int = 8,
    min_fill: float = 0.18,
) -> bool:
    """Return True if the pair's lines connect into a perpendicular wall at an endpoint.

    At an architectural wall corner both parallel lines end and a perpendicular
    wall begins.  That perpendicular wall runs *outside* the corridor — above the
    top line AND below the bottom line (for an H-pair).

    At a duct T-junction the branch duct is *inside* the corridor, so the
    perpendicular structure does not appear outside the pair's bounds.

    Uses the long-kernel perp mask so only long structural walls trigger the
    check; short equipment ends or annotation lines are ignored.

    Requires the perpendicular structure to appear on BOTH sides of the
    corridor at the same endpoint — one-sided hits are nearby unrelated lines.
    """
    if len(si) < 10 or len(sj) < 10:
        return False

    img_h, img_w = perp_mask.shape[:2]

    if axis == 'h':
        # si = top line, sj = bottom line (sorted by y_center ascending)
        # Outside bounds: above si's CC top, below sj's CC bottom.
        y_above = int(si[7])    # top of top-line CC → look above here
        y_below = int(sj[9])    # bottom of bottom-line CC → look below here
        gap     = max(int(sj[7]) - int(si[9]), 1)   # corridor interior height
        probe   = max(20, gap)                        # probe ≥ one gap-width outside

        x_left  = max(int(si[6]), int(sj[6]))        # left overlap boundary
        x_right = min(int(si[8]), int(sj[8]))        # right overlap boundary

        def _both_sides(x_probe: int) -> bool:
            x1 = max(0, x_probe - margin)
            x2 = min(img_w, x_probe + margin)
            ya1, ya2 = max(0, y_above - probe), max(0, y_above)
            yb1, yb2 = min(img_h, y_below),     min(img_h, y_below + probe)
            above = perp_mask[ya1:ya2, x1:x2]
            below = perp_mask[yb1:yb2, x1:x2]
            af = np.count_nonzero(above) / max(above.size, 1) if above.size > 0 else 0.0
            bf = np.count_nonzero(below) / max(below.size, 1) if below.size > 0 else 0.0
            return af > min_fill and bf > min_fill   # must appear on BOTH sides

        return _both_sides(x_left) or _both_sides(x_right)

    else:  # V-pair
        # si = left line, sj = right line (sorted by x_center ascending)
        # Outside bounds: left of si's CC left, right of sj's CC right.
        x_left_out  = int(si[6])    # left edge of left-line CC → look left of here
        x_right_out = int(sj[8])    # right edge of right-line CC → look right of here
        gap         = max(int(sj[6]) - int(si[8]), 1)
        probe       = max(20, gap)

        y_top = max(int(si[7]), int(sj[7]))     # top of overlap
        y_bot = min(int(si[9]), int(sj[9]))     # bottom of overlap

        def _both_sides_v(y_probe: int) -> bool:
            y1 = max(0, y_probe - margin)
            y2 = min(img_h, y_probe + margin)
            xa1, xa2 = max(0, x_left_out - probe), max(0, x_left_out)
            xb1, xb2 = min(img_w, x_right_out),    min(img_w, x_right_out + probe)
            left  = perp_mask[y1:y2, xa1:xa2]
            right = perp_mask[y1:y2, xb1:xb2]
            lf = np.count_nonzero(left)  / max(left.size,  1) if left.size  > 0 else 0.0
            rf = np.count_nonzero(right) / max(right.size, 1) if right.size > 0 else 0.0
            return lf > min_fill and rf > min_fill

        return _both_sides_v(y_top) or _both_sides_v(y_bot)


def _has_t_opening(
    roi_binary: np.ndarray,
    si: tuple, sj: tuple,
    axis: str,
    n_samples: int = 10,
    break_threshold: float = 0.30,
) -> bool:
    """Return True if one wall has a significant break (T-opening/doorway).

    A real duct has two continuous parallel walls. A wall with a doorway or
    T-junction has a gap where one side is broken. Sample both walls at
    multiple points along the overlap region. If one wall has >30% of
    samples missing while the other is continuous, it's a wall not a duct.
    """
    if len(si) < 10 or len(sj) < 10:
        return False

    img_h, img_w = roi_binary.shape[:2]

    if axis == 'h':
        x1 = int(max(si[0], sj[0]))
        x2 = int(min(si[2], sj[2]))
        y_top = int(si[1])
        y_bot = int(sj[1])
        if x2 <= x1:
            return False

        top_hits = 0
        bot_hits = 0
        for s in range(n_samples):
            sx = x1 + int((x2 - x1) * (s + 0.5) / n_samples)
            if 0 <= sx < img_w:
                # Check top wall (±3px around centerline)
                y_t = max(0, y_top - 3)
                y_tb = min(img_h, y_top + 4)
                if np.any(roi_binary[y_t:y_tb, sx] > 0):
                    top_hits += 1
                # Check bottom wall
                y_b = max(0, y_bot - 3)
                y_bb = min(img_h, y_bot + 4)
                if np.any(roi_binary[y_b:y_bb, sx] > 0):
                    bot_hits += 1
    else:
        y1 = int(max(si[1], sj[1]))
        y2 = int(min(si[3], sj[3]))
        x_left = int(si[0])
        x_right = int(sj[0])
        if y2 <= y1:
            return False

        top_hits = 0
        bot_hits = 0
        for s in range(n_samples):
            sy = y1 + int((y2 - y1) * (s + 0.5) / n_samples)
            if 0 <= sy < img_h:
                x_l = max(0, x_left - 3)
                x_lr = min(img_w, x_left + 4)
                if np.any(roi_binary[sy, x_l:x_lr] > 0):
                    top_hits += 1
                x_r = max(0, x_right - 3)
                x_rr = min(img_w, x_right + 4)
                if np.any(roi_binary[sy, x_r:x_rr] > 0):
                    bot_hits += 1

    # Both walls should be mostly continuous. If one has a large break
    # (>30% missing) while the other is solid, it's a wall with opening.
    if n_samples == 0:
        return False
    top_ratio = top_hits / n_samples
    bot_ratio = bot_hits / n_samples

    # One wall solid (>80%) and the other broken (<70%) = T-opening
    if top_ratio > 0.80 and bot_ratio < 0.70:
        return True
    if bot_ratio > 0.80 and top_ratio < 0.70:
        return True

    return False


def _build_box(
    si: tuple, sj: tuple, axis: str, offset_x: int, offset_y: int
) -> BoundingBox | None:
    """Create a full-image BoundingBox from a confirmed parallel pair."""
    if axis == 'h':
        y1 = si[1]; y2 = sj[1]
        x1 = int(max(si[0], sj[0])); x2 = int(min(si[2], sj[2]))
        if x2 <= x1: return None
        return BoundingBox(
            x=offset_x + (x1 + x2) / 2.0,
            y=offset_y + (y1 + y2) / 2.0,
            width=float(x2 - x1),
            height=float(abs(y2 - y1)),
            angle=0.0,
        )
    else:
        x1 = si[0]; x2 = sj[0]
        y1 = int(max(si[1], sj[1])); y2 = int(min(si[3], sj[3]))
        if y2 <= y1: return None
        return BoundingBox(
            x=offset_x + (x1 + x2) / 2.0,
            y=offset_y + (y1 + y2) / 2.0,
            width=float(abs(x2 - x1)),
            height=float(y2 - y1),
            angle=0.0,
        )


# ═════════════════════════════════════════════════════════════════════════════
# Stage 6 — Collinear stitching + overlap merge
# ═════════════════════════════════════════════════════════════════════════════

def _stitch_collinear(boxes: list[BoundingBox]) -> list[BoundingBox]:
    """Merge duct segments that share the same trajectory.

    A duct may appear as multiple short bounding boxes when a dimension callout,
    equipment symbol, or junction break interrupts the line.  Two segments are
    collinear and eligible for merging when:
      - Same orientation (both H or both V)
      - Centerline offset < 50% of their shared thickness
      - Thickness ratio ≤ 2:1  (same duct, not two different sizes meeting end-on)
      - Endpoint gap < 30% of combined length
    """
    if len(boxes) < 2:
        return boxes

    merged  = list(boxes)
    changed = True
    while changed:
        changed = False
        out: list[BoundingBox] = []
        used: set[int] = set()

        for i in range(len(merged)):
            if i in used:
                continue
            bi   = merged[i]
            is_h = bi.width > bi.height
            paired = -1

            for j in range(i + 1, len(merged)):
                if j in used:
                    continue
                bj = merged[j]
                if (bj.width > bj.height) != is_h:
                    continue   # different orientation

                if is_h:
                    # Centreline offset must stay within half the SMALLER box's
                    # height — using max here lets a previously-inflated box
                    # cascade-merge with progressively more distant lines.
                    if abs(bi.y - bj.y) > min(bi.height, bj.height) * 0.5:        continue
                    # Boxes with significantly different gaps are different ducts
                    if max(bi.height, bj.height) / (min(bi.height, bj.height) + 1) > 1.5: continue
                    left  = min(bi.x + bi.width / 2, bj.x + bj.width / 2)
                    right = max(bi.x - bi.width / 2, bj.x - bj.width / 2)
                    if right - left > (bi.width + bj.width) * 0.30:               continue
                else:
                    if abs(bi.x - bj.x) > min(bi.width, bj.width) * 0.5:         continue
                    if max(bi.width, bj.width) / (min(bi.width, bj.width) + 1) > 1.5:    continue
                    top = min(bi.y + bi.height / 2, bj.y + bj.height / 2)
                    bot = max(bi.y - bi.height / 2, bj.y - bj.height / 2)
                    if bot - top > (bi.height + bj.height) * 0.30:                continue

                paired = j
                break

            if paired >= 0:
                bj = merged[paired]
                if is_h:
                    nx1 = min(bi.x - bi.width / 2,  bj.x - bj.width / 2)
                    nx2 = max(bi.x + bi.width / 2,  bj.x + bj.width / 2)
                    out.append(BoundingBox(
                        x=(nx1 + nx2) / 2, y=(bi.y + bj.y) / 2,
                        width=nx2 - nx1,   height=max(bi.height, bj.height), angle=0.0,
                    ))
                else:
                    ny1 = min(bi.y - bi.height / 2, bj.y - bj.height / 2)
                    ny2 = max(bi.y + bi.height / 2, bj.y + bj.height / 2)
                    out.append(BoundingBox(
                        x=(bi.x + bj.x) / 2, y=(ny1 + ny2) / 2,
                        width=max(bi.width, bj.width), height=ny2 - ny1, angle=0.0,
                    ))
                used.add(i)
                used.add(paired)
                changed = True
            else:
                out.append(bi)

        merged = out
    return merged


def _merge_overlapping(
    boxes: list[BoundingBox], iou_threshold: float = 0.3
) -> list[BoundingBox]:
    """Remove duplicate detections using IOU and containment checks."""
    if not boxes:
        return boxes
    boxes = sorted(boxes, key=lambda b: b.width * b.height, reverse=True)
    keep: list[BoundingBox] = []
    for box in boxes:
        if not any(_iou(box, k) > iou_threshold or _containment(box, k) > 0.6
                   for k in keep):
            keep.append(box)
    return keep


# ═════════════════════════════════════════════════════════════════════════════
# Geometry utilities
# ═════════════════════════════════════════════════════════════════════════════

def _long_lines_only(mask: np.ndarray, min_length: int) -> np.ndarray:
    """Return a mask containing only CC components whose longest dimension
    exceeds min_length.  Used to strip short hatch stripes from the perp_mask
    before junction exclusion so hatched walls are not treated as hollow ducts.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        length = max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        if length >= min_length:
            out[labels == i] = 255
    return out

def _iou(a: BoundingBox, b: BoundingBox) -> float:
    ax1, ay1 = a.x - a.width / 2,  a.y - a.height / 2
    ax2, ay2 = a.x + a.width / 2,  a.y + a.height / 2
    bx1, by1 = b.x - b.width / 2,  b.y - b.height / 2
    bx2, by2 = b.x + b.width / 2,  b.y + b.height / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1: return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def _containment(a: BoundingBox, b: BoundingBox) -> float:
    """Fraction of a's area covered by b."""
    ax1, ay1 = a.x - a.width / 2,  a.y - a.height / 2
    ax2, ay2 = a.x + a.width / 2,  a.y + a.height / 2
    bx1, by1 = b.x - b.width / 2,  b.y - b.height / 2
    bx2, by2 = b.x + b.width / 2,  b.y + b.height / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1: return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = a.width * a.height
    return inter / area_a if area_a > 0 else 0.0


def _dbg(img: np.ndarray, filename: str) -> None:
    """Save a debug image to the debug/ directory."""
    dbg_dir = os.path.join(os.path.dirname(__file__), "..", "..", "debug")
    os.makedirs(dbg_dir, exist_ok=True)
    cv2.imwrite(os.path.join(dbg_dir, filename), img)
