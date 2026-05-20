"""
Label-to-duct association.

Primary:  Hungarian algorithm via scipy.optimize.linear_sum_assignment
          Builds a full cost matrix and finds the globally optimal 1-to-1
          assignment.  A greedy approach can steal a label that should belong
          to a closer duct because it processes labels in arbitrary order.

Fallback: Greedy nearest-neighbour (original behaviour) when scipy is absent.
"""

import math
import numpy as np
from app.models.schemas import BoundingBox
from app.services.ocr import OCRResult


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════

def associate_labels_optimal(
    ducts: list[BoundingBox],
    labels: list[OCRResult],
    max_distance: float = 800.0,
) -> dict[int, OCRResult]:
    """Return the globally optimal duct-index → label mapping.

    Uses scipy when available; falls back to greedy nearest-neighbour.
    """
    if not ducts or not labels:
        return {}

    try:
        from scipy.optimize import linear_sum_assignment
        return _hungarian(ducts, labels, max_distance, linear_sum_assignment)
    except ImportError:
        return _greedy(ducts, labels, max_distance)


# Backward-compat alias — old pipeline called associate_labels()
def associate_labels(
    ducts: list[BoundingBox],
    labels: list[OCRResult],
    max_distance: float = 800.0,
) -> dict[int, OCRResult]:
    return associate_labels_optimal(ducts, labels, max_distance)


# ═════════════════════════════════════════════════════════════════════════════
# Hungarian (optimal) assignment
# ═════════════════════════════════════════════════════════════════════════════

def _hungarian(ducts, labels, max_distance, lsa_fn) -> dict[int, OCRResult]:
    """Solve label-to-duct assignment optimally.

    Cost matrix shape: (n_labels × n_ducts).
    Cost(l, d) = distance × inside_penalty.
    Cells where distance > max_distance are filled with a large sentinel so
    the solver avoids infeasible assignments.

    linear_sum_assignment finds the minimum-cost matching for
    min(n_labels, n_ducts) pairs, leaving the remainder unmatched.
    """
    n_l = len(labels)
    n_d = len(ducts)
    SENTINEL = max_distance * 2.0

    cost = np.full((n_l, n_d), fill_value=SENTINEL, dtype=np.float64)

    for li, label in enumerate(labels):
        for di, duct in enumerate(ducts):
            dist = math.hypot(label.center_x - duct.x, label.center_y - duct.y)
            if dist > max_distance:
                continue
            inside       = _in_bbox(label.center_x, label.center_y, duct, padding=50)
            cost[li, di] = dist * (0.3 if inside else 1.0)

    row_ind, col_ind = lsa_fn(cost)

    associations: dict[int, OCRResult] = {}
    for li, di in zip(row_ind, col_ind):
        if cost[li, di] >= SENTINEL:
            continue  # solver filled an infeasible slot — skip
        label = labels[li]
        # Prefer higher-confidence label if the duct already has one
        if di not in associations or label.confidence > associations[di].confidence:
            associations[di] = label

    return associations


# ═════════════════════════════════════════════════════════════════════════════
# Greedy fallback
# ═════════════════════════════════════════════════════════════════════════════

def _greedy(ducts, labels, max_distance) -> dict[int, OCRResult]:
    associations: dict[int, OCRResult] = {}

    for label in labels:
        best_score = float('inf')
        best_idx   = -1

        for i, duct in enumerate(ducts):
            dist = math.hypot(label.center_x - duct.x, label.center_y - duct.y)
            if dist > max_distance:
                continue
            inside = _in_bbox(label.center_x, label.center_y, duct, padding=50)
            score  = dist * (0.3 if inside else 1.0)
            if score < best_score:
                best_score = score
                best_idx   = i

        if best_idx < 0:
            continue

        if best_idx not in associations:
            associations[best_idx] = label
        else:
            existing = associations[best_idx]
            if label.confidence > existing.confidence:
                associations[best_idx] = label
            elif label.confidence == existing.confidence:
                ed = math.hypot(existing.center_x - ducts[best_idx].x,
                                existing.center_y - ducts[best_idx].y)
                nd = math.hypot(label.center_x    - ducts[best_idx].x,
                                label.center_y    - ducts[best_idx].y)
                if nd < ed:
                    associations[best_idx] = label

    return associations


# ═════════════════════════════════════════════════════════════════════════════
# Geometry helper
# ═════════════════════════════════════════════════════════════════════════════

def _in_bbox(px: float, py: float, bbox: BoundingBox, padding: int = 0) -> bool:
    return (bbox.x - bbox.width  / 2 - padding <= px <= bbox.x + bbox.width  / 2 + padding
        and bbox.y - bbox.height / 2 - padding <= py <= bbox.y + bbox.height / 2 + padding)
