import math
from app.models.schemas import BoundingBox
from app.services.ocr import OCRResult


def associate_labels(ducts: list[BoundingBox], labels: list[OCRResult], max_distance: float = 800.0) -> dict[int, OCRResult]:
    """Associate each dimension label to its nearest duct segment.
    Prefers labels inside or very close to the duct bbox.
    When multiple labels match the same duct, prefers higher confidence (vector > OCR).
    Returns mapping: duct_index -> OCRResult.
    """
    associations = {}

    for label in labels:
        best_score = float('inf')
        best_idx = -1

        for i, duct in enumerate(ducts):
            dist = math.hypot(label.center_x - duct.x, label.center_y - duct.y)
            if dist > max_distance:
                continue

            inside = _point_in_bbox(label.center_x, label.center_y, duct, padding=50)
            score = dist * (0.3 if inside else 1.0)

            if score < best_score:
                best_score = score
                best_idx = i

        if best_idx >= 0:
            if best_idx not in associations:
                associations[best_idx] = label
            else:
                existing = associations[best_idx]
                # Prefer higher confidence (vector=0.95 beats OCR=0.3-0.7)
                if label.confidence > existing.confidence:
                    associations[best_idx] = label
                elif label.confidence == existing.confidence:
                    # Same confidence: prefer closer
                    existing_dist = math.hypot(existing.center_x - ducts[best_idx].x,
                                              existing.center_y - ducts[best_idx].y)
                    new_dist = math.hypot(label.center_x - ducts[best_idx].x,
                                          label.center_y - ducts[best_idx].y)
                    if new_dist < existing_dist:
                        associations[best_idx] = label

    return associations


def _point_in_bbox(px: float, py: float, bbox: BoundingBox, padding: int = 0) -> bool:
    """Check if point is inside bbox (with optional padding)."""
    x1 = bbox.x - bbox.width / 2 - padding
    y1 = bbox.y - bbox.height / 2 - padding
    x2 = bbox.x + bbox.width / 2 + padding
    y2 = bbox.y + bbox.height / 2 + padding
    return x1 <= px <= x2 and y1 <= py <= y2
