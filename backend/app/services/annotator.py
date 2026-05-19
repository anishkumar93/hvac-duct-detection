"""Stage 9 — Annotated Overlay Output.

Draws colored duct overlays with numbered labels on the original image.
"""
import cv2
import numpy as np
from app.models.schemas import DuctSegment, PressureClass

COLORS = {
    PressureClass.HIGH: (0, 200, 0),
    PressureClass.MEDIUM: (0, 200, 200),
    PressureClass.LOW: (200, 150, 0),
    PressureClass.UNKNOWN: (0, 180, 0),
}


def annotate_image(image: np.ndarray, ducts: list[DuctSegment], output_path: str) -> str:
    """Draw duct overlays with labels on image and save."""
    annotated = image.copy()

    for duct in ducts:
        color = COLORS.get(duct.pressure_class, (0, 180, 0))
        bbox = duct.bbox
        is_horizontal = bbox.width > bbox.height

        x1 = int(bbox.x - bbox.width / 2)
        y1 = int(bbox.y - bbox.height / 2)
        x2 = int(bbox.x + bbox.width / 2)
        y2 = int(bbox.y + bbox.height / 2)

        # Semi-transparent fill
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)

        # Border
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Numbered label
        label_x = int(bbox.x)
        label_y = int(bbox.y - bbox.height / 2 - 25) if is_horizontal else int(bbox.y)
        cv2.circle(annotated, (label_x, label_y), 18, (0, 0, 0), -1)
        cv2.circle(annotated, (label_x, label_y), 18, (255, 255, 255), 2)
        text = str(duct.id)
        sz = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.putText(annotated, text, (label_x - sz[0]//2, label_y + sz[1]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Dimension text
        if duct.dimension:
            dim_x = int(bbox.x)
            dim_y = int(bbox.y + 5)
            dsz = cv2.getTextSize(duct.dimension, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            dx = dim_x - dsz[0]//2
            dy = dim_y + dsz[1]//2
            cv2.rectangle(annotated, (dx-3, dy-dsz[1]-3), (dx+dsz[0]+3, dy+5), (255,255,255), -1)
            cv2.putText(annotated, duct.dimension, (dx, dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    cv2.imwrite(output_path, annotated)
    return output_path
