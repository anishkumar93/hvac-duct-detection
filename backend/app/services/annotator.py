import cv2
import numpy as np
from app.models.schemas import DuctSegment, PressureClass

# Color scheme (BGR)
COLORS = {
    PressureClass.HIGH: (0, 200, 0),      # Green
    PressureClass.MEDIUM: (0, 200, 200),   # Yellow
    PressureClass.LOW: (200, 150, 0),      # Blue
    PressureClass.UNKNOWN: (0, 180, 0),    # Default green
}


def annotate_image(image: np.ndarray, ducts: list[DuctSegment], output_path: str) -> str:
    """Draw duct center lines with labels on the image."""
    annotated = image.copy()

    for duct in ducts:
        color = COLORS.get(duct.pressure_class, (0, 180, 0))
        bbox = duct.bbox

        # Determine if horizontal or vertical duct
        is_horizontal = bbox.width > bbox.height

        if is_horizontal:
            # Draw thick horizontal center line
            x1 = int(bbox.x - bbox.width / 2)
            x2 = int(bbox.x + bbox.width / 2)
            cy = int(bbox.y)
            thickness = max(4, int(bbox.height * 0.4))

            # Semi-transparent line using overlay
            overlay = annotated.copy()
            cv2.line(overlay, (x1, cy), (x2, cy), color, thickness)
            cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)

            # Solid border lines (top and bottom of duct)
            y_top = int(bbox.y - bbox.height / 2)
            y_bot = int(bbox.y + bbox.height / 2)
            cv2.line(annotated, (x1, y_top), (x2, y_top), color, 2)
            cv2.line(annotated, (x1, y_bot), (x2, y_bot), color, 2)
        else:
            # Draw thick vertical center line
            y1 = int(bbox.y - bbox.height / 2)
            y2 = int(bbox.y + bbox.height / 2)
            cx = int(bbox.x)
            thickness = max(4, int(bbox.width * 0.4))

            overlay = annotated.copy()
            cv2.line(overlay, (cx, y1), (cx, y2), color, thickness)
            cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)

            x_left = int(bbox.x - bbox.width / 2)
            x_right = int(bbox.x + bbox.width / 2)
            cv2.line(annotated, (x_left, y1), (x_left, y2), color, 2)
            cv2.line(annotated, (x_right, y1), (x_right, y2), color, 2)

        # Numbered label
        label_x = int(bbox.x)
        label_y = int(bbox.y - bbox.height / 2 - 25) if is_horizontal else int(bbox.y)
        cv2.circle(annotated, (label_x, label_y), 18, (0, 0, 0), -1)
        cv2.circle(annotated, (label_x, label_y), 18, (255, 255, 255), 2)
        text = str(duct.id)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        tx = label_x - text_size[0] // 2
        ty = label_y + text_size[1] // 2
        cv2.putText(annotated, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Dimension text
        if duct.dimension:
            dim_x = int(bbox.x)
            dim_y = int(bbox.y + 5)
            dim_size = cv2.getTextSize(duct.dimension, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            dx = dim_x - dim_size[0] // 2
            dy = dim_y + dim_size[1] // 2
            cv2.rectangle(annotated, (dx - 3, dy - dim_size[1] - 3), (dx + dim_size[0] + 3, dy + 5), (255, 255, 255), -1)
            cv2.putText(annotated, duct.dimension, (dx, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    cv2.imwrite(output_path, annotated)
    return output_path
