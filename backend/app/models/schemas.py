from pydantic import BaseModel
from typing import Optional
from enum import Enum


class PressureClass(str, Enum):
    LOW = "Low Pressure"
    MEDIUM = "Medium Pressure"
    HIGH = "High Pressure"
    UNKNOWN = "Unknown"


class DuctType(str, Enum):
    SUPPLY = "Supply"
    RETURN = "Return"
    UNKNOWN = "Unknown"


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float
    angle: float = 0.0


class DuctSegment(BaseModel):
    id: int
    duct_type: DuctType = DuctType.UNKNOWN
    dimension: Optional[str] = None
    length: Optional[str] = None
    pressure_class: PressureClass = PressureClass.UNKNOWN
    bbox: BoundingBox
    confidence: float = 0.0


class DetectionResult(BaseModel):
    image_width: int
    image_height: int
    scale: Optional[str] = None
    ducts: list[DuctSegment] = []
    annotated_image_path: Optional[str] = None


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    pages: int = 1
    message: str
