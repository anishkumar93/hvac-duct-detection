from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.schemas import UploadResponse, DetectionResult
from app.services.pdf_converter import convert_to_images
from app.services.pipeline import run_detection_pipeline
import os
import uuid
import shutil

router = APIRouter()

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}


@router.post("/upload", response_model=DetectionResult)
async def upload_file(file: UploadFile = File(...), scale: str = None):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    file_id = str(uuid.uuid4())
    file_dir = os.path.join(UPLOAD_DIR, file_id)
    os.makedirs(file_dir, exist_ok=True)

    file_path = os.path.join(file_dir, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Convert PDF to images or use image directly
    if ext == ".pdf":
        image_paths = convert_to_images(file_path, file_dir)
    else:
        image_paths = [file_path]

    # Run detection on first page (extend later for multi-page)
    result = run_detection_pipeline(image_paths[0], file_id, scale=scale)
    return result
