from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.schemas import DetectionResult
from app.services.pdf_converter import convert_to_images
from app.services.pdf_analyzer import validate_mechanical_drawing
from app.services.pipeline import run_detection_pipeline
import os
import uuid
import shutil
import cv2

router = APIRouter()

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MIN_RESOLUTION = 500  # minimum width or height in pixels


@router.post("/upload", response_model=DetectionResult)
async def upload_file(file: UploadFile = File(...), scale: str = None):
    # 1. Validate filename
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    safe_filename = os.path.basename(file.filename)
    ext = os.path.splitext(safe_filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    # 2. Validate file size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, f"File too large ({len(content) // (1024*1024)}MB). Maximum: {MAX_FILE_SIZE // (1024*1024)}MB")

    if len(content) == 0:
        raise HTTPException(400, "Empty file")

    # 3. Validate content type (magic bytes)
    if not _validate_magic_bytes(content, ext):
        raise HTTPException(400, f"File content does not match extension {ext}")

    # 4. Save file
    file_id = str(uuid.uuid4())
    file_dir = os.path.join(UPLOAD_DIR, file_id)
    os.makedirs(file_dir, exist_ok=True)

    file_path = os.path.join(file_dir, safe_filename)
    with open(file_path, "wb") as f:
        f.write(content)

    # 5. Convert PDF or validate image
    if ext == ".pdf":
        # Validate it's a mechanical drawing
        is_valid, reason = validate_mechanical_drawing(file_path)
        if not is_valid:
            shutil.rmtree(file_dir, ignore_errors=True)
            raise HTTPException(422, f"Invalid drawing: {reason}")
        print(f"[Upload] PDF validation: {reason}")

        try:
            image_paths = convert_to_images(file_path, file_dir)
        except Exception as e:
            shutil.rmtree(file_dir, ignore_errors=True)
            raise HTTPException(422, f"Failed to convert PDF: {str(e)}")
    else:
        image_paths = [file_path]

    # 6. Validate image is readable and meets minimum resolution
    img = cv2.imread(image_paths[0])
    if img is None:
        shutil.rmtree(file_dir, ignore_errors=True)
        raise HTTPException(422, "Could not read image. File may be corrupt.")

    h, w = img.shape[:2]
    if w < MIN_RESOLUTION or h < MIN_RESOLUTION:
        shutil.rmtree(file_dir, ignore_errors=True)
        raise HTTPException(422, f"Image too small ({w}×{h}px). Minimum: {MIN_RESOLUTION}×{MIN_RESOLUTION}px")

    # 7. Run detection
    try:
        pdf_source = file_path if ext == '.pdf' else None
        result = run_detection_pipeline(image_paths[0], file_id, scale=scale, pdf_path=pdf_source)
    except Exception as e:
        raise HTTPException(500, f"Detection failed: {str(e)}")

    return result


def _validate_magic_bytes(content: bytes, ext: str) -> bool:
    """Check file magic bytes match the claimed extension."""
    signatures = {
        ".pdf": [b"%PDF"],
        ".png": [b"\x89PNG"],
        ".jpg": [b"\xff\xd8\xff"],
        ".jpeg": [b"\xff\xd8\xff"],
        ".tiff": [b"II\x2a\x00", b"MM\x00\x2a"],
        ".bmp": [b"BM"],
    }
    expected = signatures.get(ext, [])
    if not expected:
        return True
    return any(content[:len(sig)] == sig for sig in expected)
