from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import os

router = APIRouter()

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")


@router.get("/export/{file_id}")
async def export_annotated(file_id: str):
    annotated_path = os.path.join(OUTPUT_DIR, file_id, "annotated.png")
    if not os.path.exists(annotated_path):
        raise HTTPException(404, "Annotated file not found")
    return FileResponse(annotated_path, filename=f"annotated_{file_id}.png")
