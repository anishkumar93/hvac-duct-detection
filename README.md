# HVAC Duct Detection and Annotation System

Detects, classifies, and annotates HVAC ductwork from mechanical drawings (PDF/Image).

![Dark UI with zoom/pan canvas, duct schedule, and pressure classification](https://img.shields.io/badge/status-active-brightgreen)

## Features

- **Duct Detection** — Rule-based parallel line-pair detection (horizontal + vertical)
- **OCR Dimensions** — Multi-pass Tesseract extracts duct sizes (e.g. `18"⌀`, `22"×14"`)
- **Pressure Classification** — Size-based heuristic (High/Medium/Low)
- **Interactive Canvas** — Zoom (scroll), pan (drag), hover/click duct overlays
- **Duct Schedule** — Sortable table with pressure filter, auto-hides empty columns
- **Annotated Export** — Download full-res PNG with overlays
- **YOLO Ready** — Drop trained weights into `backend/weights/best.pt` for ML-based detection

## Architecture

```
┌─────────────────────────────────────────────┐
│            Frontend (React)                  │
│  Upload → Canvas (Zoom/Pan/SVG) → Schedule  │
└──────────────────┬──────────────────────────┘
                   │ HTTP
┌──────────────────▼──────────────────────────┐
│            Backend (FastAPI)                  │
│  PDF Convert → Preprocess → Detect → OCR →  │
│  Associate → Classify → Annotate             │
└──────────────────────────────────────────────┘
```

| Layer | Stack |
|-------|-------|
| Backend | FastAPI, OpenCV, Tesseract OCR, YOLOv8 (optional) |
| Frontend | React, Axios, SVG overlays |
| Deployment | Docker Compose, Nginx |

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- Tesseract OCR: `brew install tesseract`
- Poppler (PDF conversion): `brew install poppler`

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm start
```

Open http://localhost:3000 and upload an HVAC drawing.

### Docker

```bash
docker compose up --build
```

App available at http://localhost:3000. Backend API at http://localhost:8000.

## Detection Pipeline

1. **PDF → Image** — 300 DPI rasterization via Poppler
2. **Preprocess** — Downscale to 5000px, OTSU threshold, morphological open/close
3. **ROI Detection** — Auto-excludes title block and notes via contour analysis
4. **Line Extraction** — Morphological open with directional kernels (H and V)
5. **Line Pairing** — Pairs parallel lines (gap 10–120px, overlap > 40%) as duct walls
6. **False Positive Filter** — Rejects by thickness (20–80px), length (>100px), aspect ratio (>2.0)
7. **OCR** — Multi-pass Tesseract (OTSU + adaptive + upscaled/sharpened) on full-res image
8. **Normalization** — Fixes common misreads (`@` → `⌀`, `°6` → `"⌀`, etc.)
9. **Association** — Nearest-label matching with inside-bbox bonus
10. **Classification** — ≤12" → High, 13–24" → Medium, >24" → Low pressure
11. **Annotation** — Colored overlays + numbered labels on full-res image

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/upload` | Upload PDF/image, returns detection results |
| GET | `/api/export/{file_id}` | Download annotated image |
| GET | `/health` | Health check |

### Response Schema

```json
{
  "image_width": 10800,
  "image_height": 7200,
  "ducts": [
    {
      "id": 1,
      "duct_type": "Unknown",
      "dimension": "18\"⌀",
      "pressure_class": "Medium Pressure",
      "bbox": { "x": 3265, "y": 1906, "width": 1555, "height": 139 },
      "confidence": 0.68
    }
  ],
  "annotated_image_path": "/outputs/{file_id}/annotated.png"
}
```

## Project Structure

```
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app + CORS + static mount
│   │   ├── models/schemas.py    # Pydantic models
│   │   ├── routers/
│   │   │   ├── upload.py        # POST /api/upload
│   │   │   └── export.py        # GET /api/export/{id}
│   │   └── services/
│   │       ├── pipeline.py      # Orchestrates full detection flow
│   │       ├── preprocessor.py  # Image loading + binary threshold
│   │       ├── detector.py      # Line-pair duct detection
│   │       ├── ocr.py           # Multi-pass Tesseract + dimension regex
│   │       ├── associator.py    # Label-to-duct proximity matching
│   │       ├── classifier.py    # Pressure classification
│   │       └── annotator.py     # Draws overlays on image
│   ├── weights/                 # Place YOLO best.pt here
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.js               # Main layout + tabs
│   │   ├── App.css              # Dark theme styles
│   │   └── components/
│   │       ├── UploadPanel.jsx  # Drag-drop upload + progress
│   │       ├── DrawingCanvas.jsx # Zoom/pan canvas + SVG overlay
│   │       └── DuctSchedule.jsx # Sortable/filterable table
│   ├── .env                     # REACT_APP_API_BASE for local dev
│   ├── nginx.conf               # Production proxy config
│   └── Dockerfile
├── docker-compose.yml
├── .gitignore
└── README.md
```

## YOLOv8 Upgrade Path

The system auto-detects and uses YOLO when weights are available at `backend/weights/best.pt`, otherwise falls back to classical CV.

To train:
1. Annotate 50–100 HVAC drawings with bounding boxes ([Roboflow](https://roboflow.com) or [CVAT](https://cvat.ai))
2. Fine-tune YOLOv8:
   ```python
   from ultralytics import YOLO
   model = YOLO('yolov8m.pt')
   model.train(data='hvac_ducts.yaml', epochs=100, imgsz=1024)
   ```
3. Copy `runs/detect/train/weights/best.pt` → `backend/weights/best.pt`
4. Restart backend — detection automatically switches to YOLO

## Known Limitations

- Only detects straight horizontal/vertical ducts (no curves or elbows)
- OCR accuracy depends on text clarity and drawing quality
- Walls with parallel lines at duct-like spacing may occasionally be detected
- No supply/return classification yet (all ducts show as "Unknown" type)
