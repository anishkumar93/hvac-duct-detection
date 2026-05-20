# HVAC Duct Detection and Annotation System

Detects, classifies, and annotates HVAC ductwork from mechanical drawings (PDF/Image).

![status](https://img.shields.io/badge/status-active-brightgreen)

## Features

- **Geometry-First Detection** — Finds duct walls via morphological line isolation + parallel pairing
- **Angled Duct Support** — Detects slanting ducts at any angle using rotated morphological kernels
- **Multi-pass OCR** — Tesseract with geometry masking, upscaling, and normalization
- **Text-First Fallback** — Uses unmatched dimension labels to find ducts geometry missed
- **Damper Detection** — Identifies duct dampers (perpendicular closure lines) as positive confirmation
- **Stroke Consistency** — Splits merged grid+duct CCs by profiling thickness along length
- **Scale Calibration** — Auto-extracts drawing scale from title block, PPI-adaptive thresholds
- **Pressure Classification** — Size-based heuristic (High/Medium/Low)
- **Interactive Canvas** — Zoom (scroll), pan (drag), hover/click duct overlays
- **Duct Schedule** — Sortable table with pressure filter
- **Annotated Export** — Download full-res PNG with overlays
- **YOLO Ready** — Drop trained weights into `backend/weights/best.pt` for ML-based detection

## Architecture

```
┌─────────────────────────────────────────────────┐
│              Frontend (React)                     │
│  Upload → Canvas (Zoom/Pan) → Schedule            │
└──────────────────┬──────────────────────────────┘
                   │ HTTP (Nginx proxy)
┌──────────────────▼──────────────────────────────┐
│              Backend (FastAPI)                    │
│  Validate → Preprocess → Geometry Engine →       │
│  OCR → Associate → Classify → Annotate           │
└──────────────────────────────────────────────────┘
```

| Layer | Stack |
|-------|-------|
| Backend | FastAPI, OpenCV, Tesseract, PyMuPDF, SciPy, YOLOv8 (optional) |
| Frontend | React, Axios, SVG overlays |
| Deployment | Docker Compose, Nginx |

## Detection Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ 1. SCALE CALIBRATION                                         │
│    Title-block OCR → extract "1/4"=1'-0"" → compute PPI     │
│    PPI makes all thresholds scale-invariant                  │
├─────────────────────────────────────────────────────────────┤
│ 2. ROI DETECTION                                             │
│    Tesseract finds section headers (NOTES, DO NOT SCALE)     │
│    Excludes title block (right) + notes section (bottom)     │
│    Fallback: pixel-density grid analysis                     │
├─────────────────────────────────────────────────────────────┤
│ 3. BINARIZATION + GRID REMOVAL                               │
│    Adaptive Gaussian threshold (OTSU fallback)               │
│    Grid filter: thickness-based clustering, erase thin lines │
├─────────────────────────────────────────────────────────────┤
│ 4. LINE ISOLATION + SEGMENT EXTRACTION                       │
│    a. Morphological OPEN (H/V kernels) isolates duct walls   │
│    b. CC analysis → segment tuples (position, length, thick) │
│    c. Stroke consistency: split merged grid+duct CCs         │
│    d. Stitch broken segments (reconnect wall fragments)      │
├─────────────────────────────────────────────────────────────┤
│ 5. PARALLEL PAIRING + VALIDATION                             │
│    For each segment, find best parallel partner:             │
│    • Gap: PPI-scaled (6"–24" real)                           │
│    • Weight symmetry: ≤2.5× thickness ratio                  │
│    • Length equality: shorter ≥70% of longer                 │
│    • Co-terminus: neither extends >30% beyond the other      │
│    • Hollowness: corridor fill <20% (rejects hatched walls)  │
│    • Damper check: perpendicular closure confirms duct       │
│    • T-opening check: broken wall = reject (door/window)     │
│    • Closed rectangle: both ends walled = reject (equipment) │
├─────────────────────────────────────────────────────────────┤
│ 6. POST-PAIR FILTERS + ANGLED DETECTION                      │
│    • Min length (24" real), max length (25% ROI)             │
│    • Max gap (24"), min aspect ratio (2.5)                   │
│    • Collinear stitching + overlap merge                     │
│    • Angled ducts: rotated kernels at 30°/45°/60°/120°/135°/ │
│      150° → CC extraction → parallel pairing                 │
│      Local hatching rejection (>4 nearby parallel lines)     │
├─────────────────────────────────────────────────────────────┤
│ 7. GLOBAL OCR                                                │
│    Multi-pass Tesseract (geometry-masked + standard + upscale)│
│    Normalize misreads → filter dimension patterns (4–100")   │
├─────────────────────────────────────────────────────────────┤
│ 8. ASSOCIATION + FALLBACKS                                   │
│    a. Hungarian algorithm: optimal label-to-duct matching    │
│    b. Text-first fallback: unmatched labels → search nearby  │
│       parallel lines → create duct if found                  │
│    c. Scale validation: reject if pixel size ≠ stated dim    │
├─────────────────────────────────────────────────────────────┤
│ 9. POST-DETECTION FILTERS                                    │
│    • Equipment box filter (internal structure)               │
│    • Context filter (non-duct text nearby)                   │
│    • Scale validation for unlabelled (70%–150% of confirmed) │
│    • Connectivity (keep isolated if gap matches confirmed)   │
│    • Boundary filter (ROI edge detections)                   │
│    • Confidence scoring (multi-factor)                       │
├─────────────────────────────────────────────────────────────┤
│ 10. ANNOTATE                                                 │
│     Center line + border lines (H/V/angled aligned)          │
│     Numbered labels + dimension text                         │
└─────────────────────────────────────────────────────────────┘
```

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
      "bbox": { "x": 3265, "y": 1906, "width": 1555, "height": 139, "angle": 0.0 },
      "confidence": 0.90
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
│   │   │   ├── upload.py        # POST /api/upload (with validation)
│   │   │   └── export.py        # GET /api/export/{id}
│   │   └── services/
│   │       ├── pipeline.py      # Orchestrates full detection pipeline
│   │       ├── preprocessor.py  # Image loading + adaptive binarization
│   │       ├── geometry.py      # Geometry engine (line isolation, pairing, angled)
│   │       ├── grid_filter.py   # Grid-line removal (thickness clustering)
│   │       ├── ocr.py           # Multi-pass Tesseract + normalization
│   │       ├── pdf_analyzer.py  # Vector PDF extraction + validation
│   │       ├── pdf_converter.py # PDF → PNG rasterization
│   │       ├── associator.py    # Hungarian label-to-duct matching
│   │       ├── classifier.py    # Pressure classification
│   │       ├── post_filters.py  # False-positive filters + confidence scoring
│   │       ├── scale_extractor.py # Scale parsing + PPI computation
│   │       ├── annotator.py     # Draws overlays (H/V/angled aligned)
│   │       └── detector.py      # Legacy text-first + YOLO detection
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

## Geometry Engine Validations

A duct is a **rectangle** (horizontal, vertical, or angled). Both walls must satisfy:

| Validation | Rule |
|------------|------|
| Gap range | PPI-scaled: 6"–24" real |
| Weight symmetry | Both walls ≤2.5× thickness ratio |
| Length equality | Shorter wall ≥70% of longer |
| Co-terminus | Neither wall extends >30% beyond the other |
| Stroke consistency | Uniform thickness along length (no merged grid lines) |
| Hollowness | Corridor between walls <20% filled |
| Damper (positive) | Perpendicular closure at endpoint confirms duct |
| T-opening (negative) | Broken wall with gap = reject (door/window) |
| Closed rectangle | Both ends walled = reject (equipment box) |
| Wall junction | Perpendicular wall at endpoint = reject (architectural corner) |

## Known Limitations

- Shared-wall ducts (two ducts sharing a common wall) — only one is detected
- OCR accuracy depends on text clarity and drawing quality
- Small dimension text (<12px) may be missed
- No supply/return classification yet (all ducts show as "Unknown" type)
- Curved ducts and elbows are not detected
- Full resolution processing takes 30-40s per drawing
