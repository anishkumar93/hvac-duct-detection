# HVAC Duct Detection and Annotation System

Detects, classifies, and annotates HVAC ductwork from mechanical drawings (PDF/Image).

![status](https://img.shields.io/badge/status-active-brightgreen)

## Features

- **OCR-First Detection** — Finds duct dimensions via Tesseract + EasyOCR, then locates duct geometry
- **Text-First Approach** — Uses dimension label positions to find parallel duct walls
- **Hybrid OCR** — Tesseract for broad scanning, EasyOCR for targeted small-text reading
- **Vector PDF Extraction** — Extracts text/geometry from vector PDFs (PyMuPDF)
- **PDF Validation** — Verifies uploaded PDF is a mechanical drawing before processing
- **Pressure Classification** — Size-based heuristic (High/Medium/Low)
- **Scale Validation** — Auto-extracts drawing scale from title block, rejects dimension mismatches
- **Interactive Canvas** — Zoom (scroll), pan (drag), hover/click duct overlays
- **Duct Schedule** — Sortable table with pressure filter, auto-hides empty columns
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
│  Validate → Preprocess → OCR → Text-First →     │
│  Associate → Classify → Annotate                 │
└──────────────────────────────────────────────────┘
```

| Layer | Stack |
|-------|-------|
| Backend | FastAPI, OpenCV, Tesseract, EasyOCR, PyMuPDF, YOLOv8 (optional) |
| Frontend | React, Axios, SVG overlays |
| Deployment | Docker Compose, Nginx |

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- Tesseract OCR
- Poppler (PDF conversion)

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

```
┌─────────────────────────────────────────────────────────────┐
│ 1. PDF VALIDATION                                            │
│    Score keywords (mechanical, duct, scale, etc.)            │
│    Reject non-mechanical drawings (score < 30)               │
├─────────────────────────────────────────────────────────────┤
│ 2. PDF → IMAGE                                               │
│    300 DPI rasterization via Poppler                          │
│    Extract vector data (text + lines) via PyMuPDF            │
├─────────────────────────────────────────────────────────────┤
│ 3. PREPROCESS                                                │
│    Full resolution (configurable)                            │
│    Grayscale → OTSU threshold → morphological open/close     │
├─────────────────────────────────────────────────────────────┤
│ 4. ROI DETECTION                                             │
│    Vector-based: find section headers (NOTES, FLOOR PLAN)    │
│    Fallback: grid density analysis                           │
│    Excludes title block + notes section                      │
├─────────────────────────────────────────────────────────────┤
│ 5. OCR (Dual Engine)                                         │
│    a. Tesseract global pass on ROI (fast, broad)             │
│    b. Line-pair detection → candidate positions              │
│    c. EasyOCR targeted crops at candidates (accurate)        │
│    d. Normalize misreads: @→⌀, °6→"⌀, 66"→6"⌀              │
│    e. Filter: dimension patterns + size 4-100"               │
│    f. Boost confidence for ⌀/Ø/∅/DIA labels                 │
├─────────────────────────────────────────────────────────────┤
│ 6. TEXT-FIRST DETECTION (Primary)                            │
│    For each dimension label:                                 │
│    a. Check if text is INSIDE duct (lines on both sides)     │
│    b. Check if text is OUTSIDE duct (lines on one side)      │
│    c. Validate wall symmetry (similar line weight)           │
│    d. If no lines found → create duct at text position       │
├─────────────────────────────────────────────────────────────┤
│ 7. ASSOCIATE + CLASSIFY                                      │
│    Nearest-label matching with inside-bbox bonus             │
│    ≤12" → High, 13-24" → Medium, >24" → Low pressure        │
│    Scale validation: reject if pixel size ≠ stated dimension │
├─────────────────────────────────────────────────────────────┤
│ 8. ANNOTATE                                                  │
│    Colored overlays + numbered labels on full-res image      │
└─────────────────────────────────────────────────────────────┘
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/upload` | Upload PDF/image, returns detection results |
| GET | `/api/export/{file_id}` | Download annotated image |
| GET | `/health` | Health check |

### Upload Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | File | required | PDF or image file |
| `scale` | string | null | Drawing scale (e.g., "1/4\"=1'-0\"") |
| `resolution` | int | null | Processing resolution (null=full-res, 7000=balanced, 5000=fast) |

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
      "confidence": 0.90
    }
  ],
  "annotated_image_path": "/outputs/{file_id}/annotated.png"
}
```

### Input Validation

| Check | Error Code | Description |
|-------|-----------|-------------|
| File extension | 400 | Must be .pdf/.png/.jpg/.jpeg/.tiff/.bmp |
| File size | 413 | Maximum 50MB |
| Empty file | 400 | Rejected |
| Magic bytes | 400 | Content must match extension |
| PDF validation | 422 | Must be a mechanical drawing (keyword scoring) |
| Image readability | 422 | Must be valid, non-corrupt |
| Min resolution | 422 | At least 500×500px |

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
│   │       ├── pipeline.py      # Orchestrates: OCR → text-first → annotate
│   │       ├── preprocessor.py  # Image loading + binary threshold
│   │       ├── detector.py      # Line-pair detection + text-first detection
│   │       ├── ocr.py           # Tesseract + EasyOCR + normalization
│   │       ├── pdf_analyzer.py  # Vector PDF extraction + validation
│   │       ├── pdf_converter.py # PDF → PNG rasterization
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

## OCR Strategy

| Engine | Role | Strength |
|--------|------|----------|
| Tesseract | Global scan of drawing ROI | Fast, finds all text types |
| EasyOCR | Targeted crops at duct positions | Reads small text overlaid on lines |

**Normalization handles common misreads:**
- `66"` → `6"⌀` (leading 6 = misread ⌀ symbol)
- `18"@` → `18"⌀` (@ = misread ⌀)
- `12°6` → `12"⌀` (° = misread ", 6 = misread ⌀)
- `14°` → `14"⌀`
- `22"x14"` → `22"×14"`

## Line-Pair Detection (for EasyOCR targeting)

Line-pair detection is used to find **candidate duct positions** for EasyOCR crops, not as a duct detector itself. It validates:

1. **Gap** between parallel lines (10-120px)
2. **Overlap** ratio (>40% of shorter line)
3. **Length similarity** (prefer similar-length pairs)
4. **Line weight similarity** (reject if one line is >3x thicker than the other)


## Suggested Improvements & Scaling

### 1. PaddleOCR (Alternative OCR Engine)

**What:** PaddleOCR (PP-OCRv5) reads small engineering text better than both Tesseract and EasyOCR, but requires PaddlePaddle framework.

**Best for:**
- Reading `8"⌀` and other small dimension text with near-perfect accuracy
- Works on small crops (~500×300px) without hanging

**Trade-offs:**
| Factor | Impact |
|--------|--------|
| Accuracy | Best of all OCR engines for this use case |
| Compatibility | Hangs on large images (>2000px) on some machines |
| Install size | ~500MB (PaddlePaddle + models) |
| GPU support | Optional but significantly faster with GPU |

**Recommendation:** Use for targeted crops only (not full-image scanning). Falls back to EasyOCR if unavailable.

### 2. LLM Agentic Post-Processing

**What:** Send structured vector data (text positions + line geometry) to an LLM (Claude/GPT-4) for reasoning about duct classification and OCR validation.

**Where it fits:**
```
Current pipeline output → LLM receives JSON of detected ducts + vector text →
Returns: supply/return classification, corrected dimensions, rejected false positives
```

**Best for:**
- Supply/Return/Exhaust classification (reads nearby "S", "R", "E" labels)
- Validating OCR misreads ("is 66" actually 6"⌀?")
- Rejecting false positives using spatial context ("this is near room 101 boundary, not a duct")

**Trade-offs:**
| Factor | Impact |
|--------|--------|
| Cost | ~$0.01-0.02 per drawing (text-only JSON prompt) |
| Latency | +2-3s per drawing |
| Accuracy gain | Supply/return classification, fewer false positives |
| Dependency | Requires API key + internet connectivity |

### 3. YOLOv8 Object Detection

**What:** Train a custom YOLOv8 model on labeled HVAC drawings to detect ducts directly from pixels — no line-pair logic needed.

**Best for:**
- Detecting curved ducts, elbows, transitions, reducers
- Handling diverse drawing styles across firms
- Reducing false positives (learns what ducts look like vs walls)

**Requirements:**
- 50-100 annotated drawings (bounding boxes around ducts)
- Training: 1-3 hours on a GPU (free on Google Colab)
- Inference: <2s per drawing

**Licensing:**
| License | Usage | Cost |
|---------|-------|------|
| AGPL-3.0 (default) | Open source projects, must share source code | Free |
| Enterprise License | Commercial/proprietary products, no source sharing required | Paid (contact Ultralytics) |

> ⚠️ If deploying commercially without open-sourcing your code, you need the Enterprise license from Ultralytics.

**Trade-offs:**
| Factor | Impact |
|--------|--------|
| Upfront cost | 3-5 hours labeling + training time |
| Accuracy gain | Significant — handles curves, elbows, diverse styles |
| Speed | Faster than current pipeline (1-2s vs 30-40s) |
| Maintenance | Needs retraining when new drawing styles appear |
| Licensing | AGPL (free/open) or Enterprise (paid/commercial) |

### 4. Vision Language Model (VLM) for OCR

**What:** Send duct region image crops to a multimodal model (Claude Vision, GPT-4V) to read dimension text that Tesseract/EasyOCR can't.

**Best for:**
- Reading very small text (<12px) overlaid on duct lines
- Handling rotated/curved dimension text
- Interpreting non-standard notation

**Trade-offs:**
| Factor | Impact |
|--------|--------|
| Cost | ~$0.03-0.10 per drawing (image tokens are expensive) |
| Latency | +5-10s per drawing |
| Accuracy gain | Catches dimensions both Tesseract and EasyOCR miss |
| Dependency | API key + internet + higher token cost |

**Recommendation:** Use only as fallback for ducts where both OCR engines fail, not as primary OCR.

## Known Limitations

- Only detects straight horizontal/vertical ducts (no curves or elbows)
- OCR accuracy depends on text clarity and drawing quality
- Small dimension text (<12px) may be missed by both OCR engines
- No supply/return classification yet (all ducts show as "Unknown" type)
- Full resolution processing takes 30-40s per drawing
