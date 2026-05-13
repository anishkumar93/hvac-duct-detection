FROM python:3.11-slim

WORKDIR /app

# System dependencies: Tesseract OCR + Poppler (PDF conversion) + OpenCV deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install PyTorch CPU-only (EasyOCR dependency - skip 2GB of CUDA libs)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
# Install PaddlePaddle CPU (PaddleOCR dependency)
RUN pip install --no-cache-dir paddlepaddle paddleocr
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/

RUN mkdir -p uploads outputs tmp weights

# PaddleOCR tiling configuration (adjust based on available RAM)
ENV PADDLE_WORKERS=2
ENV PADDLE_TILE_W=1000
ENV PADDLE_TILE_H=600
ENV PADDLE_OVERLAP=150

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "120"]
