FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces and many hosts run as a non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
# CPU-only PyTorch keeps the image much smaller
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Download EasyOCR weights at build time so the first request is fast
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

COPY --chown=user . .

EXPOSE 7860
CMD uvicorn app.main:app --host 0.0.0.0 --port 7860
