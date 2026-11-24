import base64
import os
import threading
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.dates import find_invalid_date
from app.extract import extract_fields, find_company
from app.ocr import available_engines, default_engine, get_easyocr, get_trocr, group_rows, run_ocr
from app.preprocess import preprocess
from app.tracker import current_user, router as tracker_router, smart_category, to_iso

MAX_BYTES = 10 * 1024 * 1024
OCR_LOCK = threading.Lock()   # one OCR job at a time: parallel jobs only slow each other down
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')


@asynccontextmanager
async def lifespan(app):
    if os.getenv('WARMUP', '1') == '1':          # load the model once, not on the first request
        try:
            get_easyocr()                         # also TrOCR's text detector
            if default_engine() == 'trocr':
                get_trocr()
        except Exception as exc:                  # app still starts; the error shows on first use
            print(f'OCR warm-up failed: {exc}')
    yield


app = FastAPI(title='Receipt OCR', lifespan=lifespan)
app.include_router(tracker_router)


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.get('/api/engines')
def engines():
    return {'engines': available_engines()}


@app.post('/api/extract')
def extract(file: UploadFile = File(...), engine: str | None = Query(None),
            binarize: bool = Query(False), preview: bool = Query(True), _uid: int = Depends(current_user)):
    engine = engine or available_engines()[0]     # TrOCR when it is available, EasyOCR otherwise
    if engine not in available_engines():
        raise HTTPException(400, f'Engine "{engine}" is not available. Choose one of: {available_engines()}')
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(400, 'Upload an image file (JPG, PNG or WebP).')
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, 'Image is larger than 10 MB. Upload a smaller photo.')
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, 'The file could not be read as an image.')

    clean = preprocess(img, binarize=binarize)
    try:
        with OCR_LOCK:
            lines = run_ocr(clean, engine)
    except OSError as exc:                        # e.g. fine-tuned model folder missing
        raise HTTPException(503, f'Engine "{engine}" could not load: {exc}')

    rows = group_rows(lines)          # one string per printed row
    processed = None
    if preview:                                   # the /reader page shows the cleaned image; the tracker does not need it
        ok, buf = cv2.imencode('.jpg', clean, [cv2.IMWRITE_JPEG_QUALITY, 80])
        processed = 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode() if ok else None
    fields = extract_fields([r.strip() for r in rows])
    # The shop name is read only from text the OCR is fairly sure about, so background clutter
    # (keyboard keys, fingers, table edges) read as low-confidence junk like "JUCU" is not chosen.
    sure = [l for l in lines if l.get('conf') is None or l['conf'] >= 0.5]
    if sure:
        fields['company'] = find_company([r.strip() for r in group_rows(sure)]) or fields['company']
    date_iso = to_iso(fields['date'])
    bad_date = None if date_iso else find_invalid_date([r.strip() for r in rows])
    return {
        'engine': engine,
        'lines': lines,
        'text': '\n'.join(rows),
        'fields': fields,
        'date_iso': date_iso,                      # None when no valid date was found: the user must enter it
        'date_invalid': bad_date,                  # date-like text that is not a real date, e.g. 31/02/2025
        'category': smart_category(_uid, ((fields['company'] or '') + ' ') * 3 + ' '.join(rows)),   # shop name counts 3x
        'image_size': [int(clean.shape[1]), int(clean.shape[0])],
        'processed_image': processed,
    }


app.mount('/static', StaticFiles(directory=FRONTEND_DIR), name='static')


@app.get('/')
def landing_page():
    return FileResponse(os.path.join(FRONTEND_DIR, 'landing.html'))


@app.get('/app')
def tracker_page():
    return FileResponse(os.path.join(FRONTEND_DIR, 'tracker.html'))


@app.get('/reader')
def reader_page():
    return FileResponse(os.path.join(FRONTEND_DIR, 'index.html'))
