"""OCR engines behind one function: run_ocr(image, engine).

engines:
  tesseract  - classic baseline (no boxes)
  easyocr    - CRAFT detection + CRNN recognition (pre-trained)
  trocr      - EasyOCR finds the boxes, your fine-tuned TrOCR reads them
"""
import os

import cv2

LANGS = [l.strip() for l in os.getenv('OCR_LANGS', 'en').split(',') if l.strip()]
TROCR_PATH = os.getenv('TROCR_MODEL', 'models/trocr-receipt')

_easy = None
_trocr = None


def get_easyocr():
    global _easy
    if _easy is None:
        import easyocr
        _easy = easyocr.Reader(LANGS, gpu=False)
    return _easy


def get_trocr():
    global _trocr
    if _trocr is None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        proc = TrOCRProcessor.from_pretrained(TROCR_PATH)
        model = VisionEncoderDecoderModel.from_pretrained(TROCR_PATH).eval()
        _trocr = (proc, model)
    return _trocr


def available_engines():
    """trocr is offered (first, as the most accurate) when TROCR_MODEL is a local folder holding a model,
    or a Hugging Face Hub id like 'your-name/trocr-receipt'."""
    engines = ['easyocr', 'tesseract']
    is_local = os.path.isfile(os.path.join(TROCR_PATH, 'config.json'))   # an empty folder does not count
    looks_like_hub_id = '/' in TROCR_PATH and not TROCR_PATH.startswith(('models/', './', '/'))
    if is_local or looks_like_hub_id:
        engines.insert(0, 'trocr')
    return engines


def default_engine():
    """TrOCR had the best accuracy in our evaluation, so it is used whenever it is available; EasyOCR otherwise."""
    return available_engines()[0]


def _bbox(box):
    xs = [float(p[0]) for p in box]
    ys = [float(p[1]) for p in box]
    return [min(xs), min(ys), max(xs), max(ys)]


def _sort_lines(lines):
    """Top-to-bottom, then left-to-right (rows are grouped in 15px bands)."""
    return sorted(lines, key=lambda l: (round(l['box'][1] / 15), l['box'][0]) if l['box'] else (0, 0))


def _easyocr(img):
    results = get_easyocr().readtext(img)
    return [{'text': t.strip(), 'conf': float(c), 'box': _bbox(b)}
            for b, t, c in results if t.strip()]


def _trocr_engine(img):
    import torch
    from PIL import Image
    proc, model = get_trocr()
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    boxes, crops = [], []
    for b, _, _ in get_easyocr().readtext(img):
        x1, y1, x2, y2 = _bbox(b)
        x1, y1, x2, y2 = max(int(x1) - 3, 0), max(int(y1) - 3, 0), min(int(x2) + 3, w), min(int(y2) + 3, h)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        boxes.append([x1, y1, x2, y2])
        crops.append(Image.fromarray(rgb[y1:y2, x1:x2]))
    if not crops:
        return []
    texts = []
    for i in range(0, len(crops), 16):          # small batches keep memory low
        pix = proc(images=crops[i:i + 16], return_tensors='pt').pixel_values
        with torch.no_grad():
            ids = model.generate(pix, max_length=64)
        texts += proc.batch_decode(ids, skip_special_tokens=True)
    return [{'text': t.strip(), 'conf': None, 'box': b} for b, t in zip(boxes, texts) if t.strip()]


def _tesseract(img):
    import pytesseract
    text = pytesseract.image_to_string(img, config='--psm 6')
    return [{'text': l.strip(), 'conf': None, 'box': None} for l in text.splitlines() if l.strip()]


def group_rows(lines):
    """Merge boxes that sit on the same printed row into one text line.

    Gaps between boxes become spaces, so columns (item / qty / price) stay roughly aligned
    when shown in a monospace font. Engines without boxes (Tesseract) are returned as-is.
    """
    boxed = [l for l in lines if l.get('box') and l['text']]
    if not boxed:
        return [l['text'] for l in lines]

    items = []
    for l in boxed:
        x1, y1, x2, y2 = l['box']
        items.append({'t': l['text'], 'x1': x1, 'cy': (y1 + y2) / 2, 'h': max(y2 - y1, 1), 'w': max(x2 - x1, 1)})

    widths = sorted(i['w'] / max(len(i['t']), 1) for i in items)
    char_w = max(widths[len(widths) // 2], 3.0)          # typical width of one character
    left = min(i['x1'] for i in items)

    # tall, narrow boxes with several letters are rotated side notes: keep them out of the rows
    side = [i for i in items if i['h'] > 2 * i['w'] and len(i['t']) >= 3]
    main = sorted((i for i in items if i not in side), key=lambda i: i['cy'])

    rows = []
    for it in main:
        if rows:
            ref = sum(i['cy'] for i in rows[-1]) / len(rows[-1])
            avg_h = sum(i['h'] for i in rows[-1]) / len(rows[-1])
            if abs(it['cy'] - ref) <= 0.5 * min(avg_h, it['h']):
                rows[-1].append(it)
                continue
        rows.append([it])

    out = []
    for row in rows:
        text = ''
        for it in sorted(row, key=lambda i: i['x1']):
            col = min(int(round((it['x1'] - left) / char_w)), 110)
            text += (' ' * max(col - len(text), 1) if text else ' ' * col) + it['t']
        out.append(text.rstrip())
    out += [i['t'] for i in sorted(side, key=lambda i: i['x1'])]
    return out


def run_ocr(img, engine='easyocr'):
    """img: numpy image (gray or BGR). Returns a list of {text, conf, box}."""
    if engine == 'easyocr':
        lines = _easyocr(img)
    elif engine == 'trocr':
        lines = _trocr_engine(img)
    elif engine == 'tesseract':
        lines = _tesseract(img)
    else:
        raise ValueError(f'Unknown engine: {engine}')
    return _sort_lines(lines)
