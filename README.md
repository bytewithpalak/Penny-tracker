# Receipt OCR and Information Extraction

Upload a receipt photo, get back the text and the key fields (company, date, total).

```
Image -> OpenCV preprocessing -> text detection (EasyOCR/CRAFT)
      -> recognition (EasyOCR, Tesseract, or fine-tuned TrOCR)
      -> rule-based field extraction -> JSON -> web UI
```

**Live demo:** _add your link here_

![demo](docs/demo.gif)  <!-- record a short GIF and put it in docs/ -->

## Expense tracker

The home page (`/`) is a pink calendar expense tracker built on the OCR service: scan or type an expense, it lands on its date,
with email + password accounts (passwords hashed with scrypt, 14-day session cookie), monthly totals, budget and saved amount,
category breakdown, 6-month trend, search, CSV export, and separate Personal and Business profiles (SQLite, `data/tracker.db`).
The plain receipt reader is at `/reader`.

## Features

- OpenCV cleanup: resize, denoise, deskew, optional binarization
- Three interchangeable OCR engines: Tesseract (baseline), EasyOCR (pre-trained), TrOCR (fine-tuned by you on SROIE)
- Regex-based extraction of company, date and total
- FastAPI backend with a single-file web UI (upload, text boxes drawn on the cleaned image, copy JSON, download CSV)
- Evaluation script reporting CER, WER and field accuracy per engine
- Dockerfile for deployment

## Project structure

```
app/            FastAPI app, preprocessing, OCR engines, field extraction
frontend/       index.html (served by FastAPI)
training/       prepare_data.py (crop lines from SROIE), finetune_trocr.py
evaluation/     evaluate.py (CER / WER / field accuracy)
tests/          unit tests for field extraction
data/           datasets (not committed)
models/         trained models (not committed)
```

## Run locally

Requirements: Python 3.10+ and Tesseract installed on your machine
(`sudo apt install tesseract-ocr`, `brew install tesseract`, or the UB Mannheim installer on Windows).

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000 for the UI and http://127.0.0.1:8000/docs for the API docs.
The first request downloads the EasyOCR models, so it is slow once.

On Windows, if Tesseract is not found, set `pytesseract.pytesseract.tesseract_cmd` in `app/ocr.py`
to your install path.

Run the tests (no heavy dependencies needed):

```bash
python tests/test_extract.py
```

## Train your own recognition model

1. Download the SROIE dataset and arrange it as `data/sroie/{train,test}/{img,box,entities}`.
2. Crop text lines and make train/val/test CSVs:
   ```bash
   python training/prepare_data.py
   ```
3. Fine-tune TrOCR on a GPU (Google Colab or Kaggle work fine):
   ```bash
   pip install -r requirements-train.txt
   python training/finetune_trocr.py --epochs 5
   ```
   Use `--limit 200 --epochs 1` first as a smoke test. The model is saved to `models/trocr-receipt`
   and zero-shot vs fine-tuned CER is written to `training/metrics.json`.
4. Restart the app. The **TrOCR (fine-tuned)** reader appears in the dropdown when `models/trocr-receipt`
   exists. To load from the Hugging Face Hub instead, set `TROCR_MODEL=your-name/trocr-receipt`.

## Evaluate

```bash
python evaluation/evaluate.py --engines tesseract easyocr trocr --limit 100
```

Paste the printed table below. Use your own measured numbers.

| Engine | CER | WER | Company | Date | Total | Sec/receipt |
|---|---|---|---|---|---|---|
| Tesseract | | | | | | |
| EasyOCR | | | | | | |
| TrOCR (fine-tuned) | | | | | | |

### Error analysis

_Write 3 to 5 bullets: which receipts fail (faded ink, folds, tiny fonts, rotated photos) and what you tried._

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `OCR_LANGS` | `en` | EasyOCR languages, comma separated (for example `hi,en`) |
| `TROCR_MODEL` | `models/trocr-receipt` | Local folder or Hugging Face Hub id of the fine-tuned model |
| `WARMUP` | `1` | Load EasyOCR when the server starts (`0` to skip) |

## Docker

```bash
docker build -t receipt-ocr .
docker run -p 7860:7860 receipt-ocr
```

Open http://localhost:7860. To include your fine-tuned model, push it to the Hugging Face Hub and run with
`-e TROCR_MODEL=your-name/trocr-receipt`, or copy the `models/trocr-receipt` folder into the image.

## Deploy

**Hugging Face Spaces (free):** create a Space with the Docker SDK, push this repo, and add this block
at the very top of `README.md` in the Space repo:

```
---
title: Receipt OCR
sdk: docker
app_port: 7860
---
```

**Render / Railway:** connect the GitHub repo, choose Docker, and expose port 7860. Free tiers sleep when idle,
so the first request after a pause is slow.

## Limitations

- Field extraction is rule-based and tuned for English receipts with dot decimals.
- Photos with heavy glare, deep folds or extreme angles reduce accuracy.
- Text CER/WER depends on line order, so treat it as approximate; field accuracy is the stricter metric.

## Future work

- Train a small NER model for field extraction and compare it with the rules
- Hindi and Punjabi receipts with `OCR_LANGS=hi,en`
- Confidence-based review flag for uncertain fields
- Batch upload and export
