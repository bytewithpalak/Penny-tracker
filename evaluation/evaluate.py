"""Compare OCR engines on the SROIE test set: CER, WER and field accuracy.

  python evaluation/evaluate.py --engines tesseract easyocr trocr --limit 50

Expected layout: <split-dir>/img/*.jpg, <split-dir>/box/*.txt, <split-dir>/entities/*.txt
(entities files are JSON with company, date, address, total).
Line order can differ between prediction and ground truth, so text CER/WER is approximate;
field accuracy is the stricter, more meaningful number.
"""
import argparse
import json
import os
import re
import sys
import time

import cv2
import jiwer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.extract import extract_fields  # noqa: E402
from app.ocr import run_ocr  # noqa: E402
from app.preprocess import preprocess  # noqa: E402

FIELDS = ('company', 'date', 'total')


def norm_text(s):
    return re.sub(r'\s+', ' ', s.lower()).strip()


def norm_field(name, value):
    value = (value or '').lower()
    if name == 'total':
        return re.sub(r'[^0-9.]', '', value)
    return re.sub(r'[^a-z0-9]', '', value)


def read_truth_lines(box_path):
    out = []
    with open(box_path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split(',', 8)
            if len(parts) == 9 and parts[8].strip():
                out.append(parts[8].strip())
    return out


def read_entities(path):
    try:
        with open(path, encoding='utf-8', errors='ignore') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split-dir', default='data/sroie/test')
    ap.add_argument('--engines', nargs='+', default=['tesseract', 'easyocr'])
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--no-preprocess', action='store_true')
    ap.add_argument('--out', default='evaluation/results.json')
    args = ap.parse_args()

    img_dir = os.path.join(args.split_dir, 'img')
    names = sorted(n for n in os.listdir(img_dir) if n.lower().endswith(('.jpg', '.jpeg', '.png')))
    if args.limit:
        names = names[:args.limit]

    results = {}
    for engine in args.engines:
        refs, hyps = [], []
        hits = {f: 0 for f in FIELDS}
        counted = {f: 0 for f in FIELDS}
        start = time.time()
        for n, name in enumerate(names, 1):
            stem = os.path.splitext(name)[0]
            img = cv2.imread(os.path.join(img_dir, name))
            if img is None:
                continue
            clean = img if args.no_preprocess else preprocess(img)
            lines = [l['text'] for l in run_ocr(clean, engine)]
            truth_lines = read_truth_lines(os.path.join(args.split_dir, 'box', stem + '.txt'))
            refs.append(norm_text(' '.join(truth_lines)))
            hyps.append(norm_text(' '.join(lines)))

            truth = read_entities(os.path.join(args.split_dir, 'entities', stem + '.txt'))
            pred = extract_fields(lines)
            for f in FIELDS:
                if truth.get(f):
                    counted[f] += 1
                    hits[f] += norm_field(f, pred.get(f)) == norm_field(f, truth[f])
            if n % 10 == 0:
                print(f'[{engine}] {n}/{len(names)}', flush=True)

        pairs = [(r, h) for r, h in zip(refs, hyps) if r]
        results[engine] = {
            'receipts': len(pairs),
            'cer': jiwer.cer([p[0] for p in pairs], [p[1] for p in pairs]),
            'wer': jiwer.wer([p[0] for p in pairs], [p[1] for p in pairs]),
            **{f'{f}_acc': hits[f] / counted[f] if counted[f] else None for f in FIELDS},
            'seconds_per_receipt': (time.time() - start) / max(len(pairs), 1),
        }

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)

    def pct(x):
        return 'n/a' if x is None else f'{x * 100:.1f}%'
    print('\n| Engine | CER | WER | Company | Date | Total | Sec/receipt |')
    print('|---|---|---|---|---|---|---|')
    for engine, r in results.items():
        print(f"| {engine} | {pct(r['cer'])} | {pct(r['wer'])} | {pct(r['company_acc'])} | "
              f"{pct(r['date_acc'])} | {pct(r['total_acc'])} | {r['seconds_per_receipt']:.1f} |")


if __name__ == '__main__':
    main()
