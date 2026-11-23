"""Crop every labelled text line out of the SROIE receipts.

Expected input layout (rename folders if yours differ):
  data/sroie/train/img/*.jpg      data/sroie/train/box/*.txt
  data/sroie/test/img/*.jpg       data/sroie/test/box/*.txt
Each box line: x1,y1,x2,y2,x3,y3,x4,y4,text

Output:
  data/lines/{train,val,test}/*.png  and  data/lines/{train,val,test}.csv  (path,text)
A validation set is carved out of the training receipts (split by receipt, not by line)
so the test set stays untouched until the final evaluation.
"""
import argparse
import csv
import os
import random

import cv2


def read_boxes(path):
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split(',', 8)
            if len(parts) < 9 or not parts[8].strip():
                continue
            try:
                pts = list(map(int, parts[:8]))
            except ValueError:
                continue
            yield pts, parts[8].strip()


def crop_split(stems, img_dir, box_dir, out_dir, name):
    os.makedirs(os.path.join(out_dir, name), exist_ok=True)
    rows = []
    for stem, img_name in stems:
        img = cv2.imread(os.path.join(img_dir, img_name))
        box_file = os.path.join(box_dir, stem + '.txt')
        if img is None or not os.path.exists(box_file):
            continue
        h, w = img.shape[:2]
        for i, (pts, text) in enumerate(read_boxes(box_file)):
            xs, ys = pts[0::2], pts[1::2]
            x1, x2 = max(min(xs), 0), min(max(xs), w)
            y1, y2 = max(min(ys), 0), min(max(ys), h)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            path = os.path.join(out_dir, name, f'{stem}_{i}.png')
            cv2.imwrite(path, img[y1:y2, x1:x2])
            rows.append((path, text))
    with open(os.path.join(out_dir, f'{name}.csv'), 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(rows)
    print(f'{name}: {len(rows)} line crops')


def list_images(img_dir):
    return [(os.path.splitext(n)[0], n) for n in sorted(os.listdir(img_dir))
            if n.lower().endswith(('.jpg', '.jpeg', '.png'))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/sroie')
    ap.add_argument('--out', default='data/lines')
    ap.add_argument('--val-fraction', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    train = list_images(os.path.join(args.root, 'train', 'img'))
    random.Random(args.seed).shuffle(train)
    n_val = max(1, int(len(train) * args.val_fraction))
    val, train = train[:n_val], train[n_val:]
    test = list_images(os.path.join(args.root, 'test', 'img'))

    for name, stems, split in (('train', train, 'train'), ('val', val, 'train'), ('test', test, 'test')):
        crop_split(stems, os.path.join(args.root, split, 'img'),
                   os.path.join(args.root, split, 'box'), args.out, name)


if __name__ == '__main__':
    main()
