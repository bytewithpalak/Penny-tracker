"""Fine-tune TrOCR on SROIE line crops. Run on a GPU (Google Colab / Kaggle).

  python training/prepare_data.py
  python training/finetune_trocr.py --epochs 5

Reports zero-shot CER/WER (before training) and fine-tuned CER/WER on the held-out test lines,
and saves the model to models/trocr-receipt plus training/metrics.json.
"""
import argparse
import csv
import inspect
import json

import jiwer
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, TrOCRProcessor,
                          VisionEncoderDecoderModel, default_data_collator)

MAX_LEN = 64


class LineDS(Dataset):
    def __init__(self, csv_path, processor, limit=None):
        with open(csv_path, encoding='utf-8') as f:
            self.rows = list(csv.reader(f))
        if limit:
            self.rows = self.rows[:limit]
        self.processor = processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, text = self.rows[i]
        image = Image.open(path).convert('RGB')
        pixel_values = self.processor(images=image, return_tensors='pt').pixel_values[0]
        ids = self.processor.tokenizer(text, padding='max_length', max_length=MAX_LEN,
                                       truncation=True).input_ids
        pad = self.processor.tokenizer.pad_token_id
        labels = [t if t != pad else -100 for t in ids]
        return {'pixel_values': pixel_values, 'labels': torch.tensor(labels)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='microsoft/trocr-small-printed')
    ap.add_argument('--data', default='data/lines')
    ap.add_argument('--out', default='models/trocr-receipt')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--limit', type=int, default=None, help='use only N rows per split (smoke test)')
    args = ap.parse_args()

    processor = TrOCRProcessor.from_pretrained(args.base)
    model = VisionEncoderDecoderModel.from_pretrained(args.base)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    train_ds = LineDS(f'{args.data}/train.csv', processor, args.limit)
    val_ds = LineDS(f'{args.data}/val.csv', processor, args.limit)
    test_ds = LineDS(f'{args.data}/test.csv', processor, args.limit)

    def compute_metrics(pred):
        pad = processor.tokenizer.pad_token_id
        ids = np.where(pred.predictions != -100, pred.predictions, pad)
        labels = np.where(pred.label_ids != -100, pred.label_ids, pad)
        hyp = processor.batch_decode(ids, skip_special_tokens=True)
        ref = processor.batch_decode(labels, skip_special_tokens=True)
        pairs = [(r, h) for r, h in zip(ref, hyp) if r.strip()]
        ref, hyp = [p[0] for p in pairs], [p[1] for p in pairs]
        return {'cer': jiwer.cer(ref, hyp), 'wer': jiwer.wer(ref, hyp)}

    # the argument was renamed across transformers versions
    strategy_key = ('eval_strategy'
                    if 'eval_strategy' in inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
                    else 'evaluation_strategy')
    kwargs = {strategy_key: 'epoch'}
    targs = Seq2SeqTrainingArguments(
        output_dir=args.out, per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch, learning_rate=args.lr,
        num_train_epochs=args.epochs, fp16=torch.cuda.is_available(),
        save_strategy='epoch', predict_with_generate=True, generation_max_length=MAX_LEN,
        logging_steps=50, load_best_model_at_end=True, metric_for_best_model='cer',
        greater_is_better=False, save_total_limit=2, report_to='none', **kwargs)

    trainer = Seq2SeqTrainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                             data_collator=default_data_collator, compute_metrics=compute_metrics)

    before = trainer.evaluate(test_ds, metric_key_prefix='test')
    print('Zero-shot on test lines:', before)
    trainer.train()
    after = trainer.evaluate(test_ds, metric_key_prefix='test')
    print('Fine-tuned on test lines:', after)

    trainer.save_model(args.out)
    processor.save_pretrained(args.out)
    with open('training/metrics.json', 'w') as f:
        json.dump({'before': before, 'after': after}, f, indent=2)


if __name__ == '__main__':
    main()
