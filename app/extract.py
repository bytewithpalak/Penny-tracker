"""Rule-based field extraction from OCR text lines."""
import re

from app.dates import find_receipt_date


# An amount with two decimals. Not part of a longer number such as a time (10.38.03) or an ID.
# Also accepts a comma as the decimal mark ("55,00"), which OCR often produces for a dot.
MONEY = r'(?<![\d.])(\d{1,3}(?:,\d{3})+\.\d{2}|\d+\.\d{2}|\d+,\d{2})(?![\d,]|\.\d)'


def _amounts(line):
    """Amounts on a line as printed, except a comma decimal ("55,00") becomes a dot ("55.00")."""
    return [n.replace(',', '.') if re.fullmatch(r'\d+,\d{2}', n) else n for n in re.findall(MONEY, line)]
TOTAL_SKIP = ('subtotal', 'sub total', 'sub-total', 'total qty', 'total item', 'total discount', 'total tax')


def find_date(lines):
    """Raw text of the receipt's date (first valid one; a line labelled 'Date' wins), or None."""
    return find_receipt_date(lines)


def find_total(lines):
    best = None
    for i, line in enumerate(lines):
        low = line.lower()
        if 'total' not in low or any(s in low for s in TOTAL_SKIP):
            continue
        nums = _amounts(line)
        if not nums and i + 1 < len(lines):          # amount printed on the next line
            nums = _amounts(lines[i + 1])
        if nums:
            best = nums[-1]                           # last match wins (rounded totals come last)
    if best is None:                                  # fallback: largest amount on the receipt
        vals = [float(n.replace(',', '')) for l in lines for n in _amounts(l)]
        best = f'{max(vals):.2f}' if vals else None
    return best


GENERIC_HEADINGS = {'receipt', 'cash receipt', 'sales receipt', 'tax invoice', 'invoice', 'tax receipt', 'cash memo',
                    'bill', 'cash bill', 'retail invoice', 'order', 'customer copy', 'duplicate', 'original', 'welcome',
                    'thank you', 'gst invoice', 'payment receipt'}


def find_company(lines):
    for line in lines[:6]:                            # the company name is near the top
        text = re.sub(r'\s+', ' ', line).strip()
        if re.sub(r'[^a-z ]', '', text.lower()).strip() in GENERIC_HEADINGS:
            continue                                  # a heading such as "RECEIPT", not the shop name
        if len(text) > 3 and not re.search(r'\d{3,}', text):
            return text
    return None


def extract_fields(lines):
    return {'company': find_company(lines),
            'date': find_date(lines),
            'total': find_total(lines)}
