"""Receipt date parsing. Ambiguous numeric dates are read day-first (DD/MM/YYYY, Indian format).

Never falls back to today's date: unreadable or impossible dates return None.
"""
import difflib
import re
from datetime import date

MONTH_NAMES = ['january', 'february', 'march', 'april', 'may', 'june', 'july',
               'august', 'september', 'october', 'november', 'december']
MONTHS = {n: i + 1 for i, n in enumerate(MONTH_NAMES)}
MONTHS.update({n[:3]: i + 1 for i, n in enumerate(MONTH_NAMES)})
MONTHS['sept'] = 9
MIN_YEAR, MAX_YEAR = 2000, 2100          # same range the yearly analytics accept

_SEP = r'[\s\-/.,]*'
_ORD = r'(?:st|nd|rd|th)?'

# Patterns used to FIND date-like text inside an OCR line (each is validated afterwards).
FIND_PATTERNS = [
    r'(?<!\d)\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}(?!\d)',                        # 2025-04-09
    r'(?<!\d)\d{1,2}[/\-]\d{1,2}[/\-](?:\d{4}|\d{2})(?!\d)',                 # 09/04/2025, 9-4-25
    r'(?<!\d)\d{1,2}\.\d{1,2}\.\d{4}(?!\d)',                                 # 09.04.2025
    r'(?<![\dA-Za-z])\d{1,2}' + _ORD + _SEP + r'[A-Za-z]{3,9}\.?' + _SEP + r'(?:\d{4}|\d{2})(?!\d)',  # 9 April 2025, 09-Apr-25
    r'(?<![A-Za-z])[A-Za-z]{3,9}\.?\s*\d{1,2}' + _ORD + r'[\s,/\-.]+\d{4}(?!\d)',                # April 9, 2025 / April 9/2025
]

_NUM_YMD = re.compile(r'^(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})$')
_NUM_DMY = re.compile(r'^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4}|\d{2})$')
_D_MON_Y = re.compile(r'^(\d{1,2})' + _ORD + _SEP + r'([A-Za-z]{3,9})\.?' + _SEP + r'(\d{4}|\d{2})$', re.I)
_MON_D_Y = re.compile(r'^([A-Za-z]{3,9})\.?\s*(\d{1,2})' + _ORD + r'[\s,/\-.]+(\d{4})$', re.I)


def _month(word):
    w = word.lower()
    if w in MONTHS:
        return MONTHS[w]
    close = difflib.get_close_matches(w, MONTH_NAMES, n=1, cutoff=0.8)   # tolerate small OCR slips
    return MONTHS[close[0]] if close else None


def _year(y):
    y = int(y)
    return y + 2000 if y < 100 else y


def _make(y, m, d):
    if m is None or not (MIN_YEAR <= y <= MAX_YEAR):
        return None
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


def parse_date(text):
    """Return 'YYYY-MM-DD' for a date printed on a receipt, or None if it is missing or not a real date."""
    t = re.sub(r'\s+', ' ', text or '').strip()
    if not t:
        return None
    m = _NUM_YMD.match(t)
    if m:
        return _make(int(m[1]), int(m[2]), int(m[3]))
    m = _NUM_DMY.match(t)
    if m:
        a, b, y = int(m[1]), int(m[2]), _year(m[3])
        # Day-first is the rule. Month-first is used only when day-first is impossible
        # (e.g. 04/13/2025 can only mean 13 April), never to override a valid DD/MM date.
        return _make(y, b, a) or _make(y, a, b)
    m = _D_MON_Y.match(t)
    if m:
        return _make(_year(m[3]), _month(m[2]), int(m[1]))
    m = _MON_D_Y.match(t)
    if m:
        return _make(_year(m[3]), _month(m[1]), int(m[2]))
    return None


_HINT = re.compile(r'\b(date|dated|dt)\b', re.I)
_AVOID = re.compile(r'\b(due|expiry|expires|exp|valid|best before|mfg|manufactured|batch)\b', re.I)


def _candidates(lines):
    """Yield (rank, line_no, pos, raw) for every date-shaped string in the lines."""
    for i, line in enumerate(lines):
        rank = 0 if _HINT.search(line) else (2 if _AVOID.search(line) else 1)
        for pattern in FIND_PATTERNS:
            for m in re.finditer(pattern, line):
                yield rank, i, m.start(), m.group(0).strip(' ,.-/')


def find_receipt_date(lines):
    """Raw text of the best VALID date on the receipt (a line labelled 'Date' wins), else None."""
    valid = [c for c in _candidates(lines) if parse_date(c[3])]
    return min(valid)[3] if valid else None


def find_invalid_date(lines):
    """Raw text of a date-shaped string that is not a real date (e.g. 31/02/2025), else None."""
    for _, _, _, raw in _candidates(lines):
        if parse_date(raw) is None and re.search(r'\d', raw) and re.search(r'[/\-.]\d|\d[/\-.]', raw):
            return raw
    return None
