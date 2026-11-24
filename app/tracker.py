"""Expense tracker API (SQLite): expenses per profile, monthly summary, budget, category guess."""
import calendar
import hashlib
import hmac
import math
import random
import secrets
import smtplib
import time
import os
import re
import sqlite3
from contextlib import closing
from email.message import EmailMessage
from datetime import date, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from app.dates import parse_date

DB_PATH = os.getenv('TRACKER_DB', 'data/tracker.db')
PROFILE = r'^(personal|business)$'
MONTH = r'^\d{4}-(0[1-9]|1[0-2])$'
CATEGORIES = ['Food & drink', 'Groceries', 'Transport', 'Health', 'Shopping', 'Bills', 'Office', 'Other']
WORDS = {
    'Food & drink': 'cafe coffee restaurant pizza burger bakery kitchen tea grill food starbucks sandwich juice toast salad dine',
    'Groceries': 'mart supermarket grocery fresh market bazaar vegetables dairy',
    'Transport': 'uber ola fuel petrol diesel taxi metro bus parking toll railway cab roadways passenger depot fare irctc rapido travels transport',
    'Health': 'pharmacy medical chemist clinic hospital dental medicine',
    'Shopping': 'mall fashion clothing apparel electronics amazon flipkart shoes books gift macbook iphone ipad laptop headphones',
    'Bills': 'electricity water internet broadband mobile recharge rent insurance',
    'Office': 'stationery office printing courier software hosting supplies',
}
SCHEMA = """
CREATE TABLE IF NOT EXISTS expenses(id INTEGER PRIMARY KEY AUTOINCREMENT, profile TEXT NOT NULL,
  date TEXT NOT NULL, merchant TEXT NOT NULL, amount REAL NOT NULL, category TEXT NOT NULL DEFAULT 'Other',
  note TEXT DEFAULT '', ocr_text TEXT DEFAULT '', created TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS user_budgets(user_id INTEGER NOT NULL, profile TEXT NOT NULL, amount REAL NOT NULL DEFAULT 0, PRIMARY KEY(user_id, profile));
CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT NOT NULL UNIQUE, name TEXT NOT NULL,
  salt BLOB NOT NULL, pw BLOB NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS goals(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, name TEXT NOT NULL,
  target REAL NOT NULL, saved REAL NOT NULL DEFAULT 0, created TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS email_codes(email TEXT PRIMARY KEY, code_hash TEXT NOT NULL, expires REAL NOT NULL, tries INTEGER NOT NULL DEFAULT 0, sent REAL NOT NULL);
CREATE TABLE IF NOT EXISTS sessions(token TEXT PRIMARY KEY, user_id INTEGER NOT NULL, expires REAL NOT NULL);
"""
router = APIRouter(prefix='/api')


def db():
    os.makedirs(os.path.dirname(DB_PATH) or '.', exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    try:                                           # older databases had no user_id column
        con.execute('ALTER TABLE expenses ADD COLUMN user_id INTEGER')
    except sqlite3.OperationalError:
        pass
    try:                                           # accounts made before verification existed stay usable
        con.execute('ALTER TABLE users ADD COLUMN verified INTEGER NOT NULL DEFAULT 0')
        con.execute('UPDATE users SET verified=1')
        con.commit()
    except sqlite3.OperationalError:
        pass
    con.execute('CREATE INDEX IF NOT EXISTS ix_user ON expenses(user_id, profile, date)')
    return con


def guess_category(text):
    """Keyword scoring (a trained text classifier can replace this later)."""
    text = (text or '').lower()
    scores = {c: sum(bool(re.search(r'\b' + w, text)) for w in ws.split()) for c, ws in WORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] else 'Other'


def to_iso(text):
    """Turn a date printed on a receipt into YYYY-MM-DD (day-first), or None. Never returns today's date."""
    return parse_date(text)


class ExpenseIn(BaseModel):
    profile: str = Field(pattern=PROFILE)
    date: str = Field(pattern=r'^\d{4}-\d{2}-\d{2}$')
    merchant: str = Field(min_length=1, max_length=80)
    amount: float = Field(gt=0, lt=100000000)
    category: str = 'Other'
    note: str = Field('', max_length=200)
    ocr_text: str = Field('', max_length=5000)


class BudgetIn(BaseModel):
    profile: str = Field(pattern=PROFILE)
    amount: float = Field(ge=0, lt=1000000000)


EMAIL = r'^[^@\s]+@[^@\s]+\.[^@\s]{2,}$'
SESSION_DAYS = 14


class RegisterIn(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    email: str = Field(pattern=EMAIL, max_length=254)
    password: str = Field(min_length=8, max_length=128)


class LoginIn(BaseModel):
    email: str = Field(pattern=EMAIL, max_length=254)
    password: str = Field(min_length=1, max_length=128)


def _hash_pw(password, salt):
    return hashlib.scrypt(password.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _start_session(con, user_id, response):
    token = secrets.token_urlsafe(32)
    con.execute('INSERT INTO sessions(token,user_id,expires) VALUES(?,?,?)',
                (_sha(token), user_id, time.time() + SESSION_DAYS * 86400))
    response.set_cookie('pp_session', token, httponly=True, samesite='lax', max_age=SESSION_DAYS * 86400)


def current_user(request: Request):
    token = request.cookies.get('pp_session')
    if token:
        with closing(db()) as con:
            r = con.execute('SELECT user_id FROM sessions WHERE token=? AND expires>?', (_sha(token), time.time())).fetchone()
        if r:
            return r[0]
    raise HTTPException(401, 'Please sign in.')


class VerifyIn(BaseModel):
    email: str = Field(pattern=EMAIL, max_length=254)
    code: str = Field(pattern=r'^\d{6}$')


class ResendIn(BaseModel):
    email: str = Field(pattern=EMAIL, max_length=254)


def _send_code(to, code):
    """Email the code. Without SMTP settings the code is printed in the server window (development only)."""
    host = os.getenv('SMTP_HOST')
    if not host:
        print(f'\n[DEV] Email sending is not configured. Verification code for {to}: {code}\n', flush=True)
        return False
    msg = EmailMessage()
    msg['Subject'] = 'Your BillBuddy verification code'
    msg['From'] = os.getenv('SMTP_FROM') or os.getenv('SMTP_USER', '')
    msg['To'] = to
    msg.set_content(f'Your verification code is {code}. It expires in 10 minutes.\n\n'
                    'If you did not create an account, you can ignore this email.')
    try:
        with smtplib.SMTP(host, int(os.getenv('SMTP_PORT', '587')), timeout=15) as smtp:
            smtp.starttls()
            smtp.login(os.getenv('SMTP_USER', ''), os.getenv('SMTP_PASSWORD', ''))
            smtp.send_message(msg)
    except Exception as exc:
        raise HTTPException(502, 'Could not send the verification email. Check the SMTP settings on the server.') from exc
    return True


def _issue_code(con, email):
    row = con.execute('SELECT sent FROM email_codes WHERE email=?', (email,)).fetchone()
    if row and time.time() - row['sent'] < 30:
        raise HTTPException(429, 'Please wait a few seconds before asking for another code.')
    code = f'{secrets.randbelow(10 ** 6):06d}'
    con.execute('INSERT INTO email_codes(email,code_hash,expires,tries,sent) VALUES(?,?,?,0,?) '
                'ON CONFLICT(email) DO UPDATE SET code_hash=excluded.code_hash, expires=excluded.expires, tries=0, sent=excluded.sent',
                (email, _sha(code + email), time.time() + 600, time.time()))
    return code


@router.post('/auth/register')
def register(b: RegisterIn):
    email = b.email.strip().lower()
    salt = secrets.token_bytes(16)
    with closing(db()) as con, con:
        u = con.execute('SELECT id, verified FROM users WHERE email=?', (email,)).fetchone()
        if u and u['verified']:
            raise HTTPException(409, 'An account with this email already exists. Try signing in.')
        if u:                                        # unverified sign-up being retried
            con.execute('UPDATE users SET name=?, salt=?, pw=? WHERE id=?', (b.name.strip(), salt, _hash_pw(b.password, salt), u['id']))
        else:
            con.execute('INSERT INTO users(email,name,salt,pw,verified) VALUES(?,?,?,?,0)',
                        (email, b.name.strip(), salt, _hash_pw(b.password, salt)))
        sent = _send_code(email, _issue_code(con, email))
    return {'verify': True, 'email': email, 'sent': sent}


@router.post('/auth/verify')
def verify(b: VerifyIn, response: Response):
    email = b.email.strip().lower()
    with closing(db()) as con, con:
        r = con.execute('SELECT code_hash, expires, tries FROM email_codes WHERE email=?', (email,)).fetchone()
        if not r or r['expires'] < time.time():
            raise HTTPException(400, 'That code has expired. Ask for a new one.')
        if r['tries'] >= 5:
            raise HTTPException(429, 'Too many wrong tries. Ask for a new code.')
        if not hmac.compare_digest(r['code_hash'], _sha(b.code + email)):
            con.execute('UPDATE email_codes SET tries=tries+1 WHERE email=?', (email,))
            con.commit()
            raise HTTPException(400, 'That code is not correct.')
        con.execute('UPDATE users SET verified=1 WHERE email=?', (email,))
        con.execute('DELETE FROM email_codes WHERE email=?', (email,))
        u = con.execute('SELECT id, name FROM users WHERE email=?', (email,)).fetchone()
        _start_session(con, u['id'], response)
    return {'name': u['name'], 'email': email}


@router.post('/auth/resend')
def resend(b: ResendIn):
    email = b.email.strip().lower()
    with closing(db()) as con, con:
        u = con.execute('SELECT verified FROM users WHERE email=?', (email,)).fetchone()
        if u and not u['verified']:                  # same reply either way, so emails cannot be probed
            _send_code(email, _issue_code(con, email))
    return {'ok': True}


@router.post('/auth/login')
def login(b: LoginIn, response: Response):
    email = b.email.strip().lower()
    with closing(db()) as con, con:
        u = con.execute('SELECT id,name,salt,pw,verified FROM users WHERE email=?', (email,)).fetchone()
        salt = u['salt'] if u else b'\0' * 16
        ok = hmac.compare_digest(_hash_pw(b.password, salt), u['pw'] if u else b'\0' * 32) and u is not None
        if not ok:
            raise HTTPException(401, 'Incorrect email or password.')
        if not u['verified']:
            try:
                _send_code(email, _issue_code(con, email))
            except HTTPException as exc:
                if exc.status_code != 429:
                    raise
            con.commit()
            raise HTTPException(403, 'Please verify your email first. We sent you a new code.')
        _start_session(con, u['id'], response)
    return {'name': u['name'], 'email': email}


@router.post('/auth/logout')
def logout(request: Request, response: Response):
    token = request.cookies.get('pp_session')
    if token:
        with closing(db()) as con, con:
            con.execute('DELETE FROM sessions WHERE token=?', (_sha(token),))
    response.delete_cookie('pp_session')
    return {'ok': True}


@router.get('/auth/me')
def me(uid: int = Depends(current_user)):
    with closing(db()) as con:
        u = con.execute('SELECT name,email FROM users WHERE id=?', (uid,)).fetchone()
    return {'name': u['name'], 'email': u['email']}


@router.get('/expenses')
def list_expenses(profile: str = Query(..., pattern=PROFILE), month: str = Query(..., pattern=MONTH),
                  uid: int = Depends(current_user)):
    with closing(db()) as con:
        rows = con.execute('SELECT id, date, merchant, amount, category, note FROM expenses '
                           'WHERE user_id=? AND profile=? AND date LIKE ? ORDER BY date, id',
                           (uid, profile, month + '-%')).fetchall()
    return [dict(r) for r in rows]


@router.get('/search')
def search(profile: str = Query(..., pattern=PROFILE), q: str = Query(..., min_length=2, max_length=60),
           uid: int = Depends(current_user)):
    like = '%' + q.replace('\\', '').replace('%', '').replace('_', '') + '%'
    with closing(db()) as con:
        rows = con.execute('SELECT id, date, merchant, amount, category, note FROM expenses WHERE user_id=? AND profile=? '
                           'AND (merchant LIKE ? OR note LIKE ? OR category LIKE ? OR ocr_text LIKE ?) '
                           'ORDER BY date DESC LIMIT 50', (uid, profile, like, like, like, like)).fetchall()
    return [dict(r) for r in rows]


@router.post('/expenses')
def add_expense(e: ExpenseIn, uid: int = Depends(current_user)):
    try:
        date.fromisoformat(e.date)
    except ValueError:
        raise HTTPException(400, 'That date does not exist.')
    cat = e.category if e.category in CATEGORIES else 'Other'
    with closing(db()) as con, con:
        cur = con.execute('INSERT INTO expenses(user_id,profile,date,merchant,amount,category,note,ocr_text) '
                          'VALUES(?,?,?,?,?,?,?,?)',
                          (uid, e.profile, e.date, e.merchant.strip(), round(e.amount, 2), cat, e.note, e.ocr_text))
    return {'id': cur.lastrowid}


@router.delete('/expenses/{expense_id}')
def delete_expense(expense_id: int, uid: int = Depends(current_user)):
    with closing(db()) as con, con:
        n = con.execute('DELETE FROM expenses WHERE id=? AND user_id=?', (expense_id, uid)).rowcount
    if not n:
        raise HTTPException(404, 'Expense not found.')
    return {'deleted': expense_id}


@router.put('/budget')
def set_budget(b: BudgetIn, uid: int = Depends(current_user)):
    with closing(db()) as con, con:
        con.execute('INSERT INTO user_budgets(user_id, profile, amount) VALUES(?,?,?) '
                    'ON CONFLICT(user_id, profile) DO UPDATE SET amount=excluded.amount', (uid, b.profile, b.amount))
    return {'ok': True}


@router.get('/trend')
def trend(profile: str = Query(..., pattern=PROFILE), month: str = Query(..., pattern=MONTH),
          uid: int = Depends(current_user)):
    y, m = int(month[:4]), int(month[5:])
    keys = []
    for i in range(5, -1, -1):
        yy, mm = y, m - i
        while mm < 1:
            yy, mm = yy - 1, mm + 12
        keys.append(f'{yy}-{mm:02d}')
    with closing(db()) as con:
        rows = con.execute('SELECT substr(date,1,7) k, SUM(amount) t FROM expenses WHERE user_id=? AND profile=? '
                           'AND date>=? GROUP BY k', (uid, profile, keys[0] + '-01')).fetchall()
    got = {r['k']: r['t'] for r in rows}
    return [{'month': k, 'total': round(got.get(k, 0), 2)} for k in keys]


@router.get('/summary')
def summary(profile: str = Query(..., pattern=PROFILE), month: str = Query(..., pattern=MONTH),
            uid: int = Depends(current_user)):
    y, m = int(month[:4]), int(month[5:])
    py, pm = (y - 1, 12) if m == 1 else (y, m - 1)
    dim = calendar.monthrange(y, m)[1]
    with closing(db()) as con:
        rows = con.execute('SELECT date, merchant, amount, category FROM expenses WHERE user_id=? AND profile=? AND date LIKE ?',
                           (uid, profile, month + '-%')).fetchall()
        prev = con.execute('SELECT COALESCE(SUM(amount),0) FROM expenses WHERE user_id=? AND profile=? AND date LIKE ?',
                           (uid, profile, f'{py}-{pm:02d}-%')).fetchone()[0]
        b = con.execute('SELECT amount FROM user_budgets WHERE user_id=? AND profile=?', (uid, profile)).fetchone()
    budget = b[0] if b else 0
    total = round(sum(r['amount'] for r in rows), 2)
    cats, days, merchants = {}, [0.0] * dim, {}
    for r in rows:
        cats[r['category']] = cats.get(r['category'], 0) + r['amount']
        days[int(r['date'][8:10]) - 1] += r['amount']
        merchants[r['merchant']] = merchants.get(r['merchant'], 0) + r['amount']
    today = date.today()
    elapsed = today.day if (today.year, today.month) == (y, m) else (dim if (y, m) < (today.year, today.month) else 1)
    top = max(merchants.items(), key=lambda kv: kv[1]) if merchants else None
    big = max(range(dim), key=lambda i: days[i]) if rows else None
    return {
        'total': total, 'count': len(rows), 'budget': budget,
        'saved': round(budget - total, 2) if budget else None,
        'avg_per_day': round(total / elapsed, 2), 'days': [round(d, 2) for d in days],
        'by_category': sorted(({'name': k, 'total': round(v, 2)} for k, v in cats.items()), key=lambda c: -c['total']),
        'top_merchant': {'name': top[0], 'total': round(top[1], 2)} if top else None,
        'biggest_day': {'day': big + 1, 'total': round(days[big], 2)} if big is not None else None,
        'prev_total': round(prev, 2),
        'change_pct': round((total - prev) / prev * 100, 1) if prev else None,
    }


@router.get('/year')
def year_summary(profile: str = Query(..., pattern=PROFILE), year: int = Query(..., ge=2000, le=2100),
                 uid: int = Depends(current_user)):
    with closing(db()) as con:
        rows = con.execute('SELECT CAST(substr(date,6,2) AS INTEGER) m, category, amount FROM expenses '
                           'WHERE user_id=? AND profile=? AND date LIKE ?', (uid, profile, f'{year}-%')).fetchall()
    months, cats = [0.0] * 12, {}
    for r in rows:
        months[r['m'] - 1] += r['amount']
        cats[r['category']] = cats.get(r['category'], 0) + r['amount']
    total = round(sum(months), 2)
    today = date.today()
    elapsed = today.month if year == today.year else (12 if year < today.year else 1)
    top = max(range(12), key=lambda i: months[i]) if rows else None
    return {
        'year': year, 'months': [round(v, 2) for v in months], 'total': total, 'count': len(rows),
        'avg_per_month': round(total / elapsed, 2),
        'max_month': top + 1 if top is not None else None, 'max_total': round(months[top], 2) if top is not None else 0,
        'by_category': sorted(({'name': k, 'total': round(v, 2)} for k, v in cats.items()), key=lambda c: -c['total']),
    }


@router.get('/home')
def home(profile: str = Query(..., pattern=PROFILE), uid: int = Depends(current_user)):
    today = date.today()
    start = today - timedelta(days=6)
    with closing(db()) as con:
        recent = con.execute('SELECT id, date, merchant, amount, category, note FROM expenses WHERE user_id=? AND profile=? '
                             'ORDER BY date DESC, id DESC LIMIT 30', (uid, profile)).fetchall()
        wk = con.execute('SELECT date, SUM(amount) t FROM expenses WHERE user_id=? AND profile=? AND date BETWEEN ? AND ? '
                         'GROUP BY date', (uid, profile, start.isoformat(), today.isoformat())).fetchall()
    got = {r['date']: r['t'] for r in wk}
    week = [{'date': (start + timedelta(days=i)).isoformat(),
             'total': round(got.get((start + timedelta(days=i)).isoformat(), 0), 2)} for i in range(7)]
    return {'recent': [dict(r) for r in recent], 'week': week}


@router.get('/years')
def years(profile: str = Query(..., pattern=PROFILE), uid: int = Depends(current_user)):
    with closing(db()) as con:
        rows = con.execute('SELECT substr(date,1,4) y, SUM(amount) t, COUNT(*) c FROM expenses WHERE user_id=? AND profile=? '
                           'GROUP BY y ORDER BY y', (uid, profile)).fetchall()
    return [{'year': int(r['y']), 'total': round(r['t'], 2), 'count': r['c']} for r in rows]


class GoalIn(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    target: float = Field(gt=0, lt=1000000000)


class AddIn(BaseModel):
    amount: float = Field(gt=0, lt=1000000000)


@router.get('/goals')
def list_goals(uid: int = Depends(current_user)):
    with closing(db()) as con:
        rows = con.execute('SELECT id, name, target, saved FROM goals WHERE user_id=? ORDER BY id DESC', (uid,)).fetchall()
    return [dict(r) for r in rows]


@router.post('/goals')
def add_goal(g: GoalIn, uid: int = Depends(current_user)):
    with closing(db()) as con, con:
        cur = con.execute('INSERT INTO goals(user_id, name, target) VALUES(?,?,?)', (uid, g.name.strip(), g.target))
    return {'id': cur.lastrowid}


@router.post('/goals/{goal_id}/add')
def goal_add(goal_id: int, a: AddIn, uid: int = Depends(current_user)):
    with closing(db()) as con, con:
        n = con.execute('UPDATE goals SET saved=saved+? WHERE id=? AND user_id=?', (round(a.amount, 2), goal_id, uid)).rowcount
    if not n:
        raise HTTPException(404, 'Goal not found.')
    return {'ok': True}


@router.delete('/goals/{goal_id}')
def delete_goal(goal_id: int, uid: int = Depends(current_user)):
    with closing(db()) as con, con:
        n = con.execute('DELETE FROM goals WHERE id=? AND user_id=?', (goal_id, uid)).rowcount
    if not n:
        raise HTTPException(404, 'Goal not found.')
    return {'deleted': goal_id}


# Words printed on almost every receipt. They say nothing about WHAT was bought, and the model
# was learning them as if they did (one saved pharmacy scan made every receipt look like "Health").
RECEIPT_WORDS = set("""receipt invoice bill cash memo total subtotal sub grand paid balance change due amount price unit rate mrp
qty quantity item items description discount round off saved date time number num ref reference order table
vat gst cgst sgst igst tax taxes visa mastercard master credit debit card payment method upi paytm approved approval
auth transaction terminal pos bank thank thanks you your for shopping again visit welcome customer copy duplicate
keep warranty support purposes tel phone mobile email www com http https street road lane floor shop store""".split())


def _tokens(text):
    return [w for w in re.findall(r'[a-z]{3,}', (text or '').lower()) if w not in RECEIPT_WORDS]


def _nb_fit(rows):
    """Train a multinomial Naive Bayes text classifier. rows: [(text, category)]."""
    docs, counts, totals, vocab = {}, {}, {}, set()
    for text, c in rows:
        docs[c] = docs.get(c, 0) + 1
        for w in _tokens(text):
            counts.setdefault(c, {})[w] = counts.get(c, {}).get(w, 0) + 1
            totals[c] = totals.get(c, 0) + 1
            vocab.add(w)
    return docs, counts, totals, vocab


def _nb_predict(model, text, min_tokens=1):
    """Return (best category, confidence). Words never seen in training are ignored.
    With fewer than min_tokens known words there is too little evidence, so there is no answer."""
    docs, counts, totals, vocab = model
    toks = [w for w in _tokens(text) if w in vocab]
    if len(toks) < min_tokens:
        return None, 0.0
    n = sum(docs.values())
    scores = {}
    for c in docs:
        sc = math.log(docs[c] / n)
        for w in toks:
            sc += math.log((counts.get(c, {}).get(w, 0) + 1) / (totals.get(c, 0) + len(vocab) + 1))
        scores[c] = sc
    top = max(scores.values())
    probs = {c: math.exp(v - top) for c, v in scores.items()}
    best = max(probs, key=probs.get)
    return best, probs[best] / sum(probs.values())


def _labelled(con, uid):
    return [(r['t'], r['category']) for r in con.execute(
        "SELECT merchant || ' ' || merchant || ' ' || merchant || ' ' || ocr_text AS t, category FROM expenses WHERE user_id=? AND category != 'Other' "
        "ORDER BY id DESC LIMIT 3000", (uid,))]


def smart_category(uid, text):
    """Personalised category: Naive Bayes trained on this user's own saved expenses (shop name + receipt words).
    Falls back to keyword rules while there is little data or when the model is not confident."""
    with closing(db()) as con:
        rows = _labelled(con, uid)
    if len(rows) >= 8 and len({c for _, c in rows}) >= 2:
        best, conf = _nb_predict(_nb_fit(rows), text, min_tokens=2)
        if best and conf >= 0.6:
            return best
    return guess_category(text)


def _month_add(y, m, k):
    n = y * 12 + (m - 1) + k
    return n // 12, n % 12 + 1


@router.get('/ml/category-report')
def category_report(uid: int = Depends(current_user)):
    """Hold-out test of the category model: train on 80% of your expenses, test on the unseen 20%."""
    with closing(db()) as con:
        rows = _labelled(con, uid)
    if len(rows) < 40:
        raise HTTPException(400, f'Add at least 40 categorised expenses to test the model (you have {len(rows)}).')
    random.Random(1).shuffle(rows)
    cut = int(len(rows) * 0.8)
    train, test = rows[:cut], rows[cut:]
    model = _nb_fit(train)
    labels = [c for _, c in train]
    common = max(set(labels), key=labels.count)
    n = len(test)
    return {'train': len(train), 'test': n,
            'model': round(sum(_nb_predict(model, t)[0] == c for t, c in test) / n, 3),
            'rules': round(sum(guess_category(t) == c for t, c in test) / n, 3),
            'majority': round(sum(c == common for _, c in test) / n, 3)}


@router.get('/ml/insights')
def ml_insights(profile: str = Query(..., pattern=PROFILE), month: str = Query(..., pattern=MONTH),
                uid: int = Depends(current_user)):
    y, m = int(month[:4]), int(month[5:])
    (sy, sm), (ey, em) = _month_add(y, m, -11), _month_add(y, m, 1)
    with closing(db()) as con:
        rows = con.execute('SELECT date, merchant, amount, category FROM expenses WHERE user_id=? AND profile=? '
                           'AND date>=? AND date<?', (uid, profile, f'{sy}-{sm:02d}-01', f'{ey}-{em:02d}-01')).fetchall()
    totals = {}
    for r in rows:
        totals[r['date'][:7]] = totals.get(r['date'][:7], 0) + r['amount']
    series = [totals.get('%d-%02d' % _month_add(y, m, -i), 0.0) for i in range(5, -1, -1)]
    mean_y = sum(series) / 6                                    # least-squares line through the last 6 months
    slope = sum((x - 2.5) * (v - mean_y) for x, v in enumerate(series)) / sum((x - 2.5) ** 2 for x in range(6))
    next_month = max(0.0, mean_y + slope * 3.5)
    today = date.today()
    now_total = totals.get(month, 0.0)
    projected = now_total / today.day * calendar.monthrange(y, m)[1] if (today.year, today.month) == (y, m) else now_total
    past = {}
    for r in rows:
        if r['date'][:7] != month:
            past.setdefault(r['category'], []).append(r['amount'])
    unusual = []
    for r in rows:
        h = past.get(r['category'], [])
        if r['date'][:7] == month and len(h) >= 8:
            mu = sum(h) / len(h)
            sd = math.sqrt(sum((v - mu) ** 2 for v in h) / len(h))
            if r['amount'] > mu + 2.5 * sd and r['amount'] > 1.5 * mu:
                unusual.append({'date': r['date'], 'merchant': r['merchant'], 'amount': r['amount'],
                                'category': r['category'], 'typical': round(mu, 2)})
    return {'projected_end': round(projected, 2), 'next_month': round(next_month, 2),
            'history_months': sum(1 for v in series if v > 0),
            'unusual': sorted(unusual, key=lambda o: -o['amount'])[:5]}


# ---------------- sample data for demos ----------------
DEMO = {
    'Food & drink': (['Blue Tokai', 'Chai Sutta Bar', 'Haldiram', 'Punjabi Dhaba', 'Pizza Hut', 'Burger Singh'], 90, 900, 9),
    'Groceries': (['Sharma General Store', 'Big Basket', 'Reliance Smart', 'Fresh Basket'], 250, 3200, 6),
    'Transport': (['Uber', 'Ola Cab', 'Indian Oil', 'Metro Card'], 40, 1500, 6),
    'Health': (['Apollo Pharmacy', 'MedPlus', 'Dr Gupta Clinic'], 120, 2500, 1),
    'Shopping': (['Myntra', 'Amazon', 'Zara', 'Crossword'], 300, 5500, 3),
    'Bills': (['Airtel', 'Jio Recharge', 'Electricity Board', 'Broadband Plus'], 300, 4500, 2),
    'Office': (['Staples', 'Print Hub', 'Cloud Hosting'], 100, 3000, 1),
}


@router.post('/demo')
def load_demo(profile: str = Query(..., pattern=PROFILE), uid: int = Depends(current_user)):
    rnd, today = random.Random(7), date.today()
    cats = list(DEMO)
    weights = [DEMO[c][3] for c in cats]
    rows = []
    for k in range(-59, 1):                                     # the last 5 years, month by month
        y, m = _month_add(today.year, today.month, k)
        last = today.day if (y, m) == (today.year, today.month) else calendar.monthrange(y, m)[1]
        for _ in range(rnd.randint(22, 38)):
            cat = rnd.choices(cats, weights)[0]
            names, lo, hi = DEMO[cat][:3]
            amount = round(rnd.uniform(lo, hi) * (1 + 0.012 * (k + 59) / 12), 2)
            rows.append((uid, profile, f'{y}-{m:02d}-{rnd.randint(1, last):02d}', rnd.choice(names), amount, cat, 'demo'))
    with closing(db()) as con, con:
        con.executemany('INSERT INTO expenses(user_id,profile,date,merchant,amount,category,note) VALUES(?,?,?,?,?,?,?)', rows)
        con.execute('INSERT OR IGNORE INTO user_budgets(user_id, profile, amount) VALUES(?,?,40000)', (uid, profile))
    return {'added': len(rows)}


@router.delete('/demo')
def clear_demo(profile: str = Query(..., pattern=PROFILE), uid: int = Depends(current_user)):
    with closing(db()) as con, con:
        n = con.execute("DELETE FROM expenses WHERE user_id=? AND profile=? AND note='demo'", (uid, profile)).rowcount
    return {'removed': n}
