"""End-to-end: upload (mocked OCR) -> /api/extract -> /api/expenses -> calendar/summary/year/search."""
import os
import sys
import tempfile
import time

import cv2
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FLOW_DB = os.path.join(tempfile.mkdtemp(), 'test.db')
os.environ['TRACKER_DB'] = FLOW_DB
os.environ['WARMUP'] = '0'

import app.main as main  # noqa: E402
import app.tracker as tracker  # noqa: E402
from contextlib import closing  # noqa: E402

import pytest  # noqa: E402

client = TestClient(main.app)
tracker.DB_PATH = FLOW_DB


@pytest.fixture(autouse=True)
def _own_db():
    tracker.DB_PATH = FLOW_DB
    yield

OCR_ROWS = []


def _fake_ocr(img, engine):
    return [{'text': t, 'conf': 0.9, 'box': None} for t in OCR_ROWS]


main.run_ocr = _fake_ocr
main.group_rows = lambda lines: [l['text'] for l in lines]
main.available_engines = lambda: ['easyocr']

with closing(tracker.db()) as con, con:                       # a signed-in user without the email step
    con.execute("INSERT INTO users(email,name,salt,pw) VALUES('t@t.io','T',x'00',x'00')")
    uid = con.execute("SELECT id FROM users").fetchone()[0]
    con.execute('INSERT INTO sessions(token,user_id,expires) VALUES(?,?,?)', (tracker._sha('tok'), uid, time.time() + 999))
client.cookies.set('pp_session', 'tok')
PNG = cv2.imencode('.png', np.full((200, 300, 3), 255, np.uint8))[1].tobytes()


def scan(rows):
    OCR_ROWS[:] = rows
    r = client.post('/api/extract', files={'file': ('r.png', PNG, 'image/png')})
    assert r.status_code == 200, r.text
    return r.json()


def save(date, merchant='Cafe Test', amount=100.0, profile='personal'):
    return client.post('/api/expenses', json={'profile': profile, 'date': date, 'merchant': merchant, 'amount': amount})


def test_9_april_2025():
    r = scan(['SPICE CAFE', 'Date: 9 April 2025', 'TOTAL 250.00'])
    assert r['fields']['date'] == '9 April 2025' and r['date_iso'] == '2025-04-09'
    assert save(r['date_iso'], 'Spice Cafe', 250).status_code == 200
    got = client.get('/api/expenses', params={'profile': 'personal', 'month': '2025-04'}).json()
    assert [(e['date'], e['merchant']) for e in got] == [('2025-04-09', 'Spice Cafe')]       # calendar day
    assert client.get('/api/expenses', params={'profile': 'personal', 'month': '2026-11'}).json() == []  # NOT today
    assert client.get('/api/summary', params={'profile': 'personal', 'month': '2025-04'}).json()['total'] == 250


def test_15_august_2024():
    r = scan(['FRESH MART', '15 Aug 2024 11:02', 'TOTAL 80.00'])
    assert r['date_iso'] == '2024-08-15'
    save(r['date_iso'], 'Fresh Mart', 80)
    got = client.get('/api/expenses', params={'profile': 'personal', 'month': '2024-08'}).json()
    assert got[0]['date'] == '2024-08-15'


def test_numeric_is_day_first():
    assert scan(['SHOP', 'Date 09/04/2025', 'TOTAL 10.00'])['date_iso'] == '2025-04-09'


def test_no_date_requires_manual_entry():
    r = scan(['SHOP', 'TOTAL 10.00'])
    assert r['date_iso'] is None and r['fields']['date'] is None and r['date_invalid'] is None


def test_invalid_date_is_reported_not_replaced():
    r = scan(['SHOP', 'Date 31/02/2025', 'TOTAL 10.00'])
    assert r['date_iso'] is None and r['date_invalid'] == '31/02/2025'
    assert save('2025-02-31').status_code == 400          # backend also refuses impossible dates
    assert save('not-a-date').status_code == 422


def test_manual_edit_is_what_gets_saved():
    r = scan(['SHOP', 'Date: 9 April 2025', 'TOTAL 10.00'])
    assert r['date_iso'] == '2025-04-09'
    assert save('2025-04-10', 'Edited').status_code == 200      # user changed the date in the form
    got = client.get('/api/expenses', params={'profile': 'personal', 'month': '2025-04'}).json()
    assert any(e['merchant'] == 'Edited' and e['date'] == '2025-04-10' for e in got)


def test_calendar_totals_search_and_year():
    got = client.get('/api/expenses', params={'profile': 'personal', 'month': '2025-04'}).json()
    assert sum(e['amount'] for e in got) >= 250
    s = client.get('/api/search', params={'profile': 'personal', 'q': 'Spice'}).json()
    assert s and s[0]['date'] == '2025-04-09'
    y = client.get('/api/year', params={'profile': 'personal', 'year': 2025})
    assert y.status_code == 200 and y.json()
    y24 = client.get('/api/year', params={'profile': 'personal', 'year': 2024}).json()
    assert y24


def test_business_profile_is_separate():
    save('2025-04-09', 'BizCo', 5, profile='business')
    p = client.get('/api/expenses', params={'profile': 'personal', 'month': '2025-04'}).json()
    b = client.get('/api/expenses', params={'profile': 'business', 'month': '2025-04'}).json()
    assert all(e['merchant'] != 'BizCo' for e in p) and [e['merchant'] for e in b] == ['BizCo']


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
