"""Category model: receipt boilerplate must not drive the prediction; the model must still learn from the user."""
import os
import sys
import tempfile
from contextlib import closing

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest  # noqa: E402
import app.tracker as t  # noqa: E402

CAT_DB = os.path.join(tempfile.mkdtemp(), 'cat.db')


@pytest.fixture(autouse=True)
def _own_db():
    t.DB_PATH = CAT_DB
    yield

PHARMACY = ('CITY PHARMACY Date: 02/03/2025 Receipt Number: 4411 Paracetamol 2 30.00 Subtotal 60.00 VAT 5.00 '
            'Total Paid 65.00 Visa Credit Card Thank you Keep this receipt')
APPLE = ('APPLE STORE, GRAFTON STREET Date: April 9/2025 Receipt Number: 967654321 Visa Credit Card MacBook Pro 14-inch '
         '1 799.00 Subtotal 2,799.00 VAT (23%): 643.77 Total Paid 3,442.77 Thank you for shopping at Apple Keep this receipt')

t.DB_PATH = CAT_DB
with closing(t.db()) as con, con:
    con.execute("INSERT INTO users(email,name,salt,pw,verified) VALUES('a@a.io','A',x'00',x'00',1)")
t.load_demo('personal', 1)
with closing(t.db()) as con, con:      # one scanned pharmacy receipt saved earlier, like the user's DB
    con.execute("INSERT INTO expenses(user_id,profile,date,merchant,amount,category,ocr_text) "
                "VALUES(1,'personal','2025-03-02','City Pharmacy',65,'Health',?)", (PHARMACY,))


def test_boilerplate_does_not_cause_health():
    assert t.smart_category(1, APPLE) == 'Shopping'


def test_real_pharmacy_still_health():
    assert t.smart_category(1, PHARMACY) == 'Health'


def test_boilerplate_words_are_ignored():
    assert 'receipt' not in t._tokens('Receipt total visa card Pharmacy') and 'pharmacy' in t._tokens('Pharmacy')


def test_model_learns_from_user_labels():
    with closing(t.db()) as con, con:
        for i in range(12):
            con.execute("INSERT INTO expenses(user_id,profile,date,merchant,amount,category) "
                        "VALUES(1,'personal','2025-05-01','Zorba Taverna',50,'Food & drink')")
    assert t.smart_category(1, 'ZORBA TAVERNA Total 99.00 Visa') == 'Food & drink'


def test_too_little_evidence_falls_back_to_rules():
    assert t._nb_predict(t._nb_fit([('city pharmacy', 'Health'), ('big basket', 'Groceries')]), 'basket', min_tokens=2) == (None, 0.0)
