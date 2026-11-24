"""Run with `python tests/test_extract.py` or `pytest`."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.extract import extract_fields, find_company, find_date, find_total  # noqa: E402


def test_basic_receipt():
    lines = ['ACME STORE SDN BHD', 'TEL 0123456789', 'Date: 12/05/2018', 'Item A 3.00',
             'SUBTOTAL 10.00', 'TOTAL 10.60']
    assert extract_fields(lines) == {'company': 'ACME STORE SDN BHD', 'date': '12/05/2018', 'total': '10.60'}


def test_total_on_next_line():
    assert find_total(['TOTAL', '24.90']) == '24.90'


def test_total_ignores_subtotal_and_picks_last():
    assert find_total(['SUBTOTAL 9.00', 'TOTAL 9.50', 'ROUNDED TOTAL 9.50']) == '9.50'


def test_total_fallback_to_largest_amount():
    assert find_total(['Coffee 4.50', 'Cake 6.20']) == '6.20'


def test_text_dates():
    assert find_date(['Printed 5 Jan 2019']) == '5 Jan 2019'
    assert find_date(['2019-03-07 10:22']) == '2019-03-07'


def test_date_formats_day_first():
    from app.dates import parse_date
    for text in ['09/04/2025', '9/4/2025', '09-04-2025', '2025-04-09', '9 April 2025', '09 Apr 2025',
                 'April 9, 2025', 'Apr 9 2025', '9th April 2025', '9 April, 2025', '09.04.2025', '09/04/25']:
        assert parse_date(text) == '2025-04-09', text
    assert parse_date('15 August 2024') == '2024-08-15'
    assert parse_date('15 AUG 2024') == '2024-08-15'
    assert parse_date('12-Nov-26') == '2026-11-12'
    assert parse_date('9 Sept 2025') == '2025-09-09'
    assert parse_date('03/04/2025') == '2025-04-03'          # ambiguous -> DD/MM, never MM/DD
    assert parse_date('04/13/2025') == '2025-04-13'          # only readable as month-first


def test_invalid_or_missing_dates_are_none_not_today():
    from app.dates import parse_date
    for text in ['31/02/2025', '2025-13-45', '00/00/0000', '9 Foo 2025', '', None, 'hello', '09/04/1850']:
        assert parse_date(text) is None, text


def test_find_date_variants_and_priority():
    assert find_date(['Date: April 9, 2025']) == 'April 9, 2025'
    assert find_date(['EXP 01/01/2030', 'Date 09/04/2025']) == '09/04/2025'     # 'Date' line beats an expiry line
    assert find_date(['Bad 31/02/2025', 'Bill date 09/04/2025']) == '09/04/2025'  # invalid candidates are skipped
    assert find_date(['TEL 0123456789', 'TOTAL 10.60']) is None


def test_apple_receipt_from_screenshot():
    lines = ['RECEIPT', 'APPLE STORE, GRAFTON STREET', '1-6 Grafton Street, Dublin 2, D02 XH61',
             'Date: April 9/2025', 'Receipt Number: 967654321', 'Subtotal \u20ac 2,799.00', 'Total Paid \u20ac 3,442.77']
    assert extract_fields(lines) == {'company': 'APPLE STORE, GRAFTON STREET', 'date': 'April 9/2025', 'total': '3,442.77'}


def test_missing_values():
    assert find_date(['hello']) is None
    assert find_total(['hello']) is None
    assert find_company([]) is None


def test_company_skips_phone_like_lines():
    assert find_company(['0123456789', 'GOOD FOOD CAFE']) == 'GOOD FOOD CAFE'


def test_bus_ticket_time_is_not_a_total_and_comma_decimal_is_read():
    # EasyOCR rows from a Punjab Roadways ticket photo: "₹ 55.00" came out as "{55,00"
    lines = ['ROADWAYS', 'Ferozepur DEPOT', 'No:00000920 24/11/26 10.38.03', 'Tot distance: 37.0 KMs',
             'FULL: 1x55.0-755,0', '{55,00', 'not TRANSFERABLF']
    assert find_total(lines) == '55.00'
    assert find_date(lines) == '24/11/26'


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
