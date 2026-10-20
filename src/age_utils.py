"""
age_utils.py — NephroSense
Exact age from a date of birth. Used at registration (so nobody types an age) and at every login (so the age
never goes stale). Accepts DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY and YYYY-MM-DD.
"""
import re
from datetime import date, datetime

MAX_AGE = 120


def parse_dob(text):
    """-> date, or None when the text is empty / not a real calendar date."""
    if isinstance(text, datetime):
        return text.date()
    if isinstance(text, date):
        return text
    s = str(text or "").strip()
    if not s:
        return None
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        y, mo, d = (int(x) for x in m.groups())
    else:
        m = re.fullmatch(r"(\d{1,2})[/\-.\s](\d{1,2})[/\-.\s](\d{4})", s)
        if not m:
            return None
        d, mo, y = (int(x) for x in m.groups())
    try:
        return date(y, mo, d)
    except ValueError:
        return None


def calculate_age(dob, today=None):
    """Completed years on `today`. None if the date is missing, invalid, in the future or older than 120 years."""
    b = parse_dob(dob)
    today = today or date.today()
    if b is None or b > today:
        return None
    years = today.year - b.year - ((today.month, today.day) < (b.month, b.day))
    return years if 0 <= years <= MAX_AGE else None


def format_dob(dob):
    b = parse_dob(dob)
    return b.strftime("%d/%m/%Y") if b else ""