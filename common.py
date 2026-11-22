"""Shared helpers for the internship dashboards: data loading, translations, IST time gate."""
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
IST = ZoneInfo("Asia/Kolkata")

# Visibility windows in IST, 24-hour clock: visible when start <= time < end
WINDOWS = {
    1: (17, 19),   # Task 1 hexbin        5 PM - 7 PM
    2: (18, 20),   # Task 2 sunburst      6 PM - 8 PM
    3: (18, 21),   # Task 3 calendar      6 PM - 9 PM
    4: (16, 18),   # Task 4 streamgraph   4 PM - 6 PM
    5: (15, 17),   # Task 5 clustered     3 PM - 5 PM
    6: (13, 14),   # Task 6 radar         1 PM - 2 PM
}

def now_ist() -> datetime:
    return datetime.now(IST)

def is_visible(task: int, now: datetime | None = None, test_mode: bool = False) -> bool:
    """True if the chart for `task` may be shown right now (IST). test_mode shows everything."""
    if test_mode:
        return True
    now = (now or now_ist()).astimezone(IST)
    start, end = WINDOWS[task]
    return time(start) <= now.time() < time(end)

def window_label(task: int) -> str:
    fmt = lambda h: f"{(h - 1) % 12 + 1} {'AM' if h < 12 else 'PM'}"
    s, e = WINDOWS[task]
    return f"{fmt(s)} - {fmt(e)} IST"

# Category translations required by the tasks (key = raw category code in the dataset)
TRANSLATIONS = {
    "BEAUTY": "सौंदर्य",              # Hindi
    "BUSINESS": "வணிகம்",            # Tamil
    "DATING": "Partnersuche",        # German
    "TRAVEL_AND_LOCAL": "Voyage et local",   # French
    "PRODUCTIVITY": "Productividad",         # Spanish
    "PHOTOGRAPHY": "写真",                    # Japanese
}

def pretty(cat: str) -> str:
    return cat.replace("_", " ").title().replace(" And ", " & ")

def label(cat: str, translate: set | None = None) -> str:
    """Display name of a category; translated if the task asks for that category."""
    if translate is None or cat in translate:
        return TRANSLATIONS.get(cat, pretty(cat))
    return pretty(cat)

def load_apps() -> pd.DataFrame:
    """Cleaned apps merged with per-app average sentiment (see clean_data.py)."""
    apps = pd.read_csv(DATA_DIR / "cleaned_apps.csv", parse_dates=["Last_Updated"])
    sent = pd.read_csv(DATA_DIR / "cleaned_reviews_by_app.csv")
    return apps.merge(sent, on="App", how="left")

def fmt_installs(v: float) -> str:
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= div:
            return f"{v / div:.1f}{suf}"
    return f"{v:,.0f}"
