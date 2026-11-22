"""Data cleaning for the Google Play Store dataset (ElevanceSkills Python internship).

Usage:  python clean_data.py [apps_csv] [reviews_csv] [output_folder]
Outputs: cleaned_apps.csv, cleaned_reviews_by_app.csv, cleaning_log.txt
"""
import sys
import numpy as np
import pandas as pd

APPS = sys.argv[1] if len(sys.argv) > 1 else "data/googleplaystore.csv"
REVIEWS = sys.argv[2] if len(sys.argv) > 2 else "data/googleplaystore_user_reviews.csv"
OUT = sys.argv[3] if len(sys.argv) > 3 else "data"

log = []
def note(msg):
    print(msg); log.append(msg)

def size_to_mb(s):
    s = str(s).strip()
    if s.endswith("M"):
        return float(s[:-1].replace(",", ""))
    if s.endswith("k"):
        return float(s[:-1].replace(",", "")) / 1024   # kilobytes -> MB
    return np.nan                                       # "Varies with device"

apps = pd.read_csv(APPS)
note(f"Raw apps rows: {len(apps)}")

# 1. Remove the shifted/corrupted row (Category = 1.9, Genres empty)
bad = apps["Genres"].isna() | apps["Category"].astype(str).str.fullmatch(r"[\d.]+")
note(f"Corrupted rows removed: {int(bad.sum())}")
apps = apps[~bad].copy()

# 2. Remove exact duplicate rows
n = len(apps); apps = apps.drop_duplicates()
note(f"Exact duplicate rows removed: {n - len(apps)}")

# 3. Clean columns
inst = apps["Installs"].astype(str).str.replace(r"[+,]", "", regex=True)
apps["Installs"] = pd.to_numeric(inst, errors="coerce")           # junk like 'Free' -> NaN
apps["Reviews"] = pd.to_numeric(apps["Reviews"], errors="coerce")
apps["Size_MB"] = apps["Size"].map(size_to_mb)
apps["Price"] = pd.to_numeric(apps["Price"].astype(str).str.replace("$", "", regex=False), errors="coerce")
apps["Rating"] = pd.to_numeric(apps["Rating"], errors="coerce")
apps["Android_Ver"] = pd.to_numeric(
    apps["Android Ver"].astype(str).str.extract(r"^(\d+\.\d+)")[0], errors="coerce")  # 4.0.3 and up -> 4.0
apps["Last_Updated"] = pd.to_datetime(apps["Last Updated"], format="%d-%b-%y", errors="coerce")
apps["Update_Month"] = apps["Last_Updated"].dt.to_period("M").astype(str)
apps["Update_Month_Num"] = apps["Last_Updated"].dt.month

# 4. Drop rows with unusable key fields
n = len(apps)
apps = apps.dropna(subset=["Installs", "Reviews", "Last_Updated"])
note(f"Rows dropped for unusable Installs/Reviews/Date: {n - len(apps)}")

# Type: fill from price when missing, keep only Free/Paid
apps["Type"] = np.where(apps["Type"].isin(["Free", "Paid"]), apps["Type"],
                        np.where(apps["Price"] > 0, "Paid", "Free"))

# 5. One row per app: keep the row with most reviews (latest update breaks ties)
n = len(apps)
apps = (apps.sort_values(["Reviews", "Last_Updated"], ascending=False)
            .drop_duplicates(subset="App", keep="first").sort_index())
note(f"Repeated app names removed (kept highest-review row): {n - len(apps)}")

apps = apps.drop(columns=["Size", "Android Ver", "Last Updated"])
apps["Category_Clean"] = apps["Category"].str.replace("_", " ").str.title()

# 6. Reviews file: average sentiment per app
rev = pd.read_csv(REVIEWS)
note(f"Raw review rows: {len(rev)}")
rev = rev.dropna(subset=["Sentiment_Subjectivity", "Sentiment_Polarity"])
note(f"Review rows with sentiment scores: {len(rev)}")
by_app = (rev.groupby("App")
             .agg(Avg_Subjectivity=("Sentiment_Subjectivity", "mean"),
                  Avg_Polarity=("Sentiment_Polarity", "mean"),
                  Review_Count_Sentiment=("Sentiment_Subjectivity", "size"))
             .reset_index())
note(f"Apps with sentiment data: {len(by_app)}")

apps.to_csv(f"{OUT}/cleaned_apps.csv", index=False)
by_app.to_csv(f"{OUT}/cleaned_reviews_by_app.csv", index=False)

note(f"\nFinal cleaned apps: {len(apps)} rows")
note("Missing values after cleaning:\n" + apps.isna().sum().to_string())
note(f"Apps matched to sentiment: {apps['App'].isin(by_app['App']).sum()}")
note(f"Date range: {apps['Last_Updated'].min().date()} to {apps['Last_Updated'].max().date()}")
open(f"{OUT}/cleaning_log.txt", "w").write("\n".join(log))
