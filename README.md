# Google Play Store Analytics Dashboard (ElevanceSkills Python Internship)

Author: Aswadh B Pramod

Six interactive charts built with Python, Plotly and Dash on the Google Play Store dataset
(`googleplaystore.csv`, `googleplaystore_user_reviews.csv`). Each chart is shown only during its IST time window.

| Task | Chart | Visible (IST) |
|---|---|---|
| 1 | Hexbin density: app size vs rating | 5 PM - 7 PM |
| 2 | Sunburst: Country > Category > App Type > Rating Band | 6 PM - 8 PM |
| 3 | Calendar heatmap with 3-month forecast | 6 PM - 9 PM |
| 4 | Streamgraph (monthly / cumulative / growth) | 4 PM - 6 PM |
| 5 | Clustered heatmap, top 10 categories | 3 PM - 5 PM |
| 6 | Radar: free vs paid apps | 1 PM - 2 PM |

## Run
```
pip install -r requirements.txt
python clean_data.py data/googleplaystore.csv data/googleplaystore_user_reviews.csv data   # only if cleaned files are missing
python app.py          # open http://127.0.0.1:8050
```
Set `DEMO_MODE=1` (local use only) to show all charts outside their time windows.

## Tests
`python tests/test_task1.py` ... `python tests/test_task6.py`, `python tests/test_app.py`

## Notes
- The dataset has no country field, so Task 2 uses a single "Global" node (documented in the report).
- Some task filters leave very few apps (Task 1: 17, Task 3: 5); filters are applied exactly as specified.
- See the PDF report for all assumptions and definitions.
