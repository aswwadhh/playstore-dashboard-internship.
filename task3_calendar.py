"""Task 3 - interactive calendar-style heatmap of monthly installs, with MoM growth and 3-month MA forecast."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from common import label, fmt_installs, load_apps

TRANSLATE = {"BEAUTY", "BUSINESS", "DATING"}            # Hindi, Tamil, German
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
GROWTH_HIGHLIGHT = 20.0                                  # percent
MA_WINDOW = 3
HORIZON = 3                                              # forecast months
ALL = "ALL"

def filter_apps(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        df["Category"].str[0].isin(list("ECB")) & (df["Rating"] >= 4.0) & (df["Installs"] > 10_000)
        & (df["Reviews"] > 500) & df["Size_MB"].between(15, 80) & (df["Avg_Subjectivity"] > 0.5)
        & ~df["App"].str[0].str.upper().isin(list("XYZ"))               # names starting with X, Y, Z
        & ~df["App"].str.contains("s", case=False)                      # names containing S
    ].copy()

def monthly_series(d: pd.DataFrame, months: pd.PeriodIndex) -> pd.DataFrame:
    """Actual + forecast monthly table for one set of apps on a common timeline."""
    d = d.assign(month=d["Last_Updated"].dt.to_period("M"))
    inst = d.groupby("month")["Installs"].sum().reindex(months, fill_value=0.0)
    rev = d.groupby("month")["Reviews"].sum().reindex(months, fill_value=0.0)
    s = pd.DataFrame({"installs": inst.astype(float), "reviews": rev.astype(float), "forecast": False})
    s["ma"] = s["installs"].rolling(MA_WINDOW, min_periods=1).mean()
    s["ma_n"] = s["installs"].rolling(MA_WINDOW, min_periods=1).count()
    vals = list(s["installs"])
    fut = pd.period_range(months[-1] + 1, periods=HORIZON, freq="M")
    rows = []
    for m in fut:                                                       # recursive 3-month moving-average forecast
        f = float(np.mean(vals[-MA_WINDOW:])); vals.append(f)
        rows.append(pd.Series({"installs": f, "reviews": np.nan, "forecast": True, "ma": f, "ma_n": MA_WINDOW}, name=m))
    s = pd.concat([s, pd.DataFrame(rows)])
    prev = s["installs"].shift(1)
    g = (s["installs"] - prev) / prev.where(prev > 0) * 100
    g[(prev == 0) & (s["installs"] > 0)] = np.inf                        # growth from zero = new activity
    g.iloc[0] = np.nan
    s["growth"] = g
    s["highlight"] = (s["growth"] > GROWTH_HIGHLIGHT) & ~s["forecast"]   # actual months with > 20% increase
    return s

def prepare(df: pd.DataFrame) -> dict:
    d = filter_apps(df)
    cats = d.groupby("Category")["Installs"].sum().nlargest(5).index.tolist()   # top five eligible (fewer exist)
    d = d[d["Category"].isin(cats)]
    months = pd.period_range(d["Last_Updated"].min().to_period("M"), d["Last_Updated"].max().to_period("M"), freq="M")
    views = {ALL: monthly_series(d, months)}
    for c in cats:
        views[c] = monthly_series(d[d["Category"] == c], months)
    return dict(data=d, cats=cats, months=months, views=views)

def _g(v):
    return "new activity (previous month = 0)" if np.isinf(v) else ("n/a" if pd.isna(v) else f"{v:+.1f}%")

def _grid(s: pd.DataFrame):
    years = sorted({p.year for p in s.index})
    return years, {y: i for i, y in enumerate(years)}

def _view_traces(s: pd.DataFrame, name: str, vmax: float):
    years, yi = _grid(s)
    shape = (len(years), 12)
    za, zf = np.full(shape, np.nan), np.full(shape, np.nan)
    ta, tf = np.full(shape, "", dtype=object), np.full(shape, "", dtype=object)
    cd = np.full(shape + (6,), "", dtype=object)
    for p, r in s.iterrows():
        i, j = yi[p.year], p.month - 1
        if r.forecast:
            zf[i, j] = r.installs; tf[i, j] = f"F {fmt_installs(r.installs)}"
            status = "FORECAST (3-month moving average)"
        else:
            za[i, j] = r.installs; ta[i, j] = fmt_installs(r.installs) if r.installs else "0"
            status = "Actual" if r.installs else "Actual (no apps updated this month)"
        cd[i, j] = [f"{name} - {MONTHS[j]} {p.year}", fmt_installs(r.installs),
                    _g(r.growth), f"{fmt_installs(r.ma)} (avg of {int(r.ma_n)} mo)",
                    "n/a (forecast)" if r.forecast else f"{int(r.reviews):,}", status]
    hover = ("<b>%{customdata[0]}</b><br>Installs: %{customdata[1]}<br>MoM growth: %{customdata[2]}"
             "<br>Rolling avg (3 mo): %{customdata[3]}<br>Review count: %{customdata[4]}"
             "<br>Status: %{customdata[5]}<extra></extra>")
    x = list(range(12))
    common = dict(x=x, y=list(range(len(years))), hoverongaps=False, xgap=3, ygap=3)
    actual = go.Heatmap(z=za, text=ta, texttemplate="%{text}", colorscale="Blues", zmin=0, zmax=vmax,
                        customdata=cd, hovertemplate=hover,
                        colorbar=dict(title="Actual<br>installs", x=1.02, len=0.8), **common)
    fcast = go.Heatmap(z=zf, text=tf, texttemplate="%{text}", colorscale="Oranges", zmin=0, zmax=vmax,
                       customdata=cd, hovertemplate=hover,
                       showscale=False, **common)
    shapes = []
    for p, r in s.iterrows():
        if r.highlight:
            i, j = yi[p.year], p.month - 1
            shapes.append(dict(type="rect", xref="x", yref="y", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
                               line=dict(color="#d00000", width=4), fillcolor="rgba(0,0,0,0)"))
    return actual, fcast, shapes, years

def build_figure(P: dict) -> go.Figure:
    views = P["views"]; keys = [ALL] + P["cats"]
    names = {ALL: "All eligible categories"} | {c: label(c, TRANSLATE) for c in P["cats"]}
    fig, owner, shapes_by, years_by = go.Figure(), [], {}, {}
    for vi, k in enumerate(keys):
        vmax = views[k]["installs"].max() or 1
        a, f, shp, years = _view_traces(views[k], names[k], vmax)
        for t in (a, f):
            t.visible = (vi == 0); fig.add_trace(t); owner.append(vi)
        shapes_by[k], years_by[k] = shp, years
    def axis(k):
        ys = years_by[k]
        return dict(tickvals=list(range(len(ys))), ticktext=[str(y) for y in ys], autorange="reversed", title="Year")
    buttons = [dict(label=names[k], method="update",
                    args=[{"visible": [o == vi for o in owner]}, {"shapes": shapes_by[k], "yaxis": axis(k)}])
               for vi, k in enumerate(keys)]
    fig.update_layout(
        title="Calendar heatmap - monthly installs (categories starting with E, C, B)"
              f"<br><sup>{len(P['data'])} apps in {len(P['cats'])} categories (fewer than 5 qualify) | red border = MoM growth "
              f"> {GROWTH_HIGHLIGHT:.0f}% | orange cells F = 3-month moving-average forecast</sup>",
        template="plotly_white", height=560, margin=dict(t=175, b=60, r=120),
        plot_bgcolor="#e9e9e9", shapes=shapes_by[ALL],
        xaxis=dict(tickvals=list(range(12)), ticktext=MONTHS, side="top", showgrid=False),
        yaxis=axis(ALL),
        updatemenus=[dict(buttons=buttons, direction="down", x=0.0, y=1.1, xanchor="left", yanchor="bottom", showactive=True)])
    return fig

if __name__ == "__main__":
    P = prepare(load_apps())
    build_figure(P).write_html("task3_calendar.html")
    print(P["cats"]); print(P["views"][ALL][["installs", "ma", "growth", "forecast", "highlight"]])
