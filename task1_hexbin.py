"""Task 1 - interactive hexbin density chart: app size vs rating (colour = mean installs)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

from common import label, load_apps, fmt_installs

CATEGORIES = ["GAME", "BEAUTY", "BUSINESS", "COMICS", "COMMUNICATION", "DATING", "ENTERTAINMENT", "SOCIAL", "EVENTS"]
TRANSLATE = {"BEAUTY", "BUSINESS", "DATING"}               # Hindi, Tamil, German
PINK = "#ff4fa3"
HEX_R = 0.17                                               # hexagon radius in the unit-scaled space
SIZE_RANGE, RATING_RANGE = (10.0, 100.0), (3.5, 5.0)
MIN_APPS_FOR_IQR = 4

def filter_apps(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        df["Category"].isin(CATEGORIES) & (df["Rating"] > 3.5) & (df["Installs"] > 50_000)
        & (df["Reviews"] > 500) & df["Size_MB"].between(10, 100) & (df["Avg_Subjectivity"] > 0.5)
        & ~df["App"].str.contains("s", case=False)                  # no letter S, any case
    ].copy()

def to_unit(size, rating):
    return ((np.asarray(size) - SIZE_RANGE[0]) / (SIZE_RANGE[1] - SIZE_RANGE[0]),
            (np.asarray(rating) - RATING_RANGE[0]) / (RATING_RANGE[1] - RATING_RANGE[0]))

def from_unit(x, y):
    return (np.asarray(x) * (SIZE_RANGE[1] - SIZE_RANGE[0]) + SIZE_RANGE[0],
            np.asarray(y) * (RATING_RANGE[1] - RATING_RANGE[0]) + RATING_RANGE[0])

def hex_center(q, r, R=HEX_R):
    return R * 1.5 * q, R * np.sqrt(3) * (r + q / 2)

def assign_hex(x, y, R=HEX_R):
    """Flat-top hexagonal binning: returns integer axial coordinates (q, r) for each point."""
    qf = (2 / 3 * x) / R
    rf = (-1 / 3 * x + np.sqrt(3) / 3 * y) / R
    cx, cz, cy = qf, rf, -qf - rf                         # cube rounding
    rx, ry, rz = np.round(cx), np.round(cy), np.round(cz)
    dx, dy, dz = abs(rx - cx), abs(ry - cy), abs(rz - cz)
    fix_x = (dx > dy) & (dx > dz); fix_y = ~fix_x & (dy > dz)
    rx = np.where(fix_x, -ry - rz, rx); rz = np.where(~fix_x & ~fix_y, -rx - ry, rz)
    return rx.astype(int), rz.astype(int)

def hex_polygon(q, r, R=HEX_R):
    cx, cy = hex_center(q, r, R)
    ang = np.radians(np.arange(0, 361, 60))
    return cx + R * np.cos(ang), cy + R * np.sin(ang)

def iqr_outliers(d: pd.DataFrame) -> pd.DataFrame:
    """Category-level Tukey fences (1.5 x IQR) on app size, rating and installs (the chart's three numeric variables)."""
    rows = []
    for cat, g in d.groupby("Category"):
        if len(g) < MIN_APPS_FOR_IQR:
            continue
        for col, nm in (("Size_MB", "size"), ("Rating", "rating"), ("Installs", "installs")):
            q1, q3 = g[col].quantile(0.25), g[col].quantile(0.75)
            lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
            for i, v in g[col].items():
                if v < lo or v > hi:
                    rows.append((i, f"{nm} {'low' if v < lo else 'high'}"))
    if not rows:
        return d.iloc[0:0].assign(reason="")
    out = pd.DataFrame(rows, columns=["idx", "reason"]).groupby("idx")["reason"].agg(", ".join)
    return d.loc[out.index].assign(reason=out)

def prepare(df: pd.DataFrame) -> dict:
    d = filter_apps(df).reset_index(drop=True)
    x, y = to_unit(d["Size_MB"], d["Rating"])
    d["q"], d["r"] = assign_hex(x, y)
    hexes = (d.groupby(["q", "r"]).agg(n=("App", "count"), mean_installs=("Installs", "mean"),
                                       mean_rating=("Rating", "mean"), mean_size=("Size_MB", "mean"),
                                       cats=("Category", lambda s: ", ".join(sorted({label(c, TRANSLATE) for c in s}))))
             .reset_index())
    outliers = iqr_outliers(d)
    return dict(data=d, hexes=hexes, outliers=outliers)

def build_figure(P: dict) -> go.Figure:
    d, hexes, out = P["data"], P["hexes"], P["outliers"]
    fig = make_subplots(rows=2, cols=2, column_widths=[0.82, 0.18], row_heights=[0.18, 0.82],
                        shared_xaxes="columns", shared_yaxes="rows", horizontal_spacing=0.01, vertical_spacing=0.01)
    logv = np.log10(hexes["mean_installs"]); lo, hi = logv.min(), logv.max()
    t = (logv - lo) / (hi - lo) if hi > lo else logv * 0 + 0.5
    cols = sample_colorscale("Viridis", list(t))
    xs_all, ys_all = [], []
    for (_, h), colr in zip(hexes.iterrows(), cols):                 # hexagon polygons
        px, py = hex_polygon(int(h.q), int(h.r))
        dx, dy = from_unit(px, py)
        xs_all += list(dx); ys_all += list(dy)
        fig.add_shape(type="path", xref="x3", yref="y3",
                      path="M " + " L ".join(f"{a:.4f},{b:.4f}" for a, b in zip(dx, dy)) + " Z",
                      fillcolor=colr, line=dict(color="white", width=1.5), opacity=0.9, layer="below")
    cx, cy = from_unit(*hex_center(hexes["q"].values, hexes["r"].values))
    ticks = [v for v in (1e5, 1e6, 1e7, 1e8) if 10 ** lo * 0.5 <= v <= 10 ** hi * 2]
    fig.add_trace(go.Scatter(                                         # hover + colour bar for hexagons
        x=cx, y=cy, mode="markers", name="Hexagon (avg installs)", showlegend=False,
        marker=dict(size=6, color=logv, colorscale="Viridis", cmin=lo, cmax=hi, line=dict(width=0.5, color="white"),
                    colorbar=dict(title="Average<br>installs", tickvals=[np.log10(v) for v in ticks],
                                  ticktext=[fmt_installs(v) for v in ticks], x=1.03, len=0.75, y=0.4)),
        customdata=np.column_stack([hexes["n"], hexes["mean_installs"].map(fmt_installs),
                                    hexes["mean_rating"].round(2), hexes["mean_size"].round(1), hexes["cats"]]),
        hovertemplate=("<b>Hexagon</b><br>Apps: %{customdata[0]}<br>Average installs: %{customdata[1]}"
                       "<br>Avg rating: %{customdata[2]}<br>Avg size: %{customdata[3]} MB"
                       "<br>Categories: %{customdata[4]}<extra></extra>")), row=2, col=1)

    def hover(frame):
        return [f"<b>{a}</b><br>{label(c, TRANSLATE)}<br>Rating {r} | {s:.0f} MB<br>Installs {fmt_installs(i)}"
                f"<br>Subjectivity {sj:.2f}" for a, c, r, s, i, sj in
                zip(frame.App, frame.Category, frame.Rating, frame.Size_MB, frame.Installs, frame.Avg_Subjectivity)]
    other, game = d[d["Category"] != "GAME"], d[d["Category"] == "GAME"]
    for frame, name, colr in ((other, "Other apps", "#444444"), (game, "Game apps", PINK)):
        fig.add_trace(go.Scatter(x=frame.Size_MB, y=frame.Rating, mode="markers", name=name,
                                 marker=dict(size=10, color=colr, line=dict(width=1, color="white")),
                                 hovertext=hover(frame), hoverinfo="text"), row=2, col=1)
    if len(out):                                                      # IQR outliers, labelled
        fig.add_trace(go.Scatter(
            x=out.Size_MB, y=out.Rating, mode="markers+text", name="IQR outliers (category-level)",
            text=[f"{a} ({label(c, TRANSLATE)})" for a, c in zip(out.App, out.Category)], textposition="top center",
            textfont=dict(size=10),
            marker=dict(size=15, symbol="diamond", color=[PINK if c == "GAME" else "#444444" for c in out.Category],
                        line=dict(width=2.5, color="black")),
            hovertext=[h + f"<br><b>Outlier: {r}</b>" for h, r in zip(hover(out), out.reason)], hoverinfo="text"),
            row=2, col=1)
    for frame, name, colr in ((other, "Other apps", "#777777"), (game, "Game apps", PINK)):   # marginals
        fig.add_trace(go.Histogram(x=frame.Size_MB, name=name, marker_color=colr, showlegend=False, nbinsx=12,
                                   xbins=dict(start=10, end=100)), row=1, col=1)
        fig.add_trace(go.Histogram(y=frame.Rating, name=name, marker_color=colr, showlegend=False, nbinsy=8,
                                   ybins=dict(start=3.5, end=5.0)), row=2, col=2)
    fig.update_layout(
        barmode="stack", template="plotly_white", height=760, hovermode="closest",
        title="Hexbin density - app size vs rating (colour = average installs per hexagon, log scale)"
              f"<br><sup>{len(d)} apps | Game apps in pink | ◆ = IQR outliers (categories with ≥{MIN_APPS_FOR_IQR} apps)"
              " | Business: வணிகம், Dating: Partnersuche (Beauty: सौंदर्य if present)</sup>",
        legend=dict(orientation="h", y=-0.1, x=0.4), margin=dict(r=110, t=110))
    pad = 0.6
    fig.update_xaxes(title="App size (MB)", range=[min(xs_all) - 2, max(xs_all) + 2], row=2, col=1)
    fig.update_yaxes(title="Rating", range=[min(ys_all) - 0.05, max(ys_all) + 0.05], row=2, col=1)
    fig.update_yaxes(title="Count", row=1, col=1); fig.update_xaxes(title="Count", row=2, col=2)
    return fig

if __name__ == "__main__":
    P = prepare(load_apps())
    build_figure(P).write_html("task1_hexbin.html")
    print(len(P["data"]), "apps,", len(P["hexes"]), "hexagons,", len(P["outliers"]), "outliers")
    print(P["outliers"][["App", "Category", "reason"]].to_string())
