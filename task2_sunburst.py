"""Task 2 - hierarchical sunburst: Country > Category > App Type > Rating Band."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from common import label, fmt_installs, load_apps

TRANSLATE = {"BUSINESS", "TRAVEL_AND_LOCAL", "PRODUCTIVITY"}      # Tamil, French, Spanish
BANDS = [(4.0, 4.2, "4.0–4.2"), (4.2, 4.5, "4.2–4.5"), (4.5, 4.7, "4.5–4.7"), (4.7, 5.0, "4.7–5.0")]
MILLION = 1_000_000
NO_COUNTRY = "Global (no country field in dataset)"

def rating_band(r: float) -> str:
    """Bands are [low, high); the last band also includes 5.0."""
    for lo, hi, name in BANDS:
        if lo <= r < hi or (hi == 5.0 and r == 5.0):
            return name
    return None

def filter_apps(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["Rating"] >= 4.0) & (df["Installs"] > 10_000) & (df["Reviews"] > 1000)
        & df["Size_MB"].between(15, 80) & ~df["App"].str.contains(r"\d")
        & ~df["Category"].str[0].isin(list("ACGS"))
    ].copy()

def prepare(df: pd.DataFrame) -> dict:
    d = filter_apps(df)
    top5 = d.groupby("Category")["Installs"].sum().nlargest(5).index.tolist()   # global installs
    d = d[d["Category"].isin(top5)].copy()
    # the public dataset has no country; use a real Country column if one is ever supplied
    d["Country"] = d["Country"] if "Country" in d.columns else NO_COUNTRY
    d["Band"] = d["Rating"].map(rating_band)
    d["RxR"] = d["Rating"] * d["Reviews"]

    levels = ["Country", "Category", "Type", "Band"]
    frames = []
    for depth in range(1, 5):
        cols = levels[:depth]
        g = d.groupby(cols).agg(installs=("Installs", "sum"), reviews=("Reviews", "sum"),
                                rxr=("RxR", "sum"), apps=("App", "count")).reset_index()
        g["id"] = g[cols].astype(str).agg("|".join, axis=1)
        g["parent_id"] = g[cols[:-1]].astype(str).agg("|".join, axis=1) if depth > 1 else ""
        g["name"] = g[cols[-1]]
        g["level"] = depth
        frames.append(g[["id", "parent_id", "name", "level", "installs", "reviews", "rxr", "apps"]])
    nodes = pd.concat(frames, ignore_index=True)
    nodes["wrating"] = nodes["rxr"] / nodes["reviews"]                      # weighted by review count
    parent_inst = nodes.set_index("id")["installs"]
    nodes["pct_parent"] = np.where(nodes["parent_id"] == "", 100.0,
                                   nodes["installs"] / nodes["parent_id"].map(parent_inst) * 100)
    nodes["big"] = nodes["installs"] > MILLION                              # highlighted segments
    nodes["label"] = [label(n, TRANSLATE) if lv == 2 else str(n) for n, lv in zip(nodes["name"], nodes["level"])]
    return dict(data=d, top5=top5, nodes=nodes)

def build_figure(P: dict) -> go.Figure:
    n = P["nodes"]
    labels = [("★ " if b else "") + l for l, b in zip(n["label"], n["big"])]
    custom = np.column_stack([
        [fmt_installs(v) + f" ({int(v):,})" for v in n["installs"]],
        n["wrating"].round(3), n["reviews"].astype(int).map("{:,}".format),
        n["pct_parent"].round(1), n["apps"],
        np.where(n["big"], "Yes (> 1 million installs)", "No")])
    fig = go.Figure(go.Sunburst(
        ids=n["id"], parents=n["parent_id"], labels=labels, values=n["installs"],
        branchvalues="total", customdata=custom, maxdepth=-1,
        marker=dict(colors=n["wrating"], colorscale="RdYlGn", cmin=n["wrating"].min(), cmax=n["wrating"].max(),
                    colorbar=dict(title="Weighted<br>rating"),
                    line=dict(color=np.where(n["big"], "#111111", "#ffffff"),
                              width=np.where(n["big"], 3, 0.8))),
        hovertemplate=("<b>%{label}</b><br>Installs: %{customdata[0]}"
                       "<br>Weighted rating: %{customdata[1]}"
                       "<br>Review count: %{customdata[2]}"
                       "<br>Contribution to parent: %{customdata[3]}%"
                       "<br>Apps: %{customdata[4]}<br>Over 1M installs: %{customdata[5]}<extra></extra>"),
        insidetextorientation="radial"))
    fig.update_layout(
        title="Installs by Country → Category → App Type → Rating Band"
              f"<br><sup>Size = total installs | Colour = review-weighted rating | ★ and black border = over 1M installs"
              f" | {len(P['data'])} apps, top 5 categories | click a segment to drill down, centre to go back</sup>",
        height=720, margin=dict(t=110, l=10, r=10, b=10), template="plotly_white")
    return fig

if __name__ == "__main__":
    P = prepare(load_apps())
    build_figure(P).write_html("task2_sunburst.html")
    print("written", P["top5"], len(P["nodes"]), "nodes")
