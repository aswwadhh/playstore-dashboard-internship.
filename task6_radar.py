"""Task 6 - interactive radar chart: free vs paid apps (percentile-normalised)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import percentileofscore

from common import pretty, load_apps, fmt_installs

METRICS = ["Avg installs", "Weighted rating", "Total reviews", "Avg size (MB)", "Revenue ($)", "Engagement rate (%)"]
COLORS = {"Free": "#1f77b4", "Paid": "#ff7f0e"}
OVERALL = "Overall (top 5 categories)"

def filter_apps(df: pd.DataFrame) -> pd.DataFrame:
    d = df[
        (df["Installs"] >= 10_000)
        & (df["Android_Ver"] > 4.0)
        & (df["Size_MB"] > 15)
        & (df["Content Rating"] == "Everyone")
        & (df["App"].str.len() <= 30)
    ].copy()
    d["Revenue"] = np.where(d["Type"] == "Paid", d["Installs"] * d["Price"], 0.0)   # Installs x Price
    paid_ok = (d["Type"] == "Free") | ((d["Type"] == "Paid") & (d["Revenue"] > 10_000))
    return d[paid_ok]

def group_metrics(g: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "n_apps": len(g),
        "Avg installs": g["Installs"].mean(),
        "Weighted rating": (g["Rating"] * g["Reviews"]).sum() / g["Reviews"].sum(),   # weighted by reviews
        "Total reviews": g["Reviews"].sum(),
        "Avg size (MB)": g["Size_MB"].mean(),
        "Revenue ($)": g["Revenue"].sum(),
        "Engagement rate (%)": g["Reviews"].sum() / g["Installs"].sum() * 100,        # Reviews / Installs x 100
    })

def prepare(df: pd.DataFrame) -> dict:
    d = filter_apps(df)
    top5 = d.groupby("Category")["Installs"].sum().nlargest(5).index.tolist()
    d = d[d["Category"].isin(top5)]
    raw = {}                                            # view -> {type -> metrics}
    raw[OVERALL] = {t: group_metrics(g) for t, g in d.groupby("Type")}
    for c in top5:
        raw[c] = {t: group_metrics(g) for t, g in d[d["Category"] == c].groupby("Type")}
    # reference pool for percentiles = every category x type group of the top five
    pool = pd.DataFrame([m for c in top5 for m in raw[c].values()])
    norm, score = {}, {}
    for view, types in raw.items():
        norm[view], score[view] = {}, {}
        for t, m in types.items():
            n = pd.Series({k: percentileofscore(pool[k], m[k], kind="mean") for k in METRICS})
            norm[view][t] = n
            score[view][t] = n.mean()                   # composite = equal-weight mean of 6 percentiles
    return dict(data=d, top5=top5, raw=raw, norm=norm, score=score, pool=pool)

def verdict(P: dict, view: str) -> str:
    s = P["score"][view]
    if len(s) < 2:
        only = next(iter(s))
        return f"No {'Paid' if only == 'Free' else 'Free'} apps meet the filters here, so no comparison is possible."
    win = max(s, key=s.get); lose = "Paid" if win == "Free" else "Free"
    thin = [t for t in s if P["raw"][view][t]["n_apps"] < 5]
    warn = f"  (caution: only {int(P['raw'][view][thin[0]]['n_apps'])} {thin[0]} apps)" if thin else ""
    return (f"<b>Better performer: {win} apps</b> - composite score {s[win]:.1f} vs {s[lose]:.1f} for {lose}{warn}")

def _fmt(metric, v):
    if metric in ("Avg installs", "Total reviews"):
        return fmt_installs(v)
    if metric == "Revenue ($)":
        return "$" + fmt_installs(v)
    return f"{v:.2f}"

def build_figure(P: dict) -> go.Figure:
    views = [OVERALL] + P["top5"]
    fig, owner = go.Figure(), []
    theta = METRICS + [METRICS[0]]
    for vi, view in enumerate(views):
        for t in ("Free", "Paid"):
            if t in P["norm"][view]:
                n, m = P["norm"][view][t], P["raw"][view][t]
                r = [n[k] for k in METRICS] + [n[METRICS[0]]]
                custom = [f"{_fmt(k, m[k])}" for k in METRICS] + [_fmt(METRICS[0], m[METRICS[0]])]
                fig.add_trace(go.Scatterpolar(
                    r=r, theta=theta, name=f"{t} (n={int(m['n_apps'])})", fill="toself",
                    line=dict(color=COLORS[t], width=2), fillcolor=COLORS[t], opacity=0.45,
                    customdata=custom, visible=(vi == 0),
                    hovertemplate=f"<b>{t}</b> - {pretty(view) if view != OVERALL else view}<br>"
                                  "%{theta}<br>Percentile score: %{r:.1f}<br>Actual value: %{customdata}"
                                  f"<br>Composite score: {P['score'][view][t]:.1f}<extra></extra>"))
                owner.append(vi)

    def annotation(view):
        return [dict(xref="paper", yref="paper", x=0.5, y=-0.12, showarrow=False, align="center",
                     text=verdict(P, view), font=dict(size=14), bgcolor="rgba(255,243,205,0.9)",
                     bordercolor="#c9a227", borderpad=6)]

    title = lambda view: (f"Free vs Paid apps - {view if view == OVERALL else pretty(view)}"
                          "<br><sup>Percentile-normalised (0-100); hover for actual values</sup>")
    buttons = []
    for vi, view in enumerate(views):
        buttons.append(dict(label=view if view == OVERALL else pretty(view), method="update",
                            args=[{"visible": [o == vi for o in owner]},
                                  {"annotations": annotation(view), "title": title(view)}]))
    fig.update_layout(
        title=title(OVERALL), annotations=annotation(OVERALL), template="plotly_white", height=680,
        polar=dict(radialaxis=dict(range=[0, 100], tickvals=[20, 40, 60, 80, 100]),
                   angularaxis=dict(direction="clockwise")),
        legend=dict(orientation="v", x=1.0, y=0.95, xanchor="right"),
        updatemenus=[dict(buttons=buttons, direction="down", x=0.0, y=1.12, xanchor="left",
                          showactive=True)],
        margin=dict(t=130, b=110))
    return fig

if __name__ == "__main__":
    P = prepare(load_apps())
    build_figure(P).write_html("task6_radar.html")
    print("written", P["top5"])
