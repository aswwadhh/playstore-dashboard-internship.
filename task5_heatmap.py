"""Task 5 - interactive clustered heatmap of the top 10 categories."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import zscore

from common import pretty, load_apps, fmt_installs

METRICS = ["Weighted rating", "Total reviews", "Total installs", "Avg size (MB)",
           "Engagement rate (%)", "Update frequency"]
# composite-score weights (sum = 1); the task did not specify them, so they are documented choices
DEFAULT_WEIGHTS = {"Weighted rating": 0.25, "Total reviews": 0.15, "Total installs": 0.20,
                   "Avg size (MB)": 0.10, "Engagement rate (%)": 0.20, "Update frequency": 0.10}

def filter_apps(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["Rating"] >= 4.0) & (df["Size_MB"] > 10) & (df["Installs"] >= 10_000)
        & (df["Reviews"] > 1000) & (df["Update_Month_Num"] == 1)          # last updated in January
        & ~df["App"].str.contains(r"\d")
    ].copy()

def category_metrics(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("Category")
    m = pd.DataFrame({
        "Weighted rating": g.apply(lambda x: (x["Rating"] * x["Reviews"]).sum() / x["Reviews"].sum(), include_groups=False),
        "Total reviews": g["Reviews"].sum(),
        "Total installs": g["Installs"].sum(),
        "Avg size (MB)": g["Size_MB"].mean(),
        "Engagement rate (%)": g["Reviews"].sum() / g["Installs"].sum() * 100,
        # each app has a single update date, so frequency = number of apps updated in January
        "Update frequency": g["App"].count(),
    })
    m["n_apps"] = g["App"].count()
    return m

def composite(Z: pd.DataFrame, weights: dict) -> pd.Series:
    w = pd.Series(weights)[METRICS]
    return (Z[METRICS] * (w / w.sum())).sum(axis=1)

def prepare(df: pd.DataFrame, weights: dict | None = None) -> dict:
    weights = weights or DEFAULT_WEIGHTS
    d = filter_apps(df)
    top = d.groupby("Category")["Installs"].sum().nlargest(10).index
    d = d[d["Category"].isin(top)]
    raw = category_metrics(d)
    Z = pd.DataFrame(zscore(raw[METRICS], ddof=0), index=raw.index, columns=METRICS)   # Z-score normalisation
    score = composite(Z, weights).sort_values(ascending=False)
    rank = pd.Series(range(1, len(score) + 1), index=score.index)
    link = linkage(Z.values, method="ward")                                           # hierarchical clustering
    dn = dendrogram(link, no_plot=True)
    order = [raw.index[i] for i in dn["leaves"]]
    return dict(data=d, raw=raw, Z=Z, score=score, rank=rank, link=link, dendro=dn, order=order,
                weights=weights, high=score.index[:3].tolist(), low=score.index[-3:].tolist())

def _fmt(metric, v):
    if metric in ("Total reviews", "Total installs"):
        return fmt_installs(v)
    return f"{v:.2f}" if metric != "Update frequency" else f"{int(v)}"

def build_figure(P: dict) -> go.Figure:
    order, raw, Z, n = P["order"], P["raw"], P["Z"], len(P["order"])
    zmat = Z.loc[order, METRICS]; rmat = raw.loc[order, METRICS]
    colscaled = (rmat - rmat.min()) / (rmat.max() - rmat.min()).replace(0, 1)       # raw view colours
    ylab = [f"#{P['rank'][c]} {pretty(c)} (n={int(raw.loc[c, 'n_apps'])})" for c in order]
    custom = np.empty((n, len(METRICS), 3), dtype=object)
    for i, c in enumerate(order):
        for j, m in enumerate(METRICS):
            custom[i, j] = [pretty(c), _fmt(m, rmat.iloc[i, j]), f"{zmat.iloc[i, j]:.2f}"]
    hover = ("<b>%{customdata[0]}</b><br>%{x}<br>Raw value: %{customdata[1]}"
             "<br>Z-score: %{customdata[2]}<extra></extra>")

    fig = make_subplots(rows=1, cols=2, column_widths=[0.13, 0.87], shared_yaxes=True,
                        horizontal_spacing=0.17)
    dn = P["dendro"]                                                                  # dendrogram on the left
    xs, ys = [], []
    for ic, dc in zip(dn["icoord"], dn["dcoord"]):
        xs += list(dc) + [None]; ys += [(v - 5) / 10 for v in ic] + [None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#555", width=1.5),
                             hoverinfo="skip", showlegend=False), row=1, col=1)
    text_norm = [[f"{v:.2f}" for v in row] for row in zmat.values]
    text_raw = [[_fmt(m, v) for m, v in zip(METRICS, row)] for row in rmat.values]
    fig.add_trace(go.Heatmap(z=zmat.values, x=METRICS, y=list(range(n)), text=text_norm, texttemplate="%{text}",
                             colorscale="RdYlGn", zmin=-2.5, zmax=2.5, zmid=0, customdata=custom,
                             hovertemplate=hover, visible=True, name="Normalised",
                             colorbar=dict(title="Z-score", orientation="h", y=-0.22, len=0.5, thickness=12)),
                  row=1, col=2)
    fig.add_trace(go.Heatmap(z=colscaled.values, x=METRICS, y=list(range(n)), text=text_raw, texttemplate="%{text}",
                             colorscale="Blues", zmin=0, zmax=1, customdata=custom, hovertemplate=hover,
                             visible=False, name="Raw",
                             colorbar=dict(title="Column-scaled (0-1)", orientation="h", y=-0.22, len=0.5, thickness=12)),
                  row=1, col=2)

    for i, c in enumerate(order):                                                     # annotate top / bottom 3
        if c in P["high"] or c in P["low"]:
            top = c in P["high"]
            fig.add_annotation(xref="paper", yref="y2", x=1.01, y=i, xanchor="left", showarrow=False,
                               text=f"{'▲ HIGH' if top else '▼ LOW'} #{P['rank'][c]}: score {P['score'][c]:.2f}",
                               font=dict(color="#116611" if top else "#aa1111", size=12),
                               bgcolor="#e6f5e6" if top else "#fbe5e5",
                               bordercolor="#116611" if top else "#aa1111", borderpad=3)
    fig.update_yaxes(tickvals=list(range(n)), ticktext=ylab, range=[-0.5, n - 0.5], showticklabels=True,
                     tickfont=dict(size=12), row=1, col=2, side="left")
    fig.update_yaxes(showticklabels=False, range=[-0.5, n - 0.5], row=1, col=1)
    fig.update_xaxes(autorange="reversed", showticklabels=False, title="Cluster distance (Ward)", row=1, col=1)
    fig.update_xaxes(side="top", row=1, col=2)
    fig.update_layout(
        title="Clustered heatmap - top 10 categories (apps last updated in January)"
              f"<br><sup>{len(P['data'])} apps | rows ordered by hierarchical clustering | #rank = composite score "
              "| n = apps in category (small n = unreliable)</sup>",
        height=720, template="plotly_white", margin=dict(l=10, r=190, t=190, b=130),
        updatemenus=[dict(type="buttons", direction="right", x=0.0, y=1.1, xanchor="left", yanchor="bottom", showactive=True,
                          buttons=[dict(label="Normalised (Z-score)", method="update",
                                        args=[{"visible": [True, True, False]}]),
                                   dict(label="Raw values", method="update",
                                        args=[{"visible": [True, False, True]}])])])
    return fig

if __name__ == "__main__":
    P = prepare(load_apps())
    build_figure(P).write_html("task5_heatmap.html")
    print(P["score"].round(3).to_dict()); print("order", P["order"])
