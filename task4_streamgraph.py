"""Task 4 - interactive streamgraph of monthly / cumulative installs and growth %."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from common import label, fmt_installs, load_apps

TRANSLATE = {"TRAVEL_AND_LOCAL", "PRODUCTIVITY", "PHOTOGRAPHY"}   # French, Spanish, Japanese
GROWTH_THRESHOLD = 25.0     # percent
Z_THRESHOLD = 2.0
Z_WINDOW = 6                # trailing months used for the rolling Z-score
MODES = {"monthly": "Monthly installs", "cumulative": "Cumulative installs", "growth": "Growth % (MoM)"}
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#17becf", "#bcbd22"]

def filter_apps(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        df["Category"].str[0].isin(list("TPB"))
        & (df["Rating"] >= 4.2)
        & (df["Reviews"] > 1000)
        & df["Size_MB"].between(20, 80)
        & (df["Installs"] >= 10_000)
        & ~df["App"].str.contains(r"\d")
    ].copy()

def prepare(df: pd.DataFrame) -> dict:
    d = filter_apps(df)
    d["month"] = d["Last_Updated"].dt.to_period("M")
    monthly = d.pivot_table(index="month", columns="Category", values="Installs", aggfunc="sum")
    reviews = d.pivot_table(index="month", columns="Category", values="Reviews", aggfunc="sum")
    apps_n = d.pivot_table(index="month", columns="Category", values="App", aggfunc="count")
    # complete monthly timeline; a month with no updated apps in a category has 0 installs
    full = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
    monthly = monthly.reindex(full).fillna(0.0)
    reviews = reviews.reindex(full).fillna(0.0)
    apps_n = apps_n.reindex(full).fillna(0).astype(int)
    cumulative = monthly.cumsum()

    prev = monthly.shift(1)
    growth = (monthly - prev) / prev.replace(0, np.nan) * 100        # NaN when previous month is 0
    new_activity = (prev == 0) & (monthly > 0)                          # growth from zero = +inf
    growth_eff = growth.mask(new_activity, np.inf)

    hist = monthly.shift(1).rolling(Z_WINDOW, min_periods=3)            # trailing months only
    z = (monthly - hist.mean()) / hist.std().replace(0, np.nan)
    anomaly = (growth_eff > GROWTH_THRESHOLD) & (z > Z_THRESHOLD)

    order = monthly.sum().sort_values(ascending=False).index.tolist()
    return dict(filtered=d, monthly=monthly[order], cumulative=cumulative[order],
                growth=growth[order], growth_eff=growth_eff[order], z=z[order],
                anomaly=anomaly[order], new_activity=new_activity[order],
                reviews=reviews[order], apps_n=apps_n[order], order=order, months=full)

def _growth_text(g, new):
    if new:
        return "new activity (previous month = 0)"
    return "n/a" if pd.isna(g) else f"{g:+.1f}%"

def build_figure(P: dict, mode: str = "monthly") -> go.Figure:
    mode = mode if mode in MODES else "monthly"
    cats, months = P["order"], P["months"]
    x = months.to_timestamp()
    color = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(cats)}
    fig = go.Figure()

    def hover_for(c):
        i = cats.index(c)
        return [
            f"<b>{label(c, TRANSLATE)}</b> - {m.strftime('%b %Y')}"
            f"<br>Monthly: {fmt_installs(P['monthly'][c].iloc[k])} | Cumulative: {fmt_installs(P['cumulative'][c].iloc[k])}"
            f"<br>MoM growth: {_growth_text(P['growth'][c].iloc[k], P['new_activity'][c].iloc[k])}"
            f" | Z: {'n/a' if pd.isna(P['z'][c].iloc[k]) else round(P['z'][c].iloc[k], 2)}"
            f"<br>Status: {'<b>ANOMALY</b>' if P['anomaly'][c].iloc[k] else ('no data this month' if P['monthly'][c].iloc[k] == 0 else 'normal')}"
            for k, m in enumerate(x)
        ]

    if mode in ("monthly", "cumulative"):
        vals = P["monthly"] if mode == "monthly" else P["cumulative"]
        base = -vals.sum(axis=1) / 2                                    # symmetric (ThemeRiver) baseline
        lower = {}; upper = {}; run = base.copy()
        for c in cats:
            lower[c] = run.copy(); run = run + vals[c]; upper[c] = run.copy()
        for c in cats:
            xs = list(x) + list(x[::-1])
            ys = list(upper[c]) + list(lower[c][::-1])
            fig.add_trace(go.Scatter(x=xs, y=ys, fill="toself", mode="lines", name=label(c, TRANSLATE),
                                     line=dict(width=0.5, color="white"), fillcolor=color[c],
                                     opacity=0.55, hoverinfo="skip", legendgroup=c))
            mid = (upper[c] + lower[c]) / 2                             # invisible hover anchors
            fig.add_trace(go.Scatter(x=x, y=mid, mode="markers", marker=dict(size=18, opacity=0),
                                     showlegend=False, legendgroup=c, customdata=hover_for(c),
                                     hovertemplate="%{customdata}<extra></extra>"))
            # anomalies: stronger colour on the spike (previous, current, next month)
            px, py = [], []
            for k in np.where(P["anomaly"][c].values)[0]:
                lo, hi = max(k - 1, 0), min(k + 1, len(x) - 1)
                idx = list(range(lo, hi + 1))
                px += [x[j] for j in idx] + [x[j] for j in idx[::-1]] + [None]
                py += [upper[c].iloc[j] for j in idx] + [lower[c].iloc[j] for j in idx[::-1]] + [None]
            if px:
                fig.add_trace(go.Scatter(x=px, y=py, fill="toself", mode="lines", fillcolor=color[c],
                                         opacity=1, line=dict(width=1.2, color="black"),
                                         hoverinfo="skip", showlegend=False, legendgroup=c))
        # annotate the 8 largest anomalies (all anomalies are coloured and listed in hover)
        an = [(P["monthly"][c].iloc[k], c, k) for c in cats for k in np.where(P["anomaly"][c].values)[0]]
        for v, c, k in sorted(an, reverse=True)[:8]:
            y = (upper[c].iloc[k] + lower[c].iloc[k]) / 2
            fig.add_annotation(x=x[k], y=y, text=f"▲ {label(c, TRANSLATE)}<br>{fmt_installs(v)}",
                               showarrow=True, arrowhead=2, ax=0, ay=-45, font=dict(size=10),
                               bgcolor="rgba(255,255,255,0.85)", bordercolor="black")
        fig.update_yaxes(title="Installs (stream centred on zero)", showticklabels=False, zeroline=False)
    else:  # growth %: separate lines (stacking percentages would be misleading)
        for c in cats:
            g = P["growth"][c].copy()
            fig.add_trace(go.Scatter(x=x, y=g, mode="lines+markers", name=label(c, TRANSLATE),
                                     line=dict(color=color[c], width=1.5), marker=dict(size=6),
                                     connectgaps=False, customdata=hover_for(c),
                                     hovertemplate="%{customdata}<extra></extra>", legendgroup=c))
            ak = np.where(P["anomaly"][c].values & g.notna().values)[0]
            if len(ak):
                fig.add_trace(go.Scatter(x=x[ak], y=g.iloc[ak], mode="markers", showlegend=False,
                                         marker=dict(size=13, symbol="diamond", color=color[c],
                                                     line=dict(width=2, color="black")),
                                         hoverinfo="skip", legendgroup=c))
        fig.add_hline(y=GROWTH_THRESHOLD, line_dash="dash", line_color="grey",
                      annotation_text="25% threshold")
        fig.update_yaxes(title="Month-over-month growth (%)")
        fig.add_annotation(xref="paper", yref="paper", x=0, y=-0.32, showarrow=False, align="left",
                           font=dict(size=10),
                           text="Growth from a month with 0 installs is undefined and is not plotted "
                                "(shown as 'new activity' in the hover).")

    n_an = int(P["anomaly"].values.sum())
    fig.update_layout(
        title=f"Streamgraph - {MODES[mode]} by category (categories starting with T, P, B)"
              f"<br><sup>{len(P['filtered'])} apps | {n_an} anomalous category-months "
              f"(growth > {GROWTH_THRESHOLD:.0f}% AND rolling Z > {Z_THRESHOLD:.0f}) shown in strong colour with black outline</sup>",
        hovermode="closest", hoverdistance=30, height=640, legend_title="Category",
        xaxis=dict(title="Month (of last update)", rangeslider=dict(visible=True)),
        template="plotly_white", margin=dict(b=90))
    return fig

if __name__ == "__main__":
    P = prepare(load_apps())
    for m in MODES:
        build_figure(P, m).write_html(f"task4_{m}.html")
    print("written")
