"""Google Play Store analytics - internship dashboard (Dash).

Each chart is shown only inside its IST window. Outside the window the chart is never sent to the
browser (the server returns an empty slot), so it does not appear on the dashboard at all.
Set the environment variable DEMO_MODE=1 on your own computer to show every chart (for screenshots / video).
"""
import os
from datetime import datetime, timedelta

from dash import Dash, Input, Output, State, dcc, html, no_update
from dash.exceptions import PreventUpdate

import task1_hexbin as t1
import task2_sunburst as t2
import task3_calendar as t3
import task4_streamgraph as t4
import task5_heatmap as t5
import task6_radar as t6
from common import IST, WINDOWS, is_visible, load_apps, now_ist, window_label

DEMO = os.environ.get("DEMO_MODE") == "1"

def current_time() -> datetime:                       # one place to read the clock (tests replace it)
    return now_ist()

# ---- data and figures are prepared once at start-up -------------------------------------------------
DF = load_apps()
P1, P2, P3, P4, P5, P6 = (t1.prepare(DF), t2.prepare(DF), t3.prepare(DF),
                          t4.prepare(DF), t5.prepare(DF), t6.prepare(DF))
FIG = {1: t1.build_figure(P1), 2: t2.build_figure(P2), 3: t3.build_figure(P3),
       4: t4.build_figure(P4, "monthly"), 5: t5.build_figure(P5), 6: t6.build_figure(P6)}
TITLES = {1: "Hexbin density - size vs rating", 2: "Sunburst - installs hierarchy", 3: "Calendar heatmap - monthly installs",
          4: "Streamgraph - monthly / cumulative installs", 5: "Clustered heatmap - top 10 categories",
          6: "Radar - free vs paid apps"}
NOTES = {
    1: f"{len(P1['data'])} apps pass every filter. Game apps are pink; diamonds are IQR outliers (size, rating, installs).",
    2: f"{len(P2['data'])} apps in the top 5 categories. The dataset has no country field, so the top ring is a single 'Global' node.",
    3: f"{len(P3['data'])} apps in {len(P3['cats'])} categories pass the filters (fewer than five exist). Orange cells are forecasts.",
    4: f"{len(P4['filtered'])} apps; empty months are filled with 0 installs. Growth from a 0-install month is undefined.",
    5: f"{len(P5['data'])} apps updated in January. Several categories have only 1-3 apps, so treat clusters with care.",
    6: f"{len(P6['data'])} apps (free and paid). Entertainment has no qualifying paid apps; Tools and Health & Fitness have 1-2.",
}

# ---- pure helper functions (unit-tested) ------------------------------------------------------------
def visible_tasks(now: datetime | None = None) -> list[int]:
    now = now or current_time()
    return [t for t in sorted(WINDOWS) if is_visible(t, now, test_mode=DEMO)]

def next_window(now: datetime) -> tuple[int, datetime]:
    best = None
    for t, (start, _) in WINDOWS.items():
        cand = now.replace(hour=start, minute=0, second=0, microsecond=0)
        if cand <= now:
            cand += timedelta(days=1)
        if best is None or cand < best[1]:
            best = (t, cand)
    return best

def schedule_rows(now: datetime) -> list[dict]:
    vis = set(visible_tasks(now))
    return [dict(task=t, chart=TITLES[t], window=window_label(t), visible=t in vis) for t in sorted(WINDOWS)]

def section(task: int):
    controls = []
    if task == 4:
        controls = [html.Label("Show: "), dcc.RadioItems(
            id="t4-mode", value="monthly", inline=True,
            options=[{"label": f" {v}  ", "value": k} for k, v in t4.MODES.items()])]
    if task == 5:
        controls = [html.Div("Composite-score weights (the ranking updates as you move a slider):",
                             style={"fontWeight": 600, "marginBottom": 4})] + [
            html.Div([html.Span(m, style={"width": 170, "display": "inline-block", "fontSize": 13}),
                      html.Div(dcc.Slider(id=f"w-{i}", min=0, max=1, step=0.05, value=t5.DEFAULT_WEIGHTS[m],
                                          marks=None, tooltip={"placement": "right"}),
                               style={"display": "inline-block", "width": 320, "verticalAlign": "middle"})])
            for i, m in enumerate(t5.METRICS)]
    return html.Div([
        html.H3(f"Task {task}: {TITLES[task]}", style={"marginBottom": 2}),
        html.Div(f"Visible {window_label(task)}" + (" (demo mode)" if DEMO else ""), style={"color": "#666", "fontSize": 13}),
        html.Div(controls, style={"margin": "8px 0"}),
        dcc.Graph(id=f"t{task}-graph", figure=FIG[task], config={"displaylogo": False}),
        html.Div(NOTES[task], style={"fontSize": 13, "color": "#555", "marginTop": 4}),
    ], style={"background": "white", "padding": 16, "borderRadius": 8, "marginBottom": 20,
              "boxShadow": "0 1px 4px rgba(0,0,0,.15)"})

def compute_slots(now: datetime, stored: list | None):
    """Return (list of 6 slot children / no_update, new visible list). Raises PreventUpdate if nothing changed."""
    vis = visible_tasks(now)
    if stored is not None and sorted(stored) == vis:
        raise PreventUpdate
    prev = set(stored or [])
    outs = []
    for t in sorted(WINDOWS):
        if stored is not None and (t in vis) == (t in prev):
            outs.append(no_update)                          # unchanged: keep the user's zoom / selections
        else:
            outs.append(section(t) if t in vis else [])
    return outs, vis

# ---- app --------------------------------------------------------------------------------------------
app = Dash(__name__, suppress_callback_exceptions=True, title="Play Store Analytics")
server = app.server                                          # for WSGI hosting

app.layout = html.Div([
    html.H1("Google Play Store Analytics Dashboard", style={"marginBottom": 0}),
    html.Div(id="clock", style={"fontSize": 18, "margin": "6px 0"}),
    html.Div(id="schedule"),
    html.Div(id="empty-msg"),
    html.Div([html.Div(id=f"slot-{t}") for t in sorted(WINDOWS)]),
    dcc.Store(id="vis-store"),
    dcc.Interval(id="tick", interval=30_000),
], style={"maxWidth": 1250, "margin": "0 auto", "padding": 16, "fontFamily": "Segoe UI, Arial, sans-serif",
          "background": "#f3f4f7"})

@app.callback(Output("clock", "children"), Output("schedule", "children"), Output("empty-msg", "children"),
              Input("tick", "n_intervals"))
def update_clock(_):
    now = current_time().astimezone(IST)
    rows = schedule_rows(now)
    th = {"textAlign": "left", "padding": "4px 12px", "borderBottom": "2px solid #ccc"}
    td = {"padding": "4px 12px", "borderBottom": "1px solid #ddd"}
    table = html.Table([html.Thead(html.Tr([html.Th(h, style=th) for h in ("Task", "Chart", "Visible window (IST)", "Status")])),
                        html.Tbody([html.Tr([html.Td(r["task"], style=td), html.Td(r["chart"], style=td),
                                             html.Td(r["window"], style=td),
                                             html.Td("● Visible now" if r["visible"] else "○ Hidden",
                                                     style={**td, "color": "#0a7d2c" if r["visible"] else "#999", "fontWeight": 600})])
                                    for r in rows])],
                       style={"background": "white", "borderCollapse": "collapse", "marginBottom": 16})
    msg = []
    if not any(r["visible"] for r in rows):
        t, when = next_window(now)
        msg = html.Div(f"No chart is scheduled for this hour. Next: Task {t} ({TITLES[t]}) at {when:%H:%M} IST.",
                       style={"padding": 16, "background": "#fff3cd", "borderRadius": 8, "marginBottom": 16})
    banner = "DEMO MODE - all charts shown" if DEMO else "Charts appear only during their scheduled hour"
    return f"India time now: {now:%A, %d %b %Y  %H:%M:%S} IST  |  {banner}", table, msg

@app.callback([Output(f"slot-{t}", "children") for t in sorted(WINDOWS)] + [Output("vis-store", "data")],
              Input("tick", "n_intervals"), State("vis-store", "data"))
def update_slots(_, stored):
    outs, vis = compute_slots(current_time(), stored)
    return outs + [vis]

@app.callback(Output("t4-graph", "figure"), Input("t4-mode", "value"))
def update_t4(mode):
    return t4.build_figure(P4, mode)

@app.callback(Output("t5-graph", "figure"), [Input(f"w-{i}", "value") for i in range(6)])
def update_t5(*w):
    weights = dict(zip(t5.METRICS, [float(v or 0) for v in w]))
    if sum(weights.values()) == 0:
        weights = t5.DEFAULT_WEIGHTS
    return t5.build_figure(t5.prepare(DF, weights))

if __name__ == "__main__":
    app.run(debug=False, port=int(os.environ.get("PORT", 8050)))
