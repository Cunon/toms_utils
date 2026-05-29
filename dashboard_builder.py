"""
dashboard_builder.py
====================

Interactive tool for authoring a new engine-station dashboard from *your own*
cross-section image and dataframe -- no pixel-coordinate guessing required.

What it does
------------
Opens a Dash app where you:

    1. See your cross-section with a clickable coordinate grid.
    2. Click the image to drop a station (the pixel x/y is captured for you).
    3. Pick which dataframe columns that station should display.
    4. Watch a live preview of the value boxes appear on the image.
    5. Export a ready-to-run script that calls ``create_dashboard`` with the
       ``location_cols`` / ``col_labels`` you built -- or save it straight to
       disk.

Run it
------
From a dataframe in code::

    from dashboard_builder import build_builder_app
    app = build_builder_app(image="engine.png", df=df,
                            label_col="time_s", x_col="time_s")
    app.run(debug=True)

Or from the command line with a CSV::

    python dashboard_builder.py --image engine.png --csv run.csv \\
        --label-col time_s --x-col time_s
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update

from engine_dashboard import THEME, _header, _panel
from plot_engine_stations import ImageLike, _load_image


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------


def _click_grid(iw: int, ih: int) -> go.Scatter:
    """
    A dense, invisible scatter that tiles the image so a click anywhere snaps
    to the nearest grid point -- that point's (x, y) is the pixel coordinate.
    """
    step = max(1, iw // 110)
    xs, ys = [], []
    for x in range(0, iw + 1, step):
        for y in range(0, ih + 1, step):
            xs.append(x)
            ys.append(y)
    return go.Scatter(
        x=xs,
        y=ys,
        mode="markers",
        marker=dict(size=step * 1.8, color="rgba(0,0,0,0)", opacity=0),
        hovertemplate="x: %{x}<br>y: %{y}<extra>click to place</extra>",
        showlegend=False,
        name="grid",
    )


def _station_annotations(row: pd.Series, stations: list, labels: dict) -> list:
    """Value boxes for the preview row, one per station that has columns."""
    anns = []
    for s in stations:
        lines = []
        for col in s["cols"]:
            v = row.get(col)
            if v is None or pd.isna(v):
                continue
            label = labels.get(col, col)
            if isinstance(v, (int, float)):
                lines.append(f"{label}: {v:.4g}")
            else:
                lines.append(f"{label}: {v}")
        if not lines:
            continue
        anns.append(
            dict(
                x=s["x"],
                y=s["y"],
                xref="x",
                yref="y",
                text="<br>".join(lines),
                showarrow=False,
                font=dict(color="white", size=12, family="monospace"),
                align="left",
                bgcolor="rgba(20,20,20,0.78)",
                bordercolor=THEME["accent"],
                borderwidth=1,
                borderpad=4,
                xanchor="center",
                yanchor="middle",
            )
        )
    return anns


def auto_col_labels(stations: list) -> dict:
    """
    Short display label per column for the on-image boxes. The station id is
    prepended to each station's first column so every box self-identifies::

        first column -> "[id] col_name",  others -> "col_name"
    """
    labels = {}
    for s in stations:
        for i, col in enumerate(s["cols"]):
            labels[col] = f"[{s['id']}] {col}" if i == 0 else col
    return labels


def _canvas_figure(
    img,
    iw: int,
    ih: int,
    df: pd.DataFrame,
    stations: list,
    preview_row: int,
) -> go.Figure:
    """Image + click grid + placed-station markers + live value-box preview."""
    fig = go.Figure()
    fig.add_layout_image(
        dict(source=img, xref="x", yref="y", x=0, y=0,
             sizex=iw, sizey=ih, sizing="stretch", layer="below")
    )
    fig.add_trace(_click_grid(iw, ih))

    if stations:
        fig.add_trace(
            go.Scatter(
                x=[s["x"] for s in stations],
                y=[s["y"] for s in stations],
                mode="markers+text",
                marker=dict(size=14, color=THEME["accent"],
                            line=dict(width=2, color="white")),
                text=[f"<b>{s['id']}</b>" for s in stations],
                textposition="top center",
                textfont=dict(color=THEME["accent"], size=11),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    anns = []
    if stations and len(df):
        row = df.iloc[max(0, min(preview_row, len(df) - 1))]
        anns = _station_annotations(row, stations, auto_col_labels(stations))

    fig.update_layout(
        xaxis=dict(range=[0, iw], visible=False),
        yaxis=dict(range=[ih, 0], visible=False, scaleanchor="x", scaleratio=1,
                   constrain="domain"),
        autosize=True,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=anns,
        clickmode="event",
        hoverlabel=dict(bgcolor="rgba(20,20,20,0.95)",
                        font=dict(color="white", family="monospace")),
    )
    return fig


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------


def generate_script(
    stations: list,
    image_path: str,
    df: pd.DataFrame,
    label_col: Optional[str],
    x_col: Optional[str],
    title: str = "My Engine Dashboard",
    csv_path: str = "YOUR_DATA.csv",
    plot_cols: Optional[Sequence[str]] = None,
    plot_layout: str = "right",
) -> str:
    """Render a runnable ``create_dashboard`` script from the built stations."""
    loc_lines = ["location_cols = {"]
    for s in stations:
        cols = ", ".join(repr(c) for c in s["cols"])
        loc_lines.append(f"    ({s['x']}, {s['y']}): [{cols}],")
    loc_lines.append("}")

    labels = auto_col_labels(stations)
    lab_lines = ["col_labels = {"]
    for col, lab in labels.items():
        lab_lines.append(f"    {col!r}: {lab!r},")
    lab_lines.append("}")

    # Trend-plot columns: honour the caller's choice (numeric only); otherwise
    # fall back to the numeric columns the user placed on the image.
    if plot_cols:
        plot_cols = [c for c in plot_cols
                     if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not plot_cols:
        placed = [c for s in stations for c in s["cols"]]
        plot_cols = [c for c in dict.fromkeys(placed)
                     if c in df.columns and pd.api.types.is_numeric_dtype(df[c])][:6]

    return "\n".join([
        '"""Generated by dashboard_builder.py — edit freely."""',
        "import pandas as pd",
        "from engine_dashboard import create_dashboard",
        "",
        "# Load the dataframe you built this against:",
        f"df = pd.read_csv({csv_path!r})",
        "",
        "\n".join(loc_lines),
        "",
        "\n".join(lab_lines),
        "",
        "app = create_dashboard(",
        f"    image={image_path!r},",
        "    df=df,",
        "    location_cols=location_cols,",
        "    col_labels=col_labels,",
        f"    label_col={label_col!r},",
        f"    x_col={x_col!r},",
        f"    plot_cols={list(plot_cols)!r},",
        f"    title={title!r},",
        f"    plot_layout={plot_layout!r},",
        ")",
        "",
        'if __name__ == "__main__":',
        "    app.run(debug=True)",
        "",
    ])


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def build_builder_app(
    image: ImageLike,
    df: pd.DataFrame,
    label_col: Optional[str] = None,
    x_col: Optional[str] = None,
    title: str = "Dashboard Builder",
) -> Dash:
    """Build the interactive station-placement / dashboard-authoring app."""
    img = _load_image(image)
    iw, ih = img.size
    # Path/str inputs become the literal path in the generated script; an
    # in-memory PIL image gets a placeholder for the user to fill in.
    image_path = str(image) if isinstance(image, (str, Path)) else "your_image.png"
    columns = list(df.columns)
    col_options = [{"label": c, "value": c} for c in columns]
    none_opt = [{"label": "(none / row index)", "value": "__none__"}]

    app = Dash(__name__)
    app.title = title

    input_style = {
        "width": "100%", "boxSizing": "border-box", "padding": "6px 8px",
        "backgroundColor": THEME["bg"], "color": THEME["text"],
        "border": f"1px solid {THEME['panel_border']}", "borderRadius": "6px",
    }
    btn_style = {
        "padding": "8px 14px", "borderRadius": "6px", "cursor": "pointer",
        "border": f"1px solid {THEME['panel_border']}", "fontWeight": 600,
        "backgroundColor": THEME["accent"], "color": "#1a1a1a",
    }
    btn_ghost = {**btn_style, "backgroundColor": THEME["bg"],
                 "color": THEME["text"], "fontWeight": 500}

    builder_panel = html.Div(
        style={"display": "flex", "flexDirection": "column", "gap": "12px"},
        children=[
            _panel("1 · Place a station", html.Div([
                html.Div("Click the cross-section, then label the station.",
                         style={"fontSize": "12px", "color": THEME["text_muted"],
                                "marginBottom": "10px"}),
                html.Div(style={"display": "flex", "gap": "8px"}, children=[
                    html.Div(["x", dcc.Input(id="coord-x", type="number",
                              style=input_style)], style={"flex": 1}),
                    html.Div(["y", dcc.Input(id="coord-y", type="number",
                              style=input_style)], style={"flex": 1}),
                ]),
                html.Div("Station id / label", style={"fontSize": "11px",
                         "color": THEME["text_muted"], "margin": "10px 0 2px"}),
                dcc.Input(id="station-id", type="text", placeholder="e.g. 4",
                          style=input_style),
                html.Div("Columns to show here", style={"fontSize": "11px",
                         "color": THEME["text_muted"], "margin": "10px 0 2px"}),
                dcc.Dropdown(id="station-cols", options=col_options, multi=True,
                             placeholder="select columns…",
                             style={"color": "#111"}),
                html.Button("➕ Add station", id="add-btn", n_clicks=0,
                            style={**btn_style, "marginTop": "12px",
                                   "width": "100%"}),
                html.Div(id="add-status", style={"fontSize": "11px",
                         "color": THEME["accent_blue"], "marginTop": "6px",
                         "minHeight": "14px"}),
            ])),
            _panel("2 · Placed stations", html.Div([
                html.Div(id="station-list", style={"fontSize": "12px",
                         "fontFamily": "monospace", "marginBottom": "10px"}),
                html.Div(style={"display": "flex", "gap": "8px"}, children=[
                    dcc.Dropdown(id="remove-pick", placeholder="pick to remove",
                                 style={"flex": 1, "color": "#111"}),
                    html.Button("Remove", id="remove-btn", n_clicks=0,
                                style=btn_ghost),
                ]),
                html.Button("Clear all", id="clear-btn", n_clicks=0,
                            style={**btn_ghost, "marginTop": "8px",
                                   "width": "100%"}),
                html.Div("Preview row", style={"fontSize": "11px",
                         "color": THEME["text_muted"], "margin": "12px 0 2px"}),
                dcc.Slider(id="preview-row", min=0, max=max(0, len(df) - 1),
                           step=1, value=0,
                           tooltip={"placement": "bottom"}),
            ])),
            _panel("3 · Export", html.Div([
                html.Div(style={"display": "flex", "gap": "8px"}, children=[
                    html.Div(["label_col", dcc.Dropdown(
                        id="label-col", options=col_options + none_opt,
                        value=label_col or "__none__", clearable=False,
                        style={"color": "#111"})], style={"flex": 1}),
                    html.Div(["x_col", dcc.Dropdown(
                        id="x-col", options=col_options + none_opt,
                        value=x_col or "__none__", clearable=False,
                        style={"color": "#111"})], style={"flex": 1}),
                ]),
                html.Button("⤓ Generate script", id="gen-btn", n_clicks=0,
                            style={**btn_style, "margin": "12px 0 8px",
                                   "width": "100%"}),
                html.Div(style={"display": "flex", "gap": "8px",
                                "alignItems": "center"}, children=[
                    dcc.Input(id="save-name", type="text",
                              value="my_dashboard.py", style=input_style),
                    html.Button("Save", id="save-btn", n_clicks=0,
                                style=btn_ghost),
                ]),
                html.Div(id="save-status", style={"fontSize": "11px",
                         "color": THEME["accent_blue"], "minHeight": "14px",
                         "margin": "6px 0"}),
                dcc.Textarea(id="code-out", style={
                    "width": "100%", "height": "260px", "marginTop": "6px",
                    "fontFamily": "monospace", "fontSize": "11px",
                    "backgroundColor": THEME["bg"], "color": THEME["text"],
                    "border": f"1px solid {THEME['panel_border']}",
                    "borderRadius": "6px", "boxSizing": "border-box",
                    "padding": "8px"}),
            ])),
        ],
    )

    app.layout = html.Div(
        style={"backgroundColor": THEME["bg"], "color": THEME["text"],
               "fontFamily": "Inter, Arial, sans-serif", "minHeight": "100vh",
               "padding": "24px", "boxSizing": "border-box"},
        children=[
            _header(title, f"{image_path}  ·  {len(df):,} rows × "
                           f"{len(columns)} columns — click to build"),
            html.Div(
                style={"display": "grid",
                       "gridTemplateColumns": "minmax(0, 1fr) 380px",
                       "gap": "20px", "alignItems": "start"},
                children=[
                    html.Div(style={"backgroundColor": THEME["panel"],
                             "border": f"1px solid {THEME['panel_border']}",
                             "borderRadius": "8px", "padding": "12px"},
                        children=[html.Div(style={
                            "width": "100%",
                            "aspectRatio": f"{iw} / {ih}"},
                            children=[dcc.Graph(
                                id="canvas", style={"width": "100%",
                                                    "height": "100%"},
                                config={"displayModeBar": True,
                                        "responsive": True,
                                        "displaylogo": False})])]),
                    builder_panel,
                ],
            ),
            dcc.Store(id="stations-store", data=[]),
        ],
    )

    # --- click captures coordinates -------------------------------------
    @app.callback(
        Output("coord-x", "value"),
        Output("coord-y", "value"),
        Input("canvas", "clickData"),
        prevent_initial_call=True,
    )
    def _capture_click(click):
        if not click or not click.get("points"):
            return no_update, no_update
        p = click["points"][0]
        return int(round(p["x"])), int(round(p["y"]))

    # --- add / remove / clear mutate the station store ------------------
    @app.callback(
        Output("stations-store", "data"),
        Output("add-status", "children"),
        Output("station-id", "value"),
        Output("station-cols", "value"),
        Input("add-btn", "n_clicks"),
        Input("remove-btn", "n_clicks"),
        Input("clear-btn", "n_clicks"),
        State("stations-store", "data"),
        State("coord-x", "value"),
        State("coord-y", "value"),
        State("station-id", "value"),
        State("station-cols", "value"),
        State("remove-pick", "value"),
        prevent_initial_call=True,
    )
    def _mutate(_a, _r, _c, stations, cx, cy, sid, cols, remove_idx):
        stations = list(stations or [])
        trig = ctx.triggered_id
        if trig == "clear-btn":
            return [], "Cleared all stations.", no_update, no_update
        if trig == "remove-btn":
            if remove_idx is None or remove_idx >= len(stations):
                return no_update, "Pick a station to remove.", no_update, no_update
            removed = stations.pop(int(remove_idx))
            return stations, f"Removed station {removed['id']}.", no_update, no_update
        # add-btn
        if cx is None or cy is None:
            return no_update, "Click the image to set x/y first.", no_update, no_update
        if not cols:
            return no_update, "Select at least one column.", no_update, no_update
        sid = (sid or "").strip() or f"S{len(stations) + 1}"
        stations.append({"id": sid, "x": int(cx), "y": int(cy),
                         "cols": list(cols)})
        return stations, f"Added station {sid} at ({int(cx)}, {int(cy)}).", "", []

    # --- store changes redraw the canvas, list, and remove options ------
    @app.callback(
        Output("canvas", "figure"),
        Output("station-list", "children"),
        Output("remove-pick", "options"),
        Input("stations-store", "data"),
        Input("preview-row", "value"),
    )
    def _render(stations, preview_row):
        stations = stations or []
        fig = _canvas_figure(img, iw, ih, df, stations, int(preview_row or 0))
        if stations:
            items = [html.Div(f"{s['id']}  ({s['x']}, {s['y']})  →  "
                              f"{', '.join(s['cols'])}",
                              style={"marginBottom": "3px"})
                     for s in stations]
        else:
            items = [html.Div("No stations yet — click the image.",
                              style={"color": THEME["text_muted"]})]
        opts = [{"label": f"{s['id']} ({s['x']}, {s['y']})", "value": i}
                for i, s in enumerate(stations)]
        return fig, items, opts

    # --- generate script text -------------------------------------------
    @app.callback(
        Output("code-out", "value"),
        Input("gen-btn", "n_clicks"),
        State("stations-store", "data"),
        State("label-col", "value"),
        State("x-col", "value"),
        prevent_initial_call=True,
    )
    def _generate(_n, stations, label_v, x_v):
        if not stations:
            return "# Add at least one station first."
        lc = None if label_v == "__none__" else label_v
        xc = None if x_v == "__none__" else x_v
        return generate_script(stations, image_path, df, lc, xc, title=title)

    # --- save script to disk --------------------------------------------
    @app.callback(
        Output("save-status", "children"),
        Input("save-btn", "n_clicks"),
        State("stations-store", "data"),
        State("save-name", "value"),
        State("label-col", "value"),
        State("x-col", "value"),
        prevent_initial_call=True,
    )
    def _save(_n, stations, name, label_v, x_v):
        if not stations:
            return "Nothing to save — add a station first."
        out = Path(name or "my_dashboard.py").with_suffix(".py")
        marker = '"""Generated by dashboard_builder.py'
        # Don't silently clobber a hand-written file; only overwrite our own.
        if out.exists() and not out.read_text().startswith(marker):
            return f"⚠ {out.name} exists and wasn't generated here — pick another name."
        lc = None if label_v == "__none__" else label_v
        xc = None if x_v == "__none__" else x_v
        # Write the dataframe beside the script so the output runs as-is.
        csv_out = out.with_suffix(".csv")
        df.to_csv(csv_out, index=False)
        code = generate_script(stations, image_path, df, lc, xc, title=title,
                               csv_path=csv_out.name)
        out.write_text(code)
        return f"Saved → {out.resolve()}  (+ {csv_out.name})"

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="Interactive engine dashboard builder")
    ap.add_argument("--image", required=True, help="cross-section image path")
    ap.add_argument("--csv", required=True, help="dataframe CSV path")
    ap.add_argument("--label-col", default=None, help="slider/animation label column")
    ap.add_argument("--x-col", default=None, help="x-axis column for trend plots")
    ap.add_argument("--title", default="Dashboard Builder")
    ap.add_argument("--port", type=int, default=8060)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    app = build_builder_app(args.image, df, label_col=args.label_col,
                            x_col=args.x_col, title=args.title)
    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
