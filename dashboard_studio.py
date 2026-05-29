"""
dashboard_studio.py
===================

A point-and-click dashboard creator built on top of the **Unichart** plotting
tool (``unichart.UnichartNotebook``). Run it with no arguments, open the
browser, and assemble a dashboard entirely in the page:

    1. The board opens with four blank chart slots.
    2. Upload a CSV of data.
    3. (Optional) pick a column to split the data into multiple Unichart
       datasets -- e.g. one line per ``RATING`` -- so charts can overlay them.
    4. Pick a slot to edit, choose a chart type (line, scatter, bar, box,
       histogram, contour or table) and the x / y / z columns. The slot updates
       live as you edit -- toggle "Auto-refresh" off to batch several edits and
       redraw once with "Refresh now".
    5. Add / remove / reorder slots, flip the board to dark mode.
    6. Export -- download a standalone HTML file, or generate / save a runnable
       Python script that rebuilds the same dashboard with ``build_dashboard``.

The three layers are reusable on their own:

    render_figure(df, split_col, spec, darkmode) -> plotly Figure
        Turn one chart spec into a Unichart figure.

    build_dashboard(df, charts, split_col=..., title=..., darkmode=..., ncols=...)
        A read-only Dash app showing a grid of those figures (this is what the
        generated scripts call).

    build_app(title) -> Dash
        The interactive studio / creator GUI.

Run it::

    python dashboard_studio.py            # then open http://127.0.0.1:8070
"""

from __future__ import annotations

import base64
import io
import pprint
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update

from unichart import UnichartNotebook

MAX_UPLOAD_BYTES = 12_000_000  # ~12 MB; keeps "simple" from meaning "hangs".
DEFAULT_SLOTS = 4              # blank charts the board opens with

# Centralised theme so the studio chrome matches the figure styling.
THEME = {
    "bg": "#0E1117",
    "panel": "#161A22",
    "panel_border": "#2A2D34",
    "text": "#E6E6E6",
    "text_muted": "#9A9A9A",
    "accent": "#EF9F27",
    "accent_blue": "#5AA9E6",
    "danger": "#E06C75",
}

# One source of truth for a blank slot — the layout's editor defaults below
# must mirror this so a freshly opened editor agrees with the slot it edits.
BLANK_SPEC = {"type": "line", "title": "", "x": None, "y": [], "z": None,
              "by": "vars"}

# Chart types the studio offers, in menu order. Each maps to a UnichartNotebook
# method in ``render_figure``; ``needs`` drives which form fields are required.
CHART_TYPES = [
    {"value": "line",      "label": "Line",      "needs": ("x", "y")},
    {"value": "scatter",   "label": "Scatter",   "needs": ("x", "y")},
    {"value": "bar",       "label": "Bar",       "needs": ("x", "y")},
    {"value": "box",       "label": "Box",       "needs": ("x", "y")},
    {"value": "histogram", "label": "Histogram", "needs": ("y",)},
    {"value": "contour",   "label": "Contour",   "needs": ("x", "y", "z")},
    {"value": "table",     "label": "Table",     "needs": ("y",)},
]
CHART_LABELS = {c["value"]: c["label"] for c in CHART_TYPES}

# How to arrange the series within a single chart.
BY_OPTIONS = [
    {"value": "vars", "label": "One subplot per Y variable"},
    {"value": "ymult", "label": "Overlay on shared X (multi-axis)"},
    {"value": "sets", "label": "One subplot per dataset"},
]

# Short, per-type guidance shown under the form.
TYPE_HINTS = {
    "line": "Trends: pick an X and one or more Y columns.",
    "scatter": "Points only: pick an X and one or more Y columns.",
    "bar": "Pick a categorical X and one or more numeric Y columns.",
    "box": "Distribution of each Y column, grouped by a categorical X column.",
    "histogram": "Distribution of each Y column (X is ignored).",
    "contour": "Needs X, a single Y, and a Z value to colour the surface.",
    "table": "Lists the X and Y columns for every selected dataset.",
}


# ---------------------------------------------------------------------------
# Specs / slots
# ---------------------------------------------------------------------------


def _blank_spec() -> dict:
    return dict(BLANK_SPEC)


def default_charts() -> list:
    return [_blank_spec() for _ in range(DEFAULT_SLOTS)]


def _spec_ready(spec: dict) -> bool:
    """True once a slot has the fields its chart type needs to render."""
    if not spec or not spec.get("type"):
        return False
    needs = next((c["needs"] for c in CHART_TYPES if c["value"] == spec["type"]), ())
    if "x" in needs and not spec.get("x"):
        return False
    if "y" in needs and not [c for c in (spec.get("y") or []) if c]:
        return False
    if "z" in needs and not spec.get("z"):
        return False
    return True


def _slot_label(spec: dict, i: int) -> str:
    return (spec.get("title") or "").strip() or f"Slot {i + 1}"


def _slot_options(charts: Sequence[dict]) -> list:
    return [{"label": _slot_label(s, i), "value": i} for i, s in enumerate(charts)]


# ---------------------------------------------------------------------------
# Upload decoding
# ---------------------------------------------------------------------------


def _decode_csv(contents: str) -> pd.DataFrame:
    """Decode a ``dcc.Upload`` data URI into a DataFrame, or raise ValueError."""
    if not contents:
        raise ValueError("No file received.")
    try:
        _, b64 = contents.split(",", 1)
        raw = base64.b64decode(b64)
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        raise ValueError("Couldn't parse that as CSV.")
    if df.empty:
        raise ValueError("That CSV has no rows.")
    return df


# ---------------------------------------------------------------------------
# Core: turn one chart spec into a Unichart figure
# ---------------------------------------------------------------------------


def _load_notebook(df: pd.DataFrame, split_col: Optional[str],
                   darkmode: bool) -> UnichartNotebook:
    """Build a notebook with ``df`` loaded as one dataset (or one per split value)."""
    nb = UnichartNotebook()
    if split_col and split_col in df.columns:
        # One dataset per unique value, each titled by that value.
        nb.load_df(df, set_idx_column=split_col, set_name_column=split_col)
    else:
        # A title suppresses Unichart's auto-split on TITLE/SETNUMBER columns,
        # so "no split" reliably means exactly one dataset.
        nb.load_df(df, title="Data")
    if darkmode:
        nb.toggle_darkmode(True)
    nb.select("all")
    return nb


def render_figure(df: pd.DataFrame, split_col: Optional[str], spec: dict,
                  darkmode: bool = False) -> Optional[go.Figure]:
    """
    Render a single chart ``spec`` through Unichart and return its Plotly figure.

    ``spec`` keys: ``type`` (see ``CHART_TYPES``), ``x`` (str), ``y`` (list of
    str), ``z`` (str, contour only), ``by`` (see ``BY_OPTIONS``), ``title``.

    Raises ``ValueError`` with a readable message when the spec can't be drawn
    (missing column, missing required field, …); callers surface that on the
    chart card.
    """
    ctype = spec.get("type", "line")
    x = spec.get("x") or None
    y = [c for c in (spec.get("y") or []) if c]
    z = spec.get("z") or None
    by = spec.get("by", "vars")
    facet = "sets" if by == "sets" else "vars"  # bar/box/etc only know vars|sets

    def _need(cond, msg):
        if not cond:
            raise ValueError(msg)

    # Catch columns that don't exist in the current CSV up front, so a stale
    # spec (e.g. after swapping data files) gives a clear message instead of a
    # silently-empty plot or a cryptic Unichart error.
    missing = [c for c in ([x, z] + y) if c and c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not in this CSV: {', '.join(missing)}.")

    nb = _load_notebook(df, split_col, darkmode)
    if spec.get("title"):
        nb.suptitle = spec["title"]

    if ctype in ("line", "scatter"):
        _need(x and y, "Line / scatter charts need an X column and at least one Y column.")
        kwargs = {"plot_type": "scatter"} if ctype == "scatter" else {}
        if by == "ymult":
            fig = nb.plot(x=x, y=y, by="ymult")
        else:
            fig = nb.plot(x=x, y=y, by=facet, **kwargs)
    elif ctype == "bar":
        _need(x and y, "Bar charts need an X column and at least one Y column.")
        fig = nb.bar(x=x, y=y, by=facet)
    elif ctype == "box":
        _need(x and y, "Box plots need a categorical X column and at least one Y column.")
        fig = nb.box(x=x, y=y, by=facet)
    elif ctype == "histogram":
        _need(y, "Histograms need at least one Y column to bin.")
        fig = nb.histogram(x=y, by=facet)
    elif ctype == "contour":
        _need(x and y and z, "Contour plots need X, a single Y, and a Z column.")
        try:
            fig = nb.contour(x=x, y=y[0], z=z, by=facet)
        except Exception as e:  # noqa: BLE001 - griddata/Qhull errors are cryptic
            raise ValueError(
                "Couldn't interpolate a contour — X, Y and Z need to cover a "
                "2-D surface (enough spread-out points, not a single line)."
            ) from e
    elif ctype == "table":
        cols = [c for c in ([x] + y) if c]
        _need(cols, "Tables need at least one column.")
        nb.table(cols=cols, title=spec.get("title") or None)
        fig = nb.last_fig  # table() renders to last_fig and returns None
    else:
        raise ValueError(f"Unknown chart type: {ctype!r}")

    if fig is None:
        raise ValueError("Unichart produced no figure for this chart.")
    # Let the figure fill its grid card instead of its fixed design size.
    fig.update_layout(autosize=True, width=None, height=None)
    return fig


# ---------------------------------------------------------------------------
# Shared chrome
# ---------------------------------------------------------------------------


def _header(title: str, subtitle: str) -> html.Div:
    return html.Div(
        style={"marginBottom": "20px", "paddingBottom": "16px",
               "borderBottom": f"1px solid {THEME['panel_border']}"},
        children=[
            html.H1(title, style={"margin": 0, "fontSize": "26px",
                                  "fontWeight": 600, "color": THEME["text"]}),
            html.P(subtitle, style={"margin": "6px 0 0 0",
                                    "color": THEME["text_muted"], "fontSize": "13px"}),
        ],
    )


def _panel(title: str, body) -> html.Div:
    return html.Div(
        style={"backgroundColor": THEME["panel"],
               "border": f"1px solid {THEME['panel_border']}",
               "borderRadius": "8px", "padding": "16px"},
        children=[
            html.H3(title, style={"margin": "0 0 12px 0", "fontSize": "14px",
                                  "fontWeight": 600, "textTransform": "uppercase",
                                  "letterSpacing": "0.06em",
                                  "color": THEME["text_muted"]}),
            body,
        ],
    )


def _card(title: str, body, *, dark_card: bool) -> html.Div:
    """A grid cell: title bar plus an arbitrary body (graph or message)."""
    card_bg = "#11151c" if dark_card else "#ffffff"
    head = html.Div(
        title,
        style={"fontSize": "12px", "fontWeight": 600, "padding": "8px 12px",
               "color": THEME["text"] if dark_card else "#1a1a1a",
               "borderBottom": f"1px solid {THEME['panel_border'] if dark_card else '#e5e5e5'}"},
    )
    return html.Div(
        style={"backgroundColor": card_bg,
               "border": f"1px solid {THEME['panel_border']}",
               "borderRadius": "8px", "overflow": "hidden",
               "display": "flex", "flexDirection": "column", "height": "440px"},
        children=[head, html.Div(body, style={"flex": 1, "minHeight": 0})],
    )


def _graph_body(fig: go.Figure):
    return dcc.Graph(figure=fig, style={"width": "100%", "height": "100%"},
                     config={"displaylogo": False, "responsive": True,
                             "displayModeBar": False})


def _message_body(text: str, color: str):
    return html.Div(text, style={
        "height": "100%", "display": "flex", "alignItems": "center",
        "justifyContent": "center", "textAlign": "center", "padding": "16px",
        "fontSize": "13px", "color": color,
        "fontFamily": "monospace" if color == THEME["danger"] else "inherit"})


def _card_for(spec: dict, df: Optional[pd.DataFrame], split_col: Optional[str],
              darkmode: bool) -> html.Div:
    """Decide a slot's state — blank / error / figure — and build its card."""
    title = (spec.get("title") or "").strip() or CHART_LABELS.get(spec.get("type"), "Chart")
    if df is None or len(df) == 0:
        return _card(title, _message_body("Upload a CSV to render this chart.",
                     THEME["text_muted"]), dark_card=darkmode)
    if not _spec_ready(spec):
        return _card(title, _message_body("Choose a chart type and columns →",
                     THEME["text_muted"]), dark_card=darkmode)
    try:
        fig = render_figure(df, split_col, spec, darkmode)
        return _card(title, _graph_body(fig), dark_card=darkmode)
    except Exception as e:  # noqa: BLE001 - surface, don't crash the board
        return _card(title, _message_body(f"⚠ {e}", THEME["danger"]),
                     dark_card=darkmode)


def _grid(children, ncols: int) -> html.Div:
    return html.Div(
        style={"display": "grid",
               "gridTemplateColumns": f"repeat({max(1, ncols)}, minmax(0, 1fr))",
               "gap": "16px"},
        children=children,
    )


# ---------------------------------------------------------------------------
# Read-only dashboard (what generated scripts call)
# ---------------------------------------------------------------------------


def build_dashboard(df: pd.DataFrame, charts: Sequence[dict], *,
                    split_col: Optional[str] = None, title: str = "Dashboard",
                    darkmode: bool = False, ncols: int = 2) -> Dash:
    """
    Build a read-only Dash app showing ``charts`` as a responsive grid of
    Unichart figures. Each entry of ``charts`` is a spec accepted by
    ``render_figure``. This is the runtime that ``generate_script`` targets.
    """
    app = Dash(__name__)
    app.title = title

    cards = [_card_for(spec, df, split_col, darkmode) for spec in charts]
    page_bg = THEME["bg"] if darkmode else "#f4f5f7"
    text = THEME["text"] if darkmode else "#1a1a1a"
    app.layout = html.Div(
        style={"backgroundColor": page_bg, "color": text, "minHeight": "100vh",
               "padding": "24px", "boxSizing": "border-box",
               "fontFamily": "Inter, Arial, sans-serif"},
        children=[
            html.H1(title, style={"fontSize": "24px", "fontWeight": 600,
                                  "margin": "0 0 20px 0"}),
            _grid(cards, ncols) if cards
            else html.Div("No charts in this dashboard.",
                          style={"color": THEME["text_muted"]}),
        ],
    )
    return app


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------


def generate_script(charts: Sequence[dict], csv_path: str, *,
                    split_col: Optional[str], title: str, darkmode: bool,
                    ncols: int) -> str:
    """Render a runnable script that rebuilds the dashboard via ``build_dashboard``."""
    # pformat (not json.dumps) so the embedded literal is valid Python — None/True
    # rather than null/true.
    charts_repr = pprint.pformat(list(charts), indent=4, sort_dicts=False, width=88)
    return "\n".join([
        '"""Generated by dashboard_studio.py — edit freely."""',
        "import pandas as pd",
        "from dashboard_studio import build_dashboard",
        "",
        "# Data this dashboard was built against:",
        f"df = pd.read_csv({csv_path!r})",
        "",
        f"charts = {charts_repr}",
        "",
        "app = build_dashboard(",
        "    df=df,",
        "    charts=charts,",
        f"    split_col={split_col!r},",
        f"    title={title!r},",
        f"    darkmode={bool(darkmode)!r},",
        f"    ncols={int(ncols)!r},",
        ")",
        "",
        'if __name__ == "__main__":',
        "    app.run(debug=True, port=8071)",
        "",
    ])


# ---------------------------------------------------------------------------
# Studio GUI
# ---------------------------------------------------------------------------


def _chart_list(charts: Sequence[dict], selected: int) -> list:
    """Compact summary of every slot, with the one being edited marked."""
    rows = []
    for i, s in enumerate(charts):
        ready = _spec_ready(s)
        if ready:
            desc = (f"{CHART_LABELS.get(s.get('type'), '?')}  "
                    f"x={s.get('x') or '—'}  "
                    f"y={','.join(s.get('y') or []) or '—'}")
            color = THEME["text"]
        else:
            desc = "— empty —"
            color = THEME["text_muted"]
        sel = i == selected
        rows.append(html.Div(
            f"{'▶' if sel else '  '} {_slot_label(s, i)}: {desc}",
            style={"marginBottom": "3px",
                   "color": THEME["accent"] if sel else color}))
    return rows


def build_app(title: str = "Dashboard Studio") -> Dash:
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
    upload_style = {
        "width": "100%", "height": "64px", "lineHeight": "64px",
        "borderWidth": "1px", "borderStyle": "dashed",
        "borderColor": THEME["panel_border"], "borderRadius": "8px",
        "textAlign": "center", "color": THEME["text_muted"],
        "cursor": "pointer", "fontSize": "13px",
    }
    label_style = {"fontSize": "11px", "color": THEME["text_muted"],
                   "margin": "10px 0 2px"}

    init_charts = default_charts()

    load_panel = _panel("Load data", html.Div([
        dcc.Upload(id="upload-csv", style=upload_style, max_size=MAX_UPLOAD_BYTES,
                   multiple=False,
                   children="📊  Drop or click to upload a CSV (required)"),
        html.Div(id="csv-status", style={"fontSize": "11px",
                 "color": THEME["accent_blue"], "marginTop": "6px",
                 "minHeight": "14px"}),
        html.Div("Split into datasets by (optional)", style=label_style),
        dcc.Dropdown(id="split-col", options=[], value="__none__",
                     clearable=False, style={"color": "#111"}),
        html.Div("Each unique value becomes its own dataset, so charts can "
                 "overlay or facet them.", style={"fontSize": "11px",
                 "color": THEME["text_muted"], "marginTop": "4px"}),
    ]))

    edit_panel = _panel("1 · Edit a chart", html.Div([
        html.Div("Editing slot", style=label_style),
        dcc.Dropdown(id="slot-picker", options=_slot_options(init_charts),
                     value=0, clearable=False, style={"color": "#111"}),
        html.Div("Chart type", style=label_style),
        dcc.Dropdown(id="chart-type",
                     options=[{"label": c["label"], "value": c["value"]}
                              for c in CHART_TYPES],
                     value=BLANK_SPEC["type"], clearable=False,
                     style={"color": "#111"}),
        html.Div(id="type-hint", style={"fontSize": "11px",
                 "color": THEME["accent_blue"], "margin": "6px 0 0"}),
        html.Div("Title (optional)", style=label_style),
        dcc.Input(id="chart-title", type="text", debounce=True,
                  value=BLANK_SPEC["title"],
                  placeholder="e.g. Thrust vs Throttle", style=input_style),
        html.Div("X column", style=label_style),
        dcc.Dropdown(id="chart-x", options=[], value=BLANK_SPEC["x"],
                     placeholder="select…", style={"color": "#111"}),
        html.Div("Y column(s)", style=label_style),
        dcc.Dropdown(id="chart-y", options=[], value=BLANK_SPEC["y"], multi=True,
                     placeholder="select…", style={"color": "#111"}),
        html.Div(id="z-wrap", style={"display": "none"}, children=[
            html.Div("Z column (contour)", style=label_style),
            dcc.Dropdown(id="chart-z", options=[], value=BLANK_SPEC["z"],
                         placeholder="select…", style={"color": "#111"}),
        ]),
        html.Div("Layout", style=label_style),
        dcc.Dropdown(id="chart-by",
                     options=[{"label": o["label"], "value": o["value"]}
                              for o in BY_OPTIONS],
                     value=BLANK_SPEC["by"], clearable=False,
                     style={"color": "#111"}),
        html.Div(id="edit-status", style={"fontSize": "11px",
                 "color": THEME["accent_blue"], "marginTop": "8px",
                 "minHeight": "14px"}),
    ]))

    slots_panel = _panel("2 · Slots", html.Div([
        html.Div(id="chart-list", children=_chart_list(init_charts, 0),
                 style={"fontSize": "12px", "fontFamily": "monospace",
                        "marginBottom": "10px"}),
        html.Div(style={"display": "flex", "gap": "8px"}, children=[
            html.Button("↑ Up", id="up-btn", n_clicks=0, style={**btn_ghost, "flex": 1}),
            html.Button("↓ Down", id="down-btn", n_clicks=0, style={**btn_ghost, "flex": 1}),
            html.Button("✕ Remove", id="remove-btn", n_clicks=0, style={**btn_ghost, "flex": 1}),
        ]),
        html.Div(style={"display": "flex", "gap": "8px", "marginTop": "8px"}, children=[
            html.Button("➕ Add slot", id="add-slot-btn", n_clicks=0,
                        style={**btn_style, "flex": 1}),
            html.Button("Reset to 4", id="reset-btn", n_clicks=0,
                        style={**btn_ghost, "flex": 1}),
        ]),
    ]))

    board_panel = _panel("3 · Board", html.Div([
        html.Div("Dashboard title", style=label_style),
        dcc.Input(id="board-title", type="text", value="My Dashboard",
                  debounce=True, style=input_style),
        html.Div(style={"display": "flex", "gap": "8px", "alignItems": "end"},
                 children=[
            html.Div(style={"flex": 1}, children=[
                html.Div("Grid columns", style=label_style),
                dcc.Dropdown(id="ncols", options=[{"label": str(n), "value": n}
                             for n in (1, 2, 3, 4)], value=2, clearable=False,
                             style={"color": "#111"})]),
            html.Div(style={"flex": 1}, children=[
                html.Div("Theme", style=label_style),
                dcc.Dropdown(id="darkmode", clearable=False, style={"color": "#111"},
                             options=[{"label": "Light", "value": "light"},
                                      {"label": "Dark", "value": "dark"}],
                             value="light")]),
        ]),
    ]))

    export_panel = _panel("4 · Export", html.Div([
        html.Button("⬇ Download dashboard (HTML)", id="dl-btn", n_clicks=0,
                    style={**btn_style, "width": "100%"}),
        html.Div("Standalone file — opens in any browser.",
                 style={"fontSize": "11px", "color": THEME["text_muted"],
                        "margin": "6px 0 14px"}),
        html.Button("⤓ Generate script", id="gen-btn", n_clicks=0,
                    style={**btn_style, "width": "100%"}),
        html.Div(style={"display": "flex", "gap": "8px", "alignItems": "center",
                        "marginTop": "8px"}, children=[
            dcc.Input(id="save-name", type="text", value="my_unichart_dashboard.py",
                      style=input_style),
            html.Button("Save", id="save-btn", n_clicks=0, style=btn_ghost),
        ]),
        html.Div(id="export-status", style={"fontSize": "11px",
                 "color": THEME["accent_blue"], "minHeight": "14px",
                 "margin": "6px 0"}),
        dcc.Textarea(id="code-out", style={
            "width": "100%", "height": "200px", "marginTop": "6px",
            "fontFamily": "monospace", "fontSize": "11px",
            "backgroundColor": THEME["bg"], "color": THEME["text"],
            "border": f"1px solid {THEME['panel_border']}",
            "borderRadius": "6px", "boxSizing": "border-box", "padding": "8px"}),
    ]))

    side = html.Div(style={"display": "flex", "flexDirection": "column",
                           "gap": "12px"},
                    children=[load_panel, edit_panel, slots_panel,
                              board_panel, export_panel])

    # Refresh controls live above the board so they're always in view.
    refresh_bar = html.Div(
        style={"display": "flex", "alignItems": "center", "gap": "14px",
               "marginBottom": "12px"},
        children=[
            dcc.Checklist(id="auto-refresh",
                          options=[{"label": " Auto-refresh on edit", "value": "auto"}],
                          value=["auto"], inputStyle={"marginRight": "5px"},
                          style={"fontSize": "13px", "color": THEME["text"]}),
            html.Button("⟳ Refresh now", id="refresh-btn", n_clicks=0,
                        style={**btn_ghost, "padding": "6px 12px"}),
            html.Div(id="pending-indicator",
                     style={"fontSize": "12px", "fontWeight": 600,
                            "color": THEME["accent"]}),
        ],
    )

    app.layout = html.Div(
        style={"backgroundColor": THEME["bg"], "color": THEME["text"],
               "fontFamily": "Inter, Arial, sans-serif", "minHeight": "100vh",
               "padding": "24px", "boxSizing": "border-box"},
        children=[
            _header(title, "Start with four blank charts — pick a slot, set its "
                           "type and columns, and it updates live."),
            html.Div(
                style={"display": "grid",
                       "gridTemplateColumns": "minmax(0, 1fr) 380px",
                       "gap": "20px", "alignItems": "start"},
                children=[
                    html.Div([refresh_bar, html.Div(id="board")]),
                    side,
                ],
            ),
            dcc.Store(id="data-store"),
            dcc.Store(id="charts-store", data=init_charts),
            dcc.Download(id="download"),
        ],
    )

    # --- csv upload: store records, fill column-driven controls ---------
    # Specs are kept across uploads; a reference to a now-missing column shows a
    # clear "not in this CSV" card rather than being silently dropped.
    @app.callback(
        Output("data-store", "data"),
        Output("csv-status", "children"),
        Output("split-col", "options"),
        Output("chart-x", "options"),
        Output("chart-y", "options"),
        Output("chart-z", "options"),
        Input("upload-csv", "contents"),
        State("upload-csv", "filename"),
        prevent_initial_call=True,
    )
    def _on_csv(contents, filename):
        try:
            df = _decode_csv(contents)
        except ValueError as e:
            return (no_update, f"⚠ {e}", no_update, no_update, no_update, no_update)
        cols = list(df.columns)
        opts = [{"label": c, "value": c} for c in cols]
        none_opt = [{"label": "(don't split — one dataset)", "value": "__none__"}]
        store = {"records": df.to_dict("records"), "columns": cols,
                 "name": filename or "data.csv"}
        status = f"Loaded {filename} ({len(df):,} rows × {len(cols)} cols)."
        return store, status, none_opt + opts, opts, opts, opts

    # --- chart-type drives the hint + whether the Z field shows ---------
    @app.callback(
        Output("type-hint", "children"),
        Output("z-wrap", "style"),
        Input("chart-type", "value"),
    )
    def _on_type(ctype):
        z_style = {"display": "block"} if ctype == "contour" else {"display": "none"}
        return TYPE_HINTS.get(ctype, ""), z_style

    # --- selecting a slot loads its config into the editor --------------
    @app.callback(
        Output("chart-type", "value"),
        Output("chart-title", "value"),
        Output("chart-x", "value"),
        Output("chart-y", "value"),
        Output("chart-z", "value"),
        Output("chart-by", "value"),
        Input("slot-picker", "value"),
        State("charts-store", "data"),
        prevent_initial_call=True,
    )
    def _load_editor(slot, charts):
        charts = charts or []
        if slot is None or slot >= len(charts):
            return (no_update,) * 6
        s = charts[slot]
        return (s.get("type", "line"), s.get("title", ""), s.get("x"),
                s.get("y") or [], s.get("z"), s.get("by", "vars"))

    # --- editor edits + slot management both write charts-store ---------
    @app.callback(
        Output("charts-store", "data"),
        Output("slot-picker", "options"),
        Output("slot-picker", "value"),
        Output("edit-status", "children"),
        Output("chart-list", "children"),
        Input("chart-type", "value"),
        Input("chart-title", "value"),
        Input("chart-x", "value"),
        Input("chart-y", "value"),
        Input("chart-z", "value"),
        Input("chart-by", "value"),
        Input("add-slot-btn", "n_clicks"),
        Input("remove-btn", "n_clicks"),
        Input("up-btn", "n_clicks"),
        Input("down-btn", "n_clicks"),
        Input("reset-btn", "n_clicks"),
        State("charts-store", "data"),
        State("slot-picker", "value"),
        prevent_initial_call=True,
    )
    def _mutate(ctype, ititle, ix, iy, iz, iby, _add, _rm, _up, _dn, _reset,
                charts, pick):
        charts = list(charts or default_charts())
        trig = ctx.triggered_id
        pick = pick if isinstance(pick, int) else 0

        if trig == "reset-btn":
            charts = default_charts()
            return charts, _slot_options(charts), 0, "Reset to 4 blank slots.", \
                _chart_list(charts, 0)
        if trig == "add-slot-btn":
            charts.append(_blank_spec())
            new = len(charts) - 1
            return charts, _slot_options(charts), new, f"Added Slot {new + 1}.", \
                _chart_list(charts, new)
        if trig == "remove-btn":
            if len(charts) <= 1:
                return no_update, no_update, no_update, "Keep at least one slot.", no_update
            if pick >= len(charts):
                pick = len(charts) - 1
            charts.pop(pick)
            new = min(pick, len(charts) - 1)
            return charts, _slot_options(charts), new, "Removed slot.", \
                _chart_list(charts, new)
        if trig in ("up-btn", "down-btn"):
            j = pick - 1 if trig == "up-btn" else pick + 1
            if 0 <= j < len(charts):
                charts[pick], charts[j] = charts[j], charts[pick]
                return charts, _slot_options(charts), j, "Reordered.", \
                    _chart_list(charts, j)
            return no_update, no_update, no_update, "Already at the edge.", no_update

        # Otherwise an editor field changed — write it into the selected slot.
        if pick >= len(charts):
            return (no_update,) * 5
        charts[pick] = {"type": ctype, "title": (ititle or "").strip(),
                        "x": ix, "y": [c for c in (iy or []) if c],
                        "z": iz, "by": iby}
        status = f"Updated {_slot_label(charts[pick], pick)}."
        # value stays put (no_update) so editing doesn't yank the slot selector.
        return charts, _slot_options(charts), no_update, status, \
            _chart_list(charts, pick)

    # --- render the board (gated by the refresh toggle) -----------------
    # "Hold" freezes the whole board until Refresh now / re-enabling auto; a
    # fresh CSV always renders so uploaded data is never invisible.
    @app.callback(
        Output("board", "children"),
        Input("charts-store", "data"),
        Input("data-store", "data"),
        Input("split-col", "value"),
        Input("ncols", "value"),
        Input("darkmode", "value"),
        Input("refresh-btn", "n_clicks"),
        Input("auto-refresh", "value"),
    )
    def _render(charts, data_store, split_v, ncols, theme, _refresh, auto):
        held = "auto" not in (auto or [])
        trig = ctx.triggered_id
        if held and trig not in ("refresh-btn", "auto-refresh", "data-store"):
            return no_update
        charts = charts or default_charts()
        dark = theme == "dark"
        df = pd.DataFrame(data_store["records"]) if data_store else None
        split = None if split_v in (None, "__none__") else split_v
        cards = [_card_for(s, df, split, dark) for s in charts]
        return _grid(cards, int(ncols or 2))

    # --- "edits pending" cue while refresh is held ----------------------
    @app.callback(
        Output("pending-indicator", "children"),
        Input("charts-store", "data"),
        Input("refresh-btn", "n_clicks"),
        Input("auto-refresh", "value"),
        prevent_initial_call=True,
    )
    def _pending(_charts, _refresh, auto):
        held = "auto" not in (auto or [])
        if ctx.triggered_id == "charts-store" and held:
            return "● edits pending — click Refresh now"
        return ""

    # --- download standalone HTML ---------------------------------------
    @app.callback(
        Output("download", "data"),
        Input("dl-btn", "n_clicks"),
        State("charts-store", "data"),
        State("data-store", "data"),
        State("split-col", "value"),
        State("board-title", "value"),
        State("ncols", "value"),
        State("darkmode", "value"),
        prevent_initial_call=True,
    )
    def _download_html(_n, charts, data_store, split_v, btitle, ncols, theme):
        ready = [s for s in (charts or []) if _spec_ready(s)]
        if not data_store or not ready:
            return no_update
        df = pd.DataFrame(data_store["records"])
        split = None if split_v in (None, "__none__") else split_v
        html_str = _dashboard_html(df, ready, split, btitle or "Dashboard",
                                   theme == "dark", int(ncols or 2))
        return dict(content=html_str, filename="dashboard.html")

    # --- generate script text -------------------------------------------
    @app.callback(
        Output("code-out", "value"),
        Input("gen-btn", "n_clicks"),
        State("charts-store", "data"),
        State("data-store", "data"),
        State("split-col", "value"),
        State("board-title", "value"),
        State("ncols", "value"),
        State("darkmode", "value"),
        prevent_initial_call=True,
    )
    def _generate(_n, charts, data_store, split_v, btitle, ncols, theme):
        ready = [s for s in (charts or []) if _spec_ready(s)]
        if not data_store or not ready:
            return "# Upload a CSV and configure at least one chart first."
        csv_name = Path(data_store.get("name") or "data.csv").with_suffix(".csv").name
        split = None if split_v in (None, "__none__") else split_v
        return generate_script(ready, csv_name, split_col=split,
                               title=btitle or "Dashboard", darkmode=theme == "dark",
                               ncols=int(ncols or 2))

    # --- save script (+ csv) to disk ------------------------------------
    @app.callback(
        Output("export-status", "children"),
        Input("save-btn", "n_clicks"),
        State("charts-store", "data"),
        State("data-store", "data"),
        State("split-col", "value"),
        State("board-title", "value"),
        State("ncols", "value"),
        State("darkmode", "value"),
        State("save-name", "value"),
        prevent_initial_call=True,
    )
    def _save(_n, charts, data_store, split_v, btitle, ncols, theme, name):
        ready = [s for s in (charts or []) if _spec_ready(s)]
        if not data_store or not ready:
            return "Upload a CSV and configure at least one chart first."
        out = Path(name or "my_unichart_dashboard.py").with_suffix(".py")
        marker = '"""Generated by dashboard_studio.py'
        if out.exists() and not out.read_text().startswith(marker):
            return f"⚠ {out.name} exists and wasn't generated here — pick another name."

        df = pd.DataFrame(data_store["records"])
        csv_path = out.with_suffix(".csv")
        df.to_csv(csv_path, index=False)

        split = None if split_v in (None, "__none__") else split_v
        code = generate_script(ready, csv_path.name, split_col=split,
                               title=btitle or "Dashboard", darkmode=theme == "dark",
                               ncols=int(ncols or 2))
        out.write_text(code)
        n_blank = len(charts or []) - len(ready)
        note = f"  ({n_blank} blank slot(s) omitted)" if n_blank else ""
        return f"Saved → {out.resolve()} (+ {csv_path.name}){note}"

    return app


def _dashboard_html(df, charts, split_col, title, darkmode, ncols) -> str:
    """Render the whole board to a single standalone HTML string."""
    page_bg = THEME["bg"] if darkmode else "#f4f5f7"
    text = THEME["text"] if darkmode else "#1a1a1a"
    border = THEME["panel_border"]
    blocks = []
    first = True
    for spec in charts:
        try:
            fig = render_figure(df, split_col, spec, darkmode)
            inner = fig.to_html(include_plotlyjs=("cdn" if first else False),
                                full_html=False, default_height="420px")
            first = False
        except Exception as e:  # noqa: BLE001
            inner = (f'<div style="padding:16px;color:{THEME["danger"]};'
                     f'font-family:monospace">⚠ {e}</div>')
        head = (f'<div style="font-size:12px;font-weight:600;padding:8px 12px;'
                f'border-bottom:1px solid {border}">'
                f'{spec.get("title") or CHART_LABELS.get(spec.get("type"), "Chart")}</div>')
        card_bg = "#11151c" if darkmode else "#ffffff"
        blocks.append(f'<div style="background:{card_bg};border:1px solid {border};'
                      f'border-radius:8px;overflow:hidden">{head}{inner}</div>')
    grid = (f'<div style="display:grid;gap:16px;'
            f'grid-template-columns:repeat({max(1, ncols)},minmax(0,1fr))">'
            + "".join(blocks) + "</div>")
    return (f'<!doctype html><html><head><meta charset="utf-8">'
            f'<title>{title}</title></head>'
            f'<body style="background:{page_bg};color:{text};margin:0;padding:24px;'
            f'font-family:Inter,Arial,sans-serif">'
            f'<h1 style="font-size:24px;font-weight:600;margin:0 0 20px">{title}</h1>'
            f'{grid}</body></html>')


if __name__ == "__main__":
    build_app().run(debug=True, port=8070)
