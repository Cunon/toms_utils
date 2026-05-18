"""
engine_dashboard.py
===================

Dash dashboard wrapping ``overlay_values_on_image`` for transient engine
simulations. Presents the interactive Plotly figure alongside a metadata
header, a summary-statistics panel, and a current-row preview that follows
the slider.

Requirements: ``dash`` (in addition to plot_engine_stations' deps).
    pip install dash

Quick start (single view)::

    from engine_dashboard import create_dashboard

    app = create_dashboard(
        image="engine.png",
        df=df,
        location_cols={(250, 540): ["Tt_K", "Pt_kPa"], ...},
        label_col="time_s",
        x_col="time_s",
        plot_cols=["Tt_K", "Pt_kPa", "W_kgs", "Mach"],
        title="Transient Engine Simulation",
    )
    app.run(debug=True)

Tabbed view::

    app = create_dashboard(
        image="engine.png",
        df=df,
        location_cols=location_cols,
        label_col="time_s",
        x_col="time_s",
        tabs=[
            {"label": "Pressures",   "plot_cols": ["Pt_kPa", "Pt_4_kPa"]},
            {"label": "Temperatures","plot_cols": ["Tt_K", "Tt_4_K"]},
            {"label": "Flow & Mach", "plot_cols": ["W_kgs", "Mach"]},
        ],
    )
    app.run(debug=True)
"""

from __future__ import annotations
from pathlib import Path
import os, sys

github_path = Path(r"/home/tje/Documents/GitHub")

if os.path.exists(github_path):
    sys.path.append(str(github_path))

from typing import Mapping, Optional, Sequence, Tuple, Union

import pandas as pd
from dash import Dash, Input, Output, dcc, html

from toms_utils.plot_engine_stations import (
    Coord,
    ImageLike,
    overlay_values_on_image,
)


# Centralized theme so the dashboard chrome matches the figure styling.
THEME = {
    "bg": "#0E1117",
    "panel": "#161A22",
    "panel_border": "#2A2D34",
    "text": "#E6E6E6",
    "text_muted": "#9A9A9A",
    "accent": "#EF9F27",
    "accent_blue": "#5AA9E6",
}


def create_dashboard(
    image: ImageLike,
    df: pd.DataFrame,
    location_cols: Mapping[Coord, Sequence[str]],
    title: str = "Engine Stations Dashboard",
    subtitle: Optional[str] = None,
    label_col: Optional[str] = None,
    x_col: Optional[str] = None,
    plot_cols: Optional[Sequence[str]] = None,
    tabs: Optional[Sequence[dict]] = None,
    initial_row: int = 0,
    frame_duration: int = 800,
    width: int = 1400,
    plot_layout: str = "right",
    **overlay_kwargs,
) -> Dash:
    """
    Build a Dash app that displays the engine cross-section overlay plus a
    sidebar with metadata, summary statistics, and a current-row preview.

    Parameters
    ----------
    image, df, location_cols, label_col, x_col, plot_cols,
    initial_row, frame_duration, width, plot_layout, **overlay_kwargs
        Passed through to ``overlay_values_on_image``. ``plot_layout``
        defaults to ``"right"`` so line plots render alongside the image.
    tabs
        Optional list of tab configs. Each is a dict like::

            {"label": "Pressures",
             "plot_cols": ["Pt_kPa", ...],
             "location_cols": {...}}     # optional per-tab override

        When provided, the figure section becomes a tabbed view with one
        ``overlay_values_on_image`` figure per tab. ``location_cols`` and
        ``plot_cols`` fall back to the top-level values if omitted.
    title
        Dashboard header title.
    subtitle
        Optional subtitle. Defaults to a row × column summary.

    Returns
    -------
    dash.Dash
        Run with ``app.run(debug=True)``.
    """
    app = Dash(__name__)
    app.title = title

    if subtitle is None:
        subtitle = f"{len(df):,} rows × {len(df.columns)} columns"

    def _build_fig(tab_plot_cols, tab_location_cols):
        return overlay_values_on_image(
            image=image,
            df=df,
            location_cols=tab_location_cols,
            label_col=label_col,
            x_col=x_col,
            plot_cols=tab_plot_cols,
            initial_row=initial_row,
            frame_duration=frame_duration,
            width=width,
            plot_layout=plot_layout,
            **overlay_kwargs,
        )

    # Build either a single figure panel or a tabbed view.
    if tabs:
        figure_section = _build_tabs_section(
            tabs=tabs,
            default_plot_cols=plot_cols,
            default_location_cols=location_cols,
            build_fig=_build_fig,
        )
        # Summary panel uses the union of plot_cols across all tabs (or the
        # top-level plot_cols if specified) so it covers everything visible.
        summary_cols = _summary_cols_from_tabs(tabs, plot_cols, df)
    else:
        fig = _build_fig(plot_cols, location_cols)
        figure_section = _figure_panel(fig, graph_id="main-figure")
        summary_cols = (
            list(plot_cols) if plot_cols else _default_summary_cols(df)
        )

    label_steps = _slider_labels(df, label_col)

    app.layout = html.Div(
        style={
            "backgroundColor": THEME["bg"],
            "color": THEME["text"],
            "fontFamily": "Inter, Arial, sans-serif",
            "minHeight": "100vh",
            "padding": "24px",
            "boxSizing": "border-box",
        },
        children=[
            _header(title, subtitle),
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "minmax(0, 1fr) 340px",
                    "gap": "20px",
                    "alignItems": "start",
                },
                children=[
                    figure_section,
                    html.Div(
                        style={
                            "display": "flex",
                            "flexDirection": "column",
                            "gap": "16px",
                        },
                        children=[
                            _row_picker(label_steps, initial_row),
                            _row_preview_panel(),
                            _stats_panel(df, summary_cols),
                        ],
                    ),
                ],
            ),
            # Hidden store so the row-preview callback can look up values.
            dcc.Store(
                id="dataset-store",
                data={
                    "records": df.to_dict("records"),
                    "label_col": label_col,
                },
            ),
        ],
    )

    @app.callback(
        Output("row-preview", "children"),
        Input("row-slider", "value"),
        Input("dataset-store", "data"),
    )
    def _render_row_preview(row_index, store):
        records = store.get("records", [])
        if not records:
            return html.Div("No data", style={"color": THEME["text_muted"]})
        i = max(0, min(int(row_index or 0), len(records) - 1))
        return _row_preview_body(records[i], store.get("label_col"))

    return app


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def _header(title: str, subtitle: str) -> html.Div:
    return html.Div(
        style={
            "marginBottom": "20px",
            "paddingBottom": "16px",
            "borderBottom": f"1px solid {THEME['panel_border']}",
        },
        children=[
            html.H1(
                title,
                style={
                    "margin": 0,
                    "fontSize": "26px",
                    "fontWeight": 600,
                    "color": THEME["text"],
                },
            ),
            html.P(
                subtitle,
                style={
                    "margin": "6px 0 0 0",
                    "color": THEME["text_muted"],
                    "fontSize": "13px",
                },
            ),
        ],
    )


def _panel(title: str, body) -> html.Div:
    return html.Div(
        style={
            "backgroundColor": THEME["panel"],
            "border": f"1px solid {THEME['panel_border']}",
            "borderRadius": "8px",
            "padding": "16px",
        },
        children=[
            html.H3(
                title,
                style={
                    "margin": "0 0 12px 0",
                    "fontSize": "14px",
                    "fontWeight": 600,
                    "textTransform": "uppercase",
                    "letterSpacing": "0.06em",
                    "color": THEME["text_muted"],
                },
            ),
            body,
        ],
    )


def _figure_panel(fig, graph_id: str = "main-figure") -> html.Div:
    return html.Div(
        style={
            "backgroundColor": THEME["panel"],
            "border": f"1px solid {THEME['panel_border']}",
            "borderRadius": "8px",
            "padding": "12px",
        },
        children=[
            dcc.Graph(
                id=graph_id,
                figure=fig,
                config={
                    "displayModeBar": True,
                    "displaylogo": False,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                },
                style={"backgroundColor": "transparent"},
            )
        ],
    )


def _build_tabs_section(
    tabs: Sequence[dict],
    default_plot_cols: Optional[Sequence[str]],
    default_location_cols: Mapping[Coord, Sequence[str]],
    build_fig,
) -> html.Div:
    """Render the figure section as a tabbed view, one figure per tab."""
    tab_components = []
    for i, tab in enumerate(tabs):
        label = tab.get("label", f"Tab {i + 1}")
        tab_plot_cols = tab.get("plot_cols", default_plot_cols)
        tab_location_cols = tab.get("location_cols", default_location_cols)
        fig = build_fig(tab_plot_cols, tab_location_cols)
        tab_components.append(
            dcc.Tab(
                label=label,
                value=f"tab-{i}",
                style=_TAB_STYLE,
                selected_style=_TAB_SELECTED_STYLE,
                children=[
                    _figure_panel(fig, graph_id=f"main-figure-{i}"),
                ],
            )
        )

    return html.Div(
        style={
            "backgroundColor": THEME["panel"],
            "border": f"1px solid {THEME['panel_border']}",
            "borderRadius": "8px",
            "padding": "12px",
        },
        children=[
            dcc.Tabs(
                id="figure-tabs",
                value="tab-0",
                children=tab_components,
                style={"marginBottom": "8px"},
                parent_style={"backgroundColor": "transparent"},
                colors={
                    "border": THEME["panel_border"],
                    "primary": THEME["accent"],
                    "background": THEME["panel"],
                },
            )
        ],
    )


def _summary_cols_from_tabs(
    tabs: Sequence[dict],
    default_plot_cols: Optional[Sequence[str]],
    df: pd.DataFrame,
) -> list:
    """Union of every tab's plot_cols (preserves order, deduped)."""
    seen = set()
    out = []
    for tab in tabs:
        cols = tab.get("plot_cols") or default_plot_cols or []
        for c in cols:
            if c not in seen and c in df.columns:
                seen.add(c)
                out.append(c)
    return out or _default_summary_cols(df)


# Tab visual styles — kept module-level so both selected and idle tabs match
# the dark theme. dcc.Tabs applies inline styles directly.
_TAB_STYLE = {
    "backgroundColor": THEME["bg"],
    "color": THEME["text_muted"],
    "border": f"1px solid {THEME['panel_border']}",
    "borderBottom": "none",
    "padding": "10px 16px",
    "fontSize": "13px",
    "fontWeight": 500,
}

_TAB_SELECTED_STYLE = {
    "backgroundColor": THEME["panel"],
    "color": THEME["text"],
    "border": f"1px solid {THEME['panel_border']}",
    "borderBottom": f"2px solid {THEME['accent']}",
    "padding": "10px 16px",
    "fontSize": "13px",
    "fontWeight": 600,
}


def _row_picker(label_steps, initial_row: int) -> html.Div:
    n = len(label_steps)
    marks = _slider_marks(label_steps)
    return _panel(
        "Row Selector",
        html.Div(
            children=[
                html.Div(
                    id="row-slider-label",
                    style={
                        "fontSize": "12px",
                        "color": THEME["text_muted"],
                        "marginBottom": "8px",
                    },
                    children="Drives the row-preview panel below.",
                ),
                dcc.Slider(
                    id="row-slider",
                    min=0,
                    max=max(0, n - 1),
                    step=1,
                    value=initial_row,
                    marks=marks,
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ]
        ),
    )


def _row_preview_panel() -> html.Div:
    return _panel(
        "Current Row",
        html.Div(id="row-preview", style={"fontSize": "12px"}),
    )


def _row_preview_body(record: dict, label_col: Optional[str]) -> html.Div:
    items = []
    if label_col and label_col in record:
        items.append(
            html.Div(
                style={
                    "marginBottom": "8px",
                    "fontSize": "13px",
                    "color": THEME["accent"],
                    "fontWeight": 600,
                },
                children=f"{label_col}: {record[label_col]}",
            )
        )
    rows = []
    for col, val in record.items():
        if col == label_col:
            continue
        if val is None or (isinstance(val, float) and pd.isna(val)):
            display_val = "—"
        elif isinstance(val, (int, float)):
            display_val = f"{val:.4g}"
        else:
            display_val = str(val)
        rows.append(
            html.Tr(
                children=[
                    html.Td(
                        col,
                        style={
                            "padding": "4px 8px 4px 0",
                            "color": THEME["text_muted"],
                            "fontFamily": "monospace",
                        },
                    ),
                    html.Td(
                        display_val,
                        style={
                            "padding": "4px 0",
                            "textAlign": "right",
                            "fontFamily": "monospace",
                            "color": THEME["text"],
                        },
                    ),
                ]
            )
        )
    items.append(
        html.Table(
            html.Tbody(rows),
            style={"width": "100%", "borderCollapse": "collapse"},
        )
    )
    return html.Div(items)


def _stats_panel(df: pd.DataFrame, cols: Sequence[str]) -> html.Div:
    header = html.Tr(
        [
            html.Th(
                "Column",
                style={
                    "textAlign": "left",
                    "padding": "4px 8px 6px 0",
                    "color": THEME["text_muted"],
                    "fontWeight": 500,
                    "borderBottom": f"1px solid {THEME['panel_border']}",
                },
            ),
            *(
                html.Th(
                    label,
                    style={
                        "textAlign": "right",
                        "padding": "4px 0 6px 8px",
                        "color": THEME["text_muted"],
                        "fontWeight": 500,
                        "borderBottom": f"1px solid {THEME['panel_border']}",
                    },
                )
                for label in ("Min", "Max", "Mean")
            ),
        ]
    )

    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        rows.append(
            html.Tr(
                [
                    html.Td(
                        col,
                        style={
                            "padding": "4px 8px 4px 0",
                            "fontFamily": "monospace",
                            "color": THEME["text"],
                        },
                    ),
                    *(
                        html.Td(
                            f"{val:.4g}",
                            style={
                                "padding": "4px 0 4px 8px",
                                "textAlign": "right",
                                "fontFamily": "monospace",
                                "color": THEME["text"],
                            },
                        )
                        for val in (
                            series.min(),
                            series.max(),
                            series.mean(),
                        )
                    ),
                ]
            )
        )

    return _panel(
        "Summary Statistics",
        html.Table(
            [html.Thead(header), html.Tbody(rows)],
            style={
                "width": "100%",
                "borderCollapse": "collapse",
                "fontSize": "12px",
            },
        ),
    )


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _slider_labels(df: pd.DataFrame, label_col: Optional[str]) -> list:
    if label_col is not None and label_col in df.columns:
        return [str(v) for v in df[label_col].tolist()]
    return [str(i) for i in range(len(df))]


def _slider_marks(labels: list) -> dict:
    """Show at most ~10 evenly-spaced marks so the slider stays readable."""
    n = len(labels)
    if n == 0:
        return {}
    step = max(1, n // 10)
    marks = {
        i: {
            "label": labels[i],
            "style": {"color": THEME["text_muted"], "fontSize": "10px"},
        }
        for i in range(0, n, step)
    }
    marks[n - 1] = {
        "label": labels[n - 1],
        "style": {"color": THEME["text_muted"], "fontSize": "10px"},
    }
    return marks


def _default_summary_cols(df: pd.DataFrame, limit: int = 8) -> list:
    return list(df.select_dtypes("number").columns[:limit])


# ---------------------------------------------------------------------------
# Example use
# ---------------------------------------------------------------------------
#
#     import pandas as pd
#     from engine_dashboard import create_dashboard
#
#     df = pd.DataFrame(...)        # one row per time stamp
#     location_cols = {
#         (250, 540): ["Tt_K", "Pt_kPa"],
#         (420, 540): ["W_kgs", "Mach"],
#         (770, 540): ["Tt_K", "Pt_kPa"],
#     }
#
#     app = create_dashboard(
#         image="engine.png",
#         df=df,
#         location_cols=location_cols,
#         label_col="time_s",
#         x_col="time_s",
#         plot_cols=["Tt_K", "Pt_kPa", "W_kgs", "Mach"],
#     )
#     app.run(debug=True)
