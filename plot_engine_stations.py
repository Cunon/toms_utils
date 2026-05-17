"""
engine_stations.py
==================

Overlay jet-engine performance data on a high-resolution cross-section
image using Plotly. Works in Jupyter on Windows and Linux.

Two functions:

    calibrate_stations(image)
        Show the image with pixel axes so you can hover-read the (x, y)
        pixel coordinate of each station. Run this once per image to
        build your ``station_coords`` dict.

    plot_engine_stations(image, df, station_coords, ...)
        Render the cross-section with a marker at each station; hover
        any marker for the performance values from your dataframe.

Requirements: ``plotly``, ``Pillow``, ``pandas``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image, ImageDraw, ImageFont


ImageLike = Union[str, Path, Image.Image]
Coord = Tuple[float, float]


def _load_image(image: ImageLike) -> Image.Image:
    """Accept a path (str / Path) or a PIL Image; return a PIL Image."""
    if isinstance(image, Image.Image):
        return image
    return Image.open(str(image))


def plot_engine_stations(
    image: ImageLike,
    df: pd.DataFrame,
    station_coords: Mapping[Union[str, float, int], Coord],
    station_col: str = "station",
    data_cols: Optional[Sequence[str]] = None,
    hot_stations: Iterable = (),
    width: int = 1400,
    height: Optional[int] = None,
    marker_size: int = 22,
    show_labels: bool = True,
) -> go.Figure:
    """
    Overlay performance data on a jet-engine cross-section image.

    Parameters
    ----------
    image
        Path to the cross-section image, or an already-loaded PIL Image.
    df
        Performance data; one row per station.
    station_coords
        Maps a station identifier (e.g. ``"2"``, ``"4.5"``) to a
        ``(x_pixel, y_pixel)`` tuple in image-pixel coordinates with the
        origin at the top-left (the same convention every image viewer
        uses). Use ``calibrate_stations`` to find these once per image.
    station_col
        Column in ``df`` holding the station identifier. Compared as a
        string so ``"4.5"`` matches both ``"4.5"`` and ``4.5``.
    data_cols
        Columns to include in the hover tooltip. Defaults to every
        numeric column except ``station_col``.
    hot_stations
        Station IDs to render in amber (hot section). Everything else
        is rendered in blue (cold section).
    width
        Figure width in pixels. ``height`` defaults to preserve the
        image's aspect ratio.
    marker_size
        Marker diameter in pixels.
    show_labels
        If True, draws the station ID inside each marker.

    Returns
    -------
    plotly.graph_objects.Figure
        Show with ``fig.show()`` in Jupyter.
    """
    img = _load_image(image)
    iw, ih = img.size

    if height is None:
        height = int(width * ih / iw) + 20

    if data_cols is None:
        data_cols = [
            c for c in df.select_dtypes("number").columns if c != station_col
        ]

    fig = go.Figure()

    # 1) Pin the cross-section image as a layout image filling the plot.
    fig.add_layout_image(
        dict(
            source=img,
            xref="x",
            yref="y",
            x=0,
            y=0,
            sizex=iw,
            sizey=ih,
            sizing="stretch",
            layer="below",
        )
    )

    hot_set = {str(s) for s in hot_stations}

    # 2) One scatter trace per station so each gets its own hover panel.
    for sid, (xp, yp) in station_coords.items():
        sid_s = str(sid)
        rows = df[df[station_col].astype(str) == sid_s]
        if rows.empty:
            continue
        row = rows.iloc[0]

        lines = [f"<b>Station {sid}</b>"]
        for c in data_cols:
            v = row.get(c)
            if v is None or pd.isna(v):
                continue
            if isinstance(v, (int, float)):
                lines.append(f"{c}: {v:.4g}")
            else:
                lines.append(f"{c}: {v}")
        hover = "<br>".join(lines)

        is_hot = sid_s in hot_set
        marker_color = "#EF9F27" if is_hot else "#378ADD"
        text_color = "#412402" if is_hot else "#042C53"

        fig.add_trace(
            go.Scatter(
                x=[xp],
                y=[yp],
                mode="markers+text" if show_labels else "markers",
                marker=dict(
                    size=marker_size,
                    color=marker_color,
                    line=dict(width=2, color="white"),
                    symbol="circle",
                ),
                text=[f"<b>{sid}</b>"] if show_labels else None,
                textposition="middle center",
                textfont=dict(color=text_color, size=11, family="Arial Black"),
                hovertext=hover,
                hoverinfo="text",
                showlegend=False,
                name=f"Station {sid}",
            )
        )

    # 3) Axes locked to image pixels; y reversed so (0,0) is top-left.
    fig.update_layout(
        xaxis=dict(range=[0, iw], visible=False, constrain="domain"),
        yaxis=dict(
            range=[ih, 0],
            visible=False,
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
        ),
        width=width,
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(
            bgcolor="rgba(20,20,20,0.95)",
            font=dict(color="white", family="monospace", size=12),
            bordercolor="rgba(255,255,255,0.25)",
        ),
    )
    return fig


def overlay_values_on_image(
    image: ImageLike,
    df: pd.DataFrame,
    location_cols: Mapping[Coord, Sequence[str]],
    label_col: Optional[str] = None,
    initial_row: int = 0,
    width: int = 1400,
    height: Optional[int] = None,
    font_size: int = 14,
    text_color: str = "white",
    background_color: str = "rgba(20,20,20,0.75)",
    border_color: str = "rgba(255,255,255,0.4)",
    frame_duration: int = 800,
    plot_cols: Optional[Sequence[str]] = None,
    x_col: Optional[str] = None,
    plot_row_height: int = 160,
) -> go.Figure:
    """
    Render an interactive Plotly figure with dataframe values displayed as
    text labels at specified pixel locations on the image. A slider lets the
    user step through every row of the dataframe, and Play/Pause buttons
    auto-advance through the rows.

    Parameters
    ----------
    image
        Path to the image, or an already-loaded PIL Image.
    df
        Data source; each row becomes one slider step / animation frame.
    location_cols
        Maps pixel (x, y) coordinates to a list of column names.
        At each location, column values from the active row are drawn.
    label_col
        Column whose value is used as the slider step label and prefix.
        If None, the slider uses positional row indices.
    initial_row
        Which row to show first (positional, 0-based). Defaults to 0.
    width
        Figure width in pixels. ``height`` defaults to preserve aspect ratio
        plus space for the slider and Play/Pause buttons.
    font_size
        Font size in pixels. Defaults to 14.
    text_color
        Text color (named color or hex). Defaults to "white".
    background_color
        Background color of the text label box. Defaults to translucent dark.
    border_color
        Border color of the text label box. Defaults to translucent white.
    frame_duration
        Milliseconds per frame during auto-play. Defaults to 800.
    plot_cols
        If provided, render an animated line subplot below the image for each
        listed column. Useful when rows represent successive time stamps in a
        transient simulation: a marker tracks the active row across each line.
    x_col
        Column to use as the x-axis for the line subplots (e.g. ``"time"``).
        If None, the dataframe's positional index is used.
    plot_row_height
        Pixel height of each line subplot when ``plot_cols`` is provided.
        Defaults to 160.

    Returns
    -------
    plotly.graph_objects.Figure
        Show with ``fig.show()`` in Jupyter.

    Example
    -------
    >>> location_cols = {
    ...     (250, 540): ["Tt_K", "Pt_kPa"],
    ...     (420, 540): ["W_kgs", "Mach"],
    ... }
    >>> fig = overlay_values_on_image(
    ...     image="engine.png",
    ...     df=df,
    ...     location_cols=location_cols,
    ...     label_col="station",
    ... )
    >>> fig.show()
    """
    img = _load_image(image)
    iw, ih = img.size

    use_subplots = plot_cols is not None and len(plot_cols) > 0
    n_plots = len(plot_cols) if use_subplots else 0

    # Margins reserved for axis titles, slider, and play/pause buttons.
    margin_l, margin_r, margin_t, margin_b = 70, 25, 20, 80
    vertical_spacing = 0.04 if use_subplots else 0.0

    # Image is displayed at native aspect ratio inside its subplot, which is
    # (figure_width - left/right margins) wide.
    image_subplot_width = width - margin_l - margin_r
    target_image_height = max(1, int(image_subplot_width * ih / iw))

    if height is None:
        if use_subplots:
            # plot_area = height - margin_t - margin_b
            # plot_area = sum(row_heights_px) + n_plots * vertical_spacing * height
            # row_heights_px = target_image_height + n_plots * plot_row_height
            # → height (1 - n_plots * vs) = margins + image_h + n_plots * plot_row_h
            denom = 1.0 - n_plots * vertical_spacing
            height = int(
                (
                    margin_t
                    + margin_b
                    + target_image_height
                    + n_plots * plot_row_height
                )
                / max(denom, 0.2)
            )
        else:
            height = target_image_height + margin_t + margin_b

    def _annotations_for_row(row: pd.Series) -> list:
        anns = []
        for (x, y), col_names in location_cols.items():
            lines = []
            for col in col_names:
                v = row.get(col)
                if v is None or pd.isna(v):
                    continue
                if isinstance(v, (int, float)):
                    lines.append(f"{col}: {v:.4g}")
                else:
                    lines.append(f"{col}: {v}")
            if not lines:
                continue
            anns.append(
                dict(
                    x=x,
                    y=y,
                    xref="x",
                    yref="y",
                    text="<br>".join(lines),
                    showarrow=False,
                    font=dict(
                        color=text_color, size=font_size, family="monospace"
                    ),
                    align="left",
                    bgcolor=background_color,
                    bordercolor=border_color,
                    borderwidth=1,
                    borderpad=4,
                    xanchor="center",
                    yanchor="middle",
                )
            )
        return anns

    # X-axis values for the line subplots.
    if use_subplots:
        if x_col is not None and x_col in df.columns:
            x_values = list(df[x_col])
            x_axis_title = x_col
        else:
            x_values = list(range(len(df)))
            x_axis_title = "row"

        # Compute a stable x-range with a small symmetric pad so the first
        # and last markers don't sit on the axis line.
        numeric_x = [v for v in x_values if isinstance(v, (int, float))]
        if numeric_x:
            x_min, x_max = min(numeric_x), max(numeric_x)
            x_pad = (x_max - x_min) * 0.02 if x_max > x_min else 0.5
            x_range = [x_min - x_pad, x_max + x_pad]
        else:
            x_range = None

    # Build figure (subplots when line plots are requested).
    if use_subplots:
        # Absolute pixel weights — plotly normalizes them across the plot area.
        row_heights = [target_image_height] + [plot_row_height] * n_plots
        # shared_xaxes=False so line plots don't inherit the image's pixel x
        # range (0..iw). The line plots are linked to each other below via
        # `matches` so they still zoom in sync.
        fig = make_subplots(
            rows=1 + n_plots,
            cols=1,
            row_heights=row_heights,
            vertical_spacing=vertical_spacing,
            shared_xaxes=False,
        )
    else:
        fig = go.Figure()

    # 1) Pin the image as a layout image on the first subplot axes.
    fig.add_layout_image(
        dict(
            source=img,
            xref="x",
            yref="y",
            x=0,
            y=0,
            sizex=iw,
            sizey=ih,
            sizing="stretch",
            layer="below",
        )
    )

    # 2) Static line traces + animated marker traces, one pair per plot_col.
    marker_trace_indices: list = []
    if use_subplots:
        for i, col in enumerate(plot_cols):
            row_idx = 2 + i  # row 1 is the image
            if col not in df.columns:
                continue
            y_series = list(df[col])

            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_series,
                    mode="lines",
                    name=col,
                    line=dict(color="#5AA9E6", width=2, shape="spline"),
                    showlegend=False,
                    hovertemplate=f"{x_axis_title}: %{{x}}<br>{col}: %{{y:.4g}}<extra></extra>",
                ),
                row=row_idx,
                col=1,
            )

            marker_trace_indices.append(len(fig.data))
            fig.add_trace(
                go.Scatter(
                    x=[x_values[initial_row]]
                    if 0 <= initial_row < len(x_values)
                    else [x_values[0]],
                    y=[y_series[initial_row]]
                    if 0 <= initial_row < len(y_series)
                    else [y_series[0]],
                    mode="markers",
                    marker=dict(
                        size=11,
                        color="#EF9F27",
                        line=dict(color="rgba(255,255,255,0.9)", width=1.5),
                        symbol="circle",
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row_idx,
                col=1,
            )

            axis_kwargs = dict(
                showgrid=True,
                gridcolor="rgba(255,255,255,0.08)",
                gridwidth=1,
                zeroline=False,
                showline=True,
                linecolor="rgba(255,255,255,0.25)",
                linewidth=1,
                tickfont=dict(color="rgba(230,230,230,0.85)", size=10),
                ticks="outside",
                tickcolor="rgba(255,255,255,0.25)",
                ticklen=4,
            )
            fig.update_yaxes(
                title_text=col,
                row=row_idx,
                col=1,
                title_font=dict(
                    color="rgba(230,230,230,0.9)", size=11, family="Arial"
                ),
                title_standoff=8,
                nticks=4,
                **axis_kwargs,
            )
            xaxis_extra = {}
            if x_range is not None:
                xaxis_extra["range"] = x_range
            # Link all line-plot x-axes to the first one so they zoom in sync.
            if i > 0:
                xaxis_extra["matches"] = "x2"
            fig.update_xaxes(
                row=row_idx,
                col=1,
                showticklabels=(i == n_plots - 1),
                title_text=x_axis_title if i == n_plots - 1 else None,
                title_font=dict(
                    color="rgba(230,230,230,0.9)", size=11, family="Arial"
                ),
                **axis_kwargs,
                **xaxis_extra,
            )

    # 3) Pre-compute annotations, frames, and slider steps for every row.
    # Frame names use the positional index to guarantee uniqueness even if
    # the label_col has duplicate values.
    per_row_annotations = []
    frames = []
    steps = []
    plot_y_series = (
        [list(df[c]) if c in df.columns else None for c in plot_cols]
        if use_subplots
        else []
    )

    for i, (idx, row) in enumerate(df.iterrows()):
        anns = _annotations_for_row(row)
        per_row_annotations.append(anns)

        if label_col is not None and label_col in row.index:
            label = str(row[label_col])
        else:
            label = str(idx)

        frame_kwargs = dict(name=str(i), layout=dict(annotations=anns))

        if use_subplots and marker_trace_indices:
            frame_data = []
            for j, c in enumerate(plot_cols):
                if plot_y_series[j] is None:
                    continue
                frame_data.append(
                    go.Scatter(
                        x=[x_values[i]],
                        y=[plot_y_series[j][i]],
                    )
                )
            frame_kwargs["data"] = frame_data
            frame_kwargs["traces"] = marker_trace_indices

        frames.append(go.Frame(**frame_kwargs))
        steps.append(
            dict(
                method="animate",
                args=[
                    [str(i)],
                    dict(
                        mode="immediate",
                        frame=dict(duration=frame_duration, redraw=True),
                        transition=dict(duration=0),
                    ),
                ],
                label=label,
            )
        )

    active = max(0, min(initial_row, len(per_row_annotations) - 1))
    fig.frames = frames

    # 4) Play/Pause buttons drive the animation.
    play_button = dict(
        label="▶ Play",
        method="animate",
        args=[
            None,
            dict(
                frame=dict(duration=frame_duration, redraw=True),
                transition=dict(duration=0),
                fromcurrent=True,
                mode="immediate",
            ),
        ],
    )
    pause_button = dict(
        label="⏸ Pause",
        method="animate",
        args=[
            [None],
            dict(
                frame=dict(duration=0, redraw=False),
                transition=dict(duration=0),
                mode="immediate",
            ),
        ],
    )

    # 5) Lock the image axes to pixel space (y reversed so (0,0) is top-left).
    #    scaleanchor enforces native aspect ratio. Combined with our pre-sized
    #    row_heights, the image subplot fills the row exactly. If there is
    #    a small mismatch (axis padding, etc.), the y domain shrinks rather
    #    than stretching the image.
    fig.update_layout(
        xaxis=dict(
            range=[0, iw],
            visible=False,
        ),
        yaxis=dict(
            range=[ih, 0],
            visible=False,
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
        ),
        width=width,
        height=height,
        margin=dict(l=margin_l, r=margin_r, t=margin_t, b=margin_b),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=per_row_annotations[active] if per_row_annotations else [],
        font=dict(color="rgba(230,230,230,0.9)"),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                showactive=False,
                buttons=[play_button, pause_button],
                x=0.02,
                xanchor="left",
                y=-0.02,
                yanchor="top",
                pad=dict(t=10, r=10),
                bgcolor="rgba(40,40,40,0.6)",
                bordercolor="rgba(255,255,255,0.2)",
                font=dict(color="rgba(230,230,230,0.95)", size=12),
            )
        ],
        sliders=[
            dict(
                active=active,
                currentvalue=dict(
                    prefix=f"{label_col}: " if label_col else "Row: ",
                    font=dict(size=13, color="rgba(230,230,230,0.95)"),
                    xanchor="left",
                ),
                steps=steps,
                x=0.15,
                len=0.83,
                pad=dict(t=30, b=10),
                bgcolor="rgba(255,255,255,0.12)",
                bordercolor="rgba(255,255,255,0.2)",
                tickcolor="rgba(255,255,255,0.3)",
                font=dict(color="rgba(230,230,230,0.85)", size=10),
                activebgcolor="#EF9F27",
            )
        ],
    )
    return fig


def calibrate_stations(image: ImageLike, width: int = 1400) -> go.Figure:
    """
    Display the image with visible pixel-coordinate axes so you can
    hover-read the (x, y) for each station. Origin is top-left.

    Workflow
    --------
    1. ``calibrate_stations("engine.png").show()``
    2. Hover over each station to see pixel coordinates.
    3. Copy (x, y) values from the hover tooltip.
    4. Build a ``station_coords`` dict and pass to ``plot_engine_stations``.
    """
    img = _load_image(image)
    iw, ih = img.size

    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=img,
            xref="x",
            yref="y",
            x=0,
            y=0,
            sizex=iw,
            sizey=ih,
            sizing="stretch",
            layer="below",
        )
    )
    
    # Add invisible scatter trace to show pixel coordinates on hover
    step_x = max(1, iw // 50)
    step_y = max(1, ih // 50)
    x_vals = list(range(0, iw, step_x))
    y_vals = list(range(0, ih, step_y))
    
    coords_x = []
    coords_y = []
    for x in x_vals:
        for y in y_vals:
            coords_x.append(x)
            coords_y.append(y)
    
    fig.add_trace(
        go.Scatter(
            x=coords_x,
            y=coords_y,
            mode='markers',
            marker=dict(size=0, opacity=0),
            hovertemplate='x: %{x}<br>y: %{y}<extra></extra>',
            showlegend=False,
        )
    )
    
    fig.update_layout(
        xaxis=dict(
            range=[0, iw],
            showgrid=True,
            gridcolor="rgba(255, 80, 80, 0.25)",
            title="x (pixels)",
        ),
        yaxis=dict(
            range=[ih, 0],
            scaleanchor="x",
            scaleratio=1,
            showgrid=True,
            gridcolor="rgba(255, 80, 80, 0.25)",
            title="y (pixels)",
        ),
        width=width,
        height=int(width * ih / iw) + 80,
        plot_bgcolor="rgba(0, 0, 0, 0)",
        title="Hover to read pixel coordinates for each station",
        margin=dict(l=60, r=20, t=50, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Example use in a Jupyter cell
# ---------------------------------------------------------------------------
#
#     import pandas as pd
#     from engine_stations import plot_engine_stations, calibrate_stations
#
#     # First time: find the pixel coords of each station on YOUR image.
#     calibrate_stations(r"data/engine_cross_section.png").show()
#
#     # Then save them — for example:
#     station_coords = {
#         "2":   (250, 540),
#         "2.5": (420, 540),
#         "3":   (770, 540),
#         "4":   (970, 540),
#         "4.5": (1130, 540),
#         "5":   (1370, 540),
#         "8":   (1820, 540),
#     }
#
#     # Your performance dataframe (one row per station).
#     df = pd.DataFrame({
#         "station": ["2", "2.5", "3", "4",  "4.5", "5",  "8"],
#         "Tt_K":    [288,  358,  780, 1680, 1320,  920,  900],
#         "Pt_kPa":  [101,  175,  3450, 3280, 1100, 215,  105],
#         "W_kgs":   [320,   65,   65,   67,   68,   68,   68],
#         "Mach":    [0.50, 0.45, 0.30, 0.10, 0.50, 0.50, 1.00],
#     })
#
#     fig = plot_engine_stations(
#         image=r"data/engine_cross_section.png",
#         df=df,
#         station_coords=station_coords,
#         hot_stations=["4", "4.5", "5", "8"],
#     )
#     fig.show()