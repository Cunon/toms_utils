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

    if height is None:
        height = int(width * ih / iw) + 100  # room for slider + buttons

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

    # Pre-compute annotations, frames, and slider steps for every row.
    # Frame names use the positional index to guarantee uniqueness even if
    # the label_col has duplicate values.
    per_row_annotations = []
    frames = []
    steps = []
    for i, (idx, row) in enumerate(df.iterrows()):
        anns = _annotations_for_row(row)
        per_row_annotations.append(anns)

        if label_col is not None and label_col in row.index:
            label = str(row[label_col])
        else:
            label = str(idx)

        frame_name = str(i)
        frames.append(go.Frame(name=frame_name, layout=dict(annotations=anns)))
        steps.append(
            dict(
                method="animate",
                args=[
                    [frame_name],
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

    fig = go.Figure()
    fig.frames = frames

    # 1) Pin the image as a layout image filling the plot.
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

    # 2) Play/Pause buttons drive the animation.
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

    # 3) Axes locked to image pixels; y reversed so (0,0) is top-left.
    #    Initial annotations match the active slider step.
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
        margin=dict(l=0, r=0, t=0, b=80),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=per_row_annotations[active] if per_row_annotations else [],
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
            )
        ],
        sliders=[
            dict(
                active=active,
                currentvalue=dict(
                    prefix=f"{label_col}: " if label_col else "Row: ",
                    font=dict(size=14),
                ),
                steps=steps,
                x=0.15,
                len=0.83,
                pad=dict(t=30, b=10),
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