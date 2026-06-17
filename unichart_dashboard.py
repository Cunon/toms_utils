"""On-the-fly Dash dashboards that combine multiple unichart figures.

`unichart.py` plotting methods each return a Plotly ``go.Figure`` and cache the
most recent one in ``nb.last_fig``. This module wires those figures into an
interactive Dash board: a grid of panels, each with its own controls (plot type,
x / y variables, dataset on/off, suptitle, legend position) that re-plot live.

Typical use, inline in a Jupyter notebook::

    from unichart_dashboard import dashboard

    dashboard(nb, panels=[
        {'method': 'plot', 'x': 'time', 'y': 'temp'},
        {'method': 'bar',  'x': 'cat',  'y': 'val', 'datasets': [0, 1]},
    ], ncols=2)

Dash is an optional dependency; it is imported lazily so importing the core
toolkit never requires it.
"""

import threading

import plotly.graph_objects as go


# Dataset .select is shared, mutable notebook state. Panel renders flip it,
# read the figure, then restore it, so renders must not interleave.
_RENDER_LOCK = threading.Lock()

# Plot methods offered in the per-panel "plot type" switch. contour is excluded
# because it requires a z column, which this generic x/y board doesn't model.
PLOT_METHODS = ['plot', 'plot_ymult', 'bar', 'box', 'histogram']

# Methods whose signature accepts a `legend=` argument (above/right/off).
_LEGEND_METHODS = {'plot', 'plot_ymult'}

LEGEND_POSITIONS = ['above', 'right', 'off']


def _require_dash():
    """Import Dash lazily, with a friendly error if it isn't installed."""
    try:
        import dash  # noqa: F401
        from dash import Dash, dcc, html, Input, Output, MATCH
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "unichart_dashboard requires Dash. Install it with "
            "`pip install dash` (or add it to requirements.txt)."
        ) from exc
    return Dash, dcc, html, Input, Output, MATCH


def _all_columns(nb):
    """Sorted union of column names across every dataset on the notebook."""
    cols = set()
    for ds in nb.sets:
        cols.update(str(c) for c in ds.columns)
    return sorted(cols)


def _selected_indices(nb):
    """Indices of currently-selected datasets."""
    return [ds.index for ds in nb.sets if ds.select]


def render_panel(nb, method, x, y, dataset_indices, suptitle=None, legend='above'):
    """Render one panel to a ``go.Figure`` using the notebook's plot methods.

    Temporarily sets dataset selection to ``dataset_indices``, dispatches to
    ``nb.<method>`` with the given x/y/suptitle (and legend for the methods that
    support it), then restores the prior selection. Reads the figure from
    ``nb.last_fig`` so it works even under ``static_images`` mode. On any error,
    selection is restored and an empty figure carrying the error text is returned
    so one bad panel can't take down the board.

    This is the pure core of the Dash callback and is callable directly in tests.
    """
    if method not in PLOT_METHODS:
        return _error_figure(f"Unknown plot method: {method!r}")

    chosen = set(dataset_indices or [])

    with _RENDER_LOCK:
        snapshot = [(ds, ds.select) for ds in nb.sets]
        try:
            for ds in nb.sets:
                ds.select = ds.index in chosen

            kwargs = {'x': x, 'y': y}
            if suptitle:
                kwargs['suptitle'] = suptitle
            if method in _LEGEND_METHODS:
                kwargs['legend'] = legend

            getattr(nb, method)(**kwargs)
            fig = nb.last_fig
            if fig is None:
                return _error_figure("No figure produced (no data / selection?)")
            # unichart bakes figsize into width/height (see _base_layout); drop it
            # so each figure flexes to fill its grid cell instead of overflowing.
            fig.update_layout(autosize=True, width=None, height=None)
            return fig
        except Exception as exc:  # noqa: BLE001 - surface any plotting error in-panel
            return _error_figure(f"{type(exc).__name__}: {exc}")
        finally:
            for ds, was in snapshot:
                ds.select = was


def _error_figure(message):
    """A blank figure that displays an error message in the panel."""
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False,
                       xref='paper', yref='paper', x=0.5, y=0.5,
                       font=dict(color='crimson'))
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False),
                      margin=dict(l=20, r=20, t=20, b=20))
    return fig


def _normalize_y(y):
    """Y control is multi-select; normalize seeds to a list of strings."""
    if y is None:
        return []
    if isinstance(y, (list, tuple)):
        return [str(v) for v in y]
    return [str(y)]


def build_app(nb, panels, ncols=2, height=420):
    """Build (but do not run) the Dash app for the given panels.

    Returns the configured ``dash.Dash`` instance with its layout and the single
    pattern-matching callback registered. Factored out of :func:`dashboard` so
    the app/layout can be constructed and inspected without starting a server.
    """
    Dash, dcc, html, Input, Output, MATCH = _require_dash()

    if not nb.sets:
        raise ValueError("The notebook has no datasets loaded.")
    if not panels:
        raise ValueError("Provide at least one panel.")

    col_options = _all_columns(nb)
    dataset_options = [{'label': ds.title_format, 'value': ds.index}
                       for ds in nb.sets]
    default_selected = _selected_indices(nb)

    app = Dash(__name__)
    app.layout = html.Div(
        [_panel_div(html, dcc, i, panel, col_options, dataset_options,
                    default_selected, height, nb)
         for i, panel in enumerate(panels)],
        style={'display': 'grid',
               'gridTemplateColumns': f'repeat({ncols}, 1fr)',
               'gap': '16px', 'padding': '8px'},
    )

    @app.callback(
        Output({'type': 'panel-graph', 'index': MATCH}, 'figure'),
        Input({'type': 'panel-method', 'index': MATCH}, 'value'),
        Input({'type': 'panel-x', 'index': MATCH}, 'value'),
        Input({'type': 'panel-y', 'index': MATCH}, 'value'),
        Input({'type': 'panel-datasets', 'index': MATCH}, 'value'),
        Input({'type': 'panel-suptitle', 'index': MATCH}, 'value'),
        Input({'type': 'panel-legend', 'index': MATCH}, 'value'),
        prevent_initial_call=True,  # initial figures are baked into each Graph
    )
    def _update_panel(method, x, y, datasets, suptitle, legend):
        return render_panel(nb, method, x, y, datasets, suptitle, legend)

    return app


def _control(html, label, component):
    """Label a control and stack it in a small flex column."""
    return html.Div(
        [html.Label(label, style={'fontSize': '11px', 'color': '#555'}), component],
        style={'display': 'flex', 'flexDirection': 'column', 'minWidth': '120px'},
    )


def _panel_div(html, dcc, i, panel, col_options, dataset_options,
               default_selected, height, nb):
    """One panel: a row of controls above its graph."""
    method = panel.get('method', 'plot')
    x = panel.get('x')
    y = _normalize_y(panel.get('y'))
    suptitle = panel.get('suptitle')
    legend = panel.get('legend', 'above')
    selected = panel.get('datasets', default_selected)

    controls = html.Div(
        [
            _control(html, 'plot type', dcc.Dropdown(
                id={'type': 'panel-method', 'index': i},
                options=PLOT_METHODS, value=method, clearable=False)),
            _control(html, 'x', dcc.Dropdown(
                id={'type': 'panel-x', 'index': i},
                options=col_options, value=x)),
            _control(html, 'y', dcc.Dropdown(
                id={'type': 'panel-y', 'index': i},
                options=col_options, value=y, multi=True)),
            _control(html, 'datasets', dcc.Checklist(
                id={'type': 'panel-datasets', 'index': i},
                options=dataset_options, value=list(selected),
                style={'fontSize': '12px'})),
            _control(html, 'suptitle', dcc.Input(
                id={'type': 'panel-suptitle', 'index': i},
                type='text', value=suptitle or '',
                debounce=True, style={'width': '140px'})),
            _control(html, 'legend', dcc.Dropdown(
                id={'type': 'panel-legend', 'index': i},
                options=LEGEND_POSITIONS, value=legend, clearable=False)),
        ],
        style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '8px',
               'alignItems': 'flex-start', 'marginBottom': '6px'},
    )

    initial = render_panel(nb, method, x, y, selected, suptitle, legend)
    graph = dcc.Graph(id={'type': 'panel-graph', 'index': i},
                      figure=initial, responsive=True,
                      style={'height': f'{height}px', 'width': '100%'})

    return html.Div([controls, graph],
                    style={'border': '1px solid #ddd', 'borderRadius': '6px',
                           'padding': '8px'})


def dashboard(nb, panels, ncols=2, height=420, jupyter_mode='inline',
              port=8050, debug=False, **run_kwargs):
    """Build and launch an interactive Dash board combining unichart figures.

    Parameters
    ----------
    nb : UnichartNotebook
        The notebook whose datasets and plot methods drive the panels.
    panels : list[dict]
        One dict per panel. Recognized keys: ``method`` (one of
        :data:`PLOT_METHODS`, default ``'plot'``), ``x``, ``y`` (str or list),
        ``suptitle``, ``legend`` (``above``/``right``/``off``), and ``datasets``
        (list of dataset indices to select initially; defaults to the notebook's
        current selection).
    ncols : int
        Number of columns in the panel grid.
    height : int
        Height in px of each panel's graph.
    jupyter_mode : str
        Passed to ``Dash.run`` — ``'inline'`` (default) renders in the notebook
        cell; ``'external'`` / ``'tab'`` open a browser.
    port, debug, **run_kwargs
        Forwarded to ``Dash.run``.

    Returns
    -------
    dash.Dash
        The running app instance (useful for inspection / further wiring).
    """
    app = build_app(nb, panels, ncols=ncols, height=height)
    app.run(jupyter_mode=jupyter_mode, port=port, debug=debug, **run_kwargs)
    return app
