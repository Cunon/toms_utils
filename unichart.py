import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pcolors
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
import warnings
import numbers
import ipywidgets as widgets
from IPython.display import display, clear_output
import re
import inspect
from scipy.interpolate import griddata
from toms_utils.dataframe_manipulation import open_datafile, keep_columns, align_dataframes


# -----------------------------------------------------------------------------
# Constants & Mappers (Translation Layer)
# -----------------------------------------------------------------------------

MARKER_MAP_MPL_TO_PLOTLY = {
    'o': 'circle', 's': 'square', 'D': 'diamond', 'd': 'diamond-tall',
    'v': 'triangle-down', '^': 'triangle-up', '<': 'triangle-left', '>': 'triangle-right',
    'p': 'pentagon', '*': 'star', 'h': 'hexagon', 'H': 'hexagon2',
    'x': 'x', 'X': 'x-thin', '+': 'cross', '|': 'line-ns', '_': 'line-ew', '.': 'circle-dot'
}

LINESTYLE_MAP_MPL_TO_PLOTLY = {
    '-': 'solid', '--': 'dash', '-.': 'dashdot', ':': 'dot',
    'None': None, ' ': None, '': None
}

def get_plotly_marker(mpl_marker):
    return MARKER_MAP_MPL_TO_PLOTLY.get(mpl_marker, 'circle')

def get_plotly_linestyle(mpl_style):
    return LINESTYLE_MAP_MPL_TO_PLOTLY.get(mpl_style, 'solid')

def validate_color(value):
    """
    Validate if the provided value is a valid color (Hex, RGB, or standard string).
    Plotly is permissive, so we mostly check for string validity.
    """
    if not isinstance(value, str):
        return False
    # Basic check for Hex or standard names. 
    # In a full util, we might check against plotly.colors.named_colorscales()
    return True

def validate_marker(value):
    return value in MARKER_MAP_MPL_TO_PLOTLY or value is None

def validate_linestyle(value):
    return value in LINESTYLE_MAP_MPL_TO_PLOTLY or value is None

def marker_map(index):
    markers = [m for m in MARKER_MAP_MPL_TO_PLOTLY.keys() if m not in ['|', '_', '.', 'X']]
    return markers[index % len(markers)]

def _generate_contour_grid(x_data, y_data, z_data, res=100, method='linear'):
    """
    Interpolates scattered x, y, z data into a uniform 2D grid for contour plotting.
    Leaves data outside the convex hull as NaN.
    """
    # Filter out any rows where x, y, or z are NaN
    valid = ~(np.isnan(x_data) | np.isnan(y_data) | np.isnan(z_data))
    x_val = x_data[valid]
    y_val = y_data[valid]
    z_val = z_data[valid]

    # Failsafe for insufficient data
    if len(x_val) < 4:
        return x_val.values, y_val.values, z_val.values

    # Create uniform grid axes
    xi = np.linspace(x_val.min(), x_val.max(), res)
    yi = np.linspace(y_val.min(), y_val.max(), res)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    # Interpolate Z values onto the grid
    zi_grid = griddata((x_val, y_val), z_val, (xi_grid, yi_grid), method=method)

    return xi, yi, zi_grid

# -----------------------------------------------------------------------------
# Dataset Class
# -----------------------------------------------------------------------------

class Dataset:
    """
    A class to represent a dataset and manage its plotting attributes.
    Refactored to be renderer-agnostic but maintains original API.
    """

    def __init__(self, df, index=0, title=None, display_parms=None):
        self._df_full = df
        self._df_filtered = df
        self.query = None
        self._select = True
        
        # Title logic
        if title:
            self.title = title
        elif "TITLE" in df.columns:
            self.title = str(df["TITLE"].iloc[0])
        else:
            self.title = "Untitled"
            
        self.index = index
        self.title_format = f"{self.title} {index}"
        
        # Default colors (cycling through Plotly default palette)
        default_colors = px.colors.qualitative.Plotly
        self._color = default_colors[index % len(default_colors)]
        
        self._marker = marker_map(index)
        self._edge_color = "black"
        self._linestyle = None
        self.markersize = 10  # Adjusted default for Plotly
        self.alpha = 1
        self.hue = ""
        self.hue_palette = "Jet"
        self.hue_order = None
        self.reg_order = None
        self.style = None
        self.linewidth = 2
        self.edgewidth = 1
        self.set_type = 1
        self.data_type = 'discrete'
        self.delta_sets = None
        self.file_path = None
        self._display_parms = display_parms if display_parms else []
        self._plot_type = None
        self._order = None

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        if (value in self._df_full.columns) or (value is None):
            self._order = value
        else:
            raise ValueError(f"Invalid order column: {value}")

    @property
    def df(self):
        return self._df_filtered

    @df.setter
    def df(self, value):
        self._df_full = value
        self._apply_query()

    @property
    def query(self):
        return self._query

    @query.setter
    def query(self, value):
        self._query = value
        self._apply_query()

    def _apply_query(self):
        if not self._query:
            self._df_filtered = self._df_full
        else:
            try:
                result_df = self._df_full.query(self._query)
                if not result_df.empty:
                    self._df_filtered = result_df
                else:
                    print(f"No data in set {self.index} after query: {self._query}. Turning Set Off...")
                    self.select = False
                    self._df_filtered = self._df_full
            except Exception as e:
                raise ValueError(f"Query error: {e}")

    @property
    def color(self): return self._color

    @color.setter
    def color(self, value): self._color = value
        
    @property
    def select(self): return self._select

    @select.setter
    def select(self, value):
        if str(value).lower() in ['true', '1', 't', 'on']:
            self._select = True
        elif str(value).lower() in ['false', '0', 'f', 'off']:
            self._select = False
        else:
            raise ValueError(f"Invalid value for select: {value}")

    @property
    def edge_color(self): return self._edge_color

    @edge_color.setter
    def edge_color(self, value): self._edge_color = value

    @property
    def plot_type(self): return self._plot_type

    @plot_type.setter
    def plot_type(self, value):
        valid_plot_types = ['scatter', 'contour', 'histogram', 'box', 'bar', None]
        if value in valid_plot_types:
            self._plot_type = value
        else:
            raise ValueError(f"Invalid plot_type: {value}")
        
    @property
    def marker(self): return self._marker

    @marker.setter
    def marker(self, value): self._marker = value

    @property
    def linestyle(self): return self._linestyle

    @linestyle.setter
    def linestyle(self, value): self._linestyle = value

    @property
    def linewidth(self): return self._linewidth

    @linewidth.setter
    def linewidth(self, value):
        if isinstance(value, (int, float)) and value >= 0:
            self._linewidth = value
        else:
            raise ValueError(f"Invalid linewidth: {value}")

    @property
    def edgewidth(self): return self._edgewidth

    @edgewidth.setter
    def edgewidth(self, value):
        if isinstance(value, (int, float)) and value >= 0:
            self._edgewidth = value
        else:
            raise ValueError(f"Invalid edgewidth: {value}")

    @property
    def display_parms(self): return self._display_parms

    @display_parms.setter
    def display_parms(self, value):
        if isinstance(value, list):
            self._display_parms = value
        else:
            raise ValueError(f"display_parms must be a list")

    def sel_query(self, query):
        self.query = query

    def update_format_dict(self, format_options):
        for key, value in format_options.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid format key: {key}")

    def get_format_dict(self):
        return {
            'title': self.title,
            'color': self.color,
            'marker': self.marker,
            'edge_color': self.edge_color,
            'linestyle': self.linestyle,
            'markersize': self.markersize,
            'alpha': self.alpha,
            'hue': self.hue,
            'hue_palette': self.hue_palette,
            'hue_order': self.hue_order,
            'reg_order': self.reg_order,
            'index': self.index,
            'style': self.style,
            'display_parms': self.display_parms,
            'plot_type': self.plot_type,
            'linewidth': self.linewidth,
            'edgewidth': self.edgewidth,
            'delta_sets': self.delta_sets,
        }
    
    def set_format_option(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise ValueError(f"Invalid format key: {key}")

    def get_title(self):
        return self.title

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def table_read(df, x_col, y_col, x_in, kind='linear', fill_value='extrapolate', bounds_error=False):
    """
    Perform interpolation on a table. Kept identical to source for API compatibility.
    """
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Columns '{x_col}' and '{y_col}' must be present in the DataFrame.")

    df_sorted = df.sort_values(by=x_col)
    x_values = df_sorted[x_col].values
    y_values = df_sorted[y_col].values

    f = interp1d(
        x_values,
        y_values,
        kind=kind,
        fill_value=fill_value,
        bounds_error=bounds_error,
    )

    y_interp = f(x_in)
    return y_interp

def _calculate_regression(df, x_col, y_col, order):
    """Internal helper to calculate regression lines."""
    from scipy.optimize import curve_fit
    df_clean = df.dropna(subset=[x_col, y_col]).sort_values(by=x_col)
    x = df_clean[x_col]
    y = df_clean[y_col]
    
    if len(x) < 3: # Minimum points needed for most curves
        return x, y * np.nan
        
    x_lin = np.linspace(x.min(), x.max(), 100)
    
    if isinstance(order, int):
        z = np.polyfit(x, y, order)
        p = np.poly1d(z)
        y_lin = p(x_lin)
    elif isinstance(order, str):
        order = order.lower()
        try:
            if order == 'exp':
                def func(x, a, b): return a * np.exp(b * x)
                popt, _ = curve_fit(func, x, y)
                y_lin = func(x_lin, *popt)
            elif order == 'log':
                def func(x, a, b): return a * np.log(x) + b
                popt, _ = curve_fit(func, x, y)
                y_lin = func(x_lin, *popt)
            elif order == 'power':
                def func(x, a, b): return a * (x ** b)
                popt, _ = curve_fit(func, x, y)
                y_lin = func(x_lin, *popt)
            else:
                y_lin = x_lin * np.nan
        except:
            # If the curve fit fails to converge, return NaNs so the plot doesn't crash
            y_lin = x_lin * np.nan 
    else:
        y_lin = x_lin * np.nan
        
    return x_lin, y_lin

# -----------------------------------------------------------------------------
# Main Plotting Functions
# -----------------------------------------------------------------------------
def uniplot(list_of_datasets, x, y, color=None, hue=None, marker=None,
            markersize=10, marker_edge_color="black", linestyle=None, hue_palette="Jet",
            hue_order=None, line=False, suppress_msg=False, return_axes=False, axes=None,
            suptitle=None, xlabel=None, ylabel=None, subplot_titles=None,
            darkmode=False, interactive=True, display_parms=None, grid=True,
            legend='above', legend_ncols=1, figsize=(12, 8), ncols=None, nrows=None, x_lim=None, y_lim=None):
    """
    Create unified multi-dataset plots using Plotly with flexible subplot configuration.
    This function generates comprehensive visualizations for multiple datasets with support for
    subplots, regression lines, custom styling, and interactive features. It automatically
    handles subplot layout, legend grouping for trace toggling, and provides extensive
    customization options for colors, markers, and hover information.

    Parameters
    ----------
    list_of_datasets : list
        List of dataset objects containing .df (DataFrame), .select (bool), .order (str),
        and .get_format_dict() method for styling configuration.
    x : str
        Column name for x-axis values.
    y : str or list of str
        Column name(s) for y-axis values. Multiple values create subplots.
    color : str, optional
        Color for plot elements. Can be overridden by dataset format dict.
    hue : str, optional
        Column name for color-coding data points.
    marker : str, optional
        Marker symbol for scatter points.
    markersize : int, default=10
        Size of markers in scatter plots.
    marker_edge_color : str, default="black"
        Edge color for markers.
    linestyle : str, optional
        Line style for connecting points (e.g., 'solid', 'dashed').
    hue_palette : str, default="Jet"
        Color palette for hue-based coloring.
    hue_order : list, optional
        Order of categories for hue-based coloring.
    line : bool, optional
        Whether to draw lines connecting points.
    suppress_msg : bool, default=False
        Whether to suppress informational messages.
    return_axes : bool, default=False
        If True, return the figure object without displaying.
    axes : object, optional
        Pre-existing axes object to plot on.
    suptitle : str, optional
        Main title for the entire figure.
    xlabel : str, optional
        Label for x-axis.
    ylabel : str, optional
        Label for y-axis.
    subplot_titles : list of str, optional
        Titles for individual subplots.
    darkmode : bool, default=False
        If True, use dark theme template.
    interactive : bool, default=True
        Whether to enable interactive features.
    display_parms : list, optional
        Additional columns to include in hover information.
    grid : bool, default=True
        Whether to show grid lines.
    legend : str, default='above'
        Legend position ('above', 'off', or other Plotly legend positions).
    legend_ncols : int, default=1
        Number of columns for legend items.
    figsize : tuple, default=(12, 8)
        Figure size in inches (width, height).
    ncols : int, optional
        Number of columns for subplot grid.
    nrows : int, optional
        Number of rows for subplot grid.
    x_lim : tuple, optional
        X-axis limits as (min, max).
    y_lim : tuple, optional
        Y-axis limits as (min, max).

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure object containing all traces and layout.

    Raises
    ------
    ValueError
        If y contains no valid columns.

    Notes
    -----
    - Subplot layout is automatically calculated if nrows and ncols are not specified.
    - Legend groups are created per dataset to enable trace toggling across subplots.
    - Regression lines can be added by setting reg_order in dataset format dict.
    - Hover templates support custom data columns for enhanced interactivity.
    - Numeric hover values are formatted with 5 significant figures.
    - Duplicate columns in datasets are automatically removed.
    - Column names are case-insensitive for x-axis lookup.

    Examples
    --------
    >>> from toms_utils import Dataset
    >>> datasets = [Dataset(df1, select=True), Dataset(df2, select=True)]
    >>> fig = uniplot(datasets, x='time', y=['value1', 'value2'],
    ...               suptitle='Time Series Analysis', darkmode=True)
    >>> # With regression
    >>> datasets[0].format_dict['reg_order'] = 2
    >>> fig = uniplot(datasets, x='x', y='y', return_axes=True)
    >>> # Custom hover information
    >>> fig = uniplot(datasets, x='date', y='price',
    ...               display_parms=['volume', 'sector'])
    """
    y_list = y if isinstance(y, list) else [y]
    n_y = len(y_list)
    
    if n_y == 0:
        raise ValueError("y must contain at least one column.")

    # Determine subplot layout
    if nrows is None and ncols is None:
        ncols_auto = min(3, max(1, int(np.ceil(np.sqrt(n_y)))))
        nrows_auto = int(np.ceil(n_y / ncols_auto))
        nrows, ncols = nrows_auto, ncols_auto
    elif nrows is None:
        nrows = int(np.ceil(n_y / ncols))
    elif ncols is None:
        ncols = int(np.ceil(n_y / nrows))

    # Initialize subplots
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subplot_titles, shared_xaxes=False)

    # --- LAYOUT SETUP ---
    final_title = suptitle if suptitle else f"{x} vs {[str(yi) for yi in y_list]}"
    
    layout_args = {
        'template': "plotly_dark" if darkmode else "plotly_white",
        'title': {
            'text': final_title,
            'x': 0.5,
            'xanchor': 'center'
        },
        'showlegend': True if legend != 'off' else False
    }

    if figsize:
        layout_args['width'] = figsize[0] * 100
        layout_args['height'] = figsize[1] * 100

    fig.update_layout(**layout_args)

    # --- PLOT LOOP ---
    for dataset in list_of_datasets:
        if not dataset.select:
            continue
        df = dataset.df.copy()
        
        df = df.loc[:, ~df.columns.duplicated()]
        
        if dataset.order == 'index': df = df.sort_index()
        elif dataset.order: df = df.sort_values(by=dataset.order)
        else: df = df.sort_index()

        # Resolve formatting
        fmt = dataset.get_format_dict()
        cur_title = fmt.get('title')
        cur_hue = fmt.get('hue') or hue
        cur_color = color or fmt.get('color')
        cur_marker = marker or fmt.get('marker')
        cur_linestyle = linestyle or fmt.get('linestyle')
        cur_markersize = fmt.get('markersize', markersize)
        cur_linewidth = fmt.get('linewidth', 2)
        cur_alpha = fmt.get('alpha', 1)
        cur_reg_order = fmt.get('reg_order')
        cur_idx = fmt.get('index')
        hover_parms = display_parms or fmt.get('display_parms', [])

        # Determine if reg_order is valid early to strip lines from raw data points
        valid_reg = False
        if isinstance(cur_reg_order, numbers.Number) and cur_reg_order > 0:
            valid_reg = True
        elif isinstance(cur_reg_order, str) and str(cur_reg_order).lower() in ['exp', 'log', 'power']:
            valid_reg = True

        for idx_y, yi in enumerate(y_list):
            row = idx_y // ncols + 1
            col = idx_y % ncols + 1

            if yi not in df.columns:
                print(f"Skipping {yi} for set {cur_idx}: column not found.")
                continue

            # X-Column Resolution
            if x not in df.columns:
                cols_upper = {c.upper(): c for c in df.columns}
                x_key = cols_upper.get(str(x).upper())
                if not x_key:
                    print(f"Skipping plot: x='{x}' not found.")
                    continue
                x_col = x_key
            else:
                x_col = x

            # Custom Data & Hover
            hover_cols = [p for p in hover_parms if p in df.columns]
            custom_data_cols = []
            seen_cols = set()
            
            for c in [x_col, yi]:
                if c not in seen_cols:
                    custom_data_cols.append(c)
                    seen_cols.add(c)
            
            for c in hover_cols:
                if c not in seen_cols:
                    custom_data_cols.append(c)
                    seen_cols.add(c)
            
            custom_data = df[custom_data_cols]
            
            def get_cd_idx(col_name):
                return custom_data_cols.index(col_name)

            def format_row(col, num_fmt):
                """Returns the Plotly customdata string with conditional numeric formatting."""
                fmt = f":{num_fmt}" if pd.api.types.is_numeric_dtype(df[col]) else ""
                return f"{col}: %{{customdata[{get_cd_idx(col)}]{fmt}}}"

            ht_parts = [
                f"<b><u>Set: {cur_idx}</u></b>",
                f"<b>{cur_title}</b>",
                format_row(x_col, ".2f"),
                format_row(yi, ".2f")     # yi is now evaluated on its own data type
            ]
            ht_parts.extend([format_row(parm, ".5g") for parm in hover_cols])
            ht = "<br>".join(ht_parts) + "<extra></extra>"

            # Control Scatter Mode based on active regressions
            mode_parts = []
            if valid_reg:
                mode_parts.append('markers')
            else:
                if not cur_linestyle: mode_parts.append('markers')
                else: mode_parts.append('lines')
                if cur_marker: mode_parts.append('markers')
            mode = "+".join(list(dict.fromkeys(mode_parts))) if mode_parts else 'markers'

            marker_dict = dict(
                size=cur_markersize,
                symbol=get_plotly_marker(cur_marker),
                line=dict(width=fmt.get('edgewidth', 1), color=fmt.get('edge_color', 'black')),
                opacity=cur_alpha
            )
            line_dict = dict(width=cur_linewidth, dash=get_plotly_linestyle(cur_linestyle))

            if cur_hue and cur_hue in df.columns:
                hue_data = df[cur_hue]
                if pd.api.types.is_numeric_dtype(hue_data):
                    marker_dict['color'] = hue_data
                    marker_dict['colorbar'] = dict(title=cur_hue)
                else:
                    hue_series = hue_data.astype('category')
                    marker_dict['color'] = hue_series.cat.codes
                    marker_dict['colorscale'] = fmt.get('hue_palette', 'Jet')
                    marker_dict['showscale'] = False 
            else:
                marker_dict['color'] = cur_color
                line_dict['color'] = cur_color
            
            fig.add_trace(go.Scatter(
                x=df[x_col], y=df[yi], mode=mode,
                name=f"{cur_idx}: {cur_title}",
                legendgroup=f"group_{cur_idx}",
                marker=marker_dict, line=line_dict,
                customdata=custom_data, hovertemplate=ht,
                showlegend=(idx_y == 0)
            ), row=row, col=col)

            if valid_reg:
                rx, ry = _calculate_regression(df, x_col, yi, cur_reg_order)
                fig.add_trace(go.Scatter(
                    x=rx, y=ry, mode='lines',
                    name=f"{cur_idx}: {cur_title} Fit {cur_reg_order}",
                    legendgroup=f"group_{cur_idx}",
                    line=dict(color=cur_color, width=cur_linewidth, dash=get_plotly_linestyle(cur_linestyle)),
                    opacity=0.7, hoverinfo='skip', showlegend=False
                ), row=row, col=col)

    for idx_y, yi in enumerate(y_list):
        row = idx_y // ncols + 1
        col = idx_y % ncols + 1
        
        axis_title = ylabel if ylabel else yi
        x_axis_title = xlabel if xlabel else x
        
        fig.update_yaxes(title_text=axis_title, title_standoff=15, row=row, col=col)
        fig.update_xaxes(title_text=x_axis_title, title_standoff=15, row=row, col=col)

    # Global Axis Limits
    for idx_y, yi in enumerate(y_list):
        row = idx_y // ncols + 1
        col = idx_y % ncols + 1
        if y_lim: fig.update_yaxes(range=y_lim, row=row, col=col)
    
    if x_lim: fig.update_xaxes(range=x_lim)
    if not grid:
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
    if legend == 'above':
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))

    if return_axes: return fig
    fig.show()
    return fig


def uniplot_per_dataset(list_of_datasets, x, y, display_parms=None, 
                        suptitle=None, figsize=(12, 8), ncols=None, nrows=None, 
                        darkmode=True, x_lim=None, y_lim=None, axis_limits=None, return_axes=False):
    """
    Revised plotting engine for Multi-Y Axis.
    - Fixes 'ValueError: position' by shrinking X-axis domain to make room for extra axes.
    - Supports proper line widths, dash styles, and dynamically mapped regression lines.
    Regression lines inherit the dataset's linestyle; underlying scatter suppresses lines if reg_order is present.
    """
    active_datasets = [d for d in list_of_datasets if d.select]
    if not active_datasets:
        print("No datasets selected.")
        return None

    y_list = y if isinstance(y, list) else [y]
    axis_limits = axis_limits or {}
    
    # --- Grid Setup ---
    n_sets = len(active_datasets)
    if nrows is None and ncols is None:
        ncols = min(3, max(1, int(np.ceil(np.sqrt(n_sets)))))
        nrows = int(np.ceil(n_sets / ncols))
    elif nrows is None:
        nrows = int(np.ceil(n_sets / ncols))
    elif ncols is None:
        ncols = int(np.ceil(n_sets / nrows))
        
    sp_titles = [ds.title_format for ds in active_datasets]
    
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=sp_titles, shared_xaxes=False)

    layout_args = {
        'template': "plotly_dark" if darkmode else "plotly_white",
        'title': {'text': suptitle if suptitle else f"Dataset Comparison", 'x': 0.5},
        'showlegend': True,
        'margin': dict(r=50) 
    }
    if figsize:
        layout_args['width'] = figsize[0] * 100
        layout_args['height'] = figsize[1] * 100
        
    fig.update_layout(**layout_args)
    color_cycle = px.colors.qualitative.Plotly
    
    next_free_axis_idx = (nrows * ncols) + 1

    for idx_ds, dataset in enumerate(active_datasets):
        row = (idx_ds // ncols) + 1
        col = (idx_ds % ncols) + 1
        
        subplot_index = (row - 1) * ncols + col
        base_x_name = "xaxis" if subplot_index == 1 else f"xaxis{subplot_index}"
        base_y_name = "yaxis" if subplot_index == 1 else f"yaxis{subplot_index}"
        
        # --- DOMAIN MANAGEMENT ---
        try:
            old_domain = fig.layout[base_x_name].domain
            if not old_domain: old_domain = [0, 1]
        except:
            old_domain = [0, 1]
            
        d_start, d_end = old_domain
        
        extras_count = max(0, len(y_list) - 2)
        
        width_per_axis = 0.08 
        required_space = extras_count * width_per_axis
        
        new_d_end = d_end - required_space
        if new_d_end <= d_start + 0.1: new_d_end = d_end 
        
        fig.layout[base_x_name].domain = [d_start, new_d_end]

        # --- DATA PREP ---
        df = dataset.df.copy()
        if dataset.order == 'index': df = df.sort_index()
        elif dataset.order: df = df.sort_values(by=dataset.order)
        
        if x not in df.columns:
            cols_upper = {c.upper(): c for c in df.columns}
            if x.upper() in cols_upper: x_col = cols_upper[x.upper()]
            else: continue
        else:
            x_col = x

        # Early check for regression to toggle off lines for standard markers
        valid_reg = False
        cur_reg_order = dataset.reg_order
        if isinstance(cur_reg_order, numbers.Number) and cur_reg_order > 0:
            valid_reg = True
        elif isinstance(cur_reg_order, str) and str(cur_reg_order).lower() in ['exp', 'log', 'power']:
            valid_reg = True

        scatter_mode = 'markers' if valid_reg else ('lines+markers' if dataset.linestyle else 'markers')

        for idx_y, yi in enumerate(y_list):
            if yi not in df.columns: continue
            
            var_color = color_cycle[idx_y % len(color_cycle)]
            custom_data = df[[x_col, yi] + [c for c in (display_parms or []) if c in df.columns]]
            ht = f"<b>{dataset.title}</b><br>{x_col}: %{{customdata[0]:.2f}}<br>{yi}: %{{customdata[1]:.2f}}"
            specific_limit = axis_limits.get(yi, None)

            line_dict = dict(color=var_color, width=dataset.linewidth)
            if dataset.linestyle: line_dict['dash'] = get_plotly_linestyle(dataset.linestyle)

            if idx_y == 0:
                fig.add_trace(go.Scatter(
                    x=df[x_col], y=df[yi],
                    mode=scatter_mode,
                    name=yi, legendgroup=yi, showlegend=(idx_ds == 0),
                    marker=dict(color=var_color, size=dataset.markersize or 6),
                    line=line_dict,
                    customdata=custom_data, hovertemplate=ht
                ), row=row, col=col)
                
                u_dict = dict(title_text=yi, title_font=dict(color=var_color), tickfont=dict(color=var_color), row=row, col=col)
                if specific_limit: u_dict['range'] = specific_limit
                fig.update_yaxes(**u_dict)
                
                # Assign regression to primary axis
                curr_y_ref = None

            elif idx_y == 1:
                curr_y_axis = f"yaxis{next_free_axis_idx}"
                curr_y_ref = f"y{next_free_axis_idx}"
                next_free_axis_idx += 1
                
                fig.add_trace(go.Scatter(
                    x=df[x_col], y=df[yi],
                    mode=scatter_mode,
                    name=yi, legendgroup=yi, showlegend=(idx_ds == 0),
                    marker=dict(color=var_color, size=dataset.markersize or 6),
                    line=line_dict,
                    customdata=custom_data, hovertemplate=ht,
                    xaxis=base_x_name.replace("axis", ""), 
                    yaxis=curr_y_ref
                ))
                
                layout_axis = dict(
                    title=yi,
                    title_font=dict(color=var_color),
                    tickfont=dict(color=var_color),
                    anchor=base_x_name.replace("axis", ""), 
                    overlaying=base_y_name.replace("axis", ""), 
                    side='right',
                    showgrid=False
                )
                if specific_limit: layout_axis['range'] = specific_limit
                fig.update_layout({curr_y_axis: layout_axis})
                
            else:
                curr_y_axis = f"yaxis{next_free_axis_idx}"
                curr_y_ref = f"y{next_free_axis_idx}"
                next_free_axis_idx += 1
                
                extra_idx = idx_y - 1 
                pos = new_d_end + (extra_idx * width_per_axis)
                pos = min(0.99, pos) 
                
                fig.add_trace(go.Scatter(
                    x=df[x_col], y=df[yi],
                    mode=scatter_mode,
                    name=yi, legendgroup=yi, showlegend=(idx_ds == 0),
                    marker=dict(color=var_color, size=dataset.markersize or 6),
                    line=line_dict,
                    customdata=custom_data, hovertemplate=ht,
                    xaxis=base_x_name.replace("axis", ""),
                    yaxis=curr_y_ref
                ))
                
                layout_axis = dict(
                    title=yi,
                    title_font=dict(color=var_color),
                    tickfont=dict(color=var_color),
                    anchor="free",          
                    overlaying=base_y_name.replace("axis", ""),
                    side='right',
                    position=pos,            
                    showgrid=False
                )
                if specific_limit: layout_axis['range'] = specific_limit
                fig.update_layout({curr_y_axis: layout_axis})

            # --- REGRESSION LOGIC FOR PER_DATASET PLOT ---
            if valid_reg:
                rx, ry = _calculate_regression(df, x_col, yi, cur_reg_order)
                reg_trace = go.Scatter(
                    x=rx, y=ry, mode='lines',
                    name=f"{yi} Fit {cur_reg_order}",
                    legendgroup=yi,
                    line=dict(color=var_color, width=dataset.linewidth, dash=get_plotly_linestyle(dataset.linestyle)),
                    opacity=0.7, hoverinfo='skip', showlegend=False
                )
                
                # Route the trendline to the exact same sub-axis as its parent scatter data
                if idx_y == 0:
                    fig.add_trace(reg_trace, row=row, col=col)
                else:
                    reg_trace.xaxis = base_x_name.replace("axis", "")
                    reg_trace.yaxis = curr_y_ref
                    fig.add_trace(reg_trace)

        fig.update_xaxes(title_text=x, showgrid=True, row=row, col=col)
        if x_lim: fig.update_xaxes(range=x_lim, row=row, col=col)

    if return_axes: return fig 
    fig.show()
    return fig

def unibar(list_of_datasets, x, y, barmode='group', color=None, 
           suptitle=None, xlabel=None, ylabel=None, subplot_titles=None,
           darkmode=False, figsize=(12, 8), ncols=None, nrows=None, 
           y_lim=None, return_axes=False):
    """
    Grouped Bar Chart version of uniplot. 
    Subplots are organized by Y-variables.
    """
    y_list = y if isinstance(y, list) else [y]
    n_y = len(y_list)

    if nrows is None and ncols is None:
        ncols = min(3, max(1, int(np.ceil(np.sqrt(n_y)))))
        nrows = int(np.ceil(n_y / ncols))
    elif nrows is None: nrows = int(np.ceil(n_y / ncols))
    elif ncols is None: ncols = int(np.ceil(n_y / nrows))

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subplot_titles or y_list)

    layout_args = {
        'template': "plotly_dark" if darkmode else "plotly_white",
        'title': {'text': suptitle or f"Bar Comparison: {x}", 'x': 0.5},
        'barmode': barmode,
        'showlegend': True
    }
    if figsize:
        layout_args['width'], layout_args['height'] = figsize[0] * 100, figsize[1] * 100
    fig.update_layout(**layout_args)

    for ds in list_of_datasets:
        if not ds.select: continue
        df = ds.df.copy()
        
        for idx_y, yi in enumerate(y_list):
            row, col = (idx_y // ncols) + 1, (idx_y % ncols) + 1
            if yi not in df.columns: continue

            fig.add_trace(go.Bar(
                x=df[x], y=df[yi],
                name=f"{ds.index}: {ds.title}",
                legendgroup=f"group_{ds.index}",
                marker_color=ds.color if not color else color,
                opacity=ds.alpha,
                showlegend=(idx_y == 0)
            ), row=row, col=col)

    fig.update_xaxes(title_text=xlabel or x)
    fig.update_yaxes(title_text=ylabel or "Value")
    if y_lim: fig.update_yaxes(range=y_lim)

    if return_axes: return fig
    fig.show()
    return fig

def unibar_per_dataset(list_of_datasets, x, y, barmode='group',
                       suptitle=None, figsize=(12, 8), ncols=None, nrows=None, 
                       darkmode=False, y_lim=None, return_axes=False):
    """
    Grouped Bar Chart version of uniplot_per_dataset.
    Subplots are organized by Dataset.
    """
    active_ds = [d for d in list_of_datasets if d.select]
    y_list = y if isinstance(y, list) else [y]
    n_sets = len(active_ds)

    if nrows is None and ncols is None:
        ncols = min(3, max(1, int(np.ceil(np.sqrt(n_sets)))))
        nrows = int(np.ceil(n_sets / ncols))
    elif nrows is None: nrows = int(np.ceil(n_sets / ncols))
    elif ncols is None: ncols = int(np.ceil(n_sets / nrows))

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[d.title_format for d in active_ds])
    color_cycle = px.colors.qualitative.Plotly

    layout_args = {
        'template': "plotly_dark" if darkmode else "plotly_white",
        'title': {'text': suptitle or "Dataset Bar Comparison", 'x': 0.5},
        'barmode': barmode,
    }
    if figsize:
        layout_args['width'], layout_args['height'] = figsize[0] * 100, figsize[1] * 100
    fig.update_layout(**layout_args)

    for idx_ds, ds in enumerate(active_ds):
        row, col = (idx_ds // ncols) + 1, (idx_ds % ncols) + 1
        df = ds.df
        
        for idx_y, yi in enumerate(y_list):
            if yi not in df.columns: continue
            
            fig.add_trace(go.Bar(
                x=df[x], y=df[yi],
                name=yi,
                legendgroup=yi,
                marker_color=color_cycle[idx_y % len(color_cycle)],
                showlegend=(idx_ds == 0)
            ), row=row, col=col)

    fig.update_xaxes(title_text=x)
    if y_lim: fig.update_yaxes(range=y_lim)

    if return_axes: return fig
    fig.show()
    return fig

def unibox(list_of_datasets, x, y, boxmode='group', points='outliers', notched=False,
           color=None, suptitle=None, xlabel=None, ylabel=None, subplot_titles=None,
           darkmode=False, figsize=(12, 8), ncols=None, nrows=None, 
           y_lim=None, return_axes=False):
    """
    Boxplot version of uniplot.
    Subplots are organized by Y-variables.
    """
    y_list = y if isinstance(y, list) else [y]
    n_y = len(y_list)

    if nrows is None and ncols is None:
        ncols = min(3, max(1, int(np.ceil(np.sqrt(n_y)))))
        nrows = int(np.ceil(n_y / ncols))
    elif nrows is None: nrows = int(np.ceil(n_y / ncols))
    elif ncols is None: ncols = int(np.ceil(n_y / nrows))

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subplot_titles or y_list)

    layout_args = {
        'template': "plotly_dark" if darkmode else "plotly_white",
        'title': {'text': suptitle or f"Boxplot Comparison: {x}", 'x': 0.5},
        'boxmode': boxmode,
        'showlegend': True
    }
    if figsize:
        layout_args['width'], layout_args['height'] = figsize[0] * 100, figsize[1] * 100
    fig.update_layout(**layout_args)

    for ds in list_of_datasets:
        if not ds.select: continue
        df = ds.df.copy()
        
        for idx_y, yi in enumerate(y_list):
            row, col = (idx_y // ncols) + 1, (idx_y % ncols) + 1
            if yi not in df.columns: continue

            fig.add_trace(go.Box(
                x=df[x], 
                y=df[yi],
                name=f"{ds.index}: {ds.title}",
                legendgroup=f"group_{ds.index}",
                marker_color=ds.color if not color else color,
                opacity=ds.alpha,
                boxpoints=points,
                notched=notched,
                line=dict(width=ds.linewidth),
                showlegend=(idx_y == 0)
            ), row=row, col=col)

    fig.update_xaxes(title_text=xlabel or x)
    fig.update_yaxes(title_text=ylabel or "Value")
    if y_lim: fig.update_yaxes(range=y_lim)

    if return_axes: return fig
    fig.show()
    return fig

def unibox_per_dataset(list_of_datasets, x, y, boxmode='group', points='outliers', notched=False,
                       suptitle=None, figsize=(12, 8), ncols=None, nrows=None, 
                       darkmode=False, y_lim=None, return_axes=False):
    """
    Boxplot version of uniplot_per_dataset.
    Subplots are organized by Dataset.
    """
    active_ds = [d for d in list_of_datasets if d.select]
    y_list = y if isinstance(y, list) else [y]
    n_sets = len(active_ds)

    if nrows is None and ncols is None:
        ncols = min(3, max(1, int(np.ceil(np.sqrt(n_sets)))))
        nrows = int(np.ceil(n_sets / ncols))
    elif nrows is None: nrows = int(np.ceil(n_sets / ncols))
    elif ncols is None: ncols = int(np.ceil(n_sets / nrows))

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[d.title_format for d in active_ds])
    color_cycle = px.colors.qualitative.Plotly

    layout_args = {
        'template': "plotly_dark" if darkmode else "plotly_white",
        'title': {'text': suptitle or "Dataset Box Comparison", 'x': 0.5},
        'boxmode': boxmode,
    }
    if figsize:
        layout_args['width'], layout_args['height'] = figsize[0] * 100, figsize[1] * 100
    fig.update_layout(**layout_args)

    for idx_ds, ds in enumerate(active_ds):
        row, col = (idx_ds // ncols) + 1, (idx_ds % ncols) + 1
        df = ds.df
        
        for idx_y, yi in enumerate(y_list):
            if yi not in df.columns: continue
            
            fig.add_trace(go.Box(
                x=df[x], 
                y=df[yi],
                name=yi,
                legendgroup=yi,
                marker_color=color_cycle[idx_y % len(color_cycle)],
                boxpoints=points,
                notched=notched,
                showlegend=(idx_ds == 0)
            ), row=row, col=col)

    fig.update_xaxes(title_text=x)
    if y_lim: fig.update_yaxes(range=y_lim)

    if return_axes: return fig
    fig.show()
    return fig


def unihistogram(list_of_datasets, x, y=None, histfunc='sum', nbins=None, 
                 bin_size=None, bin_start=None, bin_end=None, 
                 histnorm='', barmode='overlay', opacity=0.7,
                 color=None, suptitle=None, subplot_titles=None, darkmode=False, 
                 figsize=(12, 8), ncols=None, nrows=None, x_lim=None, return_axes=False):
    """
    Create a unified histogram for a list of datasets.
    Subplots are organized by Variable (x).
    """
    x_list = x if isinstance(x, list) else [x]
    n_x = len(x_list)
    
    if nrows is None and ncols is None:
        ncols = min(3, max(1, int(np.ceil(np.sqrt(n_x)))))
        nrows = int(np.ceil(n_x / ncols))
    elif nrows is None: nrows = int(np.ceil(n_x / ncols))
    elif ncols is None: ncols = int(np.ceil(n_x / nrows))

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subplot_titles or x_list)

    layout_args = {
        'template': "plotly_dark" if darkmode else "plotly_white",
        'title': {'text': suptitle or f"Distribution Comparison", 'x': 0.5},
        'barmode': barmode,
        'showlegend': True
    }
    if figsize:
        layout_args['width'], layout_args['height'] = figsize[0] * 100, figsize[1] * 100
    
    fig.update_layout(**layout_args)
    
    xbins_dict = {}
    if bin_size is not None: xbins_dict['size'] = bin_size
    if bin_start is not None: xbins_dict['start'] = bin_start
    if bin_end is not None: xbins_dict['end'] = bin_end
    xbins = xbins_dict if xbins_dict else None

    for ds in list_of_datasets:
        if not ds.select: continue
        df = ds.df.copy()
        
        use_color = color if color else ds.color

        for idx_x, xi in enumerate(x_list):
            row, col = (idx_x // ncols) + 1, (idx_x % ncols) + 1
            
            if xi not in df.columns: continue

            subset_cols = [xi] if y is None else [xi, y]
            if y and y not in df.columns: continue
            
            clean_data = df.dropna(subset=subset_cols)
            if clean_data.empty: continue

            trace_args = dict(
                x=clean_data[xi],
                name=f"{ds.index}: {ds.title}",
                legendgroup=f"group_{ds.index}",
                marker_color=use_color,
                opacity=opacity,
                nbinsx=nbins,
                xbins=xbins,
                histnorm=histnorm,
                showlegend=(idx_x == 0)
            )

            if y:
                trace_args['y'] = clean_data[y]
                trace_args['histfunc'] = histfunc

            fig.add_trace(go.Histogram(**trace_args), row=row, col=col)

    if x_lim: fig.update_xaxes(range=x_lim)
    
    y_label = f"Sum of {y}" if y else ("Density" if "density" in histnorm else "Count")
    fig.update_yaxes(title_text=y_label)

    if return_axes: return fig
    fig.show()
    return fig

def unihistogram_by_dataset(list_of_datasets, x, y=None, histfunc='sum', nbins=None, 
                            bin_size=None, bin_start=None, bin_end=None,
                            histnorm='', barmode='overlay', opacity=0.7,
                            color=None, suptitle=None, figsize=(12, 8), ncols=None, nrows=None, 
                            darkmode=False, x_lim=None, return_axes=False):
    """
    Create a unified histogram where Subplots are organized by Dataset.
    """
    active_ds = [d for d in list_of_datasets if d.select]
    x_list = x if isinstance(x, list) else [x]
    n_sets = len(active_ds)

    if not active_ds:
        print("No datasets selected.")
        return None

    if nrows is None and ncols is None:
        ncols = min(3, max(1, int(np.ceil(np.sqrt(n_sets)))))
        nrows = int(np.ceil(n_sets / ncols))
    elif nrows is None: nrows = int(np.ceil(n_sets / ncols))
    elif ncols is None: ncols = int(np.ceil(n_sets / nrows))

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[d.title_format for d in active_ds])
    color_cycle = px.colors.qualitative.Plotly

    layout_args = {
        'template': "plotly_dark" if darkmode else "plotly_white",
        'title': {'text': suptitle or "Dataset Distribution Analysis", 'x': 0.5},
        'barmode': barmode,
        'showlegend': True
    }
    if figsize:
        layout_args['width'], layout_args['height'] = figsize[0] * 100, figsize[1] * 100
    
    fig.update_layout(**layout_args)
    
    xbins_dict = {}
    if bin_size is not None: xbins_dict['size'] = bin_size
    if bin_start is not None: xbins_dict['start'] = bin_start
    if bin_end is not None: xbins_dict['end'] = bin_end
    xbins = xbins_dict if xbins_dict else None

    for idx_ds, ds in enumerate(active_ds):
        row, col = (idx_ds // ncols) + 1, (idx_ds % ncols) + 1
        df = ds.df.copy()
        
        for idx_x, xi in enumerate(x_list):
            if xi not in df.columns: continue
            
            subset_cols = [xi] if y is None else [xi, y]
            if y and y not in df.columns: continue
            
            clean_data = df.dropna(subset=subset_cols)
            if clean_data.empty: continue
            
            if color:
                use_color = color
            elif len(x_list) == 1:
                use_color = ds.color
            else:
                use_color = color_cycle[idx_x % len(color_cycle)]

            trace_args = dict(
                x=clean_data[xi],
                name=xi,
                legendgroup=xi,
                marker_color=use_color,
                opacity=opacity,
                nbinsx=nbins,
                xbins=xbins,
                histnorm=histnorm,
                showlegend=(idx_ds == 0)
            )

            if y:
                trace_args['y'] = clean_data[y]
                trace_args['histfunc'] = histfunc

            fig.add_trace(go.Histogram(**trace_args), row=row, col=col)

    if x_lim: fig.update_xaxes(range=x_lim)
    
    y_label = f"Sum of {y}" if y else ("Density" if "density" in histnorm else "Count")
    fig.update_yaxes(title_text=y_label)

    if return_axes: return fig
    fig.show()
    return fig

def unicontour(list_of_datasets, x, y, z, contours_coloring='fill', colorscale=None,
               interpolate=True, interp_res=100, interp_method='linear',
               ncontours=None, 
               suptitle=None, xlabel=None, ylabel=None, subplot_titles=None,
               darkmode=False, figsize=(12, 8), ncols=None, nrows=None, 
               axis_limits=None, return_axes=False):
    """
    Create a unified contour plot for a list of datasets.
    Subplots are organized by Z-variables.
    """
    z_list = z if isinstance(z, list) else [z]
    n_z = len(z_list)
    active_ds = [d for d in list_of_datasets if d.select]
    num_contour_ds = sum(1 for d in active_ds if d.plot_type in ['contour', None])
    axis_limits = axis_limits or {}

    if not active_ds:
        print("No datasets selected.")
        return None

    if nrows is None and ncols is None:
        ncols = min(3, max(1, int(np.ceil(np.sqrt(n_z)))))
        nrows = int(np.ceil(n_z / ncols))
    elif nrows is None: nrows = int(np.ceil(n_z / ncols))
    elif ncols is None: ncols = int(np.ceil(n_z / nrows))

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subplot_titles or z_list, horizontal_spacing=0.15)

    layout_args = {
        'template': "plotly_dark" if darkmode else "plotly_white",
        'title': {
            'text': suptitle or f"Contour: {y} vs {x}",
            'x': 0.5, 'xanchor': 'center',
            'y': 0.98, 'yanchor': 'top', 'yref': 'container'
        },
        'showlegend': True,
        'margin': dict(r=100) 
    }
    if figsize:
        layout_args['width'], layout_args['height'] = figsize[0] * 100, figsize[1] * 100
    fig.update_layout(**layout_args)

    for idx_ds, ds in enumerate(active_ds):
        df = ds.df.copy()
        if x not in df.columns or y not in df.columns: 
            continue

        ptype = ds.plot_type if ds.plot_type else 'contour'

        for idx_z, zi in enumerate(z_list):
            row, col = (idx_z // ncols) + 1, (idx_z % ncols) + 1

            if ptype == 'contour':
                if zi not in df.columns: continue

                use_coloring = 'lines' if num_contour_ds > 1 and contours_coloring == 'fill' else contours_coloring

                clean_df = df.dropna(subset=[x, y, zi])
                if clean_df.empty: continue

                if interpolate:
                    plot_x, plot_y, plot_z = _generate_contour_grid(
                        clean_df[x], clean_df[y], clean_df[zi], 
                        res=interp_res, method=interp_method
                    )
                else:
                    plot_x, plot_y, plot_z = clean_df[x], clean_df[y], clean_df[zi]

                z_lim = axis_limits.get(zi)
                zmin, zmax = z_lim if z_lim else (None, None)

                subplot_idx = (row - 1) * ncols + col
                x_axis_name = f"xaxis{subplot_idx}" if subplot_idx > 1 else "xaxis"
                y_axis_name = f"yaxis{subplot_idx}" if subplot_idx > 1 else "yaxis"
                
                try:
                    x_domain = fig.layout[x_axis_name].domain
                    y_domain = fig.layout[y_axis_name].domain
                    cb_x = x_domain[1] + 0.01               
                    cb_y = sum(y_domain) / 2                
                    cb_len = y_domain[1] - y_domain[0]      
                except KeyError:
                    cb_x, cb_y, cb_len = 1.02, 0.5, 1.0     

                fig.add_trace(go.Contour(
                    x=plot_x, y=plot_y, z=plot_z,
                    zmin=zmin, zmax=zmax,  
                    name=f"{ds.index}: {ds.title}",
                    legendgroup=f"group_{ds.index}",
                    colorscale=colorscale or ds.hue_palette,
                    contours_coloring=use_coloring,
                    ncontours=ncontours, 
                    line=dict(width=ds.linewidth, color=ds.color if use_coloring=='lines' else None),
                    showscale=(idx_ds == 0),
                    colorbar=dict(
                        title=zi,
                        x=cb_x,
                        y=cb_y,
                        len=cb_len,
                        thickness=15
                    ),
                    hovertemplate=f"<b>{ds.title}</b><br>{x}: %{{x:.3g}}<br>{y}: %{{y:.3g}}<br>{zi}: %{{z:.3g}}<extra></extra>"
                ), row=row, col=col)
            
            elif ptype == 'scatter':
                clean_df = df.dropna(subset=[x, y])
                if clean_df.empty: continue
                
                mode_parts = []
                if not ds.linestyle: mode_parts.append('markers')
                else: mode_parts.append('lines')
                if ds.marker: mode_parts.append('markers')
                mode = "+".join(list(dict.fromkeys(mode_parts)))

                marker_dict = dict(
                    size=ds.markersize,
                    symbol=get_plotly_marker(ds.marker),
                    line=dict(width=ds.edgewidth, color=ds.edge_color),
                    opacity=ds.alpha
                )
                line_dict = dict(width=ds.linewidth, dash=get_plotly_linestyle(ds.linestyle))
                
                custom_data_cols = []
                ht = f"<b>{ds.title}</b><br>{x}: %{{x:.3g}}<br>{y}: %{{y:.3g}}"
                
                if zi in clean_df.columns:
                    custom_data_cols.append(zi)
                    ht += f"<br>{zi}: %{{customdata[0]:.3g}}"

                if ds.hue and ds.hue in clean_df.columns:
                    hue_data = clean_df[ds.hue]
                    if pd.api.types.is_numeric_dtype(hue_data):
                        marker_dict['color'] = hue_data
                    else:
                        marker_dict['color'] = hue_data.astype('category').cat.codes
                    marker_dict['colorscale'] = ds.hue_palette
                    marker_dict['showscale'] = False
                    if ds.hue != zi:
                        custom_data_cols.append(ds.hue)
                        ht += f"<br>{ds.hue}: %{{customdata[{len(custom_data_cols)-1}]}}"
                else:
                    marker_dict['color'] = ds.color
                    line_dict['color'] = ds.color
                    
                custom_data = clean_df[custom_data_cols] if custom_data_cols else None
                ht += "<extra></extra>"
                    
                fig.add_trace(go.Scatter(
                    x=clean_df[x], y=clean_df[y],
                    mode=mode,
                    name=f"{ds.index}: {ds.title}",
                    legendgroup=f"group_{ds.index}",
                    marker=marker_dict,
                    line=line_dict,
                    customdata=custom_data,
                    hovertemplate=ht,
                    showlegend=(idx_z == 0)
                ), row=row, col=col)

    fig.update_xaxes(title_text=xlabel or x)
    fig.update_yaxes(title_text=ylabel or y)

    if return_axes: return fig
    fig.show()
    return fig


def unicontour_per_dataset(list_of_datasets, x, y, z, contours_coloring='fill', colorscale=None,
                           interpolate=True, interp_res=100, interp_method='linear',
                           ncontours=None,
                           suptitle=None, figsize=(12, 8), ncols=None, nrows=None, 
                           darkmode=False, axis_limits=None, return_axes=False):
    """
    Contour plot where Subplots are organized by Dataset.
    """
    active_ds = [d for d in list_of_datasets if d.select]
    z_list = z if isinstance(z, list) else [z]
    n_sets = len(active_ds)
    axis_limits = axis_limits or {}

    if not active_ds:
        print("No datasets selected.")
        return None

    if nrows is None and ncols is None:
        ncols = min(3, max(1, int(np.ceil(np.sqrt(n_sets)))))
        nrows = int(np.ceil(n_sets / ncols))
    elif nrows is None: nrows = int(np.ceil(n_sets / ncols))
    elif ncols is None: ncols = int(np.ceil(n_sets / nrows))

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[d.title_format for d in active_ds], horizontal_spacing=0.15)

    layout_args = {
        'template': "plotly_dark" if darkmode else "plotly_white",
        'title': {'text': suptitle or f"Dataset Contour Comparison", 'x': 0.5, 'y': 0.98, 'yref': 'container'},
        'margin': dict(r=100)
    }
    if figsize:
        layout_args['width'], layout_args['height'] = figsize[0] * 100, figsize[1] * 100
    fig.update_layout(**layout_args)

    for idx_ds, ds in enumerate(active_ds):
        row, col = (idx_ds // ncols) + 1, (idx_ds % ncols) + 1
        df = ds.df.copy()
        
        if x not in df.columns or y not in df.columns: continue

        ptype = ds.plot_type if ds.plot_type else 'contour'

        for idx_z, zi in enumerate(z_list):
            if ptype == 'contour':
                if zi not in df.columns: continue
                
                use_coloring = 'lines' if len(z_list) > 1 and contours_coloring == 'fill' else contours_coloring

                clean_df = df.dropna(subset=[x, y, zi])
                if clean_df.empty: continue

                if interpolate:
                    plot_x, plot_y, plot_z = _generate_contour_grid(
                        clean_df[x], clean_df[y], clean_df[zi], 
                        res=interp_res, method=interp_method
                    )
                else:
                    plot_x, plot_y, plot_z = clean_df[x], clean_df[y], clean_df[zi]

                z_lim = axis_limits.get(zi)
                zmin, zmax = z_lim if z_lim else (None, None)

                subplot_idx = (row - 1) * ncols + col
                x_axis_name = f"xaxis{subplot_idx}" if subplot_idx > 1 else "xaxis"
                y_axis_name = f"yaxis{subplot_idx}" if subplot_idx > 1 else "yaxis"
                
                try:
                    x_domain = fig.layout[x_axis_name].domain
                    y_domain = fig.layout[y_axis_name].domain
                    cb_x = x_domain[1] + 0.01 + (idx_z * 0.05) 
                    cb_y = sum(y_domain) / 2
                    cb_len = y_domain[1] - y_domain[0]
                except KeyError:
                    cb_x, cb_y, cb_len = 1.02 + (idx_z * 0.05), 0.5, 1.0

                fig.add_trace(go.Contour(
                    x=plot_x, y=plot_y, z=plot_z,
                    zmin=zmin, zmax=zmax,  
                    name=zi,
                    legendgroup=zi,
                    colorscale=colorscale or ds.hue_palette,
                    contours_coloring=use_coloring,
                    ncontours=ncontours, 
                    showscale=(idx_ds == 0),
                    colorbar=dict(
                        title=zi,
                        x=cb_x,
                        y=cb_y,
                        len=cb_len,
                        thickness=15
                    ),
                    hovertemplate=f"<b>{zi}</b><br>{x}: %{{x:.3g}}<br>{y}: %{{y:.3g}}<br>Value: %{{z:.3g}}<extra></extra>"
                ), row=row, col=col)
            
            elif ptype == 'scatter':
                clean_df = df.dropna(subset=[x, y])
                if clean_df.empty: continue
                
                mode_parts = []
                if not ds.linestyle: mode_parts.append('markers')
                else: mode_parts.append('lines')
                if ds.marker: mode_parts.append('markers')
                mode = "+".join(list(dict.fromkeys(mode_parts)))

                marker_dict = dict(
                    size=ds.markersize,
                    symbol=get_plotly_marker(ds.marker),
                    line=dict(width=ds.edgewidth, color=ds.edge_color),
                    opacity=ds.alpha
                )
                line_dict = dict(width=ds.linewidth, dash=get_plotly_linestyle(ds.linestyle))
                
                custom_data_cols = []
                ht = f"<b>{zi} (Scatter)</b><br>{x}: %{{x:.3g}}<br>{y}: %{{y:.3g}}"
                
                if zi in clean_df.columns:
                    custom_data_cols.append(zi)
                    ht += f"<br>{zi}: %{{customdata[0]:.3g}}"

                if ds.hue and ds.hue in clean_df.columns:
                    hue_data = clean_df[ds.hue]
                    if pd.api.types.is_numeric_dtype(hue_data):
                        marker_dict['color'] = hue_data
                    else:
                        marker_dict['color'] = hue_data.astype('category').cat.codes
                    marker_dict['colorscale'] = ds.hue_palette
                    marker_dict['showscale'] = False
                    if ds.hue != zi:
                        custom_data_cols.append(ds.hue)
                        ht += f"<br>{ds.hue}: %{{customdata[{len(custom_data_cols)-1}]}}"
                else:
                    marker_dict['color'] = ds.color
                    line_dict['color'] = ds.color
                    
                custom_data = clean_df[custom_data_cols] if custom_data_cols else None
                ht += "<extra></extra>"
                    
                fig.add_trace(go.Scatter(
                    x=clean_df[x], y=clean_df[y],
                    mode=mode,
                    name=zi,
                    legendgroup=zi,
                    marker=marker_dict,
                    line=line_dict,
                    customdata=custom_data,
                    hovertemplate=ht,
                    showlegend=(idx_ds == 0)
                ), row=row, col=col)

    fig.update_xaxes(title_text=x)
    fig.update_yaxes(title_text=y)

    if return_axes: return fig
    fig.show()
    return fig

def unibar_datasets_as_x(list_of_datasets, y, agg='mean', suptitle=None, darkmode=False,
                         figsize=(12, 8), axis_limits=None, return_axes=False):
    """
    Creates a single grouped bar chart where the X-axis is the Dataset name,
    and the bars are the different Y-variables, each scaled to their own Y-axis.
    Includes an 'agg' parameter to handle multi-row datasets.
    """
    import plotly.graph_objects as go
    import plotly.express as px

    active_ds = [d for d in list_of_datasets if d.select]
    if not active_ds:
        print("No datasets selected.")
        return None

    y_list = y if isinstance(y, list) else [y]
    axis_limits = axis_limits or {}
    color_cycle = px.colors.qualitative.Plotly

    fig = go.Figure()

    # 1. Setup X-Axis Categories
    x_labels = [f"{ds.index}: {ds.title}" for ds in active_ds]

    # 2. Domain Management for Extra Y-Axes
    extras_count = max(0, len(y_list) - 2)
    width_per_axis = 0.08
    required_space = extras_count * width_per_axis
    x_domain_end = max(0.5, 1.0 - required_space) 

    # 3. Add Traces and Axes
    for idx_y, yi in enumerate(y_list):
        var_color = color_cycle[idx_y % len(color_cycle)]

        y_data = []
        for ds in active_ds:
            if yi in ds.df.columns:
                valid_data = ds.df[yi].dropna()
                if valid_data.empty:
                    val = None
                elif agg == 'mean': val = valid_data.mean()
                elif agg == 'sum': val = valid_data.sum()
                elif agg == 'max': val = valid_data.max()
                elif agg == 'min': val = valid_data.min()
                elif agg == 'median': val = valid_data.median()
                elif agg == 'first': val = valid_data.iloc[0]
                elif agg == 'last': val = valid_data.iloc[-1]
                else:
                    print(f"Warning: Unknown agg '{agg}', defaulting to mean.")
                    val = valid_data.mean()
                y_data.append(val)
            else:
                y_data.append(None)

        y_axis_name = "y" if idx_y == 0 else f"y{idx_y + 1}"

        fig.add_trace(go.Bar(
            name=yi,
            x=x_labels,
            y=y_data,
            yaxis=y_axis_name,
            offsetgroup=str(idx_y), 
            marker_color=var_color
        ))

        axis_layout = dict(
            title=yi,
            title_font=dict(color=var_color),
            tickfont=dict(color=var_color),
            showgrid=(idx_y == 0) 
        )

        if yi in axis_limits:
            axis_layout['range'] = axis_limits[yi]

        if idx_y == 0:
            fig.update_layout(yaxis=axis_layout)
        elif idx_y == 1:
            axis_layout.update(dict(overlaying='y', side='right', anchor='x'))
            fig.update_layout(yaxis2=axis_layout)
        else:
            pos = x_domain_end + ((idx_y - 1) * width_per_axis)
            axis_layout.update(dict(overlaying='y', side='right', anchor='free', position=pos))
            fig.update_layout({f"yaxis{idx_y + 1}": axis_layout})

    # 4. Final Layout Adjustments
    layout_args = {
        'template': "plotly_dark" if darkmode else "plotly_white",
        'title': {'text': suptitle or f"Variables by Dataset ({agg})", 'x': 0.5},
        'barmode': 'group',
        'xaxis': dict(domain=[0, x_domain_end], title="Dataset"),
        'margin': dict(r=50 + (extras_count * 80)) 
    }
    
    if figsize:
        layout_args['width'] = figsize[0] * 100
        layout_args['height'] = figsize[1] * 100

    fig.update_layout(**layout_args)

    if return_axes: return fig
    fig.show()
    return fig

def unibox_datasets_as_x(list_of_datasets, y, boxmode='group', points='outliers', notched=False,
                         suptitle=None, darkmode=False, figsize=(12, 8), axis_limits=None, return_axes=False):
    """
    Creates a single grouped box plot where the X-axis is the Dataset name,
    and the boxes are the different Y-variables, each scaled to their own Y-axis.
    """
    import plotly.graph_objects as go
    import plotly.express as px

    active_ds = [d for d in list_of_datasets if d.select]
    if not active_ds:
        print("No datasets selected.")
        return None

    y_list = y if isinstance(y, list) else [y]
    axis_limits = axis_limits or {}
    color_cycle = px.colors.qualitative.Plotly

    fig = go.Figure()

    # 1. Domain Management for Extra Y-Axes
    extras_count = max(0, len(y_list) - 2)
    width_per_axis = 0.08
    required_space = extras_count * width_per_axis
    x_domain_end = max(0.5, 1.0 - required_space) 

    # 2. Add Traces and Axes
    for idx_y, yi in enumerate(y_list):
        var_color = color_cycle[idx_y % len(color_cycle)]

        all_y_data = []
        all_x_labels = []
        
        for ds in active_ds:
            if yi in ds.df.columns:
                valid_data = ds.df[yi].dropna()
                if not valid_data.empty:
                    all_y_data.extend(valid_data.values)
                    label = f"{ds.index}: {ds.title}"
                    all_x_labels.extend([label] * len(valid_data))

        if not all_y_data:
            continue

        y_axis_name = "y" if idx_y == 0 else f"y{idx_y + 1}"

        fig.add_trace(go.Box(
            name=yi,
            x=all_x_labels,
            y=all_y_data,
            yaxis=y_axis_name,
            offsetgroup=str(idx_y), 
            marker_color=var_color,
            boxpoints=points,
            notched=notched,
        ))

        axis_layout = dict(
            title=yi,
            title_font=dict(color=var_color),
            tickfont=dict(color=var_color),
            showgrid=(idx_y == 0) 
        )

        if yi in axis_limits:
            axis_layout['range'] = axis_limits[yi]

        if idx_y == 0:
            fig.update_layout(yaxis=axis_layout)
        elif idx_y == 1:
            axis_layout.update(dict(overlaying='y', side='right', anchor='x'))
            fig.update_layout(yaxis2=axis_layout)
        else:
            pos = x_domain_end + ((idx_y - 1) * width_per_axis)
            axis_layout.update(dict(overlaying='y', side='right', anchor='free', position=pos))
            fig.update_layout({f"yaxis{idx_y + 1}": axis_layout})

    # 3. Final Layout Adjustments
    layout_args = {
        'template': "plotly_dark" if darkmode else "plotly_white",
        'title': {'text': suptitle or "Variables by Dataset", 'x': 0.5},
        'boxmode': boxmode,
        'xaxis': dict(domain=[0, x_domain_end], title="Dataset"),
        'margin': dict(r=50 + (extras_count * 80)) 
    }
    
    if figsize:
        layout_args['width'] = figsize[0] * 100
        layout_args['height'] = figsize[1] * 100

    fig.update_layout(**layout_args)

    if return_axes: return fig
    fig.show()
    return fig

class UnichartNotebook:
    def __init__(self):
        """
        Initialize the Notebook-based UniChart environment.
        """
        self.uset = []
        self.dataset_widget_container = widgets.VBox([])
        
        # State Memory
        self.last_x = None
        self.last_y = None
        self.last_format = 'stack'
        self.last_ymult_format = 'color'
        self.darkmode = False 
        self.last_ncols = None
        self.last_nrows = None
        self.last_fig = None 

        self.suptitle = None
        
        # Plot Decorations
        self.plot_title = None
        self.x_label = None
        self.y_label = None
        
        self.display_parms = []
        self.axis_limits = {} 
        self.lines = {}       
        self.highlights = {}  

        self.parm_description_dict = {}

        self.suptitle_size = None
        self.legend_size = None
        self.axes_title_size = None
        self.axes_tick_size = None

        self.legend = 'on'
        
        print("UniChart Notebook Environment Initialized.")

    # ------------------------------------------------------------------
    # Data Management
    # ------------------------------------------------------------------
    def load_df(self, df, title=None, set_name_column=None, set_idx_column=None, load_cols_as_vars=False):
        """
        Load a DataFrame into the environment as datasets.
        """
        # Auto-title logic
        if not title:
            if set_name_column and set_name_column in df.columns:
                pass # Title will be derived from group
            elif "TITLE" in df.columns:
                set_name_column = "TITLE"
            else:
                df["TITLE"] = "Dataset"
                set_name_column = "TITLE"

            if set_idx_column and set_idx_column in df.columns:
                pass # Index will be derived from group
            elif "SETNUMBER" in df.columns:
                set_idx_column = "SETNUMBER"
            elif "INDEX" in df.columns:
                set_idx_column = "INDEX"
            else:
                df["SETNUMBER"] = df.index

        next_index = len(self.uset)
        
        # Group and create Dataset objects
        if set_idx_column and set_idx_column in df.columns:
            for set_index, df_subset in df.groupby(set_idx_column):

                if title:
                    final_title = title
                elif set_name_column and set_name_column in df_subset.columns:
                    final_title = str(df_subset.iloc[0][set_name_column])
                elif "TITLE" in df_subset.columns:
                    final_title = str(df_subset.iloc[0]["TITLE"])
                else:
                    final_title = f"Group {set_index}"

                ds = Dataset(df_subset.copy(), index=next_index, title=final_title)
                self.uset.append(ds)
                print(f"Loaded Set {next_index}: {ds.title}")
                next_index += 1
        else:
            ds = Dataset(df.copy(), index=next_index, title=title if title else "Untitled")
            self.uset.append(ds)
            print(f"Loaded Set {next_index}: {ds.title}")

        if load_cols_as_vars:
            for column in df.columns:
                try:
                    exec(f"{column} = '{column}'", globals())
                except Exception as e:
                    print(f"Could not create variable for column '{column}': {e}")
        
        self._refresh_widgets()

    def load_clipboard(self, **kwargs):
        """Quickly load data from system clipboard."""
        try:
            df = pd.read_clipboard(**kwargs)
            self.load_df(df, title="Clipboard Data")
        except Exception as e:
            print(f"Error reading clipboard: {e}")

    def load_file(self, title=None, set_name_column=None, set_idx_column=None, 
                    load_cols_as_vars=False, initial_dir=None, force_format=None, 
                    add_file_info=False, **kwargs):
            """
            Opens a system file dialog to select a dataset and loads it into the environment.
            """
            df = open_datafile(
                initial_dir=initial_dir, 
                force_format=force_format, 
                add_file_info=add_file_info, 
                **kwargs
            )
            if df is not None:
                if title is None and add_file_info and 'FILENAME' in df.columns:
                    title = str(df['FILENAME'].iloc[0])
                    
                self.load_df(
                    df, 
                    title=title, 
                    set_name_column=set_name_column, 
                    set_idx_column=set_idx_column, 
                    load_cols_as_vars=load_cols_as_vars
                )
            else:
                print("Operation cancelled or file could not be read. No data was loaded.")

    def clear_data(self):
        self.uset = []
        self._refresh_widgets()
        print("All datasets cleared.")

    def set_title(self, uset_slice, title):
        """
        Update the title of the specified dataset(s).
        """
        for ds in self._get_uset_slice(uset_slice):
            ds.title = str(title)
            ds.title_format = f"{ds.title} {ds.index}"
        self._refresh_widgets()

    # ------------------------------------------------------------------
    # Selection & Filtering
    # ------------------------------------------------------------------
    def _get_uset_slice(self, uset_slice):
        """Helper to normalize input into a list of datasets"""
        if uset_slice is None:
            return self.uset
        elif uset_slice == 'all':
            return self.uset
        elif isinstance(uset_slice, int):
            if 0 <= uset_slice < len(self.uset):
                return [self.uset[uset_slice]]
            return []
        elif isinstance(uset_slice, list):
            return [d for i, d in enumerate(self.uset) if i in uset_slice or d in uset_slice]
        return [uset_slice]

    def select(self, uset_slice=None):
        """Select the specified dataset(s)."""
        for ds in self.uset: ds.select = False # Exclusive select logic from original
        for ds in self._get_uset_slice(uset_slice):
            ds.select = True
        self._refresh_widgets()

    def selected(self):
        """Get the currently selected datasets."""
        return [ds for ds in self.uset if ds.select]

    def omit(self, uset_slice=None):
        for ds in self._get_uset_slice(uset_slice):
            ds.select = False
        self._refresh_widgets()

    def restore(self, uset_slice=None):
        targets = self.uset if uset_slice == "all" else self._get_uset_slice(uset_slice)
        for ds in targets:
            ds.select = True
        self._refresh_widgets()

    def query(self, uset_slice=None, query_str=None):
        for ds in self._get_uset_slice(uset_slice):
            ds.query = query_str
        self._refresh_widgets()

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------
    def color(self, uset_slice, color_val):
        for ds in self._get_uset_slice(uset_slice):
            ds.color = color_val

    def marker(self, uset_slice, marker_val):
        for ds in self._get_uset_slice(uset_slice):
            ds.marker = marker_val

    def linestyle(self, uset_slice, style_val):
        for ds in self._get_uset_slice(uset_slice):
            ds.linestyle = style_val
            
    def markersize(self, uset_slice, size_val):
        for ds in self._get_uset_slice(uset_slice):
            ds.markersize = size_val
    
    def linewidth(self, uset_slice, width_val):
        for ds in self._get_uset_slice(uset_slice):
            ds.linewidth = width_val

    def hue(self, uset_slice, col_name):
        for ds in self._get_uset_slice(uset_slice):
            ds.hue = col_name

    def hue_palette(self, uset_slice, hue_palette):
        for ds in self._get_uset_slice(uset_slice):
            ds.hue_palette = hue_palette

    def alpha(self, uset_slice, alpha_val):
        for ds in self._get_uset_slice(uset_slice):
            ds.alpha = alpha_val

    def plot_type(self, uset_slice, type_val):
        for ds in self._get_uset_slice(uset_slice):
            ds.plot_type = type_val

    def set_display_parms(self, uset_slice, parms):
        if not isinstance(parms, list):
            parms = [parms]
            
        for ds in self._get_uset_slice(uset_slice):
            ds.display_parms = parms

    def toggle_darkmode(self, state=None):
        if state is not None:
            self.darkmode = bool(state)
        else:
            self.darkmode = not self.darkmode
            
        mode = "Dark" if self.darkmode else "Light"
        print(f"Plot theme set to: {mode} Mode")

    # ------------------------------------------------------------------
    # Analysis (Ported from GUI)
    # ------------------------------------------------------------------
    
    def delta(self, base_idx, study_indices, align_on=None, delta_parms=None, passed_parms=None, suffixes=("_BASE", ""), keep_study_formatting=True):
        """
        Creates a new dataset representing the difference between study and base.
        Safely ignores missing columns and skips invalid comparisons.
        """
        if align_on is None: align_on = self.last_x
        if delta_parms is None: delta_parms = [self.last_y] if isinstance(self.last_y, str) else self.last_y
        
        # Ensure list formats
        if not isinstance(delta_parms, list): delta_parms = [delta_parms]
        if passed_parms is None: passed_parms = []
        elif not isinstance(passed_parms, list): passed_parms = [passed_parms]
        
        base_ds = self.uset[base_idx]
        targets = self._get_uset_slice(study_indices)
        
        lsuffix, rsuffix = suffixes
        
        for study_ds in targets:
            if align_on not in base_ds.df.columns or align_on not in study_ds.df.columns:
                print(f"Warning: Alignment column '{align_on}' missing. Skipping study {study_ds.index}.")
                continue
                
            valid_delta_parms = [
                p for p in delta_parms 
                if p in base_ds.df.columns and p in study_ds.df.columns
            ]
            
            if not valid_delta_parms:
                print(f"Warning: No valid delta parameters shared between base {base_ds.index} and study {study_ds.index}. Skipping.")
                continue

            valid_passed_parms = [p for p in passed_parms if p in study_ds.df.columns]
            
            base_cols = list(dict.fromkeys([align_on] + valid_delta_parms))
            df_base = base_ds.df[base_cols].sort_values(align_on)
            
            study_cols = list(dict.fromkeys([align_on] + valid_delta_parms + valid_passed_parms))
            df_study = study_ds.df[study_cols].sort_values(align_on)
            
            merged = pd.merge_asof(df_base, df_study, on=align_on, suffixes=suffixes, direction='nearest')
            
            for parm in valid_delta_parms:
                b_col = f"{parm}{lsuffix}"
                s_col = f"{parm}{rsuffix}"
                merged[f"DL_{parm}"] = merged[s_col] - merged[b_col]
                merged[f"DLPCT_{parm}"] = np.where(merged[b_col] == 0, np.nan, 100 * (merged[f"DL_{parm}"] / merged[b_col]))
            
            new_title = f"Delta {base_ds.index}-{study_ds.index}"
            self.load_df(merged, title=new_title)
            self.uset[-1].delta_sets = (base_ds.index, study_ds.index)

            if keep_study_formatting:
                study_formatting_dict = study_ds.get_format_dict()
                study_formatting_dict['title'] = "DL_" + study_ds.title
                study_formatting_dict['index'] = self.uset[-1].index
                study_formatting_dict['delta_sets'] = self.uset[-1].delta_sets
                self.uset[-1].update_format_dict(study_formatting_dict)

            self.uset[-1].settype = 'delta'
            
    # ------------------------------------------------------------------
    # Axes Based Decorations (Lines/Highlights/Scale)
    # ------------------------------------------------------------------
    def line(self, column, level, color='red', dash='dash'):
        """Add a vertical or horizontal line to the next plot."""
        if level == 'clear':
            self.lines.pop(column, None)
            return
        
        if column not in self.lines: self.lines[column] = []
        self.lines[column].append({'level': level, 'color': color, 'dash': dash})

    def highlight(self, column, range_tuple, color='yellow', opacity=0.2):
        """Add a highlighted region to the next plot."""
        if range_tuple == 'clear':
            self.highlights.pop(column, None)
            return
            
        if column not in self.highlights: self.highlights[column] = []
        self.highlights[column].append({'range': range_tuple, 'color': color, 'opacity': opacity})
        
    def scale(self, column, range_tuple):
        # Normalize 'column' to always be iterable for logic flow
        # But we can't just wrap string in list because then we need to differentiate?
        # Actually, simpler to just use a helper function or logic flow.
        
        if range_tuple is None or range_tuple == 'clear':
            # Clear logic
            # Check if column is list
            if isinstance(column, (list, tuple)):
                cols = column
            else:
                cols = [column]
            
            for c in cols:
                if c in self.axis_limits:
                    del self.axis_limits[c]
                    print(f"Limits cleared for '{c}'.")
            return

        # Setting logic
        if isinstance(range_tuple, (list, tuple)) and len(range_tuple) == 2:
            # Range is valid
            # Check if column is list
            if isinstance(column, (list, tuple)):
                cols = column
            else:
                cols = [column]
            
            for c in cols:
                self.axis_limits[c] = range_tuple
                print(f"Limits set for '{c}': {range_tuple}")
        else:
            raise ValueError(...)


    # ------------------------------------------------------------------
    # Font Management & Layout Fixes
    # ------------------------------------------------------------------
    def set_font_sizes(self, preset=None, suptitle=None, legend=None, axes_title=None, axes_tick=None):
        if preset:
            presets = {
                'small':   {'suptitle': 14, 'legend': 10, 'axes_title': 12, 'axes_tick': 10},
                'med':     {'suptitle': 18, 'legend': 12, 'axes_title': 14, 'axes_tick': 12},
                'medium':  {'suptitle': 18, 'legend': 12, 'axes_title': 14, 'axes_tick': 12},
                'large':   {'suptitle': 24, 'legend': 16, 'axes_title': 18, 'axes_tick': 14},
                'x-large': {'suptitle': 30, 'legend': 20, 'axes_title': 24, 'axes_tick': 18}
            }
            p = str(preset).lower()
            if p in presets:
                self.suptitle_size = presets[p]['suptitle']
                self.legend_size = presets[p]['legend']
                self.axes_title_size = presets[p]['axes_title']
                self.axes_tick_size = presets[p]['axes_tick']
            else:
                print(f"Warning: Font preset '{preset}' not recognized. Try: {list(presets.keys())}")

        if suptitle is not None: self.suptitle_size = suptitle
        if legend is not None: self.legend_size = legend
        if axes_title is not None: self.axes_title_size = axes_title
        if axes_tick is not None: self.axes_tick_size = axes_tick

    def _apply_fonts(self, fig):
        """Internal method to apply stored font sizes and global layout fixes to a Plotly figure."""
        if fig is None: return fig
        
        layout_updates = {}
        if self.suptitle_size:
            layout_updates['title_font'] = dict(size=self.suptitle_size)
        if self.legend_size:
            layout_updates['legend'] = dict(font=dict(size=self.legend_size))
        
        # --- TITLE & LEGEND OVERLAP FIX (GLOBAL INTERCEPT) ---
        # 1. Force the title to the absolute top of the container coordinates (not paper).
        # 2. Substantially expand the top margin so horizontal legends wrapping upward don't crash into the title.
        layout_updates['title_y'] = 0.99
        layout_updates['title_yref'] = 'container'
        layout_updates['title_yanchor'] = 'top'
        layout_updates['margin_t'] = 140 # Default is ~60; expanded margin makes room for the horizontal legend
        
        if layout_updates:
            # We use `update_layout` dynamically. Plotly seamlessly merges magic kwargs like `margin_t`.
            fig.update_layout(**layout_updates)
            
        x_updates, y_updates = {}, {}
        if self.axes_title_size:
            x_updates['title_font'] = dict(size=self.axes_title_size)
            y_updates['title_font'] = dict(size=self.axes_title_size)
        if self.axes_tick_size:
            x_updates['tickfont'] = dict(size=self.axes_tick_size)
            y_updates['tickfont'] = dict(size=self.axes_tick_size)
            
        if x_updates: fig.update_xaxes(**x_updates)
        if y_updates: fig.update_yaxes(**y_updates)
        
        return fig

    # ------------------------------------------------------------------
    # Main Plot Function
    # ------------------------------------------------------------------
    def plot(self, x=None, y=None, by='vars', figsize=(12, 8), ncols=None, nrows=None, 
                subplot_titles=None, suptitle=None, suppress_legends=False, **kwargs):
            if x is None: x = self.last_x
            if y is None: y = self.last_y
            self.last_x = x
            self.last_y = y

            # Sync persistent dimensions
            if ncols is None and nrows is None:
                if self.last_ncols is not None or self.last_nrows is not None:
                    ncols, nrows = self.last_ncols, self.last_nrows
            self.last_ncols = ncols
            self.last_nrows = nrows

            if by == 'sets' or by == 'datasets':
                fig = uniplot_per_dataset(
                    list_of_datasets=self.uset,
                    x=x,
                    y=y,
                    display_parms=self.display_parms,
                    suptitle=suptitle or self.suptitle,
                    figsize=figsize,
                    ncols=ncols,
                    nrows=nrows,
                    darkmode=self.darkmode,
                    x_lim=self.axis_limits.get(x),
                    y_lim=None, 
                    axis_limits=self.axis_limits, 
                    return_axes=True 
                )
                mode = 'sets'
                
            else:
                plot_args = {
                    'list_of_datasets': self.uset,
                    'x': x, 
                    'y': y,
                    'darkmode': self.darkmode,
                    'display_parms': self.display_parms,
                    'suptitle': suptitle or self.suptitle,
                    'xlabel': self.x_label,      
                    'ylabel': self.y_label, 
                    'subplot_titles': subplot_titles,
                    'return_axes': True,
                    'figsize': figsize,
                    'ncols': ncols,
                    'nrows': nrows,
                }
                plot_args.update(kwargs)
                fig = uniplot(**plot_args)
                mode = 'vars'

            if fig is None: return

            y_list = y if isinstance(y, list) else [y]
            active_sets = [d for d in self.uset if d.select]
            
            if mode == 'vars':
                n_items = len(y_list)
            else:
                n_items = len(active_sets)

            if ncols is None and nrows is None:
                calc_ncols = min(3, max(1, int(np.ceil(np.sqrt(n_items)))))
            elif ncols is None:
                calc_ncols = int(np.ceil(n_items / nrows))
            else:
                calc_ncols = ncols
            calc_ncols = max(1, calc_ncols)

            def apply_to_all_subplots(func, **kwargs):
                for i in range(n_items):
                    r = (i // calc_ncols) + 1
                    c = (i % calc_ncols) + 1
                    func(row=r, col=c, **kwargs)

            # --- LINES ---
            for col_name, lines in self.lines.items():
                if col_name == x:
                    for l in lines:
                        fig.add_vline(x=l['level'], line_dash=l['dash'], line_color=l['color'])
                elif col_name in y_list:
                    if mode == 'vars':
                        for idx, yi in enumerate(y_list):
                            if yi == col_name:
                                r, c = (idx // calc_ncols) + 1, (idx % calc_ncols) + 1
                                for l in lines:
                                    fig.add_hline(y=l['level'], line_dash=l['dash'], line_color=l['color'], row=r, col=c)
                    else: 
                        for l in lines:
                            apply_to_all_subplots(fig.add_hline, y=l['level'], line_dash=l['dash'], line_color=l['color'])

            # --- HIGHLIGHTS ---
            for col_name, hls in self.highlights.items():
                if col_name == x:
                    for h in hls:
                        fig.add_vrect(x0=h['range'][0], x1=h['range'][1], fillcolor=h['color'], 
                                    opacity=h['opacity'], layer="below", line_width=0)
                elif col_name in y_list:
                    if mode == 'vars':
                        for idx, yi in enumerate(y_list):
                            if yi == col_name:
                                r, c = (idx // calc_ncols) + 1, (idx % calc_ncols) + 1
                                for h in hls:
                                    fig.add_hrect(y0=h['range'][0], y1=h['range'][1], fillcolor=h['color'], 
                                                opacity=h['opacity'], layer="below", line_width=0, row=r, col=c)
                    else: 
                        for h in hls:
                            apply_to_all_subplots(fig.add_hrect, y0=h['range'][0], y1=h['range'][1], 
                                                fillcolor=h['color'], opacity=h['opacity'], layer="below", line_width=0)

            if x in self.axis_limits:
                fig.update_xaxes(range=self.axis_limits[x])

            if mode == 'vars':
                for idx, yi in enumerate(y_list):
                    if yi in self.axis_limits:
                        r, c = (idx // calc_ncols) + 1, (idx % calc_ncols) + 1
                        fig.update_yaxes(range=self.axis_limits[yi], row=r, col=c)

            fig = self._apply_fonts(fig)
            if fig and suppress_legends:
                fig.update_traces(visible='legendonly') 
            
            if self.legend == 'off':
                fig.update_layout(showlegend=False)

            self.last_fig = fig
            return fig

    # ------------------------------------------------------------------
    # The bar Command
    # ------------------------------------------------------------------

    def bar(self, x=None, y=None, by='vars', barmode='group', agg='mean', figsize=(12, 8), ncols=None, nrows=None, suppress_legends=False):
            if x is None: x = self.last_x
            if y is None: y = self.last_y
            self.last_x, self.last_y = x, y

            y_list = y if isinstance(y, list) else [y]
    def help(self):
            if by == 'dataset_x':
                fig = unibar_datasets_as_x(
                    list_of_datasets=self.uset, y=y_list, agg=agg,
                    suptitle=self.suptitle, figsize=figsize, 
                    darkmode=self.darkmode, axis_limits=self.axis_limits, return_axes=True
                )
                if fig:
                    fig = self._apply_fonts(fig)
                    if suppress_legends:
                        fig.update_traces(visible='legendonly')
                    self.last_fig = fig
                return fig
                
            elif by in ['sets', 'datasets']:
                fig = unibar_per_dataset(
                    list_of_datasets=self.uset, x=x, y=y, barmode=barmode,
                    suptitle=self.suptitle, figsize=figsize, ncols=ncols, nrows=nrows,
                    darkmode=self.darkmode, return_axes=True 
                )
            else:
                fig = unibar(
                    list_of_datasets=self.uset, x=x, y=y, barmode=barmode,
                    suptitle=self.suptitle, figsize=figsize, ncols=ncols, nrows=nrows,
                    darkmode=self.darkmode, return_axes=True 
                )
                
            if fig:
                if x in self.axis_limits:
                    fig.update_xaxes(range=self.axis_limits[x])

                active_sets = [d for d in self.uset if d.select]
                n_items = len(active_sets) if by in ['sets', 'datasets'] else len(y_list)
                
                calc_ncols = ncols
                if calc_ncols is None and nrows is None:
                    calc_ncols = min(3, max(1, int(np.ceil(np.sqrt(n_items)))))
                elif calc_ncols is None:
                    calc_ncols = int(np.ceil(n_items / nrows))
                calc_ncols = max(1, calc_ncols)

                if by in ['sets', 'datasets']:
                    primary_y = y_list[0]
                    if primary_y in self.axis_limits:
                        fig.update_yaxes(range=self.axis_limits[primary_y])
                else:
                    for idx, yi in enumerate(y_list):
                        if yi in self.axis_limits:
                            r = (idx // calc_ncols) + 1
                            c = (idx % calc_ncols) + 1
                            fig.update_yaxes(range=self.axis_limits[yi], row=r, col=c)

                fig = self._apply_fonts(fig)
                if suppress_legends:
                    fig.update_traces(visible='legendonly')
                self.last_fig = fig
                
            return fig

    # ------------------------------------------------------------------
    # The box Command
    # ------------------------------------------------------------------

    def box(self, x=None, y=None, by='vars', boxmode='group', points='outliers', notched=False, 
                color=None, suptitle=None, figsize=(12, 8), ncols=None, nrows=None, suppress_legends=False):
            if x is None: x = self.last_x
            if y is None: y = self.last_y
            self.last_x, self.last_y = x, y

            y_list = y if isinstance(y, list) else [y]

            if by == 'dataset_x':
                fig = unibox_datasets_as_x(
                    list_of_datasets=self.uset, y=y_list, boxmode=boxmode, 
                    points=points, notched=notched, suptitle=suptitle or self.suptitle, 
                    figsize=figsize, darkmode=self.darkmode, axis_limits=self.axis_limits, return_axes=True
                )
                if fig:
                    fig = self._apply_fonts(fig)
                    if suppress_legends:
                        fig.update_traces(visible='legendonly')
                    self.last_fig = fig
                return fig

            elif by in ['sets', 'datasets']:
                primary_y = y_list[0]
                y_limit = self.axis_limits.get(primary_y)
                fig = unibox_per_dataset(
                    list_of_datasets=self.uset, x=x, y=y, boxmode=boxmode,
                    points=points, notched=notched,
                    suptitle=suptitle or self.suptitle, figsize=figsize, ncols=ncols, nrows=nrows,
                    darkmode=self.darkmode, y_lim=y_limit, return_axes=True
                )
            else:
                fig = unibox(
                    list_of_datasets=self.uset, x=x, y=y, boxmode=boxmode,
                    points=points, notched=notched, color=color,
                    suptitle=suptitle or self.suptitle, figsize=figsize, ncols=ncols, nrows=nrows,
                    darkmode=self.darkmode, y_lim=None, return_axes=True
                )
                
                if fig:
                    active_sets = [d for d in self.uset if d.select]
                    n_items = len(y_list)
                    calc_ncols = ncols
                    if calc_ncols is None and nrows is None:
                        calc_ncols = min(3, max(1, int(np.ceil(np.sqrt(n_items)))))
                    elif calc_ncols is None:
                        calc_ncols = int(np.ceil(n_items / nrows))
                    calc_ncols = max(1, calc_ncols)

                    for idx, yi in enumerate(y_list):
                        if yi in self.axis_limits:
                            r = (idx // calc_ncols) + 1
                            c = (idx % calc_ncols) + 1
                            fig.update_yaxes(range=self.axis_limits[yi], row=r, col=c)

            if fig:
                if x in self.axis_limits:
                    fig.update_xaxes(range=self.axis_limits[x])
                fig = self._apply_fonts(fig)
                if suppress_legends:
                    fig.update_traces(visible='legendonly')
                self.last_fig = fig
                
            
            if self.legend == 'off':
                fig.update_layout(showlegend=False)
                
            return fig               

    # ------------------------------------------------------------------
    # The histogram Command
    # ------------------------------------------------------------------
    def histogram(self, x=None, y=None, histfunc='sum', by='vars', nbins=None, 
                    bin_size=None, bin_start=None, bin_end=None,
                    histnorm='', barmode='overlay', opacity=0.7,
                    color=None, suptitle=None, figsize=(12, 8), ncols=None, nrows=None, suppress_legends=False):
            if x is None: x = self.last_x
            self.last_x = x
            
            limit = None
            if isinstance(x, str):
                limit = self.axis_limits.get(x)
            elif isinstance(x, list) and len(x) == 1:
                limit = self.axis_limits.get(x[0])

            if by in ['sets', 'datasets']:
                fig = unihistogram_by_dataset(
                    list_of_datasets=self.uset, x=x, y=y, histfunc=histfunc, nbins=nbins,
                    bin_size=bin_size, bin_start=bin_start, bin_end=bin_end, 
                    histnorm=histnorm, barmode=barmode, opacity=opacity, color=color,
                    suptitle=suptitle or self.suptitle, figsize=figsize, ncols=ncols, nrows=nrows,
                    darkmode=self.darkmode, x_lim=limit, return_axes=True
                )
            else:
                fig = unihistogram(
                    list_of_datasets=self.uset, x=x, y=y, histfunc=histfunc, nbins=nbins,
                    bin_size=bin_size, bin_start=bin_start, bin_end=bin_end, 
                    histnorm=histnorm, barmode=barmode, opacity=opacity, color=color,
                    suptitle=suptitle or self.suptitle, figsize=figsize, ncols=ncols, nrows=nrows,
                    darkmode=self.darkmode, x_lim=limit, return_axes=True
                )
                
            fig = self._apply_fonts(fig)
            if fig and suppress_legends:
                fig.update_traces(visible='legendonly')
            self.last_fig = fig
            
            if self.legend == 'off':
                fig.update_layout(showlegend=False)
            return fig
        
    def contour(self, x=None, y=None, z=None, by='vars', contours_coloring='fill', 
                colorscale=None, interpolate=True, interp_res=100, interp_method='linear',
                ncontours=None,
                suptitle=None, figsize=(12, 8), ncols=None, nrows=None, suppress_legends=False):
            if x is None: x = self.last_x
            if y is None: y = self.last_y
            self.last_x, self.last_y = x, y

            if z is None:
                print("Error: Contour plots require a 'z' variable to map to color.")
                return

            limit_x = self.axis_limits.get(x)
            limit_y = self.axis_limits.get(y)

            if by in ['sets', 'datasets']:
                fig = unicontour_per_dataset(
                    list_of_datasets=self.uset, x=x, y=y, z=z, 
                    contours_coloring=contours_coloring, colorscale=colorscale,
                    interpolate=interpolate, interp_res=interp_res, interp_method=interp_method,
                    ncontours=ncontours,
                    suptitle=suptitle or self.suptitle, figsize=figsize, ncols=ncols, nrows=nrows, 
                    darkmode=self.darkmode, axis_limits=self.axis_limits, return_axes=True
                )
            else:
                fig = unicontour(
                    list_of_datasets=self.uset, x=x, y=y, z=z,
                    contours_coloring=contours_coloring, colorscale=colorscale,
                    interpolate=interpolate, interp_res=interp_res, interp_method=interp_method,
                    ncontours=ncontours,
                    suptitle=suptitle or self.suptitle, figsize=figsize, ncols=ncols, nrows=nrows, 
                    darkmode=self.darkmode, axis_limits=self.axis_limits, return_axes=True
                )
                
            if fig:
                if limit_x: fig.update_xaxes(range=limit_x)
                if limit_y: fig.update_yaxes(range=limit_y)
                fig = self._apply_fonts(fig)
                if suppress_legends:
                    fig.update_traces(visible='legendonly')
                self.last_fig = fig
            
            if self.legend == 'off':
                fig.update_layout(showlegend=False)
                
            return fig

    # ------------------------------------------------------------------
    # The table Command
    # ------------------------------------------------------------------
    def table(self, cols=None, title=None, clipboard=False):
        if cols is None:
            if self.last_x is None or self.last_y is None:
                return "No columns specified and no previous plot variables defined."
            
            y_part = self.last_y if isinstance(self.last_y, list) else [self.last_y]
            target_cols = [self.last_x] + y_part
        else:
            target_cols = cols if isinstance(cols, list) else [cols]

        combined_dfs = []
        for ds in self.uset:
            if not ds.select:
                continue
            
            valid_cols = [c for c in target_cols if c in ds.df.columns]
            if not valid_cols:
                continue
                
            subset = ds.df[valid_cols].copy()
            subset.insert(0, 'Dataset', ds.title)
            combined_dfs.append(subset)

        if not combined_dfs:
            return "No data found for the specified columns in selected datasets."

        final_df = pd.concat(combined_dfs, ignore_index=True)
        final_df = final_df.fillna('-')

        if clipboard:
            try:
                final_df.to_clipboard(index=False, excel=True)
                table_name = title if title else "Data"
            except Exception as e:
                print(f"Clipboard copy failed: {e}\nFalling back to Tab-Separated output...\n")
        
        md_lines = []
        
        if title:
            md_lines.append(f"### {title}\n")
            
        headers = final_df.columns.tolist()
        md_lines.append("| " + " | ".join(str(h) for h in headers) + " |")
        md_lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        
        for _, row in final_df.iterrows():
            md_lines.append("| " + " | ".join(str(val) for val in row.values) + " |")

        return "\n".join(md_lines)

    def save_png(self, filename="plot.png", scale=3, width=None, height=None):
        if self.last_fig is None:
            print("No plot to save. Please run .plot() first.")
            return

        try:
            self.last_fig.write_image(filename, scale=scale, width=width, height=height)
            print(f"Plot saved to {filename}")
        except ValueError as e:
            print(f"Error saving image (ensure 'kaleido' is installed): {e}")
        except Exception as e:
            print(f"Error saving image: {e}")

    def list_sets(self):
        if not self.uset:
            print("No datasets loaded.")
            return

        rows = []
        for ds in self.uset:
            selected = "✓" if ds.select else "X"
            shape = f"{ds.df.shape[0]} x {ds.df.shape[1]}"
            query_info = str(ds.query)
            rows.append([
                f"Set {ds.index}",
                ds.title,
                selected,
                shape,
                query_info
            ])

        headers = ["Set", "Title", "Selected", "Shape", "Query?" ]
        col_widths = [
            max(len(row[i]) for row in [headers] + rows) + 2
            for i in range(len(headers))
        ]

        header_str = "".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        sep = "-" * sum(col_widths)

        output_lines = [header_str, sep]
        for row in rows:
            line = "".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row))
            output_lines.append(line)

        print("\nLoaded Datasets:")
        print("\n".join(output_lines))

    def list_parms(self, set_number=None, search_string=None, use_regex=False):
            import fnmatch
            
            if set_number is None:
                target_sets = self.selected()
                if not target_sets:
                    target_sets = self.uset
            else:
                target_sets = self._get_uset_slice(set_number)
                
            if not target_sets:
                print("No datasets available to list parameters from.")
                return []
                
            all_cols = set()
            for ds in target_sets:
                all_cols.update(ds.df.columns)
                
            if search_string:
                try:
                    if not use_regex:
                        if not any(c in search_string for c in ['*', '?', '[', ']']):
                            search_string = f"*{search_string}*"
                        pattern_str = fnmatch.translate(search_string)
                    else:
                        pattern_str = search_string
                        
                    pattern = re.compile(pattern_str, re.IGNORECASE)
                    filtered_cols = [col for col in all_cols if pattern.search(str(col))]
                    
                except re.error as e:
                    print(f"Invalid search pattern '{search_string}': {e}")
                    return []
            else:
                filtered_cols = list(all_cols)
                
            filtered_cols.sort(key=lambda x: str(x).lower())
            
            print(f"Found {len(filtered_cols)} parameters", end="")
            if search_string:
                print(f" matching '{search_string}'", end="")
            if set_number is not None:
                print(f" in set(s) {set_number}:")
            else:
                print(" in active datasets:")
                
            for col in filtered_cols:
                desc = self.parm_description_dict.get(col, "No description available.")
                print(f"  - {str(col).ljust(25)} : {desc}")
                
            return filtered_cols

    def get_fit_metrics(self, uset_slice=None, x=None, y=None):
        """
        Calculates and returns the regression equation, R-squared, and RMSE 
        for the specified datasets that have a reg_order set.
        Supports polynomial (int), exponential ('exp'), logarithmic ('log'), and power ('power') fits.
        """
        import numpy as np
        import pandas as pd
        from scipy.optimize import curve_fit
        import warnings
        
        if x is None: x = self.last_x
        if y is None: y = self.last_y
        
        if x is None or y is None:
            return "Error: Please specify x and y variables, or run a plot first."
            
        y_list = y if isinstance(y, list) else [y]
        
        # Grab the targeted datasets
        target_sets = self._get_uset_slice(uset_slice)
        
        # Filter for datasets that actually have a regression order set.
        if uset_slice is None:
            active_ds = [d for d in target_sets if d.select and d.reg_order is not None]
        else:
            active_ds = [d for d in target_sets if d.reg_order is not None]
            
        if not active_ds:
            return "No specified datasets have a valid 'reg_order' set."
            
        results = []
        
        # Suppress curve_fit warnings to keep terminal output clean if a fit fails
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        for ds in active_ds:
            # Safely drop NaNs to ensure accurate math
            df_clean = ds.df.dropna(subset=[x] + [yi for yi in y_list if yi in ds.df.columns])
            
            for yi in y_list:
                if yi not in df_clean.columns: 
                    continue
                
                order = ds.reg_order
                
                # Pre-filter data based on fit type constraints
                if isinstance(order, str) and order.lower() in ['log', 'power']:
                    # Log and Power fits require x > 0
                    valid_data = df_clean[df_clean[x] > 0]
                else:
                    valid_data = df_clean
                    
                x_data = valid_data[x]
                y_data = valid_data[yi]
                
                # Failsafe for insufficient data points
                if len(x_data) < 3:
                    continue
                
                try:
                    # 1. Calculate the fit based on type
                    if isinstance(order, int) and order > 0:
                        # Polynomial
                        z = np.polyfit(x_data, y_data, order)
                        p = np.poly1d(z)
                        y_pred = p(x_data)
                        
                        terms = []
                        for i, coef in enumerate(z):
                            pwr = order - i
                            if pwr == 0: terms.append(f"{coef:.4g}")
                            elif pwr == 1: terms.append(f"{coef:.4g}x")
                            else: terms.append(f"{coef:.4g}x^{pwr}")
                        eq_str = " + ".join(terms).replace(" + -", " - ")
                        
                    elif isinstance(order, str):
                        order_lower = order.lower()
                        if order_lower == 'exp':
                            def func(x, a, b): return a * np.exp(b * x)
                            popt, _ = curve_fit(func, x_data, y_data)
                            y_pred = func(x_data, *popt)
                            eq_str = f"{popt[0]:.4g} * e^({popt[1]:.4g}x)"
                            
                        elif order_lower == 'log':
                            def func(x, a, b): return a * np.log(x) + b
                            popt, _ = curve_fit(func, x_data, y_data)
                            y_pred = func(x_data, *popt)
                            eq_str = f"{popt[0]:.4g} * ln(x) + {popt[1]:.4g}".replace("+ -", "- ")
                            
                        elif order_lower == 'power':
                            def func(x, a, b): return a * (x ** b)
                            popt, _ = curve_fit(func, x_data, y_data)
                            y_pred = func(x_data, *popt)
                            eq_str = f"{popt[0]:.4g} * x^{popt[1]:.4g}"
                        else:
                            continue  # Skip unknown string types
                    else:
                        continue
                        
                except Exception:
                    # Fit failed to converge; skip to the next variable/dataset
                    continue

                # 2. Calculate Metrics
                residuals = y_data - y_pred
                rmse = np.sqrt(np.mean(residuals**2))
                ss_tot = np.sum((y_data - np.mean(y_data))**2)
                
                # Handle edge cases where variance is zero
                if ss_tot == 0:
                    r_squared = np.nan
                else:
                    ss_res = np.sum(residuals**2)
                    r_squared = 1 - (ss_res / ss_tot)
                
                results.append({
                    "Set": ds.index,
                    "Title": ds.title,
                    "Y-Var": yi,
                    "Order": order,
                    "R²": round(r_squared, 4),
                    "RMSE": round(rmse, 4),
                    "Equation": eq_str
                })
                
        # Restore warnings to default
        warnings.filterwarnings("default", category=RuntimeWarning)
                
        if not results:
            return "Could not calculate metrics. Check data for NaNs, insufficient rows, or non-converging fits."
            
        return pd.DataFrame(results)
  
    def summary(self, cols=None):
            if cols is None:
                y_part = self.last_y if isinstance(self.last_y, list) else [self.last_y] if self.last_y else []
                x_part = [self.last_x] if self.last_x else []
                target_cols = x_part + y_part
            else:
                target_cols = cols if isinstance(cols, list) else [cols]

            if not target_cols:
                print("No columns specified and no previous plot variables defined.")
                return

            active_ds = self.selected()
            if not active_ds:
                print("No datasets selected. Cannot generate summary.")
                return

            rows = []
            for ds in active_ds:
                if ds.query:
                    q_str = str(ds.query)
                    query_disp = (q_str[:27] + '...') if len(q_str) > 30 else q_str
                else:
                    query_disp = "-"

                for col in target_cols:
                    if col in ds.df.columns:
                        data = ds.df[col].dropna()
                        
                        if data.empty:
                            rows.append([
                                f"Set {ds.index}", str(ds.title)[:20], query_disp, 
                                str(col)[:15], "0", "-", "-", "-", "-"
                            ])
                        elif pd.api.types.is_numeric_dtype(data):
                            count = len(data)
                            vmin = data.min()
                            vmean = data.mean()
                            vmax = data.max()
                            vstd = data.std()
                            
                            rows.append([
                                f"Set {ds.index}",
                                str(ds.title)[:20],
                                query_disp,
                                str(col)[:15],
                                f"{count}",
                                f"{vmin:.4g}",
                                f"{vmean:.4g}",
                                f"{vmax:.4g}",
                                f"{vstd:.4g}" if pd.notna(vstd) else "-"
                            ])
                        else:
                            rows.append([
                                f"Set {ds.index}", str(ds.title)[:20], query_disp, 
                                str(col)[:15], f"{len(data)}", "Non-numeric", "-", "-", "-"
                            ])

            if not rows:
                print(f"None of the selected datasets contain the specified columns: {target_cols}")
                return

            headers = ["Set", "Title", "Query", "Variable", "Count", "Min", "Mean", "Max", "Std"]
            
            col_widths = [max(len(str(item)) for item in col) + 2 for col in zip(*([headers] + rows))]

            header_str = "".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
            sep = "-" * sum(col_widths)

            print(f"\nStatistical Summary for: {', '.join(target_cols)}")
            print(header_str)
            print(sep)
            for row in rows:
                print("".join(str(val).ljust(w) for val, w in zip(row, col_widths)))

    def help(self):
        print("=" * 70)
        print("📚 UnichartNotebook HELP")
        print("=" * 70)
        
        cls = self.__class__
        if cls.__doc__:
            print("\n📋 CLASS DESCRIPTION:")
            print(cls.__doc__)
        else:
            print("\n⚠️  No class docstring found.")

        methods = inspect.getmembers(cls, predicate=inspect.isfunction)
        public_methods = [m for m in methods if not m[0].startswith('_') or m[0] in ['__init__', '__repr__']]

        print("\n🔍 PUBLIC METHODS:")
        print("-" * 70)
        for name, func in public_methods:
            sig = inspect.signature(func)
            doc = inspect.getdoc(func) or "No description available."
            doc_preview = doc.split('\n')[0] if doc else ""
            print(f"• {name}{sig}")
            print(f"  → {doc_preview}")
            print()
        
        print("🛠️  PUBLIC ATTRIBUTES (Instance):")
        print("-" * 70)
        attrs = []
        for attr, value in self.__dict__.items():
            if not attr.startswith('_'):  
                attrs.append((attr, value))

        if not attrs:
            print("No public instance attributes found.")
        else:
            for name, val in sorted(attrs):
                val_str = str(val)[:100]  
                if len(str(val)) > 100:
                    val_str += "..."
                print(f"• {name}: {type(val).__name__} = {val_str}")

        print("\n💡 QUICK START TIPS:")
        print("-" * 70)
        print("1. Load data:      nb.load_df(df, title='MyData')")
        print("2. Select datasets: nb.select([0, 1])")
        print("3. Plot:           nb.plot(x='time', y='value')")
        print("4. View help:      nb.help()")
        print("\nSee method signatures above for full parameters.")

        print("\n" + "=" * 70)

    # ------------------------------------------------------------------
    # Interactive "GUI" Replacement
    # ------------------------------------------------------------------
    def _refresh_widgets(self):
        items = []
        items.append(widgets.HTML("<b>Dataset Manager</b>"))
        
        for i, ds in enumerate(self.uset):
            chk = widgets.Checkbox(value=ds.select, description=f"{i}: {ds.title}", indent=False, layout=widgets.Layout(width='300px'))
            
            def on_change(change, dataset=ds):
                dataset.select = change['new']
            chk.observe(on_change, names='value')
            
            details = widgets.Label(value=f"[Rows: {len(ds.df)}] [Query: {ds.query}]", layout=widgets.Layout(width='200px'))
            
            cp = widgets.ColorPicker(concise=True, value='blue', layout=widgets.Layout(width='30px'))
            
            def on_color_change(change, dataset=ds):
                dataset.color = change['new']
            cp.observe(on_color_change, names='value')

            row = widgets.HBox([chk, cp, details])
            items.append(row)
            
        self.dataset_widget_container.children = tuple(items)

    def gui(self):
        """Displays the interactive dataset manager widgets."""
        self._refresh_widgets()
        
        btn_refresh = widgets.Button(description="Refresh Data View")
        btn_refresh.on_click(lambda b: self._refresh_widgets())
        
        display(widgets.VBox([btn_refresh, self.dataset_widget_container]))