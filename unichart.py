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
from IPython.display import display, clear_output, HTML
import re
import inspect
from scipy.interpolate import griddata
import functools
import gc
from concurrent.futures import ThreadPoolExecutor

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

FONT_SIZE_MAP = {
    'xs':     8,
    'xsmall': 8,
    'sm':     10,
    'small':  10,
    'md':     12,
    'medium': 12,
    'base':   12,
    'lg':     14,
    'large':  14,
    'xl':     18,
    'xlarge': 18,
    'xxl':    24,
    '2xl':    24,
    'xxxl':   28,
    '3xl':    28,
    'huge':   32,
}

def get_plotly_marker(mpl_marker):
    # Pass through if already a Plotly-native name
    if mpl_marker in MARKER_MAP_MPL_TO_PLOTLY.values():
        return mpl_marker
    return MARKER_MAP_MPL_TO_PLOTLY.get(mpl_marker, 'circle')

def get_plotly_linestyle(mpl_style):
    # Pass through if already a Plotly-native name
    if mpl_style in LINESTYLE_MAP_MPL_TO_PLOTLY.values():
        return mpl_style
    return LINESTYLE_MAP_MPL_TO_PLOTLY.get(mpl_style, 'solid')

def validate_color(value):
    """Return True if value is a string (Plotly accepts named colors, hex, and rgb strings)."""
    if not isinstance(value, str):
        return False
    return True

def validate_marker(value):
    return value in MARKER_MAP_MPL_TO_PLOTLY or value is None

def validate_linestyle(value):
    return value in LINESTYLE_MAP_MPL_TO_PLOTLY or value is None

def marker_map(index):
    markers = list(MARKER_MAP_MPL_TO_PLOTLY.keys())
    return markers[index % len(markers)]

def _generate_contour_grid(x_data, y_data, z_data, res=100, method='linear'):
    """
    Interpolates scattered x, y, z data into a uniform 2D grid for contour plotting.
    Leaves data outside the convex hull as NaN.
    """
    valid = ~(np.isnan(x_data) | np.isnan(y_data) | np.isnan(z_data))
    x_val = x_data[valid]
    y_val = y_data[valid]
    z_val = z_data[valid]

    if len(x_val) < 4:
        return x_val.values, y_val.values, z_val.values

    xi = np.linspace(x_val.min(), x_val.max(), res)
    yi = np.linspace(y_val.min(), y_val.max(), res)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    zi_grid = griddata((x_val, y_val), z_val, (xi_grid, yi_grid), method=method)

    return xi, yi, zi_grid

# -----------------------------------------------------------------------------
# Private Helpers
# -----------------------------------------------------------------------------

def _calc_grid(n, nrows, ncols):
    if nrows is None and ncols is None:
        ncols = min(3, max(1, int(np.ceil(np.sqrt(n)))))
        nrows = int(np.ceil(n / ncols))
    elif nrows is None:
        nrows = int(np.ceil(n / ncols))
    elif ncols is None:
        ncols = int(np.ceil(n / nrows))
    return nrows, ncols

def _base_layout(darkmode, suptitle, figsize, **extra):
    title_defaults = {'x': 0.5, 'yref': 'container', 'y': 0.99, 'yanchor': 'top'}
    incoming_title = extra.pop('title', {})
    if isinstance(incoming_title, str):
        incoming_title = {'text': incoming_title}
    if 'text' not in incoming_title:
        incoming_title['text'] = suptitle
    merged_title = {**title_defaults, **incoming_title}

    default_margin = {'t': 100}
    incoming_margin = extra.pop('margin', {})
    merged_margin = {**default_margin, **incoming_margin}

    args = {
        'template': "plotly_dark" if darkmode else "plotly_white",
        'title': merged_title,
        'margin': merged_margin,
        **extra
    }
    if figsize:
        args['width'] = figsize[0] * 100
        args['height'] = figsize[1] * 100
    return args

def _build_xbins(bin_size, bin_start, bin_end):
    d = {}
    if bin_size is not None: d['size'] = bin_size
    if bin_start is not None: d['start'] = bin_start
    if bin_end is not None: d['end'] = bin_end
    return d or None

def _show_or_return(fig, return_axes):
    if return_axes:
        return fig
    fig.show()
    return fig

def _subplot_refs(row, col, ncols):
    """Return the (xref, yref) axis name strings for a subplot at (row, col) in an ncols grid."""
    idx = (row - 1) * ncols + col
    xref = 'x' if idx == 1 else f'x{idx}'
    yref = 'y' if idx == 1 else f'y{idx}'
    return xref, yref

# -----------------------------------------------------------------------------
# Variable Format Resolver (used by multi-y plot)
# -----------------------------------------------------------------------------
_VAR_FORMAT_KEYS = ('color', 'marker', 'linestyle', 'markersize', 'linewidth', 'alpha')

def _resolve_var_format(dataset, variable, variable_formats=None):
    """
    Per-attribute precedence: variable_formats wins, else dataset attr.

    Used by the multi-y-axis plot. Returns a flat dict containing the
    final color/marker/linestyle/markersize/linewidth/alpha that should
    be applied for a given (dataset, variable) pair.
    """
    variable_formats = variable_formats or {}
    var_fmt = variable_formats.get(variable, {})
    return {
        'color':      var_fmt.get('color',      dataset.color),
        'marker':     var_fmt.get('marker',     dataset.marker),
        'linestyle':  var_fmt.get('linestyle',  dataset.linestyle),
        'markersize': var_fmt.get('markersize', dataset.markersize),
        'linewidth':  var_fmt.get('linewidth',  dataset.linewidth),
        'alpha':      var_fmt.get('alpha',      dataset.alpha),
        'edge_color': getattr(dataset, 'edge_color', 'black'),
        'edgewidth':  getattr(dataset, 'edgewidth', 1),
    }

# -----------------------------------------------------------------------------
# Dataset Class
# -----------------------------------------------------------------------------

class Dataset:
    """Stores a DataFrame and the visual/formatting attributes used when plotting it."""

    def __init__(self, df, index=0, title=None, display_parms=None):
        self._df_full = df
        self._df_filtered = df
        self.query = None
        self._select = True
        
        if title:
            self.title = title
        elif "TITLE" in df.columns:
            self.title = str(df["TITLE"].iloc[0])
        else:
            self.title = "Untitled"
            
        self.index = index
        self.title_format = f"{self.title} {index}"
        
        default_colors = px.colors.qualitative.Plotly
        self._color = default_colors[index % len(default_colors)]
        
        self._marker = marker_map(index)
        self._edge_color = "black"
        self._linestyle = None
        self.markersize = 10
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
        self._plot_type = 'scatter'
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
        valid_plot_types = ['scatter', 'contour', 'histogram']
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
    """1D interpolation on a sorted DataFrame column."""
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

def _parse_reg_spec(spec):
    """Normalize reg_order spec into (kind, param)."""
    if not spec:
        return None, None
    if isinstance(spec, bool):
        return None, None
    if isinstance(spec, numbers.Number):
        return ('poly', int(spec)) if spec > 0 else (None, None)

    aliases = {
        'linear': ('poly', 1), 'lin': ('poly', 1),
        'quadratic': ('poly', 2), 'cubic': ('poly', 3),
        'poly': ('poly', None),
        'log': ('log', None), 'logarithmic': ('log', None),
        'exp': ('exp', None), 'exponential': ('exp', None),
        'power': ('power', None), 'pow': ('power', None),
        'lowess': ('lowess', 0.3), 'loess': ('lowess', 0.3),
        'spline': ('spline', 3), 'cubic_spline': ('spline', 3),
        'ma': ('ma', None), 'moving_average': ('ma', None), 'rolling': ('ma', None),
    }

    if isinstance(spec, str):
        s = spec.lower().strip()
        if s in aliases: return aliases[s]
        if s.startswith('poly'):
            try: return 'poly', int(s[4:])
            except ValueError: pass
        raise ValueError(f"Unknown regression type: {spec!r}")

    if isinstance(spec, (tuple, list)) and len(spec) == 2:
        kind, param = spec
        kind = str(kind).lower().strip()
        if kind in aliases:
            return aliases[kind][0], param
        raise ValueError(f"Unknown regression kind: {kind!r}")

    raise ValueError(f"Invalid reg_order spec: {spec!r}")


def _calculate_regression(df, x_col, y_col, spec):
    """
    Compute a regression curve. Returns (x_array, y_array, label).
    Returns (None, None, None) when spec is falsy or fit cannot be computed.
    """
    kind, param = _parse_reg_spec(spec)
    if kind is None:
        return None, None, None

    df_clean = df.dropna(subset=[x_col, y_col]).sort_values(by=x_col)
    x = df_clean[x_col].to_numpy(dtype=float)
    y = df_clean[y_col].to_numpy(dtype=float)
    if len(x) < 2:
        return None, None, None

    x_lin = np.linspace(x.min(), x.max(), 200)

    try:
        if kind == 'poly':
            order = int(param) if param else 1
            if len(x) < order + 1:
                return None, None, None
            p = np.poly1d(np.polyfit(x, y, order))
            label = 'Linear' if order == 1 else f'LS{order}'
            return x_lin, p(x_lin), label

        if kind == 'log':
            mask = x > 0
            if mask.sum() < 2: return None, None, None
            a, b = np.polyfit(np.log(x[mask]), y[mask], 1)
            y_lin = np.full_like(x_lin, np.nan)
            pos = x_lin > 0
            y_lin[pos] = a * np.log(x_lin[pos]) + b
            return x_lin, y_lin, 'Log'

        if kind == 'exp':
            mask = y > 0
            if mask.sum() < 2: return None, None, None
            b, log_a = np.polyfit(x[mask], np.log(y[mask]), 1)
            return x_lin, np.exp(log_a) * np.exp(b * x_lin), 'Exp'

        if kind == 'power':
            mask = (x > 0) & (y > 0)
            if mask.sum() < 2: return None, None, None
            b, log_a = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
            y_lin = np.full_like(x_lin, np.nan)
            pos = x_lin > 0
            y_lin[pos] = np.exp(log_a) * np.power(x_lin[pos], b)
            return x_lin, y_lin, 'Power'

        if kind == 'lowess':
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
            except ImportError:
                warnings.warn("LOWESS requires statsmodels. Install with `pip install statsmodels`.")
                return None, None, None
            frac = float(param) if param is not None else 0.3
            res = lowess(y, x, frac=frac, return_sorted=True)
            return res[:, 0], res[:, 1], f'LOWESS({frac:.2f})'

        if kind == 'spline':
            from scipy.interpolate import UnivariateSpline
            k = max(1, min(5, int(param) if param else 3))
            ux, idx = np.unique(x, return_index=True)
            uy = y[idx]
            if len(ux) < k + 1:
                return None, None, None
            spl = UnivariateSpline(ux, uy, k=k)
            return x_lin, spl(x_lin), f'Spline{k}'

        if kind == 'ma':
            window = int(param) if param else max(3, len(x) // 20)
            window = max(2, min(window, len(x)))
            ma = pd.Series(y).rolling(window=window, center=True, min_periods=1).mean().to_numpy()
            return x, ma, f'MA({window})'

    except Exception as e:
        warnings.warn(f"Regression '{kind}' failed: {e}")
        return None, None, None

    return None, None, None

# -----------------------------------------------------------------------------
# Main Plotting Functions
# -----------------------------------------------------------------------------
def uniplot(list_of_datasets, x, y, z=None, plot_type=None, color=None, hue=None, marker=None,
            markersize=10, marker_edge_color="black", linestyle=None, hue_palette="Jet",
            hue_order=None, line=False, suppress_msg=False, return_axes=False, axes=None,
            suptitle=None, xlabel=None, ylabel=None, subplot_titles=None,
            darkmode=False, interactive=True, display_parms=None, grid=True,
            legend='above', legend_ncols=1, figsize=(12, 8), ncols=None, nrows=None, x_lim=None, y_lim=None,
            axis_limits=None):

    axis_limits = axis_limits or {}
    x_list = x if isinstance(x, list) else [x]
    y_list = y if isinstance(y, list) else [y]

    if len(x_list) == len(y_list):
        pairs = list(zip(x_list, y_list))
    elif len(x_list) == 1:
        pairs = [(x_list[0], yi) for yi in y_list]
    elif len(y_list) == 1:
        pairs = [(xi, y_list[0]) for xi in x_list]
    else:
        raise ValueError(
            f"x and y must be the same length, or one must be a single value. "
            f"Got len(x)={len(x_list)}, len(y)={len(y_list)}."
        )

    n_plots = len(pairs)
    if n_plots == 0: raise ValueError("At least one x/y pair is required.")

    nrows, ncols = _calc_grid(n_plots, nrows, ncols)

    numeric_hue_info = {}
    for _ds in list_of_datasets:
        if not _ds.select: continue
        _fmt = _ds.get_format_dict()
        _cur_hue = _fmt.get('hue') or hue
        if not _cur_hue or _cur_hue in numeric_hue_info: continue
        _df = _ds.df
        if _cur_hue in _df.columns and pd.api.types.is_numeric_dtype(_df[_cur_hue]):
            _idx = len(numeric_hue_info) + 1
            numeric_hue_info[_cur_hue] = {
                'ca_name': 'coloraxis' if _idx == 1 else f'coloraxis{_idx}',
                'palette': _fmt.get('hue_palette', 'Jet'),
                'lim': axis_limits.get(_cur_hue),
            }

    right_margin = max(80, len(numeric_hue_info) * 90)

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subplot_titles, shared_xaxes=False)
    fig.update_layout(**_base_layout(
        darkmode, None, figsize,
        title={'text': suptitle or (f"{x} vs {[str(yi) for yi in y_list]}" if len(x_list) == 1 else f"{x_list} vs {y_list}"), 'x': 0.5, 'xanchor': 'center'},
        showlegend=(legend != 'off'),
        margin=dict(r=right_margin),
    ))

    for dataset in list_of_datasets:
        if not dataset.select: continue
        
        base_df = dataset.df
        
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

        for idx_p, (x_name, yi) in enumerate(pairs):
            row = idx_p // ncols + 1
            col = idx_p % ncols + 1

            if yi not in base_df.columns: continue

            if x_name not in base_df.columns:
                cols_upper = {c.upper(): c for c in base_df.columns}
                x_key = cols_upper.get(str(x_name).upper())
                if not x_key: continue
                x_col = x_key
            else:
                x_col = x_name

            req_cols = [x_col, yi]
            if cur_hue and cur_hue in base_df.columns: req_cols.append(cur_hue)
            valid_hover = [p for p in hover_parms if p in base_df.columns]
            req_cols.extend(valid_hover)
            if dataset.order and dataset.order != 'index' and dataset.order in base_df.columns:
                req_cols.append(dataset.order)
            
            req_cols = list(dict.fromkeys(req_cols))
            
            valid_cols_mask = ~base_df.columns.duplicated()
            df = base_df.loc[:, valid_cols_mask][req_cols]

            if dataset.order == 'index': df = df.sort_index()
            elif dataset.order: df = df.sort_values(by=dataset.order)
            else: df = df.sort_index()

            custom_data_cols = []
            seen_cols = set()
            for c in [x_col, yi] + valid_hover:
                if c not in seen_cols:
                    custom_data_cols.append(c)
                    seen_cols.add(c)
            
            custom_data = df[custom_data_cols]
            def get_cd_idx(col_name): return custom_data_cols.index(col_name)

            ht = f"<b><u>Set: {cur_idx}</u></b><br><b>{cur_title}</b><br>{x_col}: %{{customdata[{get_cd_idx(x_col)}]:.2f}}<br>{yi}: %{{customdata[{get_cd_idx(yi)}]:.2f}}"
            for parm in valid_hover:
                if pd.api.types.is_numeric_dtype(df[parm]): ht += f"<br>{parm}: %{{customdata[{get_cd_idx(parm)}]:.5g}}"
                else: ht += f"<br>{parm}: %{{customdata[{get_cd_idx(parm)}]}}"
            ht += "<extra></extra>"

            mode_parts = []
            if not cur_linestyle or cur_reg_order: mode_parts.append('markers')
            else: mode_parts.append('lines')
            if cur_marker: mode_parts.append('markers')
            mode = "+".join(mode_parts)

            marker_dict = dict(
                size=cur_markersize, symbol=get_plotly_marker(cur_marker),
                line=dict(width=fmt.get('edgewidth', 1), color=fmt.get('edge_color', 'black')),
                opacity=cur_alpha
            )
            line_dict = dict(width=cur_linewidth, dash=get_plotly_linestyle(cur_linestyle))

            if cur_hue and cur_hue in df.columns:
                hue_data = df[cur_hue]
                if pd.api.types.is_numeric_dtype(hue_data):
                    info = numeric_hue_info.get(cur_hue, {})
                    marker_dict['color'] = hue_data
                    marker_dict['coloraxis'] = info.get('ca_name', 'coloraxis')
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
                showlegend=(idx_p == 0)
            ), row=row, col=col)

            if cur_reg_order:
                rx, ry, fit_label = _calculate_regression(df, x_col, yi, cur_reg_order)
                if rx is not None:
                    fig.add_trace(go.Scatter(
                        x=rx, y=ry, mode='lines',
                        name=f"{cur_idx}: {cur_title} Fit ({fit_label})",
                        legendgroup=f"group_{cur_idx}",
                        line=dict(color=cur_color, width=cur_linewidth, dash=get_plotly_linestyle(cur_linestyle)),
                        opacity=0.7, hoverinfo='skip', showlegend=False
                    ), row=row, col=col)

    coloraxis_updates = {}
    for hue_col, info in numeric_hue_info.items():
        idx = list(numeric_hue_info.keys()).index(hue_col)
        ca_def = dict(
            colorscale=info['palette'],
            colorbar=dict(title=hue_col, x=1.02 + idx * 0.12, thickness=15),
        )
        if info['lim']:
            ca_def['cmin'] = info['lim'][0]
            ca_def['cmax'] = info['lim'][1]
        coloraxis_updates[info['ca_name']] = ca_def
    if coloraxis_updates:
        fig.update_layout(**coloraxis_updates)

    for idx_p, (x_name, yi) in enumerate(pairs):
        row = idx_p // ncols + 1
        col = idx_p % ncols + 1

        axis_title = ylabel if ylabel else yi
        x_axis_title = xlabel if xlabel else x_name

        fig.update_yaxes(title_text=axis_title, title_standoff=15, row=row, col=col)
        fig.update_xaxes(title_text=x_axis_title, title_standoff=15, row=row, col=col)

    for idx_p, (x_name, yi) in enumerate(pairs):
        row = idx_p // ncols + 1
        col = idx_p % ncols + 1
        if y_lim: fig.update_yaxes(range=y_lim, row=row, col=col)
        if x_lim: fig.update_xaxes(range=x_lim, row=row, col=col)
    if not grid:
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
    if legend == 'above':
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))

    return _show_or_return(fig, return_axes)

def uniplot_per_dataset(list_of_datasets, x, y, display_parms=None,
                        suptitle=None, figsize=(12, 8), ncols=None, nrows=None,
                        darkmode=True, x_lim=None, y_lim=None,
                        axis_limits=None, return_axes=False):

    active_datasets = [d for d in list_of_datasets if d.select]
    if not active_datasets:
        return None

    y_list = y if isinstance(y, list) else [y]
    axis_limits = axis_limits or {}
    n_sets = len(active_datasets)
    nrows, ncols = _calc_grid(n_sets, nrows, ncols)

    use_secondary = len(y_list) >= 2
    if len(y_list) > 2:
        warnings.warn(
            f"{len(y_list)} y-variables requested; only the first 2 get dedicated "
            "axes per subplot. Variables 3+ will share the secondary axis."
        )

    specs = [[{'secondary_y': use_secondary} for _ in range(ncols)] for _ in range(nrows)]
    sp_titles = [ds.title_format for ds in active_datasets]
    sp_titles += [""] * (nrows * ncols - len(sp_titles))

    fig = make_subplots(
        rows=nrows, cols=ncols, specs=specs,
        subplot_titles=sp_titles,
        horizontal_spacing=0.12 if use_secondary else 0.08,
        vertical_spacing=0.15,
    )
    fig.update_layout(**_base_layout(
        darkmode, suptitle or "Dataset Comparison", figsize,
        showlegend=True, margin=dict(r=60),
    ))

    color_cycle = px.colors.qualitative.Plotly
    primary_y = y_list[0]
    secondary_ys = y_list[1:]

    for idx_ds, dataset in enumerate(active_datasets):
        row = (idx_ds // ncols) + 1
        col = (idx_ds % ncols) + 1

        base_df = dataset.df
        if x in base_df.columns:
            x_col = x
        else:
            cols_upper = {c.upper(): c for c in base_df.columns}
            x_col = cols_upper.get(str(x).upper())
            if not x_col:
                continue

        valid_hover = [p for p in (display_parms or []) if p in base_df.columns]
        req_cols = list(dict.fromkeys(
            [x_col] + [yi for yi in y_list if yi in base_df.columns]
            + valid_hover
            + ([dataset.order] if dataset.order and dataset.order != 'index'
               and dataset.order in base_df.columns else [])
        ))
        df = base_df.loc[:, ~base_df.columns.duplicated()][req_cols]
        if dataset.order == 'index':
            df = df.sort_index()
        elif dataset.order:
            df = df.sort_values(by=dataset.order)

        line_dict = dict(width=dataset.linewidth)
        if dataset.linestyle:
            line_dict['dash'] = get_plotly_linestyle(dataset.linestyle)

        if primary_y in df.columns:
            color0 = color_cycle[0]
            cd_cols = list(dict.fromkeys([x_col, primary_y] + valid_hover))
            ht = (f"<b>{dataset.title}</b><br>{x_col}: %{{customdata[0]:.2f}}"
                  f"<br>{primary_y}: %{{customdata[1]:.2f}}<extra></extra>")
            fig.add_trace(
                go.Scatter(
                    x=df[x_col], y=df[primary_y],
                    mode='lines+markers' if dataset.linestyle else 'markers',
                    name=primary_y, legendgroup=primary_y,
                    showlegend=(idx_ds == 0),
                    marker=dict(color=color0, size=dataset.markersize or 6),
                    line=dict(color=color0, **line_dict),
                    customdata=df[cd_cols], hovertemplate=ht,
                ),
                row=row, col=col,
                secondary_y=False if use_secondary else None,
            )
            kw = dict(title_text=primary_y,
                      title_font=dict(color=color0),
                      tickfont=dict(color=color0),
                      row=row, col=col)
            if primary_y in axis_limits:
                kw['range'] = axis_limits[primary_y]
            if use_secondary:
                fig.update_yaxes(secondary_y=False, **kw)
            else:
                fig.update_yaxes(**kw)

        for k, yi in enumerate(secondary_ys):
            if yi not in df.columns:
                continue
            color_k = color_cycle[(k + 1) % len(color_cycle)]
            cd_cols = list(dict.fromkeys([x_col, yi] + valid_hover))
            ht = (f"<b>{dataset.title}</b><br>{x_col}: %{{customdata[0]:.2f}}"
                  f"<br>{yi}: %{{customdata[1]:.2f}}<extra></extra>")
            fig.add_trace(
                go.Scatter(
                    x=df[x_col], y=df[yi],
                    mode='lines+markers' if dataset.linestyle else 'markers',
                    name=yi, legendgroup=yi,
                    showlegend=(idx_ds == 0),
                    marker=dict(
                        color=color_k,
                        size=dataset.markersize or 6,
                        symbol='circle' if k == 0 else 'diamond',
                    ),
                    line=dict(color=color_k, **line_dict),
                    customdata=df[cd_cols], hovertemplate=ht,
                ),
                row=row, col=col, secondary_y=True,
            )

        if use_secondary and secondary_ys:
            color1 = color_cycle[1]
            kw = dict(title_text=secondary_ys[0],
                      title_font=dict(color=color1),
                      tickfont=dict(color=color1),
                      showgrid=False, row=row, col=col)
            if secondary_ys[0] in axis_limits:
                kw['range'] = axis_limits[secondary_ys[0]]
            fig.update_yaxes(secondary_y=True, **kw)

        fig.update_xaxes(title_text=x, row=row, col=col)
        if x_lim:
            fig.update_xaxes(range=x_lim, row=row, col=col)

    return _show_or_return(fig, return_axes)

def unibar(list_of_datasets, x, y, markers=None, variable_formats=None,
           barmode='group', color=None, 
           suptitle=None, xlabel=None, ylabel=None, subplot_titles=None,
           darkmode=False, figsize=(12, 8), ncols=None, nrows=None, 
           y_lim=None, return_axes=False):
    y_list = y if isinstance(y, list) else [y]
    markers_list = markers if isinstance(markers, list) else ([markers] if markers else [])
    variable_formats = variable_formats or {}
    n_y = len(y_list)
    nrows, ncols = _calc_grid(n_y, nrows, ncols)

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subplot_titles or y_list)
    fig.update_layout(**_base_layout(
        darkmode, suptitle or f"Bar Comparison: {x}", figsize,
        barmode=barmode, showlegend=True,
        margin=dict(t=120),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    ))

    edge_default = 'white' if darkmode else 'black'

    # Track which (marker_col, y_subplot) legend entries we've already shown,
    # for marker columns that are explicitly styled via var_format.
    marker_legend_shown = set()

    for ds in list_of_datasets:
        if not ds.select: continue
        df = ds.df
        offset_group = f"set_{ds.index}"
        
        for idx_y, yi in enumerate(y_list):
            row, col = (idx_y // ncols) + 1, (idx_y % ncols) + 1
            if yi not in df.columns: continue

            fig.add_trace(go.Bar(
                x=df[x], y=df[yi],
                name=f"{ds.index}: {ds.title}",
                legendgroup=f"group_{ds.index}",
                offsetgroup=offset_group,
                alignmentgroup="bars",
                marker_color=ds.color if not color else color,
                opacity=ds.alpha,
                showlegend=(idx_y == 0)             # bars always get one legend entry per set
            ), row=row, col=col)

            for m_idx, m_col in enumerate(markers_list):
                if m_col not in df.columns: continue

                var_fmt = variable_formats.get(m_col, {})
                styled = bool(var_fmt)              # styled = has any var_format override

                m_symbol = get_plotly_marker(var_fmt.get('marker') or marker_map(m_idx + 1))
                m_color  = var_fmt.get('color') or (color if color else ds.color)
                m_size   = var_fmt.get('markersize', max(ds.markersize, 10))
                m_alpha  = var_fmt.get('alpha', ds.alpha)

                # Legend strategy:
                #   styled marker      -> ONE entry per (marker_col, subplot), named just the column
                #   unstyled marker    -> one entry per (dataset, marker_col), grouped with dataset
                if styled:
                    legend_key = (m_col, idx_y)
                    show = legend_key not in marker_legend_shown
                    if show:
                        marker_legend_shown.add(legend_key)
                    trace_name = m_col
                    legend_group = f"marker_{m_col}"
                else:
                    show = (idx_y == 0)
                    trace_name = f"{ds.index}: {ds.title} — {m_col}"
                    legend_group = f"group_{ds.index}"

                fig.add_trace(go.Scatter(
                    x=df[x], y=df[m_col],
                    mode='markers',
                    name=trace_name,
                    legendgroup=legend_group,
                    offsetgroup=offset_group,
                    alignmentgroup="bars",
                    marker=dict(
                        symbol=m_symbol,
                        size=m_size,
                        color=m_color,
                        opacity=m_alpha,
                        line=dict(width=1.5, color=edge_default),
                    ),
                    showlegend=show,
                    hovertemplate=(f"<b>{ds.title}</b><br>{x}: %{{x}}<br>"
                                   f"{m_col}: %{{y:.4g}}<extra></extra>")
                ), row=row, col=col)

    fig.update_xaxes(title_text=xlabel or x)
    fig.update_yaxes(title_text=ylabel or "Value")
    if y_lim: fig.update_yaxes(range=y_lim)

    return _show_or_return(fig, return_axes)
def unibar_per_dataset(list_of_datasets, x, y, markers=None, variable_formats=None,
                       barmode='group',
                       suptitle=None, figsize=(12, 8), ncols=None, nrows=None, 
                       darkmode=False, y_lim=None, return_axes=False):
    """
    Grouped Bar Chart. Subplots are organized by Dataset.

    variable_formats applies to BOTH bar variables and marker columns in
    this view, since color encodes variable (not dataset) within each subplot.
    """
    active_ds = [d for d in list_of_datasets if d.select]
    y_list = y if isinstance(y, list) else [y]
    markers_list = markers if isinstance(markers, list) else ([markers] if markers else [])
    variable_formats = variable_formats or {}
    n_sets = len(active_ds)
    nrows, ncols = _calc_grid(n_sets, nrows, ncols)

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[d.title_format for d in active_ds])
    color_cycle = px.colors.qualitative.Plotly
    fig.update_layout(**_base_layout(
        darkmode, suptitle or "Dataset Bar Comparison", figsize,
        barmode=barmode,
        margin=dict(t=120),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    ))

    edge_default = 'white' if darkmode else 'black'

    for idx_ds, ds in enumerate(active_ds):
        row, col = (idx_ds // ncols) + 1, (idx_ds % ncols) + 1
        df = ds.df
        
        for idx_y, yi in enumerate(y_list):
            if yi not in df.columns: continue
            offset_group = f"var_{yi}"

            var_fmt = variable_formats.get(yi, {})
            bar_color = var_fmt.get('color') or color_cycle[idx_y % len(color_cycle)]
            bar_alpha = var_fmt.get('alpha', 1.0)

            fig.add_trace(go.Bar(
                x=df[x], y=df[yi],
                name=yi,
                legendgroup=yi,
                offsetgroup=offset_group,
                alignmentgroup="bars",
                marker_color=bar_color,
                opacity=bar_alpha,
                showlegend=(idx_ds == 0)
            ), row=row, col=col)

        for m_idx, m_col in enumerate(markers_list):
            if m_col not in df.columns: continue

            var_fmt = variable_formats.get(m_col, {})
            m_symbol = get_plotly_marker(var_fmt.get('marker') or marker_map(m_idx))
            m_color  = var_fmt.get('color') or color_cycle[(len(y_list) + m_idx) % len(color_cycle)]
            m_size   = var_fmt.get('markersize', 12)
            m_alpha  = var_fmt.get('alpha', 1.0)

            fig.add_trace(go.Scatter(
                x=df[x], y=df[m_col],
                mode='markers',
                name=m_col,
                legendgroup=f"marker_{m_col}",
                marker=dict(
                    symbol=m_symbol,
                    size=m_size,
                    color=m_color,
                    opacity=m_alpha,
                    line=dict(width=1.5, color=edge_default),
                ),
                showlegend=(idx_ds == 0),
                hovertemplate=f"<b>{m_col}</b><br>{x}: %{{x}}<br>%{{y:.4g}}<extra></extra>"
            ), row=row, col=col)

    fig.update_xaxes(title_text=x)
    if y_lim: fig.update_yaxes(range=y_lim)

    return _show_or_return(fig, return_axes)

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
    nrows, ncols = _calc_grid(n_y, nrows, ncols)

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subplot_titles or y_list)
    fig.update_layout(**_base_layout(
        darkmode, suptitle or f"Boxplot Comparison: {x}", figsize,
        boxmode=boxmode, showlegend=True
    ))

    for ds in list_of_datasets:
        if not ds.select: continue
        df = ds.df
        
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

    return _show_or_return(fig, return_axes)

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
    nrows, ncols = _calc_grid(n_sets, nrows, ncols)

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[d.title_format for d in active_ds])
    color_cycle = px.colors.qualitative.Plotly
    fig.update_layout(**_base_layout(
        darkmode, suptitle or "Dataset Box Comparison", figsize,
        boxmode=boxmode
    ))

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

    return _show_or_return(fig, return_axes)


def unihistogram(list_of_datasets, x, y=None, histfunc='sum', nbins=None,
                 bin_size=None, bin_start=None, bin_end=None,
                 histnorm='', barmode='overlay', alpha=0.7,
                 color=None, suptitle=None, subplot_titles=None, darkmode=False,
                 figsize=(12, 8), ncols=None, nrows=None, x_lim=None, return_axes=False,
                 opacity=None):
    """
    Create a unified histogram for a list of datasets.
    Subplots are organized by Variable (x).
    """
    if opacity is not None:
        warnings.warn("'opacity' is deprecated, use 'alpha'", DeprecationWarning, stacklevel=2)
        alpha = opacity
    x_list = x if isinstance(x, list) else [x]
    n_x = len(x_list)
    nrows, ncols = _calc_grid(n_x, nrows, ncols)

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subplot_titles or x_list)
    fig.update_layout(**_base_layout(
        darkmode, suptitle or "Distribution Comparison", figsize,
        barmode=barmode, showlegend=True,
        margin=dict(t=120),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    ))

    xbins = _build_xbins(bin_size, bin_start, bin_end)

    for ds in list_of_datasets:
        if not ds.select: continue
        df = ds.df
        
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
                opacity=alpha,
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

    return _show_or_return(fig, return_axes)

def unihistogram_by_dataset(list_of_datasets, x, y=None, histfunc='sum', nbins=None,
                            bin_size=None, bin_start=None, bin_end=None,
                            histnorm='', barmode='overlay', alpha=0.7,
                            color=None, suptitle=None, figsize=(12, 8), ncols=None, nrows=None,
                            darkmode=False, x_lim=None, return_axes=False,
                            opacity=None):
    """
    Create a unified histogram where Subplots are organized by Dataset.
    """
    if opacity is not None:
        warnings.warn("'opacity' is deprecated, use 'alpha'", DeprecationWarning, stacklevel=2)
        alpha = opacity
    active_ds = [d for d in list_of_datasets if d.select]
    x_list = x if isinstance(x, list) else [x]
    n_sets = len(active_ds)

    if not active_ds:
        print("No datasets selected.")
        return None

    nrows, ncols = _calc_grid(n_sets, nrows, ncols)

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[d.title_format for d in active_ds])
    color_cycle = px.colors.qualitative.Plotly
    fig.update_layout(**_base_layout(
        darkmode, suptitle or "Dataset Distribution Analysis", figsize,
        barmode=barmode, showlegend=True,
        margin=dict(t=120),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    ))

    xbins = _build_xbins(bin_size, bin_start, bin_end)

    for idx_ds, ds in enumerate(active_ds):
        row, col = (idx_ds // ncols) + 1, (idx_ds % ncols) + 1
        df = ds.df
        
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
                opacity=alpha,
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

    return _show_or_return(fig, return_axes)

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
    axis_limits = axis_limits or {}

    if not active_ds:
        print("No datasets selected.")
        return None

    nrows, ncols = _calc_grid(n_z, nrows, ncols)

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subplot_titles or z_list, horizontal_spacing=0.15)
    fig.update_layout(**_base_layout(
        darkmode, None, figsize,
        title={'text': suptitle or f"Contour: {y} vs {x}", 'x': 0.5, 'xanchor': 'center', 'y': 0.98, 'yanchor': 'top', 'yref': 'container'},
        showlegend=True, margin=dict(r=100, t=120),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    ))

    for idx_ds, ds in enumerate(active_ds):
        df = ds.df
        if x not in df.columns or y not in df.columns: 
            continue

        for idx_z, zi in enumerate(z_list):
            if zi not in df.columns: continue
            row, col = (idx_z // ncols) + 1, (idx_z % ncols) + 1

            use_coloring = 'lines' if len(active_ds) > 1 and contours_coloring == 'fill' else contours_coloring

            clean_df = df.dropna(subset=[x, y, zi])
            if clean_df.empty: continue

            if interpolate:
                plot_x, plot_y, plot_z = _generate_contour_grid(
                    clean_df[x], clean_df[y], clean_df[zi], 
                    res=interp_res, method=interp_method
                )
                if plot_x is None:
                    print(f"Skipping contour trace for set {ds.index} (Z={zi}): Insufficient points or collinear data.")
                    continue
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
                showlegend=(idx_z == 0),
                colorbar=dict(
                    title=zi,
                    x=cb_x,
                    y=cb_y,
                    len=cb_len,
                    thickness=15
                ),
                hovertemplate=f"<b>{ds.title}</b><br>{x}: %{{x:.3g}}<br>{y}: %{{y:.3g}}<br>{zi}: %{{z:.3g}}<extra></extra>"
            ), row=row, col=col)

    fig.update_xaxes(title_text=xlabel or x)
    fig.update_yaxes(title_text=ylabel or y)

    return _show_or_return(fig, return_axes)

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

    nrows, ncols = _calc_grid(n_sets, nrows, ncols)

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[d.title_format for d in active_ds], horizontal_spacing=0.15)
    fig.update_layout(**_base_layout(
        darkmode, None, figsize,
        title={'text': suptitle or "Dataset Contour Comparison", 'x': 0.5, 'y': 0.98, 'yref': 'container'},
        showlegend=True, margin=dict(r=100, t=120),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    ))

    for idx_ds, ds in enumerate(active_ds):
        row, col = (idx_ds // ncols) + 1, (idx_ds % ncols) + 1
        df = ds.df
        
        if x not in df.columns or y not in df.columns: continue

        for idx_z, zi in enumerate(z_list):
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
                showlegend=(idx_ds == 0),
                colorbar=dict(
                    title=zi,
                    x=cb_x,
                    y=cb_y,
                    len=cb_len,
                    thickness=15
                ),
                hovertemplate=f"<b>{zi}</b><br>{x}: %{{x:.3g}}<br>{y}: %{{y:.3g}}<br>Value: %{{z:.3g}}<extra></extra>"
            ), row=row, col=col)

    fig.update_xaxes(title_text=x)
    fig.update_yaxes(title_text=y)

    return _show_or_return(fig, return_axes)

def unibar_datasets_as_x(list_of_datasets, y, agg='mean', suptitle=None, darkmode=False,
                         figsize=(12, 8), axis_limits=None, return_axes=False):
    """
    Creates a single grouped bar chart where the X-axis is the Dataset name,
    and the bars are the different Y-variables, each scaled to their own Y-axis.
    Includes an 'agg' parameter to handle multi-row datasets.
    """
    active_ds = [d for d in list_of_datasets if d.select]
    if not active_ds:
        print("No datasets selected.")
        return None

    y_list = y if isinstance(y, list) else [y]
    axis_limits = axis_limits or {}
    color_cycle = px.colors.qualitative.Plotly

    fig = go.Figure()

    x_labels = [f"{ds.index}: {ds.title}" for ds in active_ds]

    extras_count = max(0, len(y_list) - 2)
    width_per_axis = 0.08
    required_space = extras_count * width_per_axis
    x_domain_end = max(0.5, 1.0 - required_space) 

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

    fig.update_layout(**_base_layout(
        darkmode, suptitle or f"Variables by Dataset ({agg})", figsize,
        barmode='group',
        xaxis=dict(domain=[0, x_domain_end], title="Dataset"),
        margin=dict(r=50 + (extras_count * 80), t=120),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    ))

    return _show_or_return(fig, return_axes)

def unibox_datasets_as_x(list_of_datasets, y, boxmode='group', points='outliers', notched=False,
                         suptitle=None, darkmode=False, figsize=(12, 8), axis_limits=None, return_axes=False):
    """
    Creates a single grouped box plot where the X-axis is the Dataset name,
    and the boxes are the different Y-variables, each scaled to their own Y-axis.
    """
    active_ds = [d for d in list_of_datasets if d.select]
    if not active_ds:
        print("No datasets selected.")
        return None

    y_list = y if isinstance(y, list) else [y]
    axis_limits = axis_limits or {}
    color_cycle = px.colors.qualitative.Plotly

    fig = go.Figure()

    extras_count = max(0, len(y_list) - 2)
    width_per_axis = 0.08
    required_space = extras_count * width_per_axis
    x_domain_end = max(0.5, 1.0 - required_space) 

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

    fig.update_layout(**_base_layout(
        darkmode, suptitle or "Variables by Dataset", figsize,
        boxmode=boxmode,
        xaxis=dict(domain=[0, x_domain_end], title="Dataset"),
        margin=dict(r=50 + (extras_count * 80))
    ))

    return _show_or_return(fig, return_axes)


# -----------------------------------------------------------------------------
# Multi-Y-Axis Scatter/Line Plot
# -----------------------------------------------------------------------------
def uniplot_ymultaxis(list_of_datasets, x, y,
                        variable_formats=None, display_parms=None,
                        suptitle=None, xlabel=None,
                        darkmode=False, figsize=(12, 8),
                        x_lim=None, axis_limits=None,
                        legend='right', legend_group_by='sets', return_axes=False):
    """
    Single-plot, multi-Y-axis scatter/line chart.

    All selected datasets are overlaid on the same x-axis. Each y variable
    in `y` gets its own y-axis (left for the first, right for the second,
    further right for the third+). One trace per (dataset × variable).

    Formatting precedence (per attribute):
        variable_formats[var][attr]   →   if set, use this
        dataset.<attr>                →   otherwise, fall back to dataset

    This is intentionally per-attribute, so you can do things like
    "set linestyle on the variable, but let color come from the dataset",
    or vice versa.

    Parameters
    ----------
    list_of_datasets : list[Dataset]
    x : str
        X-column name (shared across all traces).
    y : str | list[str]
        One or more y-column names. Each gets its own y-axis.
    variable_formats : dict[str, dict] | None
        Per-variable overrides, e.g.
            {'Temp': {'linestyle': '--'}, 'Pressure': {'color': 'blue'}}
        Recognized keys: color, marker, linestyle, markersize, linewidth, alpha.
    display_parms : list[str] | None
        Extra columns to surface in the hover tooltip.
    axis_limits : dict[str, tuple] | None
        Per-column (min, max) limits. Applies to x and to any y axis.
    legend : 'right' | 'above' | 'off'
    """
    variable_formats = variable_formats or {}
    axis_limits = axis_limits or {}

    active = [d for d in list_of_datasets if d.select]
    if not active:
        print("No datasets selected.")
        return None

    y_list = y if isinstance(y, list) else [y]
    if not y_list:
        raise ValueError("At least one y variable is required.")

    # Domain math: extra y-axes (3+) live to the right of the plot.
    extras = max(0, len(y_list) - 2)
    width_per_axis = 0.06
    x_domain_end = max(0.5, 1.0 - extras * width_per_axis)

    # Axis label colors: only apply when explicitly set in variable_formats.
    axis_label_colors = {yi: variable_formats.get(yi, {}).get('color') for yi in y_list}

    fig = go.Figure()

    for ds in active:
        base_df = ds.df
        if x not in base_df.columns:
            continue

        ds_hover = display_parms if display_parms is not None else getattr(ds, 'display_parms', [])
        valid_hover = [p for p in (ds_hover or []) if p in base_df.columns]

        for idx_y, yi in enumerate(y_list):
            if yi not in base_df.columns:
                continue

            fmt = _resolve_var_format(ds, yi, variable_formats)

            req_cols = list(dict.fromkeys([x, yi] + valid_hover))
            df = base_df.loc[:, ~base_df.columns.duplicated()][req_cols]

            order_col = getattr(ds, 'order', None)
            if order_col == 'index':
                df = df.sort_index()
            elif order_col and order_col in df.columns:
                df = df.sort_values(by=order_col)
            else:
                df = df.sort_index()

            parts = []
            if fmt['linestyle']:
                parts.append('lines')
            if fmt['marker'] or not fmt['linestyle']:
                parts.append('markers')
            mode = '+'.join(parts) if parts else 'markers'

            ht = (f"<b>Set {ds.index}: {ds.title}</b><br>"
                  f"<b>{yi}</b><br>"
                  f"{x}: %{{x:.4g}}<br>"
                  f"{yi}: %{{y:.4g}}")
            customdata = None
            if valid_hover:
                customdata = df[valid_hover]
                for i, p in enumerate(valid_hover):
                    if pd.api.types.is_numeric_dtype(df[p]):
                        ht += f"<br>{p}: %{{customdata[{i}]:.4g}}"
                    else:
                        ht += f"<br>{p}: %{{customdata[{i}]}}"
            ht += "<extra></extra>"

            y_axis_name = "y" if idx_y == 0 else f"y{idx_y + 1}"

            fig.add_trace(go.Scatter(
                x=df[x], y=df[yi],
                mode=mode,
                name=f"{ds.index}: {ds.title}" if legend_group_by == 'vars' else yi,
                yaxis=y_axis_name,
                legendgroup=f"var_{yi}" if legend_group_by == 'vars' else f"set_{ds.index}",
                legendgrouptitle_text=yi if legend_group_by == 'vars' else f"{ds.index}: {ds.title}",
                marker=dict(
                    size=fmt['markersize'],
                    symbol=get_plotly_marker(fmt['marker']),
                    color=fmt['color'],
                    opacity=fmt['alpha'],
                    line=dict(width=fmt['edgewidth'], color=fmt['edge_color']),
                ),
                line=dict(
                    width=fmt['linewidth'],
                    dash=get_plotly_linestyle(fmt['linestyle']) or 'solid',
                    color=fmt['color'],
                ),
                customdata=customdata,
                hovertemplate=ht,
            ))

    # Color axes only when explicitly set via variable_formats; otherwise use Plotly's default
    for idx_y, yi in enumerate(y_list):
        ax_color = axis_label_colors[yi]
        title_kw = dict(text=yi)
        if ax_color:
            title_kw['font'] = dict(color=ax_color)
        ax = dict(title=title_kw, showgrid=(idx_y == 0))
        if ax_color:
            ax['tickfont'] = dict(color=ax_color)
        if yi in axis_limits:
            ax['range'] = axis_limits[yi]

        if idx_y == 0:
            fig.update_layout(yaxis=ax)
        elif idx_y == 1:
            ax.update(dict(overlaying='y', side='right', anchor='x'))
            fig.update_layout(yaxis2=ax)
        else:
            pos = x_domain_end + (idx_y - 1) * width_per_axis
            ax.update(dict(overlaying='y', side='right', anchor='free', position=pos))
            fig.update_layout({f"yaxis{idx_y + 1}": ax})

    layout_extras = dict(
        showlegend=(legend != 'off'),
        xaxis=dict(domain=[0, x_domain_end], title=xlabel or x),
        margin=dict(r=60 + extras * 70, t=120 if legend == 'above' else 100),
    )
    if x_lim:
        layout_extras['xaxis']['range'] = x_lim
    elif x in axis_limits:
        layout_extras['xaxis']['range'] = axis_limits[x]

    if legend == 'above':
        layout_extras['legend'] = dict(orientation='h', yanchor='bottom',
                                       y=1.01, xanchor='center', x=0.5)

    fig.update_layout(**_base_layout(
        darkmode,
        suptitle or f"{', '.join(y_list)} vs {x}",
        figsize,
        **layout_extras,
    ))

    return _show_or_return(fig, return_axes)

class UnichartNotebook:
    def __init__(self):
        self.sets = []
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

        # Per-variable formatting overrides (used by plot_ymult / uniplot_ymultaxis).
        # Shape: {variable_name: {attr: value}} where attr ∈ _VAR_FORMAT_KEYS.
        self.variable_formats = {}

        self.suptitle_size = None
        self.legend_size = None
        self.axes_title_size = None
        self.axes_tick_size = None
        self.subplot_title_size = None
        self.colorbar_size = None
        self.hover_size = None
        self.table_header_size = None
        self.table_cell_size = None

        print("UniChart Notebook Environment Initialized.")

    # ------------------------------------------------------------------
    # Backwards compatibility
    # ------------------------------------------------------------------
    @property
    def uset(self):
        return self.sets

    @uset.setter
    def uset(self, value):
        self.sets = value

    # ------------------------------------------------------------------
    # Data Management
    # ------------------------------------------------------------------
    def load_df(self, df, title=None, set_name_column=None, set_idx_column=None, load_cols_as_vars=False):
        """Split a DataFrame into one Dataset per unique set_idx_column value, or load it as one."""
        if not title:
            if set_name_column and set_name_column in df.columns:
                pass
            elif "TITLE" in df.columns:
                set_name_column = "TITLE"
            else:
                df["TITLE"] = "Dataset"
                set_name_column = "TITLE"

            if set_idx_column and set_idx_column in df.columns:
                pass
            elif "SETNUMBER" in df.columns:
                set_idx_column = "SETNUMBER"
            elif "INDEX" in df.columns:
                set_idx_column = "INDEX"
            else:
                df["SETNUMBER"] = df.index

        next_index = len(self.sets)
        
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
                self.sets.append(ds)
                print(f"Loaded Set {next_index}: {ds.title}")
                next_index += 1
        else:
            ds = Dataset(df.copy(), index=next_index, title=title if title else "Untitled")
            self.sets.append(ds)
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

    def clear_data(self):
        self.sets = []
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
        """Normalize a selector into a list of Dataset objects.

        Accepts:
            None | 'all'    -> all datasets
            int             -> dataset at that index
            str             -> dataset(s) whose title matches exactly
            Dataset         -> wrapped in a list
            list            -> mixed list of any of the above
        Unknown inputs print a warning and return [].
        """
        if uset_slice is None or uset_slice == 'all':
            return list(self.sets)

        if isinstance(uset_slice, Dataset):
            return [uset_slice]

        if isinstance(uset_slice, int) and not isinstance(uset_slice, bool):
            if 0 <= uset_slice < len(self.sets):
                return [self.sets[uset_slice]]
            return []

        if isinstance(uset_slice, str):
            matches = [d for d in self.sets if d.title == uset_slice]
            if not matches:
                print(f"Warning: no dataset with title {uset_slice!r}.")
            return matches

        if isinstance(uset_slice, (list, tuple, set)):
            result = []
            seen_ids = set()
            for item in uset_slice:
                for d in self._get_uset_slice(item):
                    if id(d) not in seen_ids:
                        result.append(d)
                        seen_ids.add(id(d))
            return result

        print(f"Warning: don't know how to interpret {uset_slice!r} as a dataset selector.")
        return []

    def select(self, uset_slice=None):
        """Select the specified dataset(s)."""
        for ds in self.sets: ds.select = False
        for ds in self._get_uset_slice(uset_slice):
            ds.select = True
        self._refresh_widgets()

    def selected(self):
        """Get the currently selected datasets."""
        return [ds for ds in self.sets if ds.select]

    def omit(self, uset_slice=None):
        for ds in self._get_uset_slice(uset_slice):
            ds.select = False
        self._refresh_widgets()

    def restore(self, uset_slice=None):
        targets = self.sets if uset_slice == "all" else self._get_uset_slice(uset_slice)
        for ds in targets:
            ds.select = True
        self._refresh_widgets()

    def query(self, uset_slice=None, query_str=None):
        targets = list(self._get_uset_slice(uset_slice))
        if not targets:
            self._refresh_widgets()
            return

        if not query_str:
            for ds in targets:
                ds._query = query_str
                ds._df_filtered = ds._df_full
            self._refresh_widgets()
            return

        def _run(ds):
            try:
                return ds, ds._df_full.query(query_str), None
            except Exception as e:
                return ds, None, e

        max_workers = min(len(targets), 8)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(_run, targets))

        for ds, result_df, err in results:
            if err is not None:
                raise ValueError(f"Query error: {err}")
            ds._query = query_str
            if not result_df.empty:
                ds._df_filtered = result_df
            else:
                print(f"No data in set {ds.index} after query: {query_str}. Turning Set Off...")
                ds.select = False
                ds._df_filtered = ds._df_full

        self._refresh_widgets()

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------
    def color(self, uset_slice, color_val):
        """
        Set the primary color for the specified dataset(s).
        
        Args:
            uset_slice (int, list, or 'all'): The dataset index or indices to modify.
            color_val (str): A standard color name ('red'), hex code ('#FF5733'), 
                             or RGB/RGBA string ('rgb(255, 0, 0)').
        """
        for ds in self._get_uset_slice(uset_slice):
            ds.color = color_val

    def marker(self, uset_slice, marker_val):
        """
        Set the marker style for the specified dataset(s).
        
        Args:
            uset_slice (int, list, or 'all'): The dataset index or indices to modify.
            marker_val (str): Matplotlib-style marker ('o', 's', '^', 'D', '.') 
                              or Plotly-style marker ('circle', 'square').
        """
        for ds in self._get_uset_slice(uset_slice):
            ds.marker = marker_val

    def linestyle(self, uset_slice, style_val):
        """
        Set the line style for the specified dataset(s).
        
        Args:
            uset_slice (int, list, or 'all'): The dataset index or indices to modify.
            style_val (str): Matplotlib-style string ('-', '--', '-.', ':') 
                             or Plotly string ('solid', 'dash', 'dashdot', 'dot').
        """
        for ds in self._get_uset_slice(uset_slice):
            ds.linestyle = style_val
            
    def markersize(self, uset_slice, size_val):
        """
        Set the marker size for the specified dataset(s).
        """
        for ds in self._get_uset_slice(uset_slice):
            ds.markersize = size_val
    
    def linewidth(self, uset_slice, width_val):
        """
        Set the line thickness for the specified dataset(s).
        """
        for ds in self._get_uset_slice(uset_slice):
            ds.linewidth = width_val

    def hue(self, uset_slice, col_name):
        """
        Map a dataframe column to the color scale for the specified dataset(s).
        """
        for ds in self._get_uset_slice(uset_slice):
            ds.hue = col_name

    def hue_palette(self, uset_slice, hue_palette):
        """
        Set the color scale/palette used when `hue` is mapped to a variable.
        """
        for ds in self._get_uset_slice(uset_slice):
            ds.hue_palette = hue_palette

    def alpha(self, uset_slice, alpha_val):
        """
        Set the opacity (alpha) for the specified dataset(s).
        """
        for ds in self._get_uset_slice(uset_slice):
            ds.alpha = alpha_val

    def plot_type(self, uset_slice, type_val):
        for ds in self._get_uset_slice(uset_slice):
            ds.plot_type = type_val

    def reg_order(self, uset_slice, order):
        """
        Set a regression/trendline for the specified dataset(s).
        """
        for ds in self._get_uset_slice(uset_slice):
            ds.reg_order = order

    # ------------------------------------------------------------------
    # Variable-level formatting overrides
    # ------------------------------------------------------------------
    def var_format(self, variable, color=None, marker=None, linestyle=None,
                   markersize=None, linewidth=None, alpha=None):
        """
        Set persistent per-variable formatting overrides.

        Variable formatting takes precedence over Dataset formatting on a
        per-attribute basis — anything you don't set still falls back to
        the Dataset's value at plot time.

        Pass the string 'reset' as the value to clear a single attribute.

        Examples
        --------
        nb.var_format('Temperature', linestyle='--')         # all Temp lines dashed
        nb.var_format('Pressure', color='blue', marker='s')  # Pressure forced blue squares
        nb.var_format('Pressure', color='reset')             # remove just the color override
        """
        fmt = self.variable_formats.setdefault(variable, {})
        pairs = {'color': color, 'marker': marker, 'linestyle': linestyle,
                 'markersize': markersize, 'linewidth': linewidth, 'alpha': alpha}
        for k, v in pairs.items():
            if v is None:
                continue
            if v == 'reset':
                fmt.pop(k, None)
            else:
                fmt[k] = v
        if not fmt:
            del self.variable_formats[variable]
        return self.variable_formats.get(variable, {})

    def clear_var_format(self, variable=None):
        """Clear variable formatting. Pass None (or no arg) to clear everything."""
        if variable is None:
            self.variable_formats.clear()
        else:
            self.variable_formats.pop(variable, None)

    def list_var_formats(self):
        """Pretty-print current variable-level formatting."""
        if not self.variable_formats:
            print("No variable-level formatting set.")
            return
        print("Variable-level formatting (overrides dataset attributes):")
        for var, fmt in self.variable_formats.items():
            items = ", ".join(f"{k}={v!r}" for k, v in fmt.items())
            print(f"  {var}: {items}")

    def reset_format(self, uset_slice=None, sets=True, vars=True,
                     lines=True, highlights=True, scale=True, fonts=True):
        """
        Reset formatting state back to defaults.

        Parameters
        ----------
        uset_slice : int | list | 'all' | None
            Which datasets to reset. None/'all' resets every dataset.
            Ignored when `sets=False`.
        sets : bool
            Reset per-dataset visual attributes (color, marker, linestyle,
            markersize, linewidth, edgewidth, alpha, hue, hue_palette,
            reg_order) back to their Dataset.__init__ defaults.
        vars : bool
            Clear all variable-level formatting overrides (variable_formats).
        lines : bool
            Clear all stored reference lines.
        highlights : bool
            Clear all stored highlight regions.
        scale : bool
            Clear all stored axis limits (axis_limits).
        fonts : bool
            Reset all font-size overrides to None (use Plotly defaults).

        Examples
        --------
        nb.reset_format()                  # reset everything
        nb.reset_format(sets=False)        # keep per-dataset formatting, clear rest
        nb.reset_format(uset_slice=[0,1])  # only reset datasets 0 and 1
        nb.reset_format(lines=True, highlights=True, sets=False, vars=False,
                        scale=False, fonts=False)  # only clear decorations
        """
        default_colors = px.colors.qualitative.Plotly

        if sets:
            targets = (self.sets if uset_slice is None
                       else self._get_uset_slice(uset_slice))
            for ds in targets:
                ds._color     = default_colors[ds.index % len(default_colors)]
                ds._marker    = marker_map(ds.index)
                ds._linestyle = None
                ds.markersize = 10
                ds.linewidth  = 2
                ds.edgewidth  = 1
                ds.alpha      = 1
                ds.hue        = ""
                ds.hue_palette = "Jet"
                ds.hue_order  = None
                ds.reg_order  = None

        if vars:
            self.variable_formats.clear()

        if lines:
            self.lines.clear()

        if highlights:
            self.highlights.clear()

        if scale:
            self.axis_limits.clear()

        if fonts:
            for attr in ('suptitle_size', 'legend_size', 'axes_title_size',
                         'axes_tick_size', 'subplot_title_size',
                         'colorbar_size', 'hover_size'):
                setattr(self, attr, None)

        parts = []
        if sets:      parts.append("dataset formatting")
        if vars:      parts.append("variable formats")
        if lines:     parts.append("lines")
        if highlights: parts.append("highlights")
        if scale:     parts.append("axis limits")
        if fonts:     parts.append("font sizes")
        print(f"Reset: {', '.join(parts) if parts else 'nothing'}.")

    def reg_info(self, uset_slice=None):
        """Print the regression type, equation, and fit stats (R², RMSE, MAE) for each dataset."""
        KIND_LABELS = {
            'poly':   lambda p: "Linear (degree 1)" if p == 1 else f"Polynomial (degree {p})",
            'log':    lambda p: "Logarithmic",
            'exp':    lambda p: "Exponential",
            'power':  lambda p: "Power law",
            'lowess': lambda p: f"LOWESS (frac={p})",
            'spline': lambda p: "Cubic spline",
            'ma':     lambda p: f"Moving average (window={p})" if p else "Moving average",
        }

        def _fit_regression(kind, param, ds, x_col, y_col, label):
            formula = label if kind not in ('poly', 'log', 'exp', 'power') else None
            if kind is None or x_col is None or y_col is None:
                return formula, None
            df = ds.df
            if x_col not in df.columns or y_col not in df.columns:
                return formula, None
            try:
                df_c = df.dropna(subset=[x_col, y_col]).sort_values(by=x_col)
                x = df_c[x_col].to_numpy(dtype=float)
                y = df_c[y_col].to_numpy(dtype=float)
                if len(x) < 2:
                    return formula, None

                y_pred = None

                def _term(c, power, first):
                    exp_str = {0: '', 1: 'x', 2: 'x²', 3: 'x³', 4: 'x⁴', 5: 'x⁵'}.get(power, f'x^{power}')
                    if first:
                        return f"{c:.4g}{exp_str}"
                    return (f"+ {c:.4g}" if c >= 0 else f"- {abs(c):.4g}") + exp_str

                if kind == 'poly':
                    order = int(param) if param else 1
                    if len(x) < order + 1:
                        return formula, None
                    p = np.poly1d(np.polyfit(x, y, order))
                    y_pred = p(x)
                    parts = [_term(c, order - i, i == 0) for i, c in enumerate(p.coeffs)]
                    formula = "y = " + " ".join(parts)

                elif kind == 'log':
                    mask = x > 0
                    if mask.sum() < 2:
                        return formula, None
                    a, b = np.polyfit(np.log(x[mask]), y[mask], 1)
                    y_pred = np.where(x > 0, a * np.log(np.where(x > 0, x, 1)) + b, np.nan)
                    b_part = f"+ {b:.4g}" if b >= 0 else f"- {abs(b):.4g}"
                    formula = f"y = {a:.4g}·ln(x) {b_part}"

                elif kind == 'exp':
                    mask = y > 0
                    if mask.sum() < 2:
                        return formula, None
                    b, log_a = np.polyfit(x[mask], np.log(y[mask]), 1)
                    A = np.exp(log_a)
                    y_pred = A * np.exp(b * x)
                    formula = f"y = {A:.4g}·e^({b:.4g}x)"

                elif kind == 'power':
                    mask = (x > 0) & (y > 0)
                    if mask.sum() < 2:
                        return formula, None
                    b, log_a = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
                    A = np.exp(log_a)
                    y_pred = np.where(x > 0, A * np.power(np.where(x > 0, x, 1), b), np.nan)
                    formula = f"y = {A:.4g}·x^{b:.4g}"

                elif kind == 'lowess':
                    try:
                        from statsmodels.nonparametric.smoothers_lowess import lowess
                        frac = float(param) if param is not None else 0.3
                        y_pred = lowess(y, x, frac=frac, return_sorted=True)[:, 1]
                    except ImportError:
                        return formula, None

                elif kind == 'spline':
                    from scipy.interpolate import UnivariateSpline
                    k = max(1, min(5, int(param) if param else 3))
                    ux, uidx = np.unique(x, return_index=True)
                    if len(ux) < k + 1:
                        return formula, None
                    y_pred = UnivariateSpline(ux, y[uidx], k=k)(x)

                elif kind == 'ma':
                    window = int(param) if param else max(3, len(x) // 20)
                    window = max(2, min(window, len(x)))
                    y_pred = pd.Series(y).rolling(window=window, center=True, min_periods=1).mean().to_numpy()

                if y_pred is not None:
                    valid = ~(np.isnan(y) | np.isnan(y_pred))
                    y_v, yp_v = y[valid], y_pred[valid]
                    if len(y_v) >= 2:
                        ss_res = np.sum((y_v - yp_v) ** 2)
                        ss_tot = np.sum((y_v - np.mean(y_v)) ** 2)
                        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None
                        return formula, {
                            'n':    int(valid.sum()),
                            'r2':   r2,
                            'rmse': float(np.sqrt(np.mean((y_v - yp_v) ** 2))),
                            'mae':  float(np.mean(np.abs(y_v - yp_v))),
                        }

            except Exception:
                pass

            return formula, None

        lx, ly = self.last_x, self.last_y
        x_col = lx[0] if isinstance(lx, list) else lx
        y_col = ly[0] if isinstance(ly, list) else ly

        result = {}
        for ds in (self.sets if uset_slice is None or uset_slice == 'all'
                   else self._get_uset_slice(uset_slice)):
            raw = ds.reg_order
            kind, param = _parse_reg_spec(raw)
            label = KIND_LABELS.get(kind, lambda p: str(kind))(param) if kind is not None else None
            formula, stats = _fit_regression(kind, param, ds, x_col, y_col, label)
            result[ds.index] = {'raw': raw, 'kind': kind, 'param': param, 'label': label, 'formula': formula, 'stats': stats}

        any_reg = any(v['kind'] is not None for v in result.values())
        for idx, info in result.items():
            ds = self.sets[idx]
            reg_str = info['label'] or "None"
            formula_str = f"  →  {info['formula']}" if info['formula'] else ""
            print(f"Set {idx} ({ds.title}): {reg_str}{formula_str}")
            if info['stats']:
                s = info['stats']
                r2_str = f"R²={s['r2']:.4f}" if s['r2'] is not None else "R²=N/A"
                print(f"   {r2_str}   RMSE={s['rmse']:.4g}   MAE={s['mae']:.4g}   n={s['n']}")

        if not any_reg:
            print("No regression functions are currently set.")

        return result

    def set_display_parms(self, uset_slice, parms):
        """
        Update the display parameters (columns shown on hover) for the specified dataset(s).
        """
        if not isinstance(parms, list):
            parms = [parms]
            
        for ds in self._get_uset_slice(uset_slice):
            ds.display_parms = parms

    def toggle_darkmode(self, state=None):
        """
        Toggle between dark and light mode for plots.
        state: bool (optional) - Force specific state (True=Dark, False=Light)
        """
        if state is not None:
            self.darkmode = bool(state)
        else:
            self.darkmode = not self.darkmode
            
        mode = "Dark" if self.darkmode else "Light"
        print(f"Plot theme set to: {mode} Mode")

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def delta(self, base_idx, study_indices, align_on=None, delta_parms=None,
              suffixes=("_BASE", ""), direction='nearest', tolerance=None):
        """Compute delta (absolute and %) between each study dataset and the base, and load the results.

        Parameters
        ----------
        base_idx : int
            Index of the baseline dataset.
        study_indices : int | list | 'all'
            Dataset(s) to compare against the base. The base itself is always skipped.
        align_on : str | None
            Column to align on (nearest-match merge). Defaults to last_x.
        delta_parms : str | list | None
            Columns to compute deltas for. Defaults to last_y.
        suffixes : tuple[str, str]
            Suffixes applied to base and study columns during the merge. Must differ.
        direction : 'nearest' | 'forward' | 'backward'
            Passed to merge_asof — controls which study row is matched to each base row.
        tolerance : numeric | None
            Maximum allowed distance between matched align_on values. Unmatched rows get NaN.
        """
        # Resolve align_on from last plot state
        if align_on is None:
            lx = self.last_x
            align_on = lx[0] if isinstance(lx, list) else lx
        if align_on is None:
            raise ValueError("align_on is required when no prior plot exists.")
        if isinstance(align_on, list):
            raise ValueError("align_on must be a single column name, not a list.")

        # Resolve delta_parms from last plot state
        if delta_parms is None:
            ly = self.last_y
            delta_parms = ly if isinstance(ly, list) else ([ly] if ly is not None else [])
        if not isinstance(delta_parms, list):
            delta_parms = [delta_parms]
        delta_parms = [p for p in delta_parms if p is not None]
        if not delta_parms:
            raise ValueError("delta_parms is required when no prior plot exists.")

        lsuffix, rsuffix = suffixes
        if lsuffix == rsuffix:
            raise ValueError(f"suffixes must be different; both are '{lsuffix}'.")

        if not (0 <= base_idx < len(self.sets)):
            raise IndexError(f"base_idx {base_idx} is out of range (have {len(self.sets)} datasets).")

        base_ds = self.sets[base_idx]

        if align_on not in base_ds.df.columns:
            raise ValueError(f"align_on column '{align_on}' not found in base dataset '{base_ds.title}'.")

        # Exclude the base from study targets to avoid a trivial zero-delta set
        targets = [ds for ds in self._get_uset_slice(study_indices) if ds.index != base_idx]
        if not targets:
            print("No study datasets to process (base dataset excluded if present in selection).")
            return []

        created = []

        for study_ds in targets:
            if align_on not in study_ds.df.columns:
                print(f"Warning: skipping '{study_ds.title}' — align_on column '{align_on}' not found.")
                continue

            valid_parms = [p for p in delta_parms if p in base_ds.df.columns and p in study_ds.df.columns]
            skipped = sorted(set(delta_parms) - set(valid_parms))
            if skipped:
                print(f"Warning: skipping columns not present in both datasets: {skipped}")
            if not valid_parms:
                print(f"Warning: skipping '{study_ds.title}' — no valid delta columns found.")
                continue

            df_base = (base_ds.df[[align_on] + valid_parms]
                       .sort_values(align_on).reset_index(drop=True))
            df_study = (study_ds.df[[align_on] + valid_parms]
                        .sort_values(align_on).reset_index(drop=True))

            merge_kwargs = dict(on=align_on, suffixes=suffixes, direction=direction)
            if tolerance is not None:
                merge_kwargs['tolerance'] = tolerance

            merged = pd.merge_asof(df_base, df_study, **merge_kwargs)

            result = merged[[align_on]].copy()
            for parm in valid_parms:
                b_col = f"{parm}{lsuffix}"
                s_col = f"{parm}{rsuffix}"
                result[f"DL_{parm}"] = merged[s_col] - merged[b_col]
                result[f"DLPCT_{parm}"] = np.where(
                    merged[b_col] == 0, np.nan,
                    100 * (result[f"DL_{parm}"] / merged[b_col])
                )

            nan_frac = result[f"DL_{valid_parms[0]}"].isna().mean()
            if nan_frac > 0.5:
                print(f"Warning: '{study_ds.title}' — {nan_frac:.0%} of delta rows are NaN "
                      f"(large alignment gaps; consider tolerance= or a different direction=).")

            new_title = f"Delta {base_ds.index}-{study_ds.index}"
            next_index = len(self.sets)
            ds = Dataset(result, index=next_index, title=new_title)
            ds.set_type = 'delta'
            self.sets.append(ds)
            print(f"Loaded Set {next_index}: {new_title}")
            created.append(ds)

        self._refresh_widgets()
        return created

    def combine_sets(self, uset_slice, title=None, ignore_index=True):
        """Concatenate multiple datasets row-wise into a new dataset.

        Parameters
        ----------
        uset_slice : int | list | 'all'
            Datasets to combine. Must resolve to at least 2 datasets.
        title : str | None
            Title for the new dataset. Defaults to 'Combined 0-1-2-...' using source indices.
        ignore_index : bool
            Reset the row index in the combined DataFrame (default True). Set to False to
            preserve the original indices, which may be useful if they carry meaning.
        """
        sources = self._get_uset_slice(uset_slice)
        if len(sources) < 2:
            print(f"Warning: combine_sets requires at least 2 datasets (got {len(sources)}).")
            return None

        col_sets = [set(ds.df.columns) for ds in sources]
        shared = col_sets[0].intersection(*col_sets[1:])
        all_cols = set().union(*col_sets)
        only_in_some = all_cols - shared
        if only_in_some:
            print(f"Warning: {len(only_in_some)} column(s) not present in all datasets — "
                  f"those cells will be NaN: {sorted(only_in_some)}")

        combined = pd.concat([ds.df for ds in sources], ignore_index=ignore_index)

        idx_str = '-'.join(str(ds.index) for ds in sources)
        new_title = title or f"Combined {idx_str}"
        next_index = len(self.sets)
        new_ds = Dataset(combined, index=next_index, title=new_title)
        self.sets.append(new_ds)
        print(f"Loaded Set {next_index}: {new_title} ({len(combined)} rows from {len(sources)} datasets)")
        self._refresh_widgets()
        return new_ds

    # ------------------------------------------------------------------
    # Axes Based Decorations (Lines/Highlights/Scale)
    # ------------------------------------------------------------------
    def line(self, column, level, color='red', dash='dash'):
        """Add a vertical or horizontal line to the next plot."""
        if level == 'clear':
            if column == 'all':
                self.lines.clear()
            else:
                self.lines.pop(column, None)
            return
        
        if column not in self.lines: self.lines[column] = []
        plotly_dash = LINESTYLE_MAP_MPL_TO_PLOTLY.get(dash, dash)
        self.lines[column].append({'level': level, 'color': color, 'dash': plotly_dash})

    def highlight(self, column, range_tuple, color='yellow', alpha=0.2, opacity=None):
        """Add a highlighted region to the next plot."""
        if opacity is not None:
            warnings.warn("'opacity' is deprecated, use 'alpha'", DeprecationWarning, stacklevel=2)
            alpha = opacity
        if range_tuple == 'clear':
            if column == 'all':
                self.highlights.clear()
            else:
                self.highlights.pop(column, None)
            return

        if column not in self.highlights: self.highlights[column] = []
        self.highlights[column].append({'range': range_tuple, 'color': color, 'alpha': alpha})
        
    def scale(self, column, range_tuple):
        """
        Set specific axis limits for a parameter.
        """
        if range_tuple == 'clear' or range_tuple is None:
            if column in self.axis_limits:
                del self.axis_limits[column]
                print(f"Limits cleared for '{column}'.")
        else:
            if isinstance(range_tuple, (list, tuple)) and len(range_tuple) == 2:
                self.axis_limits[column] = range_tuple
                print(f"Limits set for '{column}': {range_tuple}")
            else:
                raise ValueError(f"Invalid range for {column}. Must be a tuple (min, max).")

    # ------------------------------------------------------------------
    # Font Management
    # ------------------------------------------------------------------
    def set_font_sizes(self, suptitle=None, legend=None, axes_title=None,
                    axes_tick=None, subplot_title=None, colorbar=None,
                    hover=None, table_header=None, table_cell=None, all=None, reset=False):
        """
        Configure font sizes for plot and table elements. Settings persist across plots.

        Parameters
        ----------
        suptitle, legend, axes_title, axes_tick, subplot_title, colorbar, hover : float or str
            Font sizes for plot elements.
        table_header : float or str
            Font size for table header row.
        table_cell : float or str
            Font size for table cell content.
        all : float or str
            Set all font sizes at once (overridden by individual parameters).
        reset : bool
            Reset all font sizes to defaults.
        """
        keys = ('suptitle_size', 'legend_size', 'axes_title_size', 'axes_tick_size',
                'subplot_title_size', 'colorbar_size', 'hover_size', 'table_header_size', 'table_cell_size')

        if reset:
            for k in keys:
                setattr(self, k, None)
            return

        def _validate(name, value):
            if value is None:
                return None
            if isinstance(value, str):
                resolved = FONT_SIZE_MAP.get(value.lower())
                if resolved is None:
                    valid = ', '.join(sorted(FONT_SIZE_MAP))
                    raise ValueError(f"{name}: unknown size name '{value}'. Valid names: {valid}")
                value = resolved
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric or a size name, got {type(value).__name__}")
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
            if value > 72:
                warnings.warn(f"{name}={value} is unusually large for a font size.")
            return float(value)

        base = _validate('all', all)
        resolved = {
            'suptitle_size':      _validate('suptitle', suptitle)           if suptitle      is not None else base,
            'legend_size':        _validate('legend', legend)               if legend        is not None else base,
            'axes_title_size':    _validate('axes_title', axes_title)       if axes_title    is not None else base,
            'axes_tick_size':     _validate('axes_tick', axes_tick)         if axes_tick     is not None else base,
            'subplot_title_size': _validate('subplot_title', subplot_title) if subplot_title is not None else base,
            'colorbar_size':      _validate('colorbar', colorbar)           if colorbar      is not None else base,
            'hover_size':         _validate('hover', hover)                 if hover         is not None else base,
            'table_header_size':  _validate('table_header', table_header)   if table_header  is not None else base,
            'table_cell_size':    _validate('table_cell', table_cell)       if table_cell    is not None else base,
        }
        for k, v in resolved.items():
            if v is not None:
                setattr(self, k, v)


    def get_font_sizes(self):
        """Return a dict of currently configured font sizes (None = unset/default)."""
        return {
            'suptitle':       self.suptitle_size,
            'legend':         self.legend_size,
            'axes_title':     self.axes_title_size,
            'axes_tick':      self.axes_tick_size,
            'subplot_title':  getattr(self, 'subplot_title_size', None),
            'colorbar':       getattr(self, 'colorbar_size', None),
            'hover':          getattr(self, 'hover_size', None),
            'table_header':   getattr(self, 'table_header_size', None),
            'table_cell':     getattr(self, 'table_cell_size', None),
        }

    def _apply_fonts(self, fig):
        """Apply stored font sizes to a Plotly figure."""
        if fig is None:
            return fig

        layout_updates = {}
        if self.suptitle_size is not None:
            layout_updates['title_font'] = dict(size=self.suptitle_size)
        if self.legend_size is not None:
            layout_updates['legend'] = dict(font=dict(size=self.legend_size))
        if getattr(self, 'hover_size', None) is not None:
            layout_updates['hoverlabel'] = dict(font=dict(size=self.hover_size))
        if layout_updates:
            fig.update_layout(**layout_updates)

        sp_size = getattr(self, 'subplot_title_size', None)
        if sp_size is not None and fig.layout.annotations:
            for ann in fig.layout.annotations:
                ann.font = dict(size=sp_size)

        x_updates, y_updates = {}, {}
        if self.axes_title_size is not None:
            x_updates['title_font'] = dict(size=self.axes_title_size)
            y_updates['title_font'] = dict(size=self.axes_title_size)
        if self.axes_tick_size is not None:
            x_updates['tickfont'] = dict(size=self.axes_tick_size)
            y_updates['tickfont'] = dict(size=self.axes_tick_size)
        if x_updates:
            fig.update_xaxes(**x_updates)
        if y_updates:
            fig.update_yaxes(**y_updates)

        cb_size = getattr(self, 'colorbar_size', None)
        if cb_size is not None:
            for trace in fig.data:
                cb = getattr(trace, 'colorbar', None)
                if cb is not None:
                    cb.tickfont = dict(size=cb_size)
                    if cb.title is not None:
                        try:
                            cb.title.font = dict(size=cb_size)
                        except (AttributeError, ValueError):
                            cb.title = dict(text=str(cb.title), font=dict(size=cb_size))

        return fig

    # ------------------------------------------------------------------
    # Main Plot Function
    # ------------------------------------------------------------------
    def plot(self, x=None, y=None, by='vars', figsize=(12, 8), ncols=None, nrows=None, 
                subplot_titles=None, suptitle=None, suppress_legends=False, **kwargs):
        """
        Main plotting wrapper.

        Parameters:
        -----------
        by : str, optional
            'vars'    (default) - Subplot per Y variable.
            'sets' / 'datasets' - Subplot per Dataset.
            'ymult'           - Single plot, multiple Y axes (delegates to plot_ymult).
        """
        # Delegate to the multi-y wrapper if requested
        if by == 'ymult':
            return self.plot_ymult(x=x, y=y, suptitle=suptitle,
                                     figsize=figsize,
                                     suppress_legends=suppress_legends)

        self._clear_last_fig()

        if x is None: x = self.last_x
        if y is None: y = self.last_y
        self.last_x = x
        self.last_y = y

        if ncols is None and nrows is None:
            if self.last_ncols is not None or self.last_nrows is not None:
                ncols, nrows = self.last_ncols, self.last_nrows
        self.last_ncols = ncols
        self.last_nrows = nrows

        if by == 'sets' or by == 'datasets':
            fig = uniplot_per_dataset(
                list_of_datasets=self.sets,
                x=x,
                y=y,
                display_parms=self.display_parms,
                suptitle=suptitle or self.suptitle,
                figsize=figsize,
                ncols=ncols,
                nrows=nrows,
                darkmode=self.darkmode,
                x_lim=self.axis_limits.get(x) if isinstance(x, str) else None,
                y_lim=None,
                axis_limits=self.axis_limits, 
                return_axes=True 
            )
            mode = 'sets'
            
        else:
            plot_args = {
                'list_of_datasets': self.sets,
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
                'axis_limits': self.axis_limits,
            }
            plot_args.update(kwargs)
            fig = uniplot(**plot_args)
            mode = 'vars'

        if fig is None: return

        x_list = x if isinstance(x, list) else [x]
        y_list = y if isinstance(y, list) else [y]
        active_sets = [d for d in self.sets if d.select]

        if len(x_list) == len(y_list):
            plot_pairs = list(zip(x_list, y_list))
        elif len(x_list) == 1:
            plot_pairs = [(x_list[0], yi) for yi in y_list]
        elif len(y_list) == 1:
            plot_pairs = [(xi, y_list[0]) for xi in x_list]
        else:
            plot_pairs = [(x_list[0], yi) for yi in y_list]

        if mode == 'vars':
            n_items = len(plot_pairs)
        else:
            n_items = len(active_sets)

        if ncols is None and nrows is None:
            calc_ncols = min(3, max(1, int(np.ceil(np.sqrt(n_items)))))
        elif ncols is None:
            calc_ncols = int(np.ceil(n_items / nrows))
        else:
            calc_ncols = ncols
        calc_ncols = max(1, calc_ncols)

        fig = self._apply_decorations(
            fig, x_list, y_list, mode, calc_ncols,
            plot_pairs if mode == 'vars' else None
        )

        if mode == 'vars':
            for idx, (xi, yi) in enumerate(plot_pairs):
                r, c = (idx // calc_ncols) + 1, (idx % calc_ncols) + 1
                if xi in self.axis_limits:
                    fig.update_xaxes(range=self.axis_limits[xi], row=r, col=c)
                if yi in self.axis_limits:
                    fig.update_yaxes(range=self.axis_limits[yi], row=r, col=c)

        fig = self._apply_fonts(fig)
        if fig and suppress_legends:
            fig.update_traces(visible='legendonly') 
        self.last_fig = fig
        return fig

    # ------------------------------------------------------------------
    # Multi-Y plot wrapper
    # ------------------------------------------------------------------
    def plot_ymult(self, x=None, y=None, suptitle=None, figsize=(12, 8),
                     legend='above', legend_group_by='sets', suppress_legends=False):
        """
        Single plot, multiple Y-axes. All selected datasets overlay on the same x-axis.
        Applies all notebook-level formatting: axis_limits, variable_formats, lines, highlights.
        """
        self._clear_last_fig()
        if x is None: x = self.last_x
        if y is None: y = self.last_y
        self.last_x, self.last_y = x, y

        y_list = y if isinstance(y, list) else [y]

        fig = uniplot_ymultaxis(
            list_of_datasets=self.sets,
            x=x, y=y_list,
            variable_formats=self.variable_formats,
            display_parms=self.display_parms,
            suptitle=suptitle or self.suptitle,
            xlabel=self.x_label,
            darkmode=self.darkmode,
            figsize=figsize,
            x_lim=self.axis_limits.get(x) if isinstance(x, str) else None,
            axis_limits=self.axis_limits,
            legend=legend,
            legend_group_by=legend_group_by,
            return_axes=True,
        )
        if fig is None:
            return None

        # Apply line/highlight decorations using the y_list → axis map.
        # Vertical lines on x span all axes; horizontal lines on a y-var
        # are drawn on the y-axis assigned to that variable.
        yref_for = {yi: ('y' if i == 0 else f'y{i+1}') for i, yi in enumerate(y_list)}

        for col, lines in self.lines.items():
            if col == x:
                for l in lines:
                    fig.add_vline(x=l['level'],
                                  line_dash=l['dash'] or 'solid',
                                  line_color=l['color'])
            elif col in yref_for:
                yref = yref_for[col]
                for l in lines:
                    fig.add_shape(type='line', x0=0, x1=1,
                                  y0=l['level'], y1=l['level'],
                                  xref='paper', yref=yref,
                                  line=dict(color=l['color'], dash=l['dash'] or 'solid'))

        for col, hls in self.highlights.items():
            if col == x:
                for h in hls:
                    fig.add_vrect(x0=h['range'][0], x1=h['range'][1],
                                  fillcolor=h['color'], opacity=h['alpha'],
                                  layer='below', line_width=0)
            elif col in yref_for:
                yref = yref_for[col]
                for h in hls:
                    fig.add_shape(type='rect', x0=0, x1=1,
                                  y0=h['range'][0], y1=h['range'][1],
                                  xref='paper', yref=yref,
                                  fillcolor=h['color'], opacity=h['alpha'],
                                  layer='below', line_width=0)

        fig = self._apply_fonts(fig)
        if suppress_legends:
            fig.update_traces(visible='legendonly')
        self.last_fig = fig
        return fig

    # ------------------------------------------------------------------
    # The bar Command
    # ------------------------------------------------------------------
    def bar(self, x=None, y=None, markers=None, by='vars', barmode='group', agg='mean',
            figsize=(12, 8), ncols=None, nrows=None, suppress_legends=False):
        """
        Unified interface for Bar Charts.

        Marker overlay formatting is controlled via `var_format`. Examples:
            nb.var_format('EGT_LIMIT', color='red', marker='*', markersize=18)
            nb.bar(x='PHASE', y='EGT', markers='EGT_LIMIT')
        """
        self._clear_last_fig()

        if x is None: x = self.last_x
        if y is None: y = self.last_y
        self.last_x, self.last_y = x, y

        y_list = y if isinstance(y, list) else [y]

        if by == 'dataset_x':
            if markers:
                print("Warning: `markers` is not supported with by='dataset_x'.")
            fig = unibar_datasets_as_x(
                list_of_datasets=self.sets, y=y_list, agg=agg,
                suptitle=self.suptitle, figsize=figsize, 
                darkmode=self.darkmode, axis_limits=self.axis_limits, return_axes=True
            )
            if fig:
                fig = self._apply_decorations(fig, [], y_list, 'global', 1)
                fig = self._apply_fonts(fig)
                if suppress_legends:
                    fig.update_traces(visible='legendonly')
                self.last_fig = fig
            return fig
            
        elif by in ['sets', 'datasets']:
            fig = unibar_per_dataset(
                list_of_datasets=self.sets, x=x, y=y, markers=markers,
                variable_formats=self.variable_formats,         # <-- pass through
                barmode=barmode,
                suptitle=self.suptitle, figsize=figsize, ncols=ncols, nrows=nrows,
                darkmode=self.darkmode, return_axes=True 
            )
        else:
            fig = unibar(
                list_of_datasets=self.sets, x=x, y=y, markers=markers,
                variable_formats=self.variable_formats,         # <-- pass through
                barmode=barmode,
                suptitle=self.suptitle, figsize=figsize, ncols=ncols, nrows=nrows,
                darkmode=self.darkmode, return_axes=True 
            )

        if fig:
            if x in self.axis_limits:
                fig.update_xaxes(range=self.axis_limits[x])

            active_sets = [d for d in self.sets if d.select]
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

            dec_mode = 'sets' if by in ['sets', 'datasets'] else 'vars'
            dec_items = [(x, yi) for yi in y_list] if dec_mode == 'vars' else None
            fig = self._apply_decorations(fig, [], y_list, dec_mode, calc_ncols, dec_items)

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
        """
        Unified interface for Box Plots.
        """
        self._clear_last_fig()

        if x is None: x = self.last_x
        if y is None: y = self.last_y
        self.last_x, self.last_y = x, y

        y_list = y if isinstance(y, list) else [y]

        if by == 'dataset_x':
            fig = unibox_datasets_as_x(
                list_of_datasets=self.sets, y=y_list, boxmode=boxmode,
                points=points, notched=notched, suptitle=suptitle or self.suptitle,
                figsize=figsize, darkmode=self.darkmode, axis_limits=self.axis_limits, return_axes=True
            )
            if fig:
                fig = self._apply_decorations(fig, [], y_list, 'global', 1)
                fig = self._apply_fonts(fig)
                if suppress_legends:
                    fig.update_traces(visible='legendonly')
                self.last_fig = fig
            return fig

        elif by in ['sets', 'datasets']:
            primary_y = y_list[0]
            y_limit = self.axis_limits.get(primary_y)
            fig = unibox_per_dataset(
                list_of_datasets=self.sets, x=x, y=y, boxmode=boxmode,
                points=points, notched=notched,
                suptitle=suptitle or self.suptitle, figsize=figsize, ncols=ncols, nrows=nrows,
                darkmode=self.darkmode, y_lim=y_limit, return_axes=True
            )
        else:
            fig = unibox(
                list_of_datasets=self.sets, x=x, y=y, boxmode=boxmode,
                points=points, notched=notched, color=color,
                suptitle=suptitle or self.suptitle, figsize=figsize, ncols=ncols, nrows=nrows,
                darkmode=self.darkmode, y_lim=None, return_axes=True
            )
            
            if fig:
                active_sets = [d for d in self.sets if d.select]
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
            if by in ['sets', 'datasets']:
                fig = self._apply_decorations(fig, [], y_list, 'sets', 1)
            else:
                _n = len(y_list)
                _nc = ncols if ncols is not None else (
                    int(np.ceil(_n / nrows)) if nrows is not None
                    else min(3, max(1, int(np.ceil(np.sqrt(_n)))))
                )
                _nc = max(1, _nc)
                fig = self._apply_decorations(fig, [], y_list, 'vars', _nc,
                                              [(x, yi) for yi in y_list])
            fig = self._apply_fonts(fig)
            if suppress_legends:
                fig.update_traces(visible='legendonly')
            self.last_fig = fig

        return fig

    # ------------------------------------------------------------------
    # The histogram Command
    # ------------------------------------------------------------------
    def histogram(self, x=None, y=None, histfunc='sum', by='vars', nbins=None,
                    bin_size=None, bin_start=None, bin_end=None,
                    histnorm='', barmode='overlay', alpha=0.7,
                    color=None, suptitle=None, figsize=(12, 8), ncols=None, nrows=None, suppress_legends=False,
                    opacity=None):
        """
        Unified interface for Histograms.
        """
        if opacity is not None:
            warnings.warn("'opacity' is deprecated, use 'alpha'", DeprecationWarning, stacklevel=2)
            alpha = opacity
        self._clear_last_fig()

        if x is None: x = self.last_x
        self.last_x = x
        
        limit = None
        if isinstance(x, str):
            limit = self.axis_limits.get(x)
        elif isinstance(x, list) and len(x) == 1:
            limit = self.axis_limits.get(x[0])

        x_list = x if isinstance(x, list) else [x]

        if by in ['sets', 'datasets']:
            fig = unihistogram_by_dataset(
                list_of_datasets=self.sets, x=x, y=y, histfunc=histfunc, nbins=nbins,
                bin_size=bin_size, bin_start=bin_start, bin_end=bin_end,
                histnorm=histnorm, barmode=barmode, alpha=alpha, color=color,
                suptitle=suptitle or self.suptitle, figsize=figsize, ncols=ncols, nrows=nrows,
                darkmode=self.darkmode, x_lim=limit, return_axes=True
            )
            if fig:
                fig = self._apply_decorations(fig, x_list, [], 'sets', 1)
        else:
            fig = unihistogram(
                list_of_datasets=self.sets, x=x, y=y, histfunc=histfunc, nbins=nbins,
                bin_size=bin_size, bin_start=bin_start, bin_end=bin_end,
                histnorm=histnorm, barmode=barmode, alpha=alpha, color=color,
                suptitle=suptitle or self.suptitle, figsize=figsize, ncols=ncols, nrows=nrows,
                darkmode=self.darkmode, x_lim=limit, return_axes=True
            )
            if fig:
                _n = len(x_list)
                _nc = ncols if ncols is not None else (
                    int(np.ceil(_n / nrows)) if nrows is not None
                    else min(3, max(1, int(np.ceil(np.sqrt(_n)))))
                )
                _nc = max(1, _nc)
                fig = self._apply_decorations(fig, x_list, [], 'vars', _nc,
                                              [(xi, None) for xi in x_list])

        fig = self._apply_fonts(fig)
        if fig and suppress_legends:
            fig.update_traces(visible='legendonly')
        self.last_fig = fig
        return fig
        
    # ------------------------------------------------------------------
    # The contour Command
    # ------------------------------------------------------------------
    def contour(self, x=None, y=None, z=None, by='vars', contours_coloring='fill', 
                    colorscale=None, interpolate=True, interp_res=100, interp_method='linear',
                    ncontours=None,
                    suptitle=None, figsize=(12, 8), ncols=None, nrows=None, suppress_legends=False):
        """
        Unified interface for Contour Plots.
        """
        self._clear_last_fig()

        if x is None: x = self.last_x
        if y is None: y = self.last_y
        if z is None: z = getattr(self, 'last_z', None) 
        
        self.last_x, self.last_y, self.last_z = x, y, z

        if z is None:
            print("Error: Contour plots require a 'z' variable to map to color.")
            return

        limit_x = self.axis_limits.get(x)
        limit_y = self.axis_limits.get(y)

        if by in ['sets', 'datasets']:
            fig = unicontour_per_dataset(
                list_of_datasets=self.sets, x=x, y=y, z=z, 
                contours_coloring=contours_coloring, colorscale=colorscale,
                interpolate=interpolate, interp_res=interp_res, interp_method=interp_method,
                ncontours=ncontours,
                suptitle=suptitle or self.suptitle, figsize=figsize, ncols=ncols, nrows=nrows, 
                darkmode=self.darkmode, axis_limits=self.axis_limits, return_axes=True
            )
        else:
            fig = unicontour(
                list_of_datasets=self.sets, x=x, y=y, z=z,
                contours_coloring=contours_coloring, colorscale=colorscale,
                interpolate=interpolate, interp_res=interp_res, interp_method=interp_method,
                ncontours=ncontours,
                suptitle=suptitle or self.suptitle, figsize=figsize, ncols=ncols, nrows=nrows, 
                darkmode=self.darkmode, axis_limits=self.axis_limits, return_axes=True
            )
            
        if fig:
            z_list = z if isinstance(z, list) else [z]
            if by in ['sets', 'datasets']:
                fig = self._apply_decorations(fig, [x], [y], 'sets', 1)
            else:
                _n = len(z_list)
                _nc = ncols if ncols is not None else (
                    int(np.ceil(_n / nrows)) if nrows is not None
                    else min(3, max(1, int(np.ceil(np.sqrt(_n)))))
                )
                _nc = max(1, _nc)
                fig = self._apply_decorations(fig, [x], [y], 'vars', _nc,
                                              [(x, y) for _ in z_list])
            fig = self._apply_fonts(fig)
            if limit_x: fig.update_xaxes(range=limit_x)
            if limit_y: fig.update_yaxes(range=limit_y)
            if suppress_legends:
                fig.update_traces(visible='legendonly')
            self.last_fig = fig

        return fig

    # ------------------------------------------------------------------
    # The table Command
    # ------------------------------------------------------------------
    def table(self, cols=None, title=None):
        """
        Display a Plotly Table of specific columns from selected datasets.
        """
        if cols is None:
            if self.last_x is None or self.last_y is None:
                print("No columns specified and no previous plot variables defined.")
                return
            
            y_part = self.last_y if isinstance(self.last_y, list) else [self.last_y]
            target_cols = [self.last_x] + y_part
        else:
            target_cols = cols if isinstance(cols, list) else [cols]

        combined_dfs = []
        
        for ds in self.sets:
            if not ds.select:
                continue
            
            valid_cols = [c for c in target_cols if c in ds.df.columns]
            
            if not valid_cols:
                continue
                
            subset = ds.df[valid_cols].copy()           # <-- copy to avoid mutating the dataset
            subset.insert(0, 'Dataset', ds.title)
            subset.insert(0, 'Set', ds.index)
            combined_dfs.append(subset)

        if not combined_dfs:
            print("No data found for the specified columns in selected datasets.")
            return

        final_df = pd.concat(combined_dfs, ignore_index=True)
        final_df = final_df.fillna('-')

        if self.darkmode:
            header_color = 'rgb(30, 30, 30)'
            cell_color = 'rgb(50, 50, 50)'
            font_color = 'white'
            line_color = 'rgb(70, 70, 70)'
        else:
            header_color = 'rgb(230, 230, 230)'
            cell_color = 'white'
            font_color = 'black'
            line_color = 'rgb(200, 200, 200)'

        # Bold header text via HTML — works in all Plotly versions
        header_values = [f"<b>{c}</b>" for c in final_df.columns]

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=header_values,
                fill_color=header_color,
                align='left',
                font=dict(color=font_color, size=12),     # <-- removed weight='bold'
                line_color=line_color
            ),
            cells=dict(
                values=[final_df[k].tolist() for k in final_df.columns],
                fill_color=cell_color,
                align='left',
                font=dict(color=font_color, size=11),
                line_color=line_color,
                height=25
            )
        )])

        layout_args = {
            'title': {'text': title or "Data Table", 'x': 0.5},
            'template': "plotly_dark" if self.darkmode else "plotly_white",
            'margin': dict(l=20, r=20, t=50, b=20),
        }
        fig.update_layout(**layout_args)

        self.last_fig = fig                             # <-- so save_png works
        fig = self._apply_fonts(fig)

        display_df = final_df.copy()
        for col in display_df.columns:
            if display_df[col].dtype in ['float64', 'float32']:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.5g}" if isinstance(x, (int, float)) and x != '-' else x)

        header_size = self.table_header_size or 22
        cell_size = self.table_cell_size or 20
        title_size = self.suptitle_size or header_size + 4

        html_table = display_df.to_html(index=False, escape=False)
        if title:
            caption_html = (
                f'<caption style="caption-side:top;text-align:center;'
                f'font-weight:600;color:#000;font-size:{title_size}px;'
                f'padding:6px 8px;background-color:#e8e8e8;'
                f'border:1px solid #ccc;border-bottom:none;'
                f'font-family:-apple-system, BlinkMacSystemFont, \'Segoe UI\', Arial, sans-serif;">'
                f'{title}</caption>'
            )
            html_table = html_table.replace('<table', '<table', 1)
            html_table = html_table.replace('>', f'>{caption_html}', 1)

        styled_html = f"""
        <div style="margin-top:8px; margin-bottom:8px; overflow-x:auto;">
            {html_table}
        </div>
        <style>
        table {{
            border-collapse: collapse;
            width: auto;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
        }}
        th {{
            background-color: #e8e8e8;
            color: #000;
            font-weight: 600;
            font-size: {header_size}px;
            padding: 3px 8px;
            text-align: center;
            border: 1px solid #ccc;
        }}
        td {{
            color: #000;
            font-size: {cell_size}px;
            padding: 2px 8px;
            text-align: center;
            border: 1px solid #ddd;
        }}
        tr:nth-child(odd) td {{ background-color: #ffffff; }}
        tr:nth-child(even) td {{ background-color: #f3f3f3; }}
        tr:hover td {{ background-color: #ececec; }}
        </style>
        """

        display(HTML(styled_html))

    def save_png(self, filename="plot.png", scale=3, width=None, height=None):
        """
        Save the last generated plot to a PNG file.
        Requires the 'kaleido' package to be installed.
        """
        if self.last_fig is None:
            print("No plot to save. Please run .plot() first.")
            return

        try:
            import plotly.io as pio
            pio.defaults.mathjax = None 

            self.last_fig.write_image(filename, scale=scale, width=width, height=height)
            print(f"Plot saved to {filename}")

        except ValueError as e:
            print(f"Error saving image (ensure 'kaleido' is installed): {e}")
        except Exception as e:
            print(f"Error saving image: {e}")

    def list_sets(self):
        """
        Print a formatted table of all loaded datasets.
        """
        if not self.sets:
            print("No datasets loaded.")
            return

        rows = []
        for ds in self.sets:
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
        """
        List the parameters (columns) available in the loaded datasets.
        """
        import fnmatch
        
        if set_number is None:
            target_sets = self.selected()
            if not target_sets:
                target_sets = self.sets
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

    def summary(self, cols=None):
        """
        Print a formatted statistical summary table.
        """
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
        """
        Display help information for the UnichartNotebook class.
        """
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
        print("1. Load data:       nb.load_df(df, title='MyData')")
        print("2. Select datasets: nb.select([0, 1])")
        print("3. Plot:            nb.plot(x='time', y='value')")
        print("4. Multi-Y plot:    nb.plot_ymult(x='time', y=['Temp', 'Pressure'])")
        print("5. Variable format: nb.var_format('Temp', linestyle='--')")
        print("6. View help:       nb.help()")
        print("\nSee method signatures above for full parameters.")

        print("\n" + "=" * 70)

    # ------------------------------------------------------------------
    # Interactive "GUI" Replacement
    # ------------------------------------------------------------------
    def _clear_last_fig(self):
        """Drop the previous figure's trace data to free memory before the next plot."""
        if self.last_fig is not None:
            self.last_fig.data = []
            self.last_fig.layout = {}
            self.last_fig = None

    def _apply_decorations(self, fig, x_vars, y_vars, mode, calc_ncols, plot_items=None):
        """
        Apply stored lines and highlights to a figure.
        """
        x_list = x_vars if isinstance(x_vars, list) else ([x_vars] if x_vars else [])
        y_list = y_vars if isinstance(y_vars, list) else ([y_vars] if y_vars else [])

        for col_name, col_lines in self.lines.items():
            if col_name in x_list:
                if mode == 'vars' and plot_items:
                    for idx, (xi, yi) in enumerate(plot_items):
                        if xi == col_name:
                            r, c = (idx // calc_ncols) + 1, (idx % calc_ncols) + 1
                            xref, yref = _subplot_refs(r, c, calc_ncols)
                            for l in col_lines:
                                fig.add_shape(
                                    type='line', x0=l['level'], x1=l['level'], y0=0, y1=1,
                                    xref=xref, yref=f'{yref} domain',
                                    line=dict(color=l['color'], dash=l['dash'] or 'solid')
                                )
                else:
                    for l in col_lines:
                        fig.add_vline(x=l['level'], line_dash=l['dash'] or 'solid', line_color=l['color'])

            if col_name in y_list:
                if mode == 'vars' and plot_items:
                    for idx, (xi, yi) in enumerate(plot_items):
                        if yi == col_name:
                            r, c = (idx // calc_ncols) + 1, (idx % calc_ncols) + 1
                            xref, yref = _subplot_refs(r, c, calc_ncols)
                            for l in col_lines:
                                fig.add_shape(
                                    type='line', x0=0, x1=1, y0=l['level'], y1=l['level'],
                                    xref=f'{xref} domain', yref=yref,
                                    line=dict(color=l['color'], dash=l['dash'] or 'solid')
                                )
                else:
                    for l in col_lines:
                        fig.add_hline(y=l['level'], line_dash=l['dash'] or 'solid', line_color=l['color'])

        for col_name, hls in self.highlights.items():
            if col_name in x_list:
                if mode == 'vars' and plot_items:
                    for idx, (xi, yi) in enumerate(plot_items):
                        if xi == col_name:
                            r, c = (idx // calc_ncols) + 1, (idx % calc_ncols) + 1
                            xref, yref = _subplot_refs(r, c, calc_ncols)
                            for h in hls:
                                fig.add_shape(
                                    type='rect', x0=h['range'][0], x1=h['range'][1], y0=0, y1=1,
                                    xref=xref, yref=f'{yref} domain',
                                    fillcolor=h['color'], opacity=h['alpha'], layer='below', line_width=0
                                )
                else:
                    for h in hls:
                        fig.add_vrect(x0=h['range'][0], x1=h['range'][1], fillcolor=h['color'],
                                      opacity=h['alpha'], layer='below', line_width=0)

            if col_name in y_list:
                if mode == 'vars' and plot_items:
                    for idx, (xi, yi) in enumerate(plot_items):
                        if yi == col_name:
                            r, c = (idx // calc_ncols) + 1, (idx % calc_ncols) + 1
                            xref, yref = _subplot_refs(r, c, calc_ncols)
                            for h in hls:
                                fig.add_shape(
                                    type='rect', x0=0, x1=1, y0=h['range'][0], y1=h['range'][1],
                                    xref=f'{xref} domain', yref=yref,
                                    fillcolor=h['color'], opacity=h['alpha'], layer='below', line_width=0
                                )
                else:
                    for h in hls:
                        fig.add_hrect(y0=h['range'][0], y1=h['range'][1], fillcolor=h['color'],
                                      opacity=h['alpha'], layer='below', line_width=0)

        return fig

    def _refresh_widgets(self):
        """Rebuild the dataset widget list, closing old widgets first to avoid memory leaks."""
        if hasattr(self, 'dataset_widget_container') and self.dataset_widget_container.children:
            for child in self.dataset_widget_container.children:
                if hasattr(child, 'children'):
                    for sub_child in child.children:
                        sub_child.close()
                child.close()
                
        items = []
        items.append(widgets.HTML("<b>Dataset Manager</b>"))
        
        def update_select(change, dataset):
            dataset.select = change['new']

        def update_color(change, dataset):
            dataset.color = change['new']

        for i, ds in enumerate(self.sets):
            
            chk = widgets.Checkbox(
                value=ds.select, 
                description=f"{i}: {ds.title}", 
                indent=False, 
                layout=widgets.Layout(width='300px')
            )
            
            chk.observe(functools.partial(update_select, dataset=ds), names='value')
            
            details = widgets.Label(
                value=f"[Rows: {len(ds.df)}] [Query: {ds.query}]", 
                layout=widgets.Layout(width='200px')
            )
            
            current_color = ds.color if isinstance(ds.color, str) else 'blue'
            cp = widgets.ColorPicker(
                concise=True, 
                value=current_color, 
                layout=widgets.Layout(width='30px')
            )
            
            cp.observe(functools.partial(update_color, dataset=ds), names='value')

            row = widgets.HBox([chk, cp, details])
            items.append(row)
            
        self.dataset_widget_container.children = tuple(items)
        

    def gui(self):
        """Displays the interactive dataset manager widgets."""
        self._refresh_widgets()
        
        btn_refresh = widgets.Button(description="Refresh Data View")
        btn_refresh.on_click(lambda b: self._refresh_widgets())
        
        display(widgets.VBox([btn_refresh, self.dataset_widget_container]))
