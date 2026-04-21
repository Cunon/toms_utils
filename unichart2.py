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
import functools
import gc

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
    """
    Translates MPL styles to Plotly. 
    NOTE: Since we are using Scattergl (WebGL) for performance, we are strictly
    limited to 'solid', 'dash', 'dot', and 'dashdot'. Complex dash arrays will
    fail to render in WebGL, so this includes a failsafe fallback to 'solid'.
    """
    # If already a valid WebGL plotly dash string
    if mpl_style in ['solid', 'dash', 'dot', 'dashdot', None]:
        return mpl_style
        
    style = LINESTYLE_MAP_MPL_TO_PLOTLY.get(mpl_style, 'solid')
    
    # Failsafe for complex or unsupported dashes mapped by accident
    if style not in ['solid', 'dash', 'dot', 'dashdot', None]:
        return 'solid'
        
    return style

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
    markers = list(MARKER_MAP_MPL_TO_PLOTLY.keys())
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
        self.hue_palette = "Jet" # Default Plotly colorscale
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
    """Internal helper to calculate polynomial regression lines."""
    df_clean = df.dropna(subset=[x_col, y_col]).sort_values(by=x_col)
    x = df_clean[x_col]
    y = df_clean[y_col]
    
    if len(x) < order + 1:
        return x, y * np.nan
        
    z = np.polyfit(x, y, order)
    p = np.poly1d(z)
    
    # Create smooth line
    x_lin = np.linspace(x.min(), x.max(), 100)
    y_lin = p(x_lin)
    return x_lin, y_lin

# -----------------------------------------------------------------------------
# Main Plotting Functions
# -----------------------------------------------------------------------------
def uniplot(list_of_datasets, x, y, z=None, plot_type=None, color=None, hue=None, marker=None,
            markersize=10, marker_edge_color="black", linestyle=None, hue_palette="Jet",
            hue_order=None, line=False, suppress_msg=False, return_axes=False, axes=None,
            suptitle=None, xlabel=None, ylabel=None, subplot_titles=None,
            darkmode=False, interactive=True, display_parms=None, grid=True,
            legend='above', legend_ncols=1, figsize=(12, 8), ncols=None, nrows=None, x_lim=None, y_lim=None):
    
    y_list = y if isinstance(y, list) else [y]
    n_y = len(y_list)
    if n_y == 0: raise ValueError("y must contain at least one column.")

    if nrows is None and ncols is None:
        ncols_auto = min(3, max(1, int(np.ceil(np.sqrt(n_y)))))
        nrows_auto = int(np.ceil(n_y / ncols_auto))
        nrows, ncols = nrows_auto, ncols_auto
    elif nrows is None: nrows = int(np.ceil(n_y / ncols))
    elif ncols is None: ncols = int(np.ceil(n_y / nrows))

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subplot_titles, shared_xaxes=False)

    layout_args = {
        'template': "plotly_dark" if darkmode else "plotly_white",
        'title': {'text': suptitle if suptitle else f"{x} vs {[str(yi) for yi in y_list]}", 'x': 0.5, 'xanchor': 'center'},
        'showlegend': True if legend != 'off' else False
    }

    if figsize:
        layout_args['width'] = figsize[0] * 100
        layout_args['height'] = figsize[1] * 100

    fig.update_layout(**layout_args)

    for dataset in list_of_datasets:
        if not dataset.select: continue
        
        # Pull reference instead of full copy
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

        for idx_y, yi in enumerate(y_list):
            row = idx_y // ncols + 1
            col = idx_y % ncols + 1

            if yi not in base_df.columns: continue

            # X-Column Resolution
            if x not in base_df.columns:
                cols_upper = {c.upper(): c for c in base_df.columns}
                x_key = cols_upper.get(str(x).upper())
                if not x_key: continue
                x_col = x_key
            else:
                x_col = x

            # MICRO-SUBSET: Build list of specifically required columns
            req_cols = [x_col, yi]
            if cur_hue and cur_hue in base_df.columns: req_cols.append(cur_hue)
            valid_hover = [p for p in hover_parms if p in base_df.columns]
            req_cols.extend(valid_hover)
            if dataset.order and dataset.order != 'index' and dataset.order in base_df.columns:
                req_cols.append(dataset.order)
            
            # Remove duplicates to prevent Pandas indexing errors
            req_cols = list(dict.fromkeys(req_cols))
            
            # Filter duplicated columns in base df dynamically, then slice our Micro-Subset
            valid_cols_mask = ~base_df.columns.duplicated()
            df = base_df.loc[:, valid_cols_mask][req_cols]

            # Apply sorting ONLY to the micro-subset
            if dataset.order == 'index': df = df.sort_index()
            elif dataset.order: df = df.sort_values(by=dataset.order)
            else: df = df.sort_index()

            # Custom Data & Hover (using sliced df)
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
            if not cur_linestyle: mode_parts.append('markers')
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

            # USE SCATTERGL
            fig.add_trace(go.Scattergl(
                x=df[x_col], y=df[yi], mode=mode,
                name=f"{cur_idx}: {cur_title}",
                legendgroup=f"group_{cur_idx}",
                marker=marker_dict, line=line_dict,
                customdata=custom_data, hovertemplate=ht,
                showlegend=(idx_y == 0)
            ), row=row, col=col)

            if isinstance(cur_reg_order, numbers.Number) and cur_reg_order > 0:
                rx, ry = _calculate_regression(df, x_col, yi, cur_reg_order)
                # USE SCATTERGL
                fig.add_trace(go.Scattergl(
                    x=rx, y=ry, mode='lines',
                    name=f"{cur_idx}: {cur_title} Fit LS {cur_reg_order}",
                    legendgroup=f"group_{cur_idx}",
                    line=dict(color=cur_color, width=cur_linewidth, dash='dash'),
                    opacity=0.7, hoverinfo='skip', showlegend=False
                ), row=row, col=col)

    for idx_y, yi in enumerate(y_list):
        row = idx_y // ncols + 1
        col = idx_y % ncols + 1
        
        axis_title = ylabel if ylabel else yi
        x_axis_title = xlabel if xlabel else x
        
        fig.update_yaxes(title_text=axis_title, title_standoff=15, row=row, col=col)
        fig.update_xaxes(title_text=x_axis_title, title_standoff=15, row=row, col=col)

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

    active_datasets = [d for d in list_of_datasets if d.select]
    if not active_datasets: return None

    y_list = y if isinstance(y, list) else [y]
    axis_limits = axis_limits or {}
    
    n_sets = len(active_datasets)
    if nrows is None and ncols is None:
        ncols = min(3, max(1, int(np.ceil(np.sqrt(n_sets)))))
        nrows = int(np.ceil(n_sets / ncols))
    elif nrows is None: nrows = int(np.ceil(n_sets / ncols))
    elif ncols is None: ncols = int(np.ceil(n_sets / nrows))
        
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

        # MICRO-SUBSET Logic
        base_df = dataset.df
        
        if x not in base_df.columns:
            cols_upper = {c.upper(): c for c in base_df.columns}
            if x.upper() in cols_upper: x_col = cols_upper[x.upper()]
            else: continue
        else: x_col = x

        req_cols = [x_col] + [yi for yi in y_list if yi in base_df.columns]
        valid_hover = [p for p in (display_parms or []) if p in base_df.columns]
        req_cols.extend(valid_hover)
        if dataset.order and dataset.order != 'index' and dataset.order in base_df.columns:
            req_cols.append(dataset.order)
            
        req_cols = list(dict.fromkeys(req_cols))
        df = base_df.loc[:, ~base_df.columns.duplicated()][req_cols]
        
        if dataset.order == 'index': df = df.sort_index()
        elif dataset.order: df = df.sort_values(by=dataset.order)

        for idx_y, yi in enumerate(y_list):
            if yi not in df.columns: continue
            
            var_color = color_cycle[idx_y % len(color_cycle)]
            custom_data = df[[x_col, yi] + valid_hover]
            ht = f"<b>{dataset.title}</b><br>{x_col}: %{{customdata[0]:.2f}}<br>{yi}: %{{customdata[1]:.2f}}"
            specific_limit = axis_limits.get(yi, None)

            line_dict = dict(color=var_color, width=dataset.linewidth)
            if dataset.linestyle: line_dict['dash'] = get_plotly_linestyle(dataset.linestyle)

            if idx_y == 0:
                # USE SCATTERGL
                fig.add_trace(go.Scattergl(
                    x=df[x_col], y=df[yi],
                    mode='lines+markers' if dataset.linestyle else 'markers',
                    name=yi, legendgroup=yi, showlegend=(idx_ds == 0),
                    marker=dict(color=var_color, size=dataset.markersize or 6),
                    line=line_dict, customdata=custom_data, hovertemplate=ht
                ), row=row, col=col)
                
                u_dict = dict(title_text=yi, title_font=dict(color=var_color), tickfont=dict(color=var_color), row=row, col=col)
                if specific_limit: u_dict['range'] = specific_limit
                fig.update_yaxes(**u_dict)

            elif idx_y == 1:
                curr_y_axis = f"yaxis{next_free_axis_idx}"
                curr_y_ref = f"y{next_free_axis_idx}"
                next_free_axis_idx += 1
                
                # USE SCATTERGL
                fig.add_trace(go.Scattergl(
                    x=df[x_col], y=df[yi],
                    mode='lines+markers' if dataset.linestyle else 'markers',
                    name=yi, legendgroup=yi, showlegend=(idx_ds == 0),
                    marker=dict(color=var_color, size=dataset.markersize or 6),
                    line=line_dict, customdata=custom_data, hovertemplate=ht,
                    xaxis=base_x_name.replace("axis", ""),
                    yaxis=curr_y_ref
                ))
                
                layout_axis = dict(
                    title=yi, title_font=dict(color=var_color), tickfont=dict(color=var_color),
                    anchor=base_x_name.replace("axis", ""), overlaying=base_y_name.replace("axis", ""),
                    side='right', showgrid=False
                )
                if specific_limit: layout_axis['range'] = specific_limit
                fig.update_layout({curr_y_axis: layout_axis})
                
            else:
                curr_y_axis = f"yaxis{next_free_axis_idx}"
                curr_y_ref = f"y{next_free_axis_idx}"
                next_free_axis_idx += 1
                
                extra_idx = idx_y - 1 
                pos = min(0.99, new_d_end + (extra_idx * width_per_axis))
                
                # USE SCATTERGL
                fig.add_trace(go.Scattergl(
                    x=df[x_col], y=df[yi],
                    mode='lines+markers' if dataset.linestyle else 'markers',
                    name=yi, legendgroup=yi, showlegend=(idx_ds == 0),
                    marker=dict(color=var_color, size=dataset.markersize or 6),
                    line=line_dict, customdata=custom_data, hovertemplate=ht,
                    xaxis=base_x_name.replace("axis", ""),
                    yaxis=curr_y_ref
                ))
                
                layout_axis = dict(
                    title=yi, title_font=dict(color=var_color), tickfont=dict(color=var_color),
                    anchor="free", overlaying=base_y_name.replace("axis", ""),
                    side='right', position=pos, showgrid=False
                )
                if specific_limit: layout_axis['range'] = specific_limit
                fig.update_layout({curr_y_axis: layout_axis})

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
        df = ds.df
        
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

    # Final Axis formatting
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
    
    Parameters:
    - points: 'all', 'outliers', 'suspectedoutliers', or False
    - notched: True/False (for confidence intervals)
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
        df = ds.df
        
        for idx_y, yi in enumerate(y_list):
            row, col = (idx_y // ncols) + 1, (idx_y % ncols) + 1
            if yi not in df.columns: continue

            # Create Box Trace
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

    # Final Axis formatting
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
    
    # Construct explicit binning logic
    xbins_dict = {}
    if bin_size is not None: xbins_dict['size'] = bin_size
    if bin_start is not None: xbins_dict['start'] = bin_start
    if bin_end is not None: xbins_dict['end'] = bin_end
    xbins = xbins_dict if xbins_dict else None

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
                opacity=opacity,
                nbinsx=nbins,
                xbins=xbins,  # Injects exact bin width/range if specified
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
    
    # Construct explicit binning logic
    xbins_dict = {}
    if bin_size is not None: xbins_dict['size'] = bin_size
    if bin_start is not None: xbins_dict['start'] = bin_start
    if bin_end is not None: xbins_dict['end'] = bin_end
    xbins = xbins_dict if xbins_dict else None

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
                opacity=opacity,
                nbinsx=nbins,
                xbins=xbins, # Injects exact bin width/range if specified
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
            else:
                plot_x, plot_y, plot_z = clean_df[x], clean_df[y], clean_df[zi]

            # Resolve Z-limits for color scaling
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

            # Resolve Z-limits for color scaling
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

        # Extract data with the chosen aggregation method
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

        # Add the Bar Trace
        fig.add_trace(go.Bar(
            name=yi,
            x=x_labels,
            y=y_data,
            yaxis=y_axis_name,
            offsetgroup=str(idx_y), # Forces grouping across multiple Y-axes
            marker_color=var_color
        ))

        # Configure the Layout for this Axis
        axis_layout = dict(
            title=yi,
            title_font=dict(color=var_color),
            tickfont=dict(color=var_color),
            showgrid=(idx_y == 0) 
        )

        if yi in axis_limits:
            axis_layout['range'] = axis_limits[yi]

        # Anchor logic for positioning axes
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

        # Extract all valid data for this specific variable across all datasets
        all_y_data = []
        all_x_labels = []
        
        for ds in active_ds:
            if yi in ds.df.columns:
                valid_data = ds.df[yi].dropna()
                if not valid_data.empty:
                    # Append the raw values
                    all_y_data.extend(valid_data.values)
                    # Append the corresponding dataset name for each value
                    label = f"{ds.index}: {ds.title}"
                    all_x_labels.extend([label] * len(valid_data))

        # Skip trace if no data was found for this variable across all datasets
        if not all_y_data:
            continue

        y_axis_name = "y" if idx_y == 0 else f"y{idx_y + 1}"

        # Add the Box Trace
        fig.add_trace(go.Box(
            name=yi,
            x=all_x_labels,
            y=all_y_data,
            yaxis=y_axis_name,
            offsetgroup=str(idx_y), # Forces grouping across multiple Y-axes
            marker_color=var_color,
            boxpoints=points,
            notched=notched,
        ))

        # Configure the Layout for this Axis
        axis_layout = dict(
            title=yi,
            title_font=dict(color=var_color),
            tickfont=dict(color=var_color),
            showgrid=(idx_y == 0) 
        )

        if yi in axis_limits:
            axis_layout['range'] = axis_limits[yi]

        # Anchor logic for positioning axes
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
            # Check bounds
            if 0 <= uset_slice < len(self.uset):
                return [self.uset[uset_slice]]
            return []
        elif isinstance(uset_slice, list):
            # List of indices or objects
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
        
        Args:
            uset_slice (int, list, or 'all'): The dataset index or indices to modify.
            size_val (int or float): Size of the marker (e.g., 5, 10.5).
        """
        for ds in self._get_uset_slice(uset_slice):
            ds.markersize = size_val
    
    def linewidth(self, uset_slice, width_val):
        """
        Set the line thickness for the specified dataset(s).
        
        Args:
            uset_slice (int, list, or 'all'): The dataset index or indices to modify.
            width_val (int or float): Thickness of the line (e.g., 1, 2.5).
        """
        for ds in self._get_uset_slice(uset_slice):
            ds.linewidth = width_val

    def hue(self, uset_slice, col_name):
        """
        Map a dataframe column to the color scale for the specified dataset(s).
        
        Args:
            uset_slice (int, list, or 'all'): The dataset index or indices to modify.
            col_name (str): Name of the column to color by (e.g., 'Temperature', 'Category').
        """
        for ds in self._get_uset_slice(uset_slice):
            ds.hue = col_name

    def hue_palette(self, uset_slice, hue_palette):
        """
        Set the color scale/palette used when `hue` is mapped to a variable.
        
        Args:
            uset_slice (int, list, or 'all'): The dataset index or indices to modify.
            hue_palette (str or list): A Plotly colorscale name (e.g., 'Viridis', 'Plasma', 'Inferno')
                                       or a discrete sequence (e.g., px.colors.qualitative.Pastel).
        """
        for ds in self._get_uset_slice(uset_slice):
            ds.hue_palette = hue_palette

    def alpha(self, uset_slice, alpha_val):
        """
        Set the opacity (alpha) for the specified dataset(s).
        
        Args:
            uset_slice (int, list, or 'all'): The dataset index or indices to modify.
            alpha_val (float): Opacity value between 0.0 (fully transparent) and 1.0 (fully opaque).
                               Example: 0.5.
        """
        for ds in self._get_uset_slice(uset_slice):
            ds.alpha = alpha_val

    def plot_type(self, uset_slice, type_val):
        for ds in self._get_uset_slice(uset_slice):
            ds.plot_type = type_val

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
    # Analysis (Ported from GUI)
    # ------------------------------------------------------------------
    def delta(self, base_idx, study_indices, align_on=None, delta_parms=None, suffixes=("_BASE", "")):
        """
        Creates a new dataset representing the difference between study and base.
        """
        if align_on is None: align_on = self.last_x
        if delta_parms is None: delta_parms = [self.last_y] if isinstance(self.last_y, str) else self.last_y
        
        # Ensure list format
        if not isinstance(delta_parms, list): delta_parms = [delta_parms]
        
        base_ds = self.uset[base_idx]
        targets = self._get_uset_slice(study_indices)
        
        lsuffix, rsuffix = suffixes
        
        for study_ds in targets:
            df_base = base_ds.df[[align_on] + delta_parms].sort_values(align_on)
            df_study = study_ds.df[[align_on] + delta_parms].sort_values(align_on)
            
            merged = pd.merge_asof(df_base, df_study, on=align_on, suffixes=suffixes, direction='nearest')
            
            for parm in delta_parms:
                b_col = f"{parm}{lsuffix}"
                s_col = f"{parm}{rsuffix}"
                merged[f"DL_{parm}"] = merged[s_col] - merged[b_col]
                # Avoid div by zero
                merged[f"DLPCT_{parm}"] = np.where(merged[b_col] == 0, np.nan, 100 * (merged[f"DL_{parm}"] / merged[b_col]))
            
            # Create new dataset
            new_title = f"Delta {base_ds.index}-{study_ds.index}"
            self.load_df(merged, title=new_title)
            
            # Tag it
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
            """
            Set specific axis limits for a parameter.
            
            Parameters:
            -----------
            column : str
                The name of the column (parameter) to constrain.
            range_tuple : tuple, list, or 'clear'
                The (min, max) range for the axis. 
                Pass 'clear' or None to remove the restriction.
                
            Example:
            --------
            UC.scale('pressure', (900, 1100))
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
    def set_font_sizes(self, suptitle=None, legend=None, axes_title=None, axes_tick=None):
        """Set independent font sizes for various plot elements."""
        if suptitle is not None: self.suptitle_size = suptitle
        if legend is not None: self.legend_size = legend
        if axes_title is not None: self.axes_title_size = axes_title
        if axes_tick is not None: self.axes_tick_size = axes_tick

    def _apply_fonts(self, fig):
        """Internal method to apply stored font sizes to a Plotly figure."""
        if fig is None: return fig
        
        layout_updates = {}
        if self.suptitle_size:
            layout_updates['title_font'] = dict(size=self.suptitle_size)
        if self.legend_size:
            layout_updates['legend'] = dict(font=dict(size=self.legend_size))
        
        if layout_updates:
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
            """
            Main plotting wrapper.
            
            Parameters:
            -----------
            by : str, optional
                'vars' (default) - Creates a subplot for each Y variable.
                                Best for comparing multiple datasets on specific metrics.
                'sets'           - Creates a subplot for each Dataset.
                                Best for looking at all metrics for a specific dataset.
            """
            self._clear_last_fig()

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

            # Calculate grid dimensions to know where to place lines/highlights
            y_list = y if isinstance(y, list) else [y]
            active_sets = [d for d in self.uset if d.select]
            
            if mode == 'vars':
                n_items = len(y_list)
            else:
                n_items = len(active_sets)

            # Standard grid calc used by both plotters
            if ncols is None and nrows is None:
                calc_ncols = min(3, max(1, int(np.ceil(np.sqrt(n_items)))))
            elif ncols is None:
                calc_ncols = int(np.ceil(n_items / nrows))
            else:
                calc_ncols = ncols
            calc_ncols = max(1, calc_ncols)

            # Decoration Helper
            def apply_to_all_subplots(func, **kwargs):
                for i in range(n_items):
                    r = (i // calc_ncols) + 1
                    c = (i % calc_ncols) + 1
                    func(row=r, col=c, **kwargs)

            # --- LINES ---
            for col_name, lines in self.lines.items():
                if col_name == x:
                    # Vertical lines on X axis apply to everyone
                    for l in lines:
                        fig.add_vline(x=l['level'], line_dash=l['dash'], line_color=l['color'])
                elif col_name in y_list:
                    # Horizontal lines differ based on mode
                    if mode == 'vars':
                        # Only apply to the specific subplot for this variable
                        for idx, yi in enumerate(y_list):
                            if yi == col_name:
                                r, c = (idx // calc_ncols) + 1, (idx % calc_ncols) + 1
                                for l in lines:
                                    fig.add_hline(y=l['level'], line_dash=l['dash'], line_color=l['color'], row=r, col=c)
                    else: # mode == 'sets'
                        # Variable exists on ALL subplots
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
                    else: # mode == 'sets'
                        for h in hls:
                            apply_to_all_subplots(fig.add_hrect, y0=h['range'][0], y1=h['range'][1], 
                                                fillcolor=h['color'], opacity=h['opacity'], layer="below", line_width=0)

            # X limits
            if x in self.axis_limits:
                fig.update_xaxes(range=self.axis_limits[x])

            # Y limits
            # In 'sets' mode, limits were already applied inside uniplot_per_dataset.
            if mode == 'vars':
                for idx, yi in enumerate(y_list):
                    if yi in self.axis_limits:
                        r, c = (idx // calc_ncols) + 1, (idx % calc_ncols) + 1
                        fig.update_yaxes(range=self.axis_limits[yi], row=r, col=c)

            fig = self._apply_fonts(fig)
            if fig and suppress_legends:
                fig.update_traces(visible='legendonly') 
            self.last_fig = fig
            return fig


    # ------------------------------------------------------------------
    # The bar Command
    # ------------------------------------------------------------------

    def bar(self, x=None, y=None, by='vars', barmode='group', agg='mean', figsize=(12, 8), ncols=None, nrows=None, suppress_legends=False):
            """
            Unified interface for Bar Charts.
            """
            self._clear_last_fig()

            if x is None: x = self.last_x
            if y is None: y = self.last_y
            self.last_x, self.last_y = x, y

            y_list = y if isinstance(y, list) else [y]

            # --- 1. DISPATCH BLOCK ---
            if by == 'dataset_x':
                fig = unibar_datasets_as_x(
                    list_of_datasets=self.uset, y=y_list, agg=agg,
                    suptitle=self.suptitle, figsize=figsize, 
                    darkmode=self.darkmode, axis_limits=self.axis_limits, return_axes=True
                )
                # dataset_x handles its own axis limits internally, so we return early
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
                
            # --- 2. APPLY LIMITS (Only for original vars/sets modes) ---
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

                # --- 3. FINAL POLISH ---
                fig = self._apply_fonts(fig)
                if suppress_legends:
                    fig.update_traces(visible='legendonly')
                self.last_fig = fig
                
            # Ensure we return the figure object so fig1.show() works!
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

            # --- 1. DISPATCH BLOCK ---
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
                # Resolve limits if they exist for the primary Y
                primary_y = y_list[0]
                y_limit = self.axis_limits.get(primary_y)
                fig = unibox_per_dataset(
                    list_of_datasets=self.uset, x=x, y=y, boxmode=boxmode,
                    points=points, notched=notched,
                    suptitle=suptitle or self.suptitle, figsize=figsize, ncols=ncols, nrows=nrows,
                    darkmode=self.darkmode, y_lim=y_limit, return_axes=True
                )
            else:
                # vars mode limits are handled inside unibox
                fig = unibox(
                    list_of_datasets=self.uset, x=x, y=y, boxmode=boxmode,
                    points=points, notched=notched, color=color,
                    suptitle=suptitle or self.suptitle, figsize=figsize, ncols=ncols, nrows=nrows,
                    darkmode=self.darkmode, y_lim=None, return_axes=True
                )
                
                # Apply limits for vars mode
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
                
            return fig               
        # ------------------------------------------------------------------
    # The histogram Command
    # ------------------------------------------------------------------
    def histogram(self, x=None, y=None, histfunc='sum', by='vars', nbins=None, 
                    bin_size=None, bin_start=None, bin_end=None,
                    histnorm='', barmode='overlay', opacity=0.7,
                    color=None, suptitle=None, figsize=(12, 8), ncols=None, nrows=None, suppress_legends=False):
            """
            Unified interface for Histograms.
            
            Parameters:
            -----------
            x : str or list
                The column(s) to distribute (the bins).
            y : str, optional
                The column to use for weighting the histogram (e.g., time spent).
            histfunc : str, optional
                The aggregation function for 'y' (default is 'sum'). Can be 'count', 'sum', 'avg', 'min', 'max'.
            nbins : int, optional
                Target number of bins. Plotly will attempt to create approximately this many.
            bin_size : float, optional
                Forces the exact width of each bin.
            bin_start : float, optional
                Forces the starting value of the first bin.
            bin_end : float, optional
                Forces the ending value of the last bin.
            by : str
                'vars' (default) - Subplots by variable (compare datasets).
                'sets'           - Subplots by dataset (compare variables).
            histnorm : str
                '' (default, count), 'percent', 'probability', 'density', 'probability density'
            """
            self._clear_last_fig()

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
            return fig
        
    def contour(self, x=None, y=None, z=None, by='vars', contours_coloring='fill', 
                colorscale=None, interpolate=True, interp_res=100, interp_method='linear',
                ncontours=None,
                suptitle=None, figsize=(12, 8), ncols=None, nrows=None, suppress_legends=False):
            """

            Unified interface for Contour Plots.
            
            Parameters:
            -----------
            x : str
                The X-axis variable (e.g., 'Mach')
            y : str
                The Y-axis variable (e.g., 'Altitude')
            z : str or list
                The variable(s) to color the contour by (e.g., 'T4F')
            by : str
                'vars' (default) - Subplots by Z-variable.
                'sets'           - Subplots by dataset.
            interpolate : bool
                If True, uses scipy.griddata to mesh scattered x/y points.
            interp_res : int
                The resolution of the interpolation grid (default 100x100).
            suptitle : str, optional
                A title for the entire figure.
            """

            self._clear_last_fig()

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
                
            return fig

    # ------------------------------------------------------------------
    # The table Command
    # ------------------------------------------------------------------
    def table(self, cols=None, title=None):
        """
        Display a Plotly Table of specific columns from selected datasets.
        
        Parameters:
        -----------
        cols : str or list, optional
            The columns to display.
            If None, defaults to [self.last_x] + [self.last_y].
        title : str, optional
            Title for the table layout.
        """
        # 1. Resolve Columns
        if cols is None:
            if self.last_x is None or self.last_y is None:
                print("No columns specified and no previous plot variables defined.")
                return
            
            # Construct list from last_x and last_y (which might be a list or str)
            y_part = self.last_y if isinstance(self.last_y, list) else [self.last_y]
            target_cols = [self.last_x] + y_part
        else:
            target_cols = cols if isinstance(cols, list) else [cols]

        # 2. Aggregate Data from Selected Datasets
        combined_dfs = []
        
        for ds in self.uset:
            if not ds.select:
                continue
            
            # Filter for existing columns only
            valid_cols = [c for c in target_cols if c in ds.df.columns]
            
            if not valid_cols:
                continue
                
            subset = ds.df[valid_cols]
            # Add a source column to identify the dataset
            subset.insert(0, 'Dataset', ds.title)
            combined_dfs.append(subset)

        if not combined_dfs:
            print("No data found for the specified columns in selected datasets.")
            return

        final_df = pd.concat(combined_dfs, ignore_index=True)
        
        # Fill NaNs for display purposes (optional, but looks better in tables)
        final_df = final_df.fillna('-')

        # 3. Define Styling based on Darkmode
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

        # 4. Create Plotly Table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(final_df.columns),
                fill_color=header_color,
                align='left',
                font=dict(color=font_color, size=12, weight='bold'),
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

        # Layout updates
        layout_args = {
            'title': {'text': title or "Data Table", 'x': 0.5},
            'template': "plotly_dark" if self.darkmode else "plotly_white",
            'margin': dict(l=20, r=20, t=50, b=20),
        }
        fig.update_layout(**layout_args)

    def save_png(self, filename="plot.png", scale=3, width=None, height=None):
        """
        Save the last generated plot to a PNG file.
        Requires the 'kaleido' package to be installed.
        """
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
        """
        Print a formatted table of all loaded datasets, including:
        - Set index and title
        - Whether selected (✓/✗)
        - DataFrame shape (rows, columns)
        - Whether a query is applied (if non-None and non-empty)

        The table is printed using standard console formatting (monospace-friendly).
        """
        if not self.uset:
            print("No datasets loaded.")
            return

        # Prepare rows for tabular output
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

        # Column names and widths
        headers = ["Set", "Title", "Selected", "Shape", "Query?" ]
        # Calculate column widths (with padding)
        col_widths = [
            max(len(row[i]) for row in [headers] + rows) + 2
            for i in range(len(headers))
        ]

        # Format header and separator
        header_str = "".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        sep = "-" * sum(col_widths)

        # Build output
        output_lines = [header_str, sep]
        for row in rows:
            line = "".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row))
            output_lines.append(line)

        print("\nLoaded Datasets:")
        print("\n".join(output_lines))

    def list_parms(self, set_number=None, search_string=None, use_regex=False):
            """
            List the parameters (columns) available in the loaded datasets.
            Includes descriptions if available in self.parm_description_dict.
            
            Parameters:
            -----------
            set_number : int, list, 'all', or None, optional
                The specific dataset index to inspect.
            search_string : str, optional
                A substring or wildcard to filter the parameter names.
            use_regex : bool, optional
                If True, treats the search_string as a strict regular expression.
                
            Returns:
            --------
            list
                A sorted list of matching parameter (column) names.
            """
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
                
            # --- MODIFIED OUTPUT LOGIC ---
            for col in filtered_cols:
                desc = self.parm_description_dict.get(col, "No description available.")
                print(f"  - {str(col).ljust(25)} : {desc}")
                
            return filtered_cols

    def summary(self, cols=None):
            """
            Print a formatted statistical summary table for specific columns
            across all selected datasets, noting any applied queries.

            Parameters:
            -----------
            cols : str or list, optional
                The columns to summarize. If None, defaults to the last plotted x and y variables.
            """
            # 1. Resolve Columns
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

            # 2. Gather Data
            rows = []
            for ds in active_ds:
                # Format the query string for display
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

            # 3. Format and Print Table
            headers = ["Set", "Title", "Query", "Variable", "Count", "Min", "Mean", "Max", "Std"]
            
            # Calculate column widths with 2 spaces padding
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
        Display help information for the UnichartNotebook class:
        - Class docstring
        - Public methods and their signatures/descriptions
        - Public attributes (non-callable properties) with their types and values (if not too long)
        - Example usage tips
        """
        print("=" * 70)
        print("📚 UnichartNotebook HELP")
        print("=" * 70)
        
        # 1. Class docstring
        cls = self.__class__
        if cls.__doc__:
            print("\n📋 CLASS DESCRIPTION:")
            print(cls.__doc__)
        else:
            print("\n⚠️  No class docstring found.")

        # 2. Public methods (exclude dunder & private unless overridden)
        methods = inspect.getmembers(cls, predicate=inspect.isfunction)
        public_methods = [m for m in methods if not m[0].startswith('_') or m[0] in ['__init__', '__repr__']]

        print("\n🔍 PUBLIC METHODS:")
        print("-" * 70)
        for name, func in public_methods:
            sig = inspect.signature(func)
            doc = inspect.getdoc(func) or "No description available."
            # Shorten doc to first line for brevity + show full in next block if needed
            doc_preview = doc.split('\n')[0] if doc else ""
            print(f"• {name}{sig}")
            print(f"  → {doc_preview}")
            print()
        
        # 3. Public attributes (instance variables set in __init__)
        print("🛠️  PUBLIC ATTRIBUTES (Instance):")
        print("-" * 70)
        attrs = []
        for attr, value in self.__dict__.items():
            if not attr.startswith('_'):  # skip private/internal
                attrs.append((attr, value))

        if not attrs:
            print("No public instance attributes found.")
        else:
            for name, val in sorted(attrs):
                val_str = str(val)[:100]  # truncate long reprs
                if len(str(val)) > 100:
                    val_str += "..."
                print(f"• {name}: {type(val).__name__} = {val_str}")

        # 4. Tips / Quick Start
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
    def _clear_last_fig(self):
        """
        Actively hollows out the massive JSON payload of the previous Plotly figure 
        before dropping the reference. This prevents rapid RAM inflation (high-water marks) 
        during tight plotting loops.
        """
        if self.last_fig is not None:
            # Empty the heavy data payloads
            self.last_fig.data = []
            self.last_fig.layout = {}
            # Drop the pointer
            self.last_fig = None

    def _refresh_widgets(self):
        """
        Rebuilds the dataset management widget list, cleanly destroying 
        old widgets to prevent memory leaks in the kernel and the browser.
        """
        
        # 1. THE PURGE: Cleanly destroy existing widgets
        # Simply unassigning them from .children leaves them in memory.
        if hasattr(self, 'dataset_widget_container') and self.dataset_widget_container.children:
            for child in self.dataset_widget_container.children:
                # HBox is a container. We must dig down and close its children first.
                if hasattr(child, 'children'):
                    for sub_child in child.children:
                        sub_child.close()
                # Close the parent HBox/HTML widget
                child.close()
                
        items = []
        
        # Header
        items.append(widgets.HTML("<b>Dataset Manager</b>"))
        
        # 2. DECOUPLED CALLBACKS: Define these outside the loop
        # This prevents creating a new function object in memory for every dataset.
        def update_select(change, dataset):
            dataset.select = change['new']

        def update_color(change, dataset):
            dataset.color = change['new']

        # 3. WIDGET REBUILD
        for i, ds in enumerate(self.uset):
            
            # Checkbox for Selection
            chk = widgets.Checkbox(
                value=ds.select, 
                description=f"{i}: {ds.title}", 
                indent=False, 
                layout=widgets.Layout(width='300px')
            )
            
            # functools.partial safely binds the specific dataset to the callback 
            # without capturing the entire local scope in a messy closure.
            chk.observe(functools.partial(update_select, dataset=ds), names='value')
            
            # Metadata display
            details = widgets.Label(
                value=f"[Rows: {len(ds.df)}] [Query: {ds.query}]", 
                layout=widgets.Layout(width='200px')
            )
            
            # Color Picker
            # Fixed a bug from the original code: it was hardcoded to 'blue'. 
            # Now it dynamically pulls the actual dataset color.
            current_color = ds.color if isinstance(ds.color, str) else 'blue'
            cp = widgets.ColorPicker(
                concise=True, 
                value=current_color, 
                layout=widgets.Layout(width='30px')
            )
            
            cp.observe(functools.partial(update_color, dataset=ds), names='value')

            # Group and append
            row = widgets.HBox([chk, cp, details])
            items.append(row)
            
        # Assign the fresh batch of widgets to the container
        self.dataset_widget_container.children = tuple(items)
        

    def gui(self):
        """Displays the interactive dataset manager widgets."""
        self._refresh_widgets()
        
        btn_refresh = widgets.Button(description="Refresh Data View")
        btn_refresh.on_click(lambda b: self._refresh_widgets())
        
        display(widgets.VBox([btn_refresh, self.dataset_widget_container]))