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
    markers = list(MARKER_MAP_MPL_TO_PLOTLY.keys())
    return markers[index % len(markers)]

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
        self.hue_palette = "Viridis" # Default Plotly colorscale
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
            markersize=10, marker_edge_color="black", linestyle=None, hue_palette="Viridis",
            hue_order=None, line=False, suppress_msg=False, return_axes=False, axes=None,
            suptitle=None, xlabel=None, ylabel=None, subplot_titles=None,
            darkmode=False, interactive=True, display_parms=None, grid=True,
            legend='above', legend_ncols=1, figsize=(12, 8), ncols=None, nrows=None, x_lim=None, y_lim=None):
    """
    Create a unified plot for a list of datasets using Plotly.
    Updated: Subplot titles default to None; Main title is centered.
    Includes legendgroup fix to toggle traces across subplots.
    """
    from plotly.subplots import make_subplots
    
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
        # 'xaxis_title': xlabel if xlabel else x, # REMOVED: Applied in loop below instead
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

            ht = f"<b>{cur_title}</b><br>{x_col}: %{{customdata[{get_cd_idx(x_col)}]:.2f}}<br>{yi}: %{{customdata[{get_cd_idx(yi)}]:.2f}}"
            for parm in hover_cols:
                # Check if the column is numeric to apply 5 sig fig formatting
                if pd.api.types.is_numeric_dtype(df[parm]):
                    ht += f"<br>{parm}: %{{customdata[{get_cd_idx(parm)}]:.5g}}"
                else:
                    ht += f"<br>{parm}: %{{customdata[{get_cd_idx(parm)}]}}"
            ht += "<extra></extra>"

            mode_parts = []
            if not cur_linestyle: mode_parts.append('markers')
            else: mode_parts.append('lines')
            if cur_marker: mode_parts.append('markers')
            mode = "+".join(mode_parts)

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
                    marker_dict['colorscale'] = fmt.get('hue_palette', 'Viridis')
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

            if isinstance(cur_reg_order, numbers.Number) and cur_reg_order > 0:
                rx, ry = _calculate_regression(df, x_col, yi, cur_reg_order)
                fig.add_trace(go.Scatter(
                    x=rx, y=ry, mode='lines',
                    name=f"{cur_idx}: {cur_title} Fit LS {cur_reg_order}",
                    legendgroup=f"group_{cur_idx}",
                    line=dict(color=cur_color, width=cur_linewidth, dash='dash'),
                    opacity=0.7, hoverinfo='skip', showlegend=False
                ), row=row, col=col)

    # --- AXIS CONFIGURATION (FIXED) ---
    for idx_y, yi in enumerate(y_list):
        row = idx_y // ncols + 1
        col = idx_y % ncols + 1
        
        axis_title = ylabel if ylabel else yi
        # NEW: Explicitly define x-axis title for this subplot
        x_axis_title = xlabel if xlabel else x
        
        fig.update_yaxes(title_text=axis_title, title_standoff=15, row=row, col=col)
        # NEW: Added title_text=x_axis_title to update_xaxes
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
    - Supports proper line widths and dash styles.
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
    
    # We DO NOT set a huge right margin here anymore; we handle spacing via domain shrinking.
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=sp_titles, shared_xaxes=False)

    layout_args = {
        'template': "plotly_dark" if darkmode else "plotly_white",
        'title': {'text': suptitle if suptitle else f"Dataset Comparison", 'x': 0.5},
        'showlegend': True,
        # Standard margin is usually fine now, or slight increase
        'margin': dict(r=50) 
    }
    if figsize:
        layout_args['width'] = figsize[0] * 100
        layout_args['height'] = figsize[1] * 100
        
    fig.update_layout(**layout_args)
    color_cycle = px.colors.qualitative.Plotly
    
    # Counter for generating new axis IDs (starts after the grid axes)
    next_free_axis_idx = (nrows * ncols) + 1

    for idx_ds, dataset in enumerate(active_datasets):
        row = (idx_ds // ncols) + 1
        col = (idx_ds % ncols) + 1
        
        # Resolve Axis Names
        subplot_index = (row - 1) * ncols + col
        base_x_name = "xaxis" if subplot_index == 1 else f"xaxis{subplot_index}"
        base_y_name = "yaxis" if subplot_index == 1 else f"yaxis{subplot_index}"
        
        # --- DOMAIN MANAGEMENT ---
        # 1. Get original domain (default [0,1] if not found)
        # make_subplots pre-populates these, so we can read them.
        try:
            old_domain = fig.layout[base_x_name].domain
            if not old_domain: old_domain = [0, 1]
        except:
            old_domain = [0, 1]
            
        d_start, d_end = old_domain
        
        # 2. Calculate space needed for EXTRA axes (vars 3, 4, etc.)
        # Variable 1 is Left. Variable 2 is Right (anchored to X). 
        # Variable 3+ need their own space.
        extras_count = max(0, len(y_list) - 2)
        
        # We reserve roughly 0.08 (8% of width) per extra axis
        # You can tweak 'width_per_axis' if labels are getting cut off
        width_per_axis = 0.08 
        required_space = extras_count * width_per_axis
        
        # 3. Shrink the X-axis to make room
        new_d_end = d_end - required_space
        # Safety check: don't invert axis if too many vars
        if new_d_end <= d_start + 0.1: new_d_end = d_end 
        
        # Apply new domain to the subplot's X-axis
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

        for idx_y, yi in enumerate(y_list):
            if yi not in df.columns: continue
            
            var_color = color_cycle[idx_y % len(color_cycle)]
            custom_data = df[[x_col, yi] + [c for c in (display_parms or []) if c in df.columns]]
            ht = f"<b>{dataset.title}</b><br>{x_col}: %{{customdata[0]:.2f}}<br>{yi}: %{{customdata[1]:.2f}}"
            specific_limit = axis_limits.get(yi, None)

            # Styling
            line_dict = dict(color=var_color, width=dataset.linewidth)
            if dataset.linestyle: line_dict['dash'] = get_plotly_linestyle(dataset.linestyle)

            if idx_y == 0:
                # --- PRIMARY AXIS (LEFT) ---
                fig.add_trace(go.Scatter(
                    x=df[x_col], y=df[yi],
                    mode='lines+markers' if dataset.linestyle else 'markers',
                    name=yi, legendgroup=yi, showlegend=(idx_ds == 0),
                    marker=dict(color=var_color, size=dataset.markersize or 6),
                    line=line_dict,
                    customdata=custom_data, hovertemplate=ht
                ), row=row, col=col)
                
                # Update base Y-axis
                u_dict = dict(title_text=yi, title_font=dict(color=var_color), tickfont=dict(color=var_color), row=row, col=col)
                if specific_limit: u_dict['range'] = specific_limit
                fig.update_yaxes(**u_dict)

            elif idx_y == 1:
                # --- SECONDARY AXIS (Standard Right) ---
                # This one is "free" (doesn't need custom positioning), 
                # it just anchors to the (now shrunk) X-axis and sits on the right edge.
                
                # We need a new axis object, but we anchor it to the subplot's x
                curr_y_axis = f"yaxis{next_free_axis_idx}"
                curr_y_ref = f"y{next_free_axis_idx}"
                next_free_axis_idx += 1
                
                fig.add_trace(go.Scatter(
                    x=df[x_col], y=df[yi],
                    mode='lines+markers' if dataset.linestyle else 'markers',
                    name=yi, legendgroup=yi, showlegend=(idx_ds == 0),
                    marker=dict(color=var_color, size=dataset.markersize or 6),
                    line=line_dict,
                    customdata=custom_data, hovertemplate=ht,
                    xaxis=base_x_name.replace("axis", ""), # e.g. "x2"
                    yaxis=curr_y_ref
                ))
                
                layout_axis = dict(
                    title=yi,
                    title_font=dict(color=var_color),
                    tickfont=dict(color=var_color),
                    anchor=base_x_name.replace("axis", ""), # Anchors to x, so sits at x-domain end
                    overlaying=base_y_name.replace("axis", ""), # y, y2, etc
                    side='right',
                    showgrid=False
                )
                if specific_limit: layout_axis['range'] = specific_limit
                fig.update_layout({curr_y_axis: layout_axis})
                
            else:
                # --- TERTIARY+ AXES (Floating Right) ---
                # These sit in the empty space we created by shrinking the domain.
                
                curr_y_axis = f"yaxis{next_free_axis_idx}"
                curr_y_ref = f"y{next_free_axis_idx}"
                next_free_axis_idx += 1
                
                # We place them iteratively
                extra_idx = idx_y - 1 # 1st extra, 2nd extra...
                pos = new_d_end + (extra_idx * width_per_axis)
                # Cap at 1.0 just in case
                pos = min(0.99, pos) 
                
                fig.add_trace(go.Scatter(
                    x=df[x_col], y=df[yi],
                    mode='lines+markers' if dataset.linestyle else 'markers',
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
                    anchor="free",          # Free floating
                    overlaying=base_y_name.replace("axis", ""),
                    side='right',
                    position=pos,           # Explicit position in [0, 1]
                    showgrid=False
                )
                if specific_limit: layout_axis['range'] = specific_limit
                fig.update_layout({curr_y_axis: layout_axis})

        # Final cleanup for the subplot
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
        df = ds.df.copy()
        
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


def unihistogram(list_of_datasets, x, nbins=None, histnorm='', barmode='overlay', opacity=0.7,
                 color=None, suptitle=None, subplot_titles=None, darkmode=False, 
                 figsize=(12, 8), ncols=None, nrows=None, x_lim=None, return_axes=False):
    """
    Create a unified histogram for a list of datasets.
    Subplots are organized by Variable (x).
    
    Parameters:
    - color: str, optional (Overrides dataset colors if provided)
    """
    x_list = x if isinstance(x, list) else [x]
    n_x = len(x_list)
    
    # --- Grid Setup ---
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

    for ds in list_of_datasets:
        if not ds.select: continue
        df = ds.df.copy()
        
        # Use explicit color arg if provided, otherwise use the Dataset's class color
        use_color = color if color else ds.color

        for idx_x, xi in enumerate(x_list):
            row, col = (idx_x // ncols) + 1, (idx_x % ncols) + 1
            
            if xi not in df.columns: continue

            clean_data = df[xi].dropna()

            fig.add_trace(go.Histogram(
                x=clean_data,
                name=f"{ds.index}: {ds.title}",
                legendgroup=f"group_{ds.index}",
                marker_color=use_color,
                opacity=opacity,
                nbinsx=nbins,
                histnorm=histnorm,
                showlegend=(idx_x == 0)
            ), row=row, col=col)

    # Final Formatting
    if x_lim: fig.update_xaxes(range=x_lim)
    
    y_label = "Density" if "density" in histnorm else "Count"
    fig.update_yaxes(title_text=y_label)

    if return_axes: return fig
    fig.show()
    return fig


def unihistogram_by_dataset(list_of_datasets, x, nbins=None, histnorm='', barmode='overlay', opacity=0.7,
                            color=None, suptitle=None, figsize=(12, 8), ncols=None, nrows=None, 
                            darkmode=False, x_lim=None, return_axes=False):
    """
    Create a unified histogram where Subplots are organized by Dataset.
    
    Color Logic:
    - If plotting 1 variable: Uses the Dataset's unique color.
    - If plotting >1 variable: Uses a color cycle (to distinguish variables), 
      unless 'color' override is provided.
    """
    active_ds = [d for d in list_of_datasets if d.select]
    x_list = x if isinstance(x, list) else [x]
    n_sets = len(active_ds)

    if not active_ds:
        print("No datasets selected.")
        return None

    # --- Grid Setup ---
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

    for idx_ds, ds in enumerate(active_ds):
        row, col = (idx_ds // ncols) + 1, (idx_ds % ncols) + 1
        df = ds.df.copy()
        
        for idx_x, xi in enumerate(x_list):
            if xi not in df.columns: continue
            
            clean_data = df[xi].dropna()
            
            # Color Logic:
            # 1. Global override 'color'
            # 2. If single variable: use Dataset color (ds.color)
            # 3. If multi variable: use Cycle (to distinguish x1 vs x2)
            if color:
                use_color = color
            elif len(x_list) == 1:
                use_color = ds.color
            else:
                use_color = color_cycle[idx_x % len(color_cycle)]

            fig.add_trace(go.Histogram(
                x=clean_data,
                name=xi,
                legendgroup=xi,
                marker_color=use_color,
                opacity=opacity,
                nbinsx=nbins,
                histnorm=histnorm,
                showlegend=(idx_ds == 0)
            ), row=row, col=col)

    if x_lim: fig.update_xaxes(range=x_lim)
    
    y_label = "Density" if "density" in histnorm else "Count"
    fig.update_yaxes(title_text=y_label)

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
        self.darkmode = True 
        self.last_ncols = None
        self.last_nrows = None
        self.last_fig = None # New: Store last figure for export

        self.suptitle = None
        
        # Plot Decorations
        self.plot_title = None
        self.x_label = None
        self.y_label = None
        
        self.display_parms = []
        self.axis_limits = {} 
        self.lines = {}       
        self.highlights = {}  
        
        print("UniChart Notebook Environment Initialized.")

    # ------------------------------------------------------------------
    # Data Management
    # ------------------------------------------------------------------
    def load_df(self, df, title=None, set_column=None, load_cols_as_vars=False):
        """
        Load a DataFrame into the environment as datasets.
        """
        # Auto-title logic
        if not title:
            if set_column and set_column in df.columns:
                pass # Title will be derived from group
            elif "TITLE" in df.columns:
                set_column = "TITLE"
            else:
                df["TITLE"] = "Dataset"
                set_column = "TITLE"
        
        next_index = len(self.uset)
        
        # Group and create Dataset objects
        if set_column and set_column in df.columns:
            for group_name, df_subset in df.groupby(set_column):
                # Handle cases where title might be implied
                actual_title = title if title else str(group_name)
                
                ds = Dataset(df_subset.copy(), index=next_index, title=actual_title)
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
        for ds in self.uset: ds.select = False # Exclusive select logic from original
        for ds in self._get_uset_slice(uset_slice):
            ds.select = True
        self._refresh_widgets()

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

    def hue(self, uset_slice, col_name):
        for ds in self._get_uset_slice(uset_slice):
            ds.hue = col_name

    def alpha(self, uset_slice, alpha_val):
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
    # Decorations (Lines/Highlights)
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

    # ------------------------------------------------------------------
    # Decorations (Lines/Highlights)
    # ------------------------------------------------------------------
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
    # The Plot Command
    # ------------------------------------------------------------------
    def plot(self, x=None, y=None, by='vars', figsize=(12, 8), ncols=None, nrows=None, 
                subplot_titles=None, suptitle=None, **kwargs):
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
            # 1. State Management & Defaults
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

            # 2. Dispatch Logic
            if by == 'sets' or by == 'datasets':
                # --- ROUTE TO DATASET GRID ---
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
                    axis_limits=self.axis_limits,  # <--- NEW ARGUMENT
                    return_axes=True 
                )
                # Set flag for decoration logic
                mode = 'sets'
                
            else:
                # --- ROUTE TO VARIABLE GRID (Original) ---
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
                    'return_axes': True, # Important: get figure back instead of showing immediately
                    'figsize': figsize,
                    'ncols': ncols,
                    'nrows': nrows,
                }
                plot_args.update(kwargs)
                fig = uniplot(**plot_args)
                mode = 'vars'

            if fig is None: return

            # 3. Apply Decorations (Unified Logic)
            # We need to calculate grid dimensions to know where to place lines/highlights
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

            # --- LIMITS (UPDATED) ---
            # X limits
            if x in self.axis_limits:
                fig.update_xaxes(range=self.axis_limits[x])

            # Y limits
            # CHANGE: Only apply post-hoc limits if we are in 'vars' mode. 
            # In 'sets' mode, limits were already applied inside uniplot_per_dataset.
            if mode == 'vars':
                for idx, yi in enumerate(y_list):
                    if yi in self.axis_limits:
                        r, c = (idx // calc_ncols) + 1, (idx % calc_ncols) + 1
                        fig.update_yaxes(range=self.axis_limits[yi], row=r, col=c)

            self.last_fig = fig
            return fig


    # ------------------------------------------------------------------
    # The bar Command
    # ------------------------------------------------------------------

    def bar(self, x=None, y=None, by='vars', barmode='group', figsize=(12, 8), ncols=None, nrows=None):
        """
        Unified interface for Bar Charts.
        """
        if x is None: x = self.last_x
        if y is None: y = self.last_y
        self.last_x, self.last_y = x, y

        if by in ['sets', 'datasets']:
            return unibar_per_dataset(
                list_of_datasets=self.uset, x=x, y=y, barmode=barmode,
                suptitle=self.suptitle, figsize=figsize, ncols=ncols, nrows=nrows,
                darkmode=self.darkmode, y_lim=self.axis_limits.get(y[0] if isinstance(y, list) else y)
            )
        else:
            return unibar(
                list_of_datasets=self.uset, x=x, y=y, barmode=barmode,
                suptitle=self.suptitle, figsize=figsize, ncols=ncols, nrows=nrows,
                darkmode=self.darkmode
            )

    # ------------------------------------------------------------------
    # The box Command
    # ------------------------------------------------------------------

    def box(self, x=None, y=None, by='vars', boxmode='group', points='outliers', notched=False, 
                color=None, suptitle=None, figsize=(12, 8), ncols=None, nrows=None):
            """
            Unified interface for Box Plots.
            
            Parameters:
            -----------
            by : str
                'vars' (default) - Subplots by variable (compare datasets side-by-side).
                'sets' - Subplots by dataset (compare variables side-by-side).
            points : str
                'outliers' (default), 'all', 'suspectedoutliers', or False.
            notched : bool
                Whether to draw a notched boxplot (useful for rough confidence interval comparison).
            color : str, optional
                Override color for all boxes (only applies when by='vars').
            """
            if x is None: x = self.last_x
            if y is None: y = self.last_y
            self.last_x, self.last_y = x, y

            # Resolve limits if they exist for the primary Y
            primary_y = y[0] if isinstance(y, list) else y
            y_limit = self.axis_limits.get(primary_y)

            if by in ['sets', 'datasets']:
                self.last_fig = unibox_per_dataset(
                    list_of_datasets=self.uset, x=x, y=y, boxmode=boxmode,
                    points=points, notched=notched,
                    suptitle=suptitle or self.suptitle, figsize=figsize, ncols=ncols, nrows=nrows,
                    darkmode=self.darkmode, y_lim=y_limit, return_axes=True
                )
            else:
                self.last_fig = unibox(
                    list_of_datasets=self.uset, x=x, y=y, boxmode=boxmode,
                    points=points, notched=notched,
                    color=color, # <--- Added this passthrough
                    suptitle=suptitle or self.suptitle, figsize=figsize, ncols=ncols, nrows=nrows,
                    darkmode=self.darkmode, y_lim=y_limit, return_axes=True
                )
                
            if self.last_fig:
                self.last_fig.show()
    # ------------------------------------------------------------------
    # The histogram Command
    # ------------------------------------------------------------------
    def histogram(self, x=None, by='vars', nbins=None, histnorm='', barmode='overlay', opacity=0.7,
                  color=None, suptitle=None, figsize=(12, 8), ncols=None, nrows=None):
        """
        Unified interface for Histograms.
        
        Parameters:
        -----------
        x : str or list
            The column(s) to distribute.
        by : str
            'vars' (default) - Subplots by variable (compare datasets).
                               Uses Dataset colors.
            'sets'           - Subplots by dataset (compare variables).
                               Uses Dataset color (if 1 var) or Color Cycle (if >1 var).
        histnorm : str
            '' (default, count), 'percent', 'probability', 'density', 'probability density'
        color : str, optional
            Override the color for all traces.
        """
        if x is None: x = self.last_x
        self.last_x = x
        
        # Resolve limit
        limit = None
        if isinstance(x, str):
            limit = self.axis_limits.get(x)
        elif isinstance(x, list) and len(x) == 1:
            limit = self.axis_limits.get(x[0])

        if by in ['sets', 'datasets']:
            self.last_fig = unihistogram_by_dataset(
                list_of_datasets=self.uset, x=x, nbins=nbins, histnorm=histnorm,
                barmode=barmode, opacity=opacity, color=color,
                suptitle=suptitle or self.suptitle, figsize=figsize, ncols=ncols, nrows=nrows,
                darkmode=self.darkmode, x_lim=limit, return_axes=True
            )
        else:
            self.last_fig = unihistogram(
                list_of_datasets=self.uset, x=x, nbins=nbins, histnorm=histnorm,
                barmode=barmode, opacity=opacity, color=color,
                suptitle=suptitle or self.suptitle, figsize=figsize, ncols=ncols, nrows=nrows,
                darkmode=self.darkmode, x_lim=limit, return_axes=True
            )
            
        if self.last_fig:
            self.last_fig.show()
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
                
            subset = ds.df[valid_cols].copy()
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
        
        fig.show()

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
    def _refresh_widgets(self):
        """Rebuilds the dataset management widget list."""
        items = []
        
        # Header
        items.append(widgets.HTML("<b>Dataset Manager</b>"))
        
        for i, ds in enumerate(self.uset):
            # Checkbox for Selection
            chk = widgets.Checkbox(value=ds.select, description=f"{i}: {ds.title}", indent=False, layout=widgets.Layout(width='300px'))
            
            # Observe changes to update dataset object immediately
            def on_change(change, dataset=ds):
                dataset.select = change['new']
            chk.observe(on_change, names='value')
            
            # Metadata display
            details = widgets.Label(value=f"[Rows: {len(ds.df)}] [Query: {ds.query}]", layout=widgets.Layout(width='200px'))
            
            # Color Picker
            # Try to convert mpl color/plotly color to hex for widget
            # Simple fallback to white if unknown format
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