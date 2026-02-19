import pandas as pd
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import os

def open_datafile(initial_dir=None, force_format=None, add_file_info=False, **kwargs):
    """
    Opens a system file dialog to select a dataset and loads it into a Pandas DataFrame.
    
    Parameters:
    - initial_dir (str): Directory to start the search in.
    - force_format (str): Force a specific read method (e.g., 'csv') regardless of extension.
    - add_file_info (bool): If True, adds 'FILENAME' and 'FILEPATH' columns to the DataFrame.
    - **kwargs: Arbitrary keyword arguments passed directly to the pandas read function.
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    file_path = filedialog.askopenfilename(
        initialdir=initial_dir or os.getcwd(),
        title="Select Data File",
        filetypes=[
            ("Data Files", "*.csv *.txt *.dat *.xlsx *.xls *.parquet *.json *.pkl"),
            ("All Files", "*.*")
        ]
    )
    
    root.destroy()
    
    if not file_path:
        print("Selection cancelled.")
        return None

    # Normalization logic
    if force_format:
        load_type = force_format.lower().strip('.')
    else:
        _, ext = os.path.splitext(file_path)
        load_type = ext.lower().strip('.')

    try:
        df = None
        
        # Load logic
        if load_type in ['csv', 'txt', 'dat']:
            df = pd.read_csv(file_path, **kwargs)
            print(f"pd.read_csv(r'{str(file_path)}')")
        elif load_type in ['xlsx', 'xls', 'excel']:
            df = pd.read_excel(file_path, **kwargs)
            print(f"pd.read_excel(r'{str(file_path)}', **kwargs)")
        elif load_type == 'parquet':
            df = pd.read_parquet(file_path, **kwargs)
            print(f"pd.read_parquet(r'{str(file_path)}', **kwargs)")
        elif load_type == 'json':
            df = pd.read_json(file_path, **kwargs)
            print(f"pd.read_json(r'{str(file_path)}', **kwargs)")
        elif load_type in ['pkl', 'pickle']:
            df = pd.read_pickle(file_path, **kwargs)
            print(f"pd.read_pickle(r'{str(file_path)}', **kwargs)")
        else:
            print(f"Error: Format '{load_type}' not supported.")
            return None

        # Add Metadata columns if requested
        if add_file_info and df is not None:
            df['FILENAME'] = os.path.basename(file_path)
            df['FILEPATH'] = os.path.abspath(file_path)
            
        return df

    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def keep_columns(df, columns):
    """
    Returns a dataframe with only the specified columns if they exist.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to select
    
    Returns:
        DataFrame with only existing columns from the input list
    """
    existing_columns = [col for col in columns if col in df.columns]
    return df[existing_columns]

def align_dataframes(
    df1,
    df2,
    match_columns,
    column_mode="original",
    fill_method="inner",
    interpolate_kwargs=None,
):
    """
    Align two DataFrames on the given key columns so the result DataFrames have
    identical row order/length and can be compared elementwise.

    Args:
        df1, df2: DataFrames to align.
        match_columns: list of column names present in both frames, used as keys.
        column_mode: how to align columns after row alignment:
            - "original" (default): keep each frame's original columns.
            - "intersection": keep only columns present in both frames.
            - "union": keep union of columns; missing columns are added and filled with NA.
        fill_method: strategy for aligning rows:
            - "inner" (default):
                Keep only rows whose keys appear in both df1 and df2.
                This is effectively an inner join on `match_columns` (plus a
                per-key row counter to handle duplicates deterministically).
            - "interpolate":
                Use the union of keys from df1 and df2 instead of an inner join.
                Each frame is reindexed onto the union grid and then interpolated
                separately to fill numeric gaps created by the outer alignment.
                Requires that `match_columns` uniquely identify rows in each frame
                (no duplicates).
        interpolate_kwargs: dict of kwargs passed to DataFrame.interpolate
            when fill_method == "interpolate". Ignored for fill_method == "inner".

    Returns:
        (aligned_left, aligned_right): two DataFrames with matching row order/length.

        Each returned DataFrame also contains a `_interpolated` column with:
            - 0 for rows that existed in the original DataFrame.
            - 1 for rows that were introduced by the union-of-keys grid for that
              particular frame (i.e., interpolated rows when fill_method == "interpolate").
    """
    if not match_columns:
        raise ValueError("match_columns must be a non-empty list")

    missing = [c for c in match_columns if c not in df1.columns or c not in df2.columns]
    if missing:
        raise ValueError(f"Columns not in both dataframes: {missing}")

    if column_mode not in {"original", "intersection", "union"}:
        raise ValueError('column_mode must be "original", "intersection", or "union"')

    if fill_method not in {"inner", "interpolate"}:
        raise ValueError('fill_method must be "inner" or "interpolate"')

    # -------------------------------------------------------------------------
    # Row alignment
    # -------------------------------------------------------------------------
    if fill_method == "inner":
        # Original behavior: keep only rows that exist in both (inner join).
        # Use a per-key row_id so duplicates are paired deterministically.
        left = df1.reset_index().rename(columns={"index": "_left_index"})
        right = df2.reset_index().rename(columns={"index": "_right_index"})

        left["_row_id"] = left.groupby(match_columns).cumcount()
        right["_row_id"] = right.groupby(match_columns).cumcount()

        merged = left.merge(
            right,
            on=match_columns + ["_row_id"],
            how="outer",
            indicator=True,
            suffixes=("_left", "_right"),
        )

        keep = merged[merged["_merge"] == "both"]

        aligned_left = df1.loc[keep["_left_index"]].reset_index(drop=True)
        aligned_right = df2.loc[keep["_right_index"]].reset_index(drop=True)

        # No synthetic rows in the inner-join case
        aligned_left["_interpolated"] = 0
        aligned_right["_interpolated"] = 0

    else:  # fill_method == "interpolate"
        if df1.duplicated(subset=match_columns).any():
            dup = df1[df1.duplicated(subset=match_columns, keep=False)][match_columns]
            raise ValueError(
                "fill_method='interpolate' requires df1 to have unique keys in "
                f"match_columns. Found duplicates:\n{dup}"
            )

        if df2.duplicated(subset=match_columns).any():
            dup = df2[df2.duplicated(subset=match_columns, keep=False)][match_columns]
            raise ValueError(
                "fill_method='interpolate' requires df2 to have unique keys in "
                f"match_columns. Found duplicates:\n{dup}"
            )

        # Use match_columns as index to build the union grid
        left_idx = df1.set_index(match_columns)
        right_idx = df2.set_index(match_columns)

        union_index = left_idx.index.union(right_idx.index)

        # Try to sort for a sensible interpolation order; fall back if not sortable.
        try:
            union_index = union_index.sort_values()
        except Exception:
            pass

        # Boolean masks: which keys existed in the original frames?
        left_original_mask = union_index.isin(left_idx.index)
        right_original_mask = union_index.isin(right_idx.index)

        aligned_left = left_idx.reindex(union_index)
        aligned_right = right_idx.reindex(union_index)

        kwargs = interpolate_kwargs or {}

        # Interpolate numeric columns; non-numeric cols keep NaNs where no data.
        aligned_left = aligned_left.interpolate(**kwargs)
        aligned_right = aligned_right.interpolate(**kwargs)

        # Mark rows that were not present in the original frame as interpolated.
        aligned_left["_interpolated"] = (~left_original_mask).astype(int)
        aligned_right["_interpolated"] = (~right_original_mask).astype(int)

        # Bring key columns back out of the index.
        aligned_left = aligned_left.reset_index()
        aligned_right = aligned_right.reset_index()

    # -------------------------------------------------------------------------
    # Column alignment
    # -------------------------------------------------------------------------
    if column_mode == "intersection":
        cols = [c for c in aligned_left.columns if c in aligned_right.columns]
        aligned_left = aligned_left[cols]
        aligned_right = aligned_right[cols]

    elif column_mode == "union":
        cols = list(dict.fromkeys(list(aligned_left.columns) + list(aligned_right.columns)))
        for col in cols:
            if col not in aligned_left.columns:
                aligned_left[col] = pd.NA
            if col not in aligned_right.columns:
                aligned_right[col] = pd.NA
        aligned_left = aligned_left[cols]
        aligned_right = aligned_right[cols]

    return aligned_left, aligned_right
