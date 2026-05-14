# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Personal Python utility library for Jupyter notebook-based data analysis workflows. The library is imported directly into notebooks — there is no package installation step.

## Creating New Notebooks

```bash
./new_py.sh
```

Prompts for an optional description, creates a dated `.ipynb` from `template_notebook.ipynb` with substituted placeholders (`%DATE%`, `%DESCRIPTION%`, `%WORKING_DIRECTORY%`, `%GITHUB_DIR%`), and opens VS Code. Requires `.local_config` for machine-specific paths (gitignored).

## Machine-Specific Config

`.local_config` (gitignored, created per-machine) overrides defaults used by `new_py.sh`:

```bash
NOTEBOOK_DIR="/path/to/notebooks"
GITHUB_DIR="/path/to/GitHub"
VSCODE_CMD="code"   # or e.g. "flatpak run com.visualstudio.code"
```

## Module Architecture

### `file_manipulation.py`
Unix-like command wrappers (`ls`, `find`, `grep`, `sed`, `cp`, `mv`, `rm`, `du`, `tree`, etc.) implemented in Python for use in notebooks. `ls()` supports sorting, long format, and filtering. `grep()` does regex search across files. `sed()` does in-place regex replacement with optional backup. All path arguments accept glob patterns.

### `dataframe_manipulation.py`
- `open_datafile()` — GUI file picker that loads CSV/Excel/Parquet/JSON/Pickle into a DataFrame, with optional filename/filepath metadata columns.
- `align_dataframes()` — Aligns two DataFrames on key columns with three fill strategies (`inner`, `union`, `interpolate`) and three column modes (`original`, `intersection`, `union`). Adds an `_interpolated` boolean column when interpolation is used.

### `unichart.py`
Large (~3500 line) Plotly-based interactive visualization framework for Jupyter notebooks. Two core classes:

- **`Dataset`** — Wraps a DataFrame with visual attributes (color, marker, linestyle, opacity, query filter, etc.)
- **`UnichartNotebook`** — Orchestrates multiple `Dataset` objects; renders interactive Plotly figures with ipywidgets controls.

Plot types: `uniplot` (scatter), `unibar`, `unibox`, `unihistogram`, `unicontour`. Each has a `_per_dataset` variant that creates subplots. `uniplot_ymultaxis` supports per-variable y-axis formatting. Regression/trendline types: linear, polynomial, log, exponential, power, LOWESS, spline.

Dataset selection throughout uses `_get_uset_slice()`, which normalizes int/str/list/`Dataset` into a consistent slice — use this pattern when adding new methods.

### `whisper_functions.py`
Wraps OpenAI Whisper for batch audio/video transcription. `transcribe_media()` accepts a file or directory and writes `.json` + `.txt` output alongside each source file. Whisper is not in `requirements.txt` and must be installed separately.

## Dependencies

Managed via `requirements.txt`. Key packages: `pandas`, `numpy`, `scipy`, `plotly`, `ipywidgets`, `matplotlib`. Install with:

```bash
pip install -r requirements.txt
```

## No Build, Test, or Lint System

There is no test suite, build pipeline, or linting configuration. `setup.py` is empty.
