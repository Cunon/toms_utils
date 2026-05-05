#!/bin/bash

# Resolve the directory this script lives in (handles symlinks)
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"

# Repo-relative paths — these travel with the script
template_notebook="$SCRIPT_DIR/template_notebook.ipynb"
workspace_file="$SCRIPT_DIR/linux_profile.code-workspace"

# Load machine-specific overrides if present (gitignored)
local_config="$SCRIPT_DIR/.local_config"
[[ -f "$local_config" ]] && source "$local_config"

# Sensible defaults; override per-machine via .local_config
default_dir="${NOTEBOOK_DIR:-$HOME/Documents/python_notebooks}"
vscode_cmd="${VSCODE_CMD:-code}"

# --- everything below is your original logic, lightly tweaked ---
date_str=$(date +"%y%m%d")
working_dir="$default_dir"
[[ "$PWD" != "$HOME" ]] && working_dir="$PWD"

base_name="${date_str}_New_Notebook.ipynb"
filename="$default_dir/$base_name"
counter=1
while [[ -f "$filename" ]]; do
    filename="${default_dir}/${date_str}_New_Notebook_${counter}.ipynb"
    ((counter++))
done

mkdir -p "$default_dir"
cp "$template_notebook" "$filename"
sed -i "s|%DATE%|$date_str|g" "$filename"
sed -i "s|%WORKING_DIRECTORY%|$working_dir|g" "$filename"

$vscode_cmd "$workspace_file" &
sleep 1
$vscode_cmd "$filename" &

exit 0
