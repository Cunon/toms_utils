#!/bin/bash

# Get the current date in YYMMDD format
date_str=$(date +"%y%m%d")

# Get the current notebook name
current_notebook=/home/tom/Documents/GitHub/toms_utils/template_notebook.ipynb

# Set the target directory
default_dir="/home/tom/Documents/misc_projects/python_notebooks"
workspace_file="/home/tom/Documents/GitHub/toms_utils/linux_profile.code-workspace"

working_dir=$default_dir

# Check if the current directory is not the home directory
if [[ "$PWD" != "$HOME" ]]; then
    working_dir="$PWD"; 
fi

# Generate initial filename
base_name="${date_str}_New_Notebook.ipynb"
filename="$default_dir/$base_name"

# Check if file exists and increment number until we find an available name
counter=1
while [[ -f "$filename" ]]; do
    filename="${default_dir}/${date_str}_New_Notebook_${counter}.ipynb"
    ((counter++))
done

# Create the target directory if it doesn't exist
mkdir -p "$default_dir"

# Copy and rename the notebook
cp "$current_notebook" "$filename"

# Replace %DATE% in the notebook with the current date
sed -i "s|%DATE%|$date_str|g" "$filename"
sed -i "s|%WORKING_DIRECTORY%|$working_dir|g" "$filename"

# 1. Use '&' to run these in the background so the script doesn't wait for you to close VS Code
# 2. We launch the workspace first, then the file
flatpak run com.visualstudio.code "$workspace_file" & 

# Small sleep to ensure the workspace loads before the file command hits
sleep 1 

flatpak run com.visualstudio.code "$filename" &

# 3. Exit the script naturally. 
# If you are running this from a terminal, the terminal will return to the prompt.
# If you are running this via a Desktop Shortcut/Launcher, the window will close.
exit 0
