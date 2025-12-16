import os
import platform
from pathlib import Path
import re
import glob
import shutil

def open_directory(input_dir=None):
    """
    Open the parent directory of a file in the default file explorer/browser.
    If no path is provided, opens the current working directory.
    
    Args:
        input_dir (str or Path, optional): Path to the file whose parent directory 
                                              should be opened. If None, opens current working directory.
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if input_dir is None:
            parent_dir = Path.cwd()
        else:
            filepath = Path(input_dir)
            parent_dir = filepath.parent
        
        system = platform.system()
        if system == "Windows":
            os.startfile(parent_dir)
        elif system == "Darwin":  # macOS
            os.system(f'open "{parent_dir}"')
        else:  # Linux 
            os.system(f'xdg-open "{parent_dir}"')
        return True
        
    except Exception as e:
        print(f"Error opening directory: {e}")
        return False

def pwd():
    """
    os.getcwd()
    """
    return os.getcwd()

def cd(path, *args, **kwargs):
    """
    os.chdir(path)
    """
    os.chdir(path, *args, **kwargs)

def mkdir(path, parents=False, exist_ok=False, *args, **kwargs):
    """
    Path(path).mkdir wrapper
    
    Args:
        path (str or Path): Path of the directory to create.
        parents (bool): If True, create parent directories if they don't exist.
        exist_ok (bool): If True, do not raise an error if the directory already exists.
    """
    Path(path).mkdir(parents=parents, exist_ok=exist_ok, *args, **kwargs)

def ls(pattern="*", show_hidden=False, long_format=False, recursive=False):
    """
    List files matching a glob pattern with optional flags.
    
    Args:
        pattern (str): Glob pattern to match files.
        show_hidden (bool): If True, show hidden files and directories.
        long_format (bool): If True, display detailed information.
        recursive (bool): If True, list subdirectories recursively.
    """
    def list_items(current_path, level=0):
        returned_items = None
        try:
            items = Path(current_path).glob(pattern)
            if not show_hidden:
                items = [item for item in items if not item.name.startswith('.')]
            
            for item in sorted(items):
                if long_format:
                    stat = item.stat()
                    print(f"{stat.st_mode} {stat.st_size} {item.name}")
                else:
                    print(item.name)
                
                if recursive and item.is_dir():
                    print(f"{item}/")
                    list_items(str(item) + "/**/*", level + 1)
            returned_items = list(items)
        except PermissionError:
            print(f"Permission denied: {current_path}")
    
    returned_items = list_items(".")
    return returned_items


def tree(path=None, level=0, max_level=2, show_hidden=False, show_dirs_only=False, show_full_path=False,
         pattern=None, show_permissions=False, show_sizes=True):
    """
    Display the directory structure in a tree-like format.
    Args:
        path (str or Path): Root directory to display.
        level (int): Current depth level for indentation.
        max_level (int): Maximum recursion depth.
        show_hidden (bool): Whether to show hidden files/directories.
        show_dirs_only (bool): Show only directories.
        show_full_path (bool): Show full paths instead of relative paths.
        pattern (str or None): Regex pattern to filter files by name.
        show_permissions (bool): Show file permissions.
        show_sizes (bool): Show the size of each file and directory.
    """
    if path is None:
        path = Path.cwd()
    else:
        path = Path(path)
        
    if not path.exists():
        print(f"Path {path} does not exist.")
        return
        
    if max_level is not None and level > max_level:
        return
      
    try:
        entries = sorted(path.iterdir(), key=lambda x: x.name.lower())
    except PermissionError:
        print(f"Permission denied: {path}")
        return
      
    for i, entry in enumerate(entries):
        if not show_hidden and entry.name.startswith('.'):
            continue
            
        # Handle pattern matching properly
        if pattern is not None:
            # If pattern looks like a glob pattern (contains * or ? or []), convert it to regex
            if '*' in pattern or '?' in pattern or '[' in pattern:
                regex_pattern = pattern.replace('.', r'\.').replace('*', '.*').replace('?', '.')
                # Ensure we match the entire filename
                if not re.match(f"^{regex_pattern}$", entry.name):
                    continue
            else:
                # Treat as regular regex
                if not re.match(pattern, entry.name):
                    continue
      
        is_last = (i == len(entries) - 1)
      
        if level > 0:
            prefix = "│   " * (level - 1) + ("└── " if is_last else "├── ")
        else:
            prefix = ""
      
        if show_full_path:
            display_name = str(entry)
        else:
            display_name = entry.name
      
        # Add permissions and sizes if requested
        extra_info = ""
        if show_permissions and entry.is_file():
            perm_str = oct(entry.stat().st_mode)[-3:]
            extra_info += f" {perm_str}"
        if show_sizes and entry.is_file():
            size_str = f" ({entry.stat().st_size} bytes)"
            extra_info += size_str
      
        print(prefix + display_name + extra_info)
      
        # Recursively process directories
        if entry.is_dir() and not show_dirs_only:
            new_level = level + 1
            tree(entry, new_level, max_level, show_hidden, show_dirs_only, show_full_path,
                 pattern, show_permissions, show_sizes)

import re

def grep(pattern, path=".", recursive=False, ignore_case=True, line_numbers=True, 
         invert_match=False, count=True, show_filename=True):
    """
    Search for a regex pattern within files.
    
    Args:
        pattern (str): The regex pattern to search for.
        path (str or Path): The directory or file to search in.
        recursive (bool): If True, search directories recursively.
        ignore_case (bool): If True, perform case-insensitive matching.
        line_numbers (bool): If True, include line numbers in console output.
        invert_match (bool): If True, select lines that do NOT match the pattern.
        count (bool): If True, print only the count of matching lines per file to console.
        show_filename (bool): If True, prefix the console output with the filename.
        
    Returns:
        list: A list of dicts: [{'file': Path, 'count': int, 'lines': [int, int, ...]}]
    """
    path_obj = Path(path)
    files_to_search = []
    results = []

    if path_obj.is_file():
        files_to_search = [path_obj]
    elif path_obj.is_dir():
        if recursive:
            files_to_search = [p for p in path_obj.rglob("*") if p.is_file()]
        else:
            files_to_search = [p for p in path_obj.glob("*") if p.is_file()]
    else:
        files_to_search = list(Path(".").glob(str(path)))
        if not files_to_search:
            print(f"grep: {path}: No such file or directory")
            return []

    flags = re.IGNORECASE if ignore_case else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        print(f"grep: Invalid regular expression: {e}")
        return []

    for file_path in files_to_search:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                matching_line_numbers = []
                
                for i, line in enumerate(f, 1):
                    line_content = line.rstrip('\n')
                    match = regex.search(line_content)
                    
                    # Handle Invert Match logic
                    is_hit = not match if invert_match else bool(match)

                    if is_hit:
                        matching_line_numbers.append(i)
                        
                        # Handle Console Output
                        if not count:
                            output_parts = []
                            if show_filename:
                                output_parts.append(f"{file_path}")
                            if line_numbers:
                                output_parts.append(f"{i}")
                            output_parts.append(line_content)
                            print(":".join(output_parts))
                
                if matching_line_numbers:
                    results.append({
                        'file': file_path,
                        'count': len(matching_line_numbers),
                        'lines': matching_line_numbers
                    })
                    
                    if count:
                        print(f"{file_path}:{len(matching_line_numbers)}")

        except PermissionError:
            print(f"grep: {file_path}: Permission denied")
        except Exception:
            pass
            
    return results

def find(pattern=None, path=".", file_type=None, absolute_paths=False):
    """
    Find files and directories using glob patterns with Unix-like options.
    
    Args:
        path (str): Directory to search in.
        pattern (str): Glob pattern to match filenames.
        file_type (str): Type of files to find ('f' for files, 'd' for directories).
        absolute_paths (bool): If True, return and print absolute (resolved) paths.
    Returns:
        list: List of found items (Path objects; resolved if absolute_paths=True).
    """
    search_path = Path(path)
    glob_pattern = pattern if pattern else "*"
    
    items = list(search_path.rglob(glob_pattern))

    if file_type == 'f':
        items = [item for item in items if item.is_file()]
    elif file_type == 'd':
        items = [item for item in items if item.is_dir()]
    
    if absolute_paths:
        items = [item.resolve() for item in items]
    
    for item in sorted(items, key=lambda p: str(p).lower()):
        print(item)
        
    return list(items)


def rm(path, recursive=False, force=False, verbose=True):
    """
    Remove files or directories.
    
    Args:
        path (str or Path): Path to remove.
        recursive (bool): If True, remove directories and their contents recursively.
        force (bool): If True, ignore non-existent files and never prompt.
        verbose (bool): If True, print information about each operation.
    """
    def remove_item(item_path):
        try:
            path_obj = Path(item_path)
            
            if not path_obj.exists():
                if not force:
                    print(f"rm: cannot remove '{item_path}': No such file or directory")
                return
            
            if path_obj.is_dir():
                if recursive:
                    if verbose:
                        print(f"rm -r {item_path}")
                    shutil.rmtree(path_obj)
                else:
                    if verbose:
                        print(f"rm: cannot remove '{item_path}': Is a directory")
            else:
                if verbose:
                    print(f"rm {item_path}")
                os.remove(path_obj)
                
        except PermissionError:
            print(f"rm: cannot remove '{item_path}': Permission denied")
        except Exception as e:
            print(f"rm: cannot remove '{item_path}': {e}")
    
    # Handle glob patterns
    if "*" in str(path) or "?" in str(path) or "[" in str(path):
        for item in Path().glob(path):
            remove_item(item)
    else:
        remove_item(path)

def mv(src, dst):
    """
    Wrapper for shutil.move
    
    Args:
        src (str or Path): Source path.
        dst (str or Path): Destination path.
    """
    try:
        shutil.move(str(src), str(dst))
    except Exception as e:
        print(f"Error moving {src} to {dst}: {e}")

def cp(src, dst, recursive=False):
    """
    Wrapped for shutil.copy2
    
    Args:
        src (str or Path): Source path.
        dst (str or Path): Destination path.
        recursive (bool): If True, copy directories recursively.
    """
    try:
        src_path = Path(src)
        dst_path = Path(dst)
        
        if src_path.is_dir():
            if recursive:
                shutil.copytree(src_path, dst_path)
            else:
                print(f"cp: -r not specified; omitting directory '{src}'")
        else:
            shutil.copy2(src_path, dst_path)
    except Exception as e:
        print(f"Error copying {src} to {dst}: {e}")