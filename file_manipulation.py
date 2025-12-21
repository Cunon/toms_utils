import os
import platform
import shutil
import glob
import re
import time
import difflib
import inspect
from pathlib import Path

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

def head(path, lines=10):
    """
    Print the first N lines of a file.
    
    Args:
        path (str): Path to the file.
        lines (int): Number of lines to print (default: 10).
    """
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f, 1):
                if i > lines:
                    break
                print(line, end='')
    except Exception as e:
        print(f"head error: {e}")

def tail(path, lines=10):
    """
    Print the last N lines of a file.
    
    Args:
        path (str): Path to the file.
        lines (int): Number of lines to print (default: 10).
    """
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            # Read all lines and get the last N
            all_lines = f.readlines()
            start_index = max(0, len(all_lines) - lines)
            for line in all_lines[start_index:]:
                print(line, end='')
    except Exception as e:
        print(f"tail error: {e}")


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

def mkdir(path, parents=False, exist_ok=True, *args, **kwargs):
    """
    Path(path).mkdir wrapper
    
    Args:
        path (str or Path): Path of the directory to create.
        parents (bool): If True, create parent directories if they don't exist.
        exist_ok (bool): If True, do not raise an error if the directory already exists.
    """
    try:
        Path(path).mkdir(parents=parents, exist_ok=exist_ok, *args, **kwargs)
    except Exception as e:
        print(f"mkdir error: {e}")

import stat
import time
from pathlib import Path


def ls(
    path=".",
    pattern="*",
    show_hidden=False,
    long_format=False,
    recursive=False,
    absolute_paths=False,
    sort="name",           # "name" | "size" | "mtime" | "type"
    reverse=False,
    dirs_first=False,
    human_readable=True,
):
    """
    List files matching a glob pattern with optional flags.

    Args:
        path (str | Path): Base directory OR a glob pattern (if it contains wildcards).
        pattern (str): Glob pattern within `path` directory (ignored if `path` contains wildcards).
        show_hidden (bool): If True, include dotfiles and files under hidden directories.
        long_format (bool): If True, print permissions, size, and mtime.
        recursive (bool): If True, recurse into subdirectories.
        absolute_paths (bool): If True, print/return resolved absolute paths.
        sort (str): Sorting key: "name", "size", "mtime", "type".
        reverse (bool): Reverse sort order.
        dirs_first (bool): If True, list directories before files.
        human_readable (bool): If True, show sizes as B/KB/MB/... in long format.

    Returns:
        list[Path]: Paths in the order printed.
    """
    def _has_wildcards(s: str) -> bool:
        return any(ch in s for ch in ("*", "?", "["))

    def _is_hidden(p: Path, base: Path) -> bool:
        # Hidden if any path component in the relative path starts with '.'
        try:
            rel = p.relative_to(base)
            parts = rel.parts
        except Exception:
            parts = p.parts
        return any(part.startswith(".") for part in parts)

    def _fmt_size(n: int) -> str:
        if not human_readable:
            return str(n)
        val = float(n)
        for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
            if val < 1024.0:
                return f"{val:>7.2f}{unit}"
            val /= 1024.0
        return f"{val:>7.2f}EB"

    def _fmt_mtime(ts: float) -> str:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))

    path_str = str(path)
    base = Path(".")
    items: list[Path] = []

    try:
        # If `path` itself is a glob, treat it as the full pattern.
        if _has_wildcards(path_str):
            candidates = list(base.glob(path_str))
            base_for_hidden = base
        else:
            base = Path(path_str)
            base_for_hidden = base

            if base.is_file():
                candidates = [base]
            elif base.is_dir():
                candidates = list(base.rglob(pattern)) if recursive else list(base.glob(pattern))
            else:
                print(f"ls: {path}: No such file or directory")
                return []

        for p in candidates:
            # Skip hidden (including anything under a hidden dir) unless requested
            if not show_hidden and _is_hidden(p, base_for_hidden):
                continue
            items.append(p)

        # Sorting
        def sort_key(p: Path):
            try:
                st = p.stat()
            except Exception:
                st = None

            if sort == "size":
                key = st.st_size if (st and p.is_file()) else -1
            elif sort == "mtime":
                key = st.st_mtime if st else 0
            elif sort == "type":
                # dirs first if dirs_first else natural; "type" groups dirs/files
                key = (0 if p.is_dir() else 1, p.name.lower())
            else:  # "name"
                key = p.name.lower()

            if dirs_first and sort != "type":
                return (0 if p.is_dir() else 1, key)
            return key

        items.sort(key=sort_key, reverse=reverse)

        # Print
        for p in items:
            display = p.resolve() if absolute_paths else p
            if long_format:
                try:
                    st = p.stat()
                    perms = stat.filemode(st.st_mode)
                    size = _fmt_size(st.st_size) if p.is_file() else " " * 9
                    mtime = _fmt_mtime(st.st_mtime)
                except Exception:
                    perms = "??????????"
                    size = " " * 9
                    mtime = "????-??-?? ??:??"

                suffix = "/" if p.is_dir() else ""
                print(f"{perms} {size} {mtime} {display}{suffix}")
            else:
                suffix = "/" if p.is_dir() else ""
                print(f"{display}{suffix}")

        return items

    except PermissionError:
        print(f"ls: Permission denied: {path}")
        return []
    except Exception as e:
        print(f"ls error: {e}")
        return []

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

def sed(path, find, replace, recursive=False, in_place=True, backup_extension=None, flags=0):
    """
    Find and replace text in files.
    
    Args:
        find (str): Regex find to find.
        replace (str): Text to replace matches with.
        path (str or Path): File, directory, or glob find to process.
        recursive (bool): If True, process directories recursively.
        in_place (bool): If True, modify the file. If False, print to stdout.
        backup_extension (str): If provided, create a backup (e.g., '.bak').
        flags (int): Regex flags.
        
    Returns:
        Union[list, dict]: 
            If in_place=True: List of Path objects for files that were modified.
            If in_place=False: Dictionary {Path: str} of files and their potential new content.
    """
    path_obj = Path(path)
    files_to_process = []
    
    # Return structures
    modified_files = [] 
    preview_results = {}

    # 1. Resolve files
    if path_obj.is_file():
        files_to_process = [path_obj]
    elif path_obj.is_dir():
        if recursive:
            files_to_process = [p for p in path_obj.rglob("*") if p.is_file()]
        else:
            files_to_process = [p for p in path_obj.glob("*") if p.is_file()]
    else:
        files_to_process = list(Path(".").glob(str(path)))
        if not files_to_process:
            print(f"sed: {path}: No such file or directory")
            return [] if in_place else {}

    # 2. Compile Regex
    try:
        regex = re.compile(find, flags)
    except re.error as e:
        print(f"sed: Invalid regular expression: {e}")
        return [] if in_place else {}

    # 3. Process Files
    for file_path in files_to_process:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            new_content = regex.sub(replace, content)

            # Skip if no changes found
            if content == new_content:
                continue

            if in_place:
                # Backup
                if backup_extension:
                    backup_path = file_path.with_suffix(file_path.suffix + backup_extension)
                    shutil.copy2(file_path, backup_path)
                    print(f"Backup created: {backup_path}")

                # Write
                with open(file_path, 'w', encoding='utf-8', newline='') as f:
                    f.write(new_content)
                print(f"Modified: {file_path}")
                modified_files.append(file_path)
            else:
                # Dry run output
                print(f"--- Output for {file_path} ---")
                print(new_content)
                print(f"--- End of {file_path} ---\n")
                preview_results[file_path] = new_content

        except PermissionError:
            print(f"sed: {file_path}: Permission denied")
        except UnicodeDecodeError:
            print(f"sed: {file_path}: Binary or unreadable file skipped")
        except Exception as e:
            print(f"sed: Error processing {file_path}: {e}")
            
    return modified_files if in_place else preview_results

def grep(pattern, path=".", recursive=False, ignore_case=True, line_numbers=True, 
         invert_match=False, count=True, show_filename=True):
    """
    Search for a regex pattern within files.
    
    Args:
        pattern (str): The regex pattern to search for.
        path (str or Path): The directory or file to search in.
        recursive (bool): If True, search directories recursively.
        ignore_case (bool): If True, perform case-insensitive matching.
        line_numbers (bool): If True, include line numbers in output AND return dictionary.
        invert_match (bool): If True, select lines that do NOT match the pattern.
        count (bool): If True, print only the count of matching lines per file.
        show_filename (bool): If True, prefix the output with the filename.
        
    Returns:
        list: A list of dicts. 
              Always contains: {'file': Path, 'count': int}
              Conditionally contains: 'lines': [int, int...] (only if line_numbers=True)
    """
    path_obj = Path(path)
    files_to_search = []
    results = []

    # 1. Resolve files
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

    # 2. Compile Regex
    flags = re.IGNORECASE if ignore_case else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        print(f"grep: Invalid regular expression: {e}")
        return []

    # 3. Search Files
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
                
                # If matches were found in this file, add to results
                if matching_line_numbers:
                    # Base dictionary with always-present keys
                    file_result = {
                        'file': file_path,
                        'count': len(matching_line_numbers)
                    }
                    
                    # Conditionally add the 'lines' key
                    if line_numbers:
                        file_result['lines'] = matching_line_numbers
                        
                    results.append(file_result)
                    
                    # Console output for 'count' mode
                    if count:
                        print(f"{file_path}:{len(matching_line_numbers)}")

        except PermissionError:
            print(f"grep: {file_path}: Permission denied")
        except Exception:
            pass
            
    return results

def find(pattern=None, path=".", file_type=None, absolute_paths=False, max_level=None):
    """
    Find files and directories using glob patterns with Unix-like options.
    
    Args:
        path (str): Directory to search in.
        pattern (str): Glob pattern to match filenames.
        file_type (str): Type of files to find ('f' for files, 'd' for directories).
        absolute_paths (bool): If True, return and print absolute (resolved) paths.
        max_level (int): Maximum depth to search. 1 = immediate children only.
                         None = infinite recursion.
    Returns:
        list: List of found items (Path objects; resolved if absolute_paths=True).
    """
    search_path = Path(path)
    glob_pattern = pattern if pattern else "*"
    
    # Start with a generator from rglob
    items = search_path.rglob(glob_pattern)

    # Filter by max_level if specified
    if max_level is not None:
        items = (
            item for item in items 
            if len(item.relative_to(search_path).parts) <= max_level
        )

    if file_type == 'f':
        items = [item for item in items if item.is_file()]
    elif file_type == 'd':
        items = [item for item in items if item.is_dir()]
    else:
        # If no file_type specified, ensure we consume the generator into a list
        items = list(items)
    
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

# ==========================================
#  NEW Additions
# ==========================================

def touch(path, exist_ok=True):
    """
    Create an empty file or update the timestamp of an existing file.
    
    Args:
        path (str): Path to the file.
        exist_ok (bool): If True, update timestamp if file exists. 
                         If False, raise error if file exists.
    """
    try:
        p = Path(path)
        if p.exists():
            if exist_ok:
                current_time = time.time()
                os.utime(p, (current_time, current_time))
                print(f"Updated timestamp: {p}")
            else:
                print(f"touch: file {p} already exists")
        else:
            p.touch()
            print(f"Created file: {p}")
    except Exception as e:
        print(f"touch error: {e}")

def cat(path, show_line_numbers=False):
    """
    Print the contents of a file to stdout.
    
    Args:
        path (str): Path to the file.
        show_line_numbers (bool): Prefix lines with numbers.
    """
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f, 1):
                prefix = f"{i:4} | " if show_line_numbers else ""
                print(f"{prefix}{line}", end='')
            print() # Ensure newline at end
    except Exception as e:
        print(f"cat error: {e}")

def du(path=".", human_readable=True):
    """
    Calculate disk usage of a directory recursively.
    
    Args:
        path (str): Directory or file path.
        human_readable (bool): Return string (e.g. '5 MB') instead of bytes.
    """
    path_obj = Path(path)
    total = 0
    try:
        if path_obj.is_file():
            total = path_obj.stat().st_size
        else:
            for item in path_obj.rglob('*'):
                if item.is_file():
                    total += item.stat().st_size
        
        if human_readable:
            for unit in ['B', 'KB', 'MB', 'GB']:
                if total < 1024:
                    res = f"{total:.2f} {unit}"
                    print(res)
                    return res
                total /= 1024
            print(f"{total:.2f} TB")
        else:
            print(total)
            return total
    except Exception as e:
        print(f"du error: {e}")

def archive(source_path, output_name, format="zip"):
    """
    Compress a directory into an archive file.
    
    Args:
        source_path (str): Directory to compress.
        output_name (str): Output filename (without extension).
        format (str): 'zip', 'tar', 'gztar', 'bztar', or 'xztar'.
    """
    try:
        shutil.make_archive(output_name, format, source_path)
        print(f"Archived '{source_path}' -> '{output_name}.{format}'")
    except Exception as e:
        print(f"archive error: {e}")

def extract(archive_path, extract_dir="."):
    """
    Extract an archive file.
    
    Args:
        archive_path (str): Path to the zip/tar file.
        extract_dir (str): Directory to extract into.
    """
    try:
        shutil.unpack_archive(archive_path, extract_dir)
        print(f"Extracted '{archive_path}' -> '{extract_dir}'")
    except Exception as e:
        print(f"extract error: {e}")

def diff(file1, file2, context_lines=3):
    """
    Compare two files line by line (Unified Diff).
    
    Args:
        file1 (str): Path to first file.
        file2 (str): Path to second file.
        context_lines (int): Number of context lines to show around changes.
    """
    try:
        f1 = Path(file1).read_text(encoding='utf-8', errors='ignore').splitlines()
        f2 = Path(file2).read_text(encoding='utf-8', errors='ignore').splitlines()
        
        diff_gen = difflib.unified_diff(
            f1, f2,
            fromfile=file1, tofile=file2,
            lineterm="", n=context_lines
        )
        
        diffs = list(diff_gen)
        if not diffs:
            print("Files are identical.")
        else:
            for line in diffs:
                print(line)
    except Exception as e:
        print(f"diff error: {e}")

# ==========================================
#  Help System
# ==========================================

def manual(command=None):
    """
    Show the toolbox manual.
    
    Usage:
        manual()          -> Lists all available commands.
        manual('grep')    -> Shows detailed help for 'grep'.
        manual(grep)      -> Same as above.
    """
    # Helper to gather all functions defined in this module
    # We filter out imports and internal variables
    all_globals = globals().copy()
    toolbox_funcs = {
        name: func for name, func in all_globals.items()
        if inspect.isfunction(func) and func.__module__ == __name__ and name != 'manual'
    }
    
    if command is None:
        print(f"{'COMMAND':<15} | {'DESCRIPTION'}")
        print("-" * 60)
        for name in sorted(toolbox_funcs):
            func = toolbox_funcs[name]
            # Get the first line of the docstring
            doc = inspect.getdoc(func) or "No description."
            summary = doc.split('\n')[0]
            print(f"{name:<15} | {summary}")
        print("-" * 60)
        print("Type manual('command_name') for details.")
        
    else:
        # Resolve command if it's a string
        target_func = None
        if isinstance(command, str):
            target_func = toolbox_funcs.get(command)
        elif hasattr(command, '__name__'):
            target_func = command
            
        if target_func:
            print(f"--- Manual: {target_func.__name__} ---")
            print(inspect.getdoc(target_func))
            print("-------------------------------------")
        else:
            print(f"Unknown command: {command}")