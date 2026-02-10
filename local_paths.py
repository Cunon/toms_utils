import os
from pathlib import Path

class LocalPaths:
    """Store and provide access to local paths for this project."""
    
    def __init__(self):
        """Initialize paths relative to this file's location."""
        self.file_dir = Path(__file__).resolve().parent
        self.project_root = self.file_dir
    
    @property
    def local_dir(self) -> Path:
        """Get the directory containing this file."""
        return self.file_dir
    
    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        return self._project_root
    
    @project_root.setter
    def project_root(self, value: Path) -> None:
        """Set the project root directory."""
        self._project_root = Path(value).resolve()
    
    def get_path(self, relative_path: str) -> Path:
        """
        Get an absolute path from a relative path.
        
        Args:
            relative_path: Path relative to project root
            
        Returns:
            Absolute Path object
        """
        return (self.project_root / relative_path).resolve()
    
    def __repr__(self) -> str:
        return f"LocalPaths(root={self.project_root})"


# Create a singleton instance for easy importing
paths = LocalPaths()