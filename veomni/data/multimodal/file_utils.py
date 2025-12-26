import os
from typing import Optional


def resolve_relative_path(file_path: str, train_path: Optional[str] = None) -> str:
    """
    Resolve relative file path relative to train_path directory.

    Args:
        file_path: File path (can be absolute, relative, or URL)
        train_path: Path to training data file or directory. If None, returns original path.

    Returns:
        Resolved absolute path if file_path is relative and train_path is provided,
        otherwise returns original file_path.
    """
    # Skip resolution for URLs or absolute paths
    if file_path.startswith(("http://", "https://")) or os.path.isabs(file_path):
        return file_path

    # Resolve relative path relative to train_path directory
    if train_path:
        # Get the directory containing train_path (could be a file or directory)
        if os.path.isfile(train_path):
            train_dir = os.path.dirname(train_path)
        else:
            train_dir = train_path
        # Resolve relative path
        return os.path.join(train_dir, file_path)

    return file_path

