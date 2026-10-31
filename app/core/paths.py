"""Resolve configured paths independently of the process working directory."""
from pathlib import Path
import os
import shutil
import sys

BACKEND_ROOT = Path(__file__).resolve().parents[2]


def backend_path(value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = BACKEND_ROOT / path
    return os.path.abspath(path)


def python_executable(value: str = "") -> str:
    """Accept an interpreter path, a command on PATH, or the current Python."""
    value = value.strip()
    if not value:
        return sys.executable
    if Path(value).is_absolute() or "/" in value or "\\" in value:
        return backend_path(value)
    return shutil.which(value) or value
