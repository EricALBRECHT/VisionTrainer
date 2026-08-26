from __future__ import annotations

from pathlib import Path


def file_size(path: Path) -> int:
    try:
        if path.is_file():
            return int(path.stat().st_size)
    except OSError:
        return 0
    return 0


def directory_size(path: Path) -> int:
    """Recursive size of a directory; missing paths return 0."""
    try:
        if not path.exists():
            return 0
        if path.is_file():
            return file_size(path)
    except OSError:
        return 0

    total = 0
    try:
        for child in path.rglob("*"):
            try:
                if child.is_file():
                    total += int(child.stat().st_size)
            except OSError:
                continue
    except OSError:
        return total
    return total


def path_size(path: Path) -> int:
    try:
        if path.is_dir():
            return directory_size(path)
        if path.is_file():
            return file_size(path)
    except OSError:
        return 0
    return 0


def format_bytes(num_bytes: int | float | None) -> str:
    if num_bytes is None:
        return "—"
    value = float(max(0, num_bytes))
    units = ("o", "Ko", "Mo", "Go", "To")
    index = 0
    while value >= 1024.0 and index < len(units) - 1:
        value /= 1024.0
        index += 1
    if index == 0:
        return f"{int(value)} {units[index]}"
    return f"{value:.2f} {units[index]}".replace(".", ",")
