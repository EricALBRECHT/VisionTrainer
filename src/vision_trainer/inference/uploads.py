from __future__ import annotations

import re
import secrets
from pathlib import Path, PurePosixPath

SUPPORTED_UPLOAD_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


class UnsafeUploadNameError(ValueError):
    """Raised when an upload name cannot be safely mapped inside a temp directory."""


def display_upload_name(original_name: str) -> str:
    """Return a basename suitable for UI / download naming (never used as a path)."""
    name = PurePosixPath(str(original_name).replace("\\", "/")).name
    return name or "image.jpg"


def safe_internal_upload_path(destination_dir: Path, original_name: str) -> Path:
    """
    Build a path inside ``destination_dir`` using a generated internal filename.

    The user-provided name is never joined into the filesystem path.
    """
    destination_dir = destination_dir.resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)

    display = display_upload_name(original_name)
    suffix = Path(display).suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
        suffix = ".jpg"

    internal_name = f"upload_{secrets.token_hex(8)}{suffix}"
    if re.search(r"[\\/]", internal_name) or ".." in internal_name:
        raise UnsafeUploadNameError("Nom interne invalide.")

    target = (destination_dir / internal_name).resolve()
    try:
        target.relative_to(destination_dir)
    except ValueError as exc:
        raise UnsafeUploadNameError(
            f"Le chemin d'upload sortirait du dossier temporaire : {original_name!r}."
        ) from exc
    return target
