from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from vision_trainer.yolo.models import DatasetInfo, SplitInfo

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_KEYS = ("train", "val", "test")

# ZIP import safety limits (generous for vision datasets; ~215 MB zip must pass).
MAX_ZIP_MEMBERS = 200_000
MAX_ZIP_UNCOMPRESSED_BYTES = 20 * 1024 * 1024 * 1024  # 20 GiB


class ZipExtractionError(Exception):
    """Raised when a ZIP archive fails security validation before extraction."""


def find_data_yaml(root: Path) -> Path | None:
    """
    Return the first data.yaml found under ``root``.

    Search order: ``root/data.yaml``, then each direct child, then a shallow
    recursive fallback (depth ≤ 3) for ZIPs with an extra nesting level.
    """
    root = root.resolve()
    direct = root / "data.yaml"
    if direct.is_file():
        return direct

    for child in sorted(root.iterdir()):
        if child.is_dir():
            candidate = child / "data.yaml"
            if candidate.is_file():
                return candidate

    for path in sorted(root.rglob("data.yaml")):
        try:
            depth = len(path.relative_to(root).parts)
        except ValueError:
            continue
        if depth <= 3 and path.is_file():
            return path
    return None


def find_yolo_layout_root(root: Path) -> Path | None:
    """Find a YOLO detect layout root (train/images + train/labels)."""
    root = root.resolve()
    candidates: list[Path] = [root]
    for child in sorted(root.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            candidates.append(child)
            for grand in sorted(child.iterdir()):
                if grand.is_dir() and not grand.name.startswith("."):
                    candidates.append(grand)

    for candidate in candidates:
        if _has_yolo_split_dirs(candidate, "train"):
            return candidate
    return None


def _has_yolo_split_dirs(base: Path, split: str) -> bool:
    return (base / split / "images").is_dir() and (base / split / "labels").is_dir()


def find_classes_txt(layout_root: Path) -> Path | None:
    """Prefer train/classes.txt, then val/, then layout root."""
    for candidate in (
        layout_root / "train" / "classes.txt",
        layout_root / "val" / "classes.txt",
        layout_root / "classes.txt",
    ):
        if candidate.is_file():
            return candidate
    return None


def parse_classes_txt(path: Path) -> list[str]:
    names: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        name = line.strip()
        if not name or name.startswith("#"):
            continue
        names.append(name)
    return names


def generate_data_yaml_from_layout(
    layout_root: Path,
    *,
    class_names: list[str],
    destination: Path | None = None,
) -> Path:
    """Write an internal data.yaml for a folder layout (never overwrites existing)."""
    target = destination if destination is not None else layout_root / "data.yaml"
    if target.is_file() and destination is None:
        return target

    payload: dict[str, Any] = {
        "path": ".",
        "train": "train/images",
        "names": {index: name for index, name in enumerate(class_names)},
    }
    if _has_yolo_split_dirs(layout_root, "val"):
        payload["val"] = "val/images"
    if _has_yolo_split_dirs(layout_root, "test"):
        payload["test"] = "test/images"

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return target


def ensure_data_yaml_for_layout(root: Path) -> tuple[Path | None, str | None]:
    """
    If data.yaml is missing but train/images+labels and classes.txt exist,
    generate an internal data.yaml. Existing YAML is never overwritten.
    """
    existing = find_data_yaml(root)
    if existing is not None:
        return existing, None

    layout = find_yolo_layout_root(root)
    if layout is None or not _has_yolo_split_dirs(layout, "train"):
        return None, None

    classes_path = find_classes_txt(layout)
    if classes_path is None:
        return None, None

    names = parse_classes_txt(classes_path)
    if not names:
        return None, None

    yaml_path = generate_data_yaml_from_layout(layout, class_names=names)
    try:
        rel = classes_path.relative_to(layout)
    except ValueError:
        rel = classes_path.name
    message = (
        f"data.yaml généré automatiquement depuis `{rel}` "
        f"({len(names)} classe(s)) → `{yaml_path}`."
    )
    return yaml_path, message


def extract_zip_dataset(zip_path: Path, destination: Path) -> Path:
    """Extract a YOLO dataset ZIP after validating every archive member."""
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()

    try:
        archive = zipfile.ZipFile(zip_path, "r")
    except zipfile.BadZipFile as exc:
        raise ZipExtractionError("Archive ZIP invalide.") from exc

    with archive:
        members = archive.infolist()
        if len(members) > MAX_ZIP_MEMBERS:
            raise ZipExtractionError(
                f"Archive trop volumineuse : {len(members)} entrées "
                f"(limite = {MAX_ZIP_MEMBERS})."
            )
        total_uncompressed = sum(max(0, int(member.file_size)) for member in members)
        if total_uncompressed > MAX_ZIP_UNCOMPRESSED_BYTES:
            raise ZipExtractionError(
                f"Taille décompressée annoncée trop élevée : {total_uncompressed} octets "
                f"(limite = {MAX_ZIP_UNCOMPRESSED_BYTES})."
            )

        planned_extractions: list[tuple[zipfile.ZipInfo, Path]] = []
        for member in members:
            target = _validate_zip_member(member, destination_resolved)
            planned_extractions.append((member, target))

        for member, target in planned_extractions:
            _extract_zip_member(archive, member, target)

    return destination


def _validate_zip_member(member: zipfile.ZipInfo, destination_resolved: Path) -> Path:
    member_name = member.filename

    if not member_name or member_name.startswith(("/", "\\")):
        raise ZipExtractionError(f"Chemin absolu non autorisé dans l'archive : {member_name!r}.")

    if re.match(r"^[A-Za-z]:[/\\]", member_name):
        raise ZipExtractionError(f"Chemin absolu non autorisé dans l'archive : {member_name!r}.")

    pure_path = PurePosixPath(member_name)
    if pure_path.is_absolute():
        raise ZipExtractionError(f"Chemin absolu non autorisé dans l'archive : {member_name!r}.")

    if ".." in pure_path.parts:
        raise ZipExtractionError(f"Traversée de répertoire non autorisée : {member_name!r}.")

    is_symlink = ((member.external_attr >> 16) & 0o170000) == 0o120000
    if is_symlink:
        raise ZipExtractionError(f"Lien symbolique non autorisé : {member_name!r}.")

    target = (destination_resolved / Path(*pure_path.parts)).resolve()
    try:
        target.relative_to(destination_resolved)
    except ValueError as exc:
        raise ZipExtractionError(
            f"Destination hors du dossier cible : {member_name!r}."
        ) from exc

    return target


def _extract_zip_member(
    archive: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    target: Path,
) -> None:
    if member.is_dir() or member.filename.endswith("/"):
        target.mkdir(parents=True, exist_ok=True)
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member) as source, target.open("wb") as destination:
        shutil.copyfileobj(source, destination)


def load_dataset_from_directory(
    root: Path,
    *,
    containment_root: Path | None = None,
) -> tuple[DatasetInfo | None, list[str]]:
    """
    Parse data.yaml and build dataset metadata.

    When ``containment_root`` is set (ZIP imports), resolved ``path``/split directories
    must remain inside that root. Roboflow ``../train/images`` fallbacks that land
    inside the extract root remain allowed.

    If data.yaml is absent but train/images+labels and classes.txt are present,
    an internal data.yaml is generated (existing YAML is never overwritten).
    Soft informational messages may be returned alongside hard errors.
    """
    infos: list[str] = []
    yaml_path = find_data_yaml(root)
    if yaml_path is None:
        yaml_path, generated_msg = ensure_data_yaml_for_layout(root)
        if generated_msg:
            infos.append(generated_msg)
    if yaml_path is None:
        return None, ["data.yaml introuvable dans le dataset."] + infos

    try:
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return None, [f"data.yaml illisible : {exc}"] + infos

    if not isinstance(raw, dict):
        return None, ["data.yaml doit contenir un mapping YAML."] + infos

    errors: list[str] = []
    class_names = _parse_class_names(raw.get("names"), errors)
    if class_names is None:
        return None, errors + infos

    yaml_dir = yaml_path.parent.resolve()
    containment = containment_root.resolve() if containment_root is not None else None

    dataset_root = _resolve_dataset_root(yaml_path, raw.get("path"))
    if containment is not None and not _is_within_root(dataset_root, containment):
        # Absolute/external path: not applicable for ZIP datasets.
        dataset_root = yaml_dir

    if not _path_is_applicable(dataset_root, raw, yaml_dir):
        dataset_root = yaml_dir

    if containment is not None and not _is_within_root(dataset_root, containment):
        return None, [
            f"Le chemin 'path' sort de la racine du dataset extrait : {dataset_root}"
        ] + infos

    splits = _parse_splits(
        raw,
        dataset_root,
        yaml_dir,
        errors,
        containment_root=containment,
    )

    if errors:
        return None, errors + infos

    info = DatasetInfo(
        root=dataset_root,
        yaml_path=yaml_path,
        class_names=class_names,
        splits=splits,
    )
    return info, infos


def _parse_class_names(names_raw: Any, errors: list[str]) -> dict[int, str] | None:
    if names_raw is None:
        errors.append("Le champ 'names' est absent de data.yaml.")
        return None

    if isinstance(names_raw, dict):
        return _parse_class_names_dict(names_raw, errors)

    if isinstance(names_raw, list):
        return _parse_class_names_list(names_raw, errors)

    errors.append("'names' doit être une liste ou un dictionnaire.")
    return None


def _parse_class_names_dict(names_raw: dict[Any, Any], errors: list[str]) -> dict[int, str] | None:
    if not names_raw:
        errors.append("Le champ 'names' est vide.")
        return None

    parsed: dict[int, str] = {}
    invalid = False

    for key, value in names_raw.items():
        try:
            class_id = int(key)
        except (TypeError, ValueError):
            errors.append(f"Identifiant de classe invalide dans names : {key!r}.")
            invalid = True
            continue

        if class_id < 0:
            errors.append(f"Identifiant de classe négatif dans names : {class_id}.")
            invalid = True
            continue

        if not isinstance(value, str) or not value.strip():
            errors.append(f"Nom de classe invalide pour l'id {class_id}.")
            invalid = True
            continue

        parsed[class_id] = value.strip()

    if invalid:
        return None

    expected_keys = list(range(len(parsed)))
    actual_keys = sorted(parsed.keys())
    if actual_keys != expected_keys:
        errors.append("Les indices de 'names' doivent être contigus à partir de 0.")
        return None

    return parsed


def _parse_class_names_list(names_raw: list[Any], errors: list[str]) -> dict[int, str] | None:
    if not names_raw:
        errors.append("Le champ 'names' est vide.")
        return None

    parsed: dict[int, str] = {}
    invalid = False

    for index, name in enumerate(names_raw):
        if not isinstance(name, str):
            errors.append(
                f"Entrée names[{index}] invalide : doit être une chaîne, reçu {type(name).__name__}."
            )
            invalid = True
            continue
        if not name.strip():
            errors.append(f"Entrée names[{index}] vide.")
            invalid = True
            continue
        parsed[index] = name.strip()

    if invalid:
        return None

    return parsed


def _resolve_dataset_root(yaml_path: Path, path_value: Any) -> Path:
    """
    Resolve the dataset root for relative split paths.

    Relative ``path:`` values are always anchored to the directory that contains
    data.yaml — never to the process CWD or the temporary import parent folder.
    Absolute or relative ``path:`` values are used only when applicable (the
    directory exists and can locate at least one declared relative split).
    Otherwise the directory containing data.yaml is used.
    """
    yaml_dir = yaml_path.parent.resolve()
    if not isinstance(path_value, str) or not path_value.strip():
        return yaml_dir

    candidate = Path(path_value.strip())
    if candidate.is_absolute():
        resolved = candidate
    else:
        resolved = (yaml_dir / candidate).resolve()

    return resolved


def _path_is_applicable(
    dataset_root: Path,
    raw: dict[str, Any],
    yaml_dir: Path,
) -> bool:
    """Return True when ``path:`` is a usable base for relative split paths."""
    if dataset_root == yaml_dir:
        return True
    if not dataset_root.is_dir():
        return False

    relative_splits = [
        Path(value.strip())
        for key in SPLIT_KEYS
        for value in [raw.get(key)]
        if isinstance(value, str) and value.strip() and not Path(value.strip()).is_absolute()
    ]
    if not relative_splits:
        return True

    return any((dataset_root / split_ref).exists() for split_ref in relative_splits)


def _parse_splits(
    raw: dict[str, Any],
    dataset_root: Path,
    yaml_dir: Path,
    errors: list[str],
    *,
    containment_root: Path | None = None,
) -> dict[str, SplitInfo]:
    splits: dict[str, SplitInfo] = {}
    yaml_dir = yaml_dir.resolve()
    split_base = dataset_root if _path_is_applicable(dataset_root, raw, yaml_dir) else yaml_dir
    containment = containment_root.resolve() if containment_root is not None else None

    for split_name in SPLIT_KEYS:
        split_ref = raw.get(split_name)
        if split_ref is None:
            continue

        if not isinstance(split_ref, str):
            errors.append(
                f"Le split '{split_name}' doit être une chaîne de chemin de dossier, "
                f"reçu : {type(split_ref).__name__}."
            )
            continue

        if not split_ref.strip():
            errors.append(f"Le split '{split_name}' est présent mais vide.")
            continue

        declared = split_ref.strip()
        images_dir, used_fallback, fallback_ref = _resolve_split_path_tolerant(
            declared,
            split_base=split_base,
            extracted_root=yaml_dir,
            containment_root=containment,
        )

        if containment is not None and not _is_within_root(images_dir, containment):
            errors.append(
                f"Le split '{split_name}' ({declared}) sort de la racine du dataset extrait."
            )
            continue

        labels_dir = _infer_labels_dir(images_dir)
        splits[split_name] = SplitInfo(
            name=split_name,
            images_dir=images_dir,
            labels_dir=labels_dir,
            declared_ref=declared,
            resolved_via_fallback=used_fallback,
            fallback_ref=fallback_ref,
        )

    return splits


def _resolve_split_path(split_ref: str, dataset_root: Path) -> Path:
    """Resolve a split path against the dataset root (yaml ``path:`` or yaml dir)."""
    candidate = Path(split_ref)
    if candidate.is_absolute():
        return candidate
    return (dataset_root / candidate).resolve()


def _leading_parent_stripped_ref(split_ref: str) -> str | None:
    """Strip only leading ``..`` components from a relative split path."""
    candidate = Path(split_ref)
    if candidate.is_absolute():
        return None

    parts = candidate.parts
    index = 0
    while index < len(parts) and parts[index] == "..":
        index += 1

    if index == 0:
        return None
    if index >= len(parts):
        return None

    return str(Path(*parts[index:]))


def _is_within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _resolve_split_path_tolerant(
    split_ref: str,
    *,
    split_base: Path,
    extracted_root: Path,
    containment_root: Path | None = None,
) -> tuple[Path, bool, str | None]:
    """
    Resolve a split path, with a contained fallback when leading ``../`` miss.

    Returns (resolved_path, used_fallback, fallback_ref).
    """
    containment = containment_root.resolve() if containment_root is not None else None
    declared_path = _resolve_split_path(split_ref, split_base)

    declared_ok = declared_path.exists()
    if declared_ok and containment is not None and not _is_within_root(declared_path, containment):
        declared_ok = False

    if declared_ok:
        return declared_path, False, None

    stripped = _leading_parent_stripped_ref(split_ref)
    if stripped is None:
        return declared_path, False, None

    fallback_path = (extracted_root / stripped).resolve()
    if not _is_within_root(fallback_path, extracted_root):
        return declared_path, False, None
    if containment is not None and not _is_within_root(fallback_path, containment):
        return declared_path, False, None
    if not fallback_path.exists():
        return declared_path, False, None

    return fallback_path, True, stripped


def _infer_labels_dir(images_dir: Path) -> Path | None:
    parts = list(images_dir.parts)
    if "images" in parts:
        labels_parts = ["labels" if part == "images" else part for part in parts]
        return Path(*labels_parts)
    sibling = images_dir.parent / "labels" / images_dir.name
    return sibling


def list_images(directory: Path | None) -> list[Path]:
    if directory is None or not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def relative_path_without_extension(path: Path, base_dir: Path) -> str:
    return str(path.relative_to(base_dir).with_suffix(""))


def label_path_for_image(
    image_path: Path,
    images_dir: Path,
    labels_dir: Path,
) -> Path:
    relative = relative_path_without_extension(image_path, images_dir)
    return labels_dir / f"{relative}.txt"


def image_path_for_label(
    label_path: Path,
    labels_dir: Path,
    images_dir: Path,
) -> Path | None:
    relative = relative_path_without_extension(label_path, labels_dir)
    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{relative}{extension}"
        if candidate.is_file():
            return candidate
    return images_dir / f"{relative}.jpg"
