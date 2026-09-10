"""Shared Streamlit UI for ZIP vs external dataset sources."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import streamlit as st

from vision_trainer.datasets.external import (
    ExternalDatasetError,
    external_root_exists,
    list_external_dataset_dirs,
    resolve_external_dataset_path,
)
from vision_trainer.paths import get_external_datasets_root

SourceMode = Literal["zip", "local"]


def render_source_mode_selector(*, key: str) -> SourceMode:
    choice = st.segmented_control(
        "Source du dataset",
        options=["Import ZIP", "Dossier local"],
        default="Import ZIP",
        key=key,
        required=True,
    )
    if choice == "Dossier local":
        return "local"
    return "zip"


def render_external_dataset_picker(*, key_prefix: str) -> Path | None:
    """
    Show first-level external folders and return the selected path, or None.

    Does not validate the dataset content — caller must verify explicitly.
    """
    root = get_external_datasets_root()
    st.caption(f"Racine autorisée : `{root}` (lecture seule recommandée).")

    if not external_root_exists(root):
        st.warning(
            f"Aucun volume de datasets externes détecté (`{root}`). "
            "Montez un dossier hôte vers `/datasets` dans docker-compose "
            "(ex. `/mnt/d/Datasets:/datasets:ro`)."
        )
        return None

    dirs = list_external_dataset_dirs(root)
    if not dirs:
        st.info(f"Aucun dataset externe trouvé dans `{root}`.")
        return None

    labels = [d.name for d in dirs]
    selected = st.selectbox(
        "Dataset externe",
        options=labels,
        key=f"{key_prefix}_external_name",
    )
    try:
        path = resolve_external_dataset_path(str(selected), root=root)
    except ExternalDatasetError as exc:
        st.error(str(exc))
        return None

    st.code(str(path), language=None)
    return path


def show_session_dataset_summary(
    payload: dict,
    *,
    title: str = "Dataset actif",
) -> None:
    from vision_trainer.datasets.external import (
        SOURCE_EXTERNAL,
        normalize_source_type,
        source_type_label_fr,
    )

    st.info(title)
    name = payload.get("display_name") or payload.get("dataset_id") or "—"
    st.write(f"**{name}** — {source_type_label_fr(payload.get('source_type'))}")
    root = payload.get("root") or payload.get("extract_dir") or ""
    if root:
        st.code(str(root), language=None)
    if normalize_source_type(payload.get("source_type")) == SOURCE_EXTERNAL:
        path = Path(str(root))
        if not path.is_dir():
            st.error(f"Le dataset externe n'est plus accessible : {path}")
