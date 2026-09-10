from __future__ import annotations

from pathlib import Path

import streamlit as st

from vision_trainer.results.models import SESSION_INFERENCE_WEIGHTS_KEY
from vision_trainer.storage import (
    CLEANABLE_CATEGORIES,
    CategoryKind,
    DeletionPlan,
    StorageManager,
    format_bytes,
)
from vision_trainer.training.trainer import SESSION_ACTIVE_RUN_KEY


def render(*, embedded: bool = False) -> None:
    if not embedded:
        st.title("Gestion du stockage")
    st.markdown(
        "Visualisez et libérez l'espace occupé par les datasets, runs et modèles. "
        "Aucune suppression n'est automatique."
    )


    def _protected_paths_from_session() -> list[Path]:
        paths: list[Path] = []
        active = st.session_state.get(SESSION_ACTIVE_RUN_KEY)
        if active:
            paths.append(Path(active))
        preferred = st.session_state.get(SESSION_INFERENCE_WEIGHTS_KEY)
        if preferred:
            paths.append(Path(preferred))
        return paths


    def _manager() -> StorageManager:
        return StorageManager(protected_paths=_protected_paths_from_session())


    def _reset_selection_state() -> None:
        for key in list(st.session_state.keys()):
            if str(key).startswith("storage_sel_") or str(key).startswith("storage_cat_"):
                del st.session_state[key]
        st.session_state.pop("storage_pending_plan", None)
        st.session_state.pop("storage_confirm_final_model", None)


    col_refresh, _ = st.columns([1, 4])
    with col_refresh:
        if st.button("Actualiser", key="storage_refresh"):
            st.session_state.pop("storage_overview_cache", None)
            st.rerun()

    manager = _manager()

    flash = st.session_state.pop("storage_flash", None)
    if flash:
        st.success(flash.get("title", "Suppression terminée"))
        for line in flash.get("lines", []):
            st.write(line)
        for line in flash.get("errors", []):
            st.error(line)

    overview = manager.build_overview()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Espace utilisé", format_bytes(overview.total_bytes))
    m2.metric("Datasets", overview.dataset_count)
    m3.metric("Runs", overview.run_count)
    m4.metric("Modèles best.*", overview.model_count)
    m5.metric(
        "Disque libre",
        format_bytes(overview.free_disk_bytes) if overview.free_disk_bytes is not None else "—",
    )

    if not overview.groups:
        st.info("Aucun fichier géré trouvé sous le dossier de données (datasets / runs).")
        st.stop()

    st.subheader("Groupes")

    # Bulk selection helpers
    b1, b2, b3 = st.columns(3)
    with b1:
        select_all = st.button("Sélectionner tout", key="storage_select_all")
    with b2:
        clear_all = st.button("Tout désélectionner", key="storage_clear_all")
    with b3:
        pass

    if select_all:
        for group in overview.groups:
            st.session_state[f"storage_sel_{group.group_id}"] = True
    if clear_all:
        for group in overview.groups:
            st.session_state[f"storage_sel_{group.group_id}"] = False

    selected_group_ids: list[str] = []
    selected_plans: list[DeletionPlan] = []

    for group in overview.groups:
        box = st.container(border=True)
        with box:
            header_cols = st.columns([0.08, 0.62, 0.3])
            with header_cols[0]:
                checked = st.checkbox(
                    "sel",
                    key=f"storage_sel_{group.group_id}",
                    label_visibility="collapsed",
                    disabled=group.protected,
                )
            with header_cols[1]:
                st.markdown(f"**{group.title}**")
                st.caption(group.subtitle)
                if group.protected:
                    st.warning(group.protection_reason or "Protégé")
                if group.legacy:
                    st.caption("Session non identifiée / métadonnées incomplètes")
            with header_cols[2]:
                st.write(format_bytes(group.size_bytes))

            with st.expander("Détails", expanded=False):
                cat_kinds_selected: list[CategoryKind] = []
                for category in group.categories:
                    if category.size_bytes <= 0 and not category.paths:
                        continue
                    label = (
                        f"{category.label} — {format_bytes(category.size_bytes)}"
                        + (" · partagé" if category.shared else "")
                        + (" · important" if category.important else "")
                    )
                    default_on = category.kind in CLEANABLE_CATEGORIES
                    cat_key = f"storage_cat_{group.group_id}_{category.kind.value}"
                    if cat_key not in st.session_state:
                        st.session_state[cat_key] = default_on and not category.important
                    cat_checked = st.checkbox(
                        label,
                        key=cat_key,
                        disabled=group.protected,
                    )
                    if cat_checked:
                        cat_kinds_selected.append(category.kind)

                action_cols = st.columns(3)
                with action_cols[0]:
                    if st.button(
                        "Nettoyer",
                        key=f"clean_{group.group_id}",
                        disabled=group.protected,
                        help="Supprime checkpoints intermédiaires et temporaires. Conserve dataset et best.*.",
                    ):
                        plan = manager.plan_clean(group)
                        st.session_state["storage_pending_plan"] = {
                            "plan": plan,
                            "label": f"Nettoyage — {group.title}",
                        }
                        st.rerun()
                with action_cols[1]:
                    if st.button(
                        "Supprimer le dataset",
                        key=f"del_ds_{group.group_id}",
                        disabled=group.protected or group.dataset_dir is None,
                    ):
                        plan = manager.plan_delete_dataset(group)
                        st.session_state["storage_pending_plan"] = {
                            "plan": plan,
                            "label": f"Suppression dataset — {group.title}",
                        }
                        st.rerun()
                with action_cols[2]:
                    if st.button(
                        "Supprimer tout",
                        key=f"del_all_{group.group_id}",
                        disabled=group.protected,
                        type="primary",
                    ):
                        plan = manager.plan_delete_all(group)
                        st.session_state["storage_pending_plan"] = {
                            "plan": plan,
                            "label": f"Suppression totale — {group.title}",
                            "require_final_ack": True,
                        }
                        st.rerun()

                if st.button(
                    "Supprimer les catégories cochées",
                    key=f"del_cats_{group.group_id}",
                    disabled=group.protected or not cat_kinds_selected,
                ):
                    plan = manager.plan_delete_categories(group, cat_kinds_selected)
                    st.session_state["storage_pending_plan"] = {
                        "plan": plan,
                        "label": f"Suppression partielle — {group.title}",
                        "require_final_ack": any(
                            kind == CategoryKind.FINAL_MODEL for kind in cat_kinds_selected
                        ),
                    }
                    st.rerun()

            if checked and not group.protected:
                selected_group_ids.append(group.group_id)
                # Group-level selection = delete all for that group
                selected_plans.append(manager.plan_delete_all(group))

    st.divider()
    merged = manager.merge_plans(selected_plans) if selected_plans else DeletionPlan()
    st.write(f"**{len(selected_group_ids)} groupe(s) sélectionné(s)**")
    st.write(f"Espace libérable estimé : **{format_bytes(merged.estimated_bytes)}**")
    if merged.blocked:
        for message in merged.blocked:
            st.caption(f"⚠ {message}")

    if st.button(
        "Supprimer la sélection",
        type="primary",
        disabled=not selected_group_ids,
        key="storage_delete_selection",
    ):
        st.session_state["storage_pending_plan"] = {
            "plan": merged,
            "label": "Suppression de la sélection",
            "require_final_ack": merged.includes_final_model,
        }
        st.rerun()

    pending = st.session_state.get("storage_pending_plan")
    if pending:
        plan: DeletionPlan = pending["plan"]
        st.subheader("Confirmation de suppression")
        st.warning("Cette action est irréversible après confirmation.")
        st.write(pending.get("label", "Suppression"))

        if plan.blocked and not plan.items:
            st.error("Rien à supprimer (éléments bloqués).")
            for message in plan.blocked:
                st.write(f"- {message}")
            if st.button("Fermer", key="storage_close_blocked"):
                st.session_state.pop("storage_pending_plan", None)
                st.rerun()
            st.stop()

        n_files = sum(1 for item in plan.items if not item.is_dir)
        n_dirs = sum(1 for item in plan.items if item.is_dir)
        n_models = sum(1 for item in plan.items if item.category == CategoryKind.FINAL_MODEL)
        n_datasets = sum(
            1
            for item in plan.items
            if item.category
            in {CategoryKind.DATASET_ARCHIVE, CategoryKind.DATASET_EXTRACTED}
            or (item.is_dir and "datasets" in item.path.parts)
        )
        n_runs = sum(
            1
            for item in plan.items
            if item.is_dir and "runs" in item.path.parts and item.path.parent.name == "runs"
        )

        st.markdown(
            f"""
    Vous allez supprimer :

    - **{n_datasets}** élément(s) dataset
    - **{n_runs}** dossier(s) d'entraînement
    - **{n_files}** fichier(s) / **{n_dirs}** dossier(s) au total
    - **{n_models}** modèle(s) final/best explicitement ciblé(s)

    Espace libéré estimé : **environ {format_bytes(plan.estimated_bytes)}**
    """
        )
        if plan.blocked:
            st.info("Certains éléments ont été exclus :")
            for message in plan.blocked:
                st.write(f"- {message}")

        require_final = bool(pending.get("require_final_ack") or plan.includes_final_model)
        final_ok = True
        if require_final:
            final_ok = st.checkbox(
                "Je confirme vouloir supprimer un ou plusieurs modèles finaux (best.*).",
                key="storage_confirm_final_model",
            )

        c_ok, c_cancel = st.columns(2)
        with c_cancel:
            if st.button("Annuler", key="storage_cancel_delete"):
                st.session_state.pop("storage_pending_plan", None)
                st.session_state.pop("storage_confirm_final_model", None)
                st.rerun()
        with c_ok:
            confirm = st.button(
                "Confirmer la suppression",
                type="primary",
                disabled=require_final and not final_ok,
                key="storage_confirm_delete",
            )

        if confirm:
            result = manager.execute_plan(plan)
            st.session_state.pop("storage_pending_plan", None)
            st.session_state.pop("storage_confirm_final_model", None)
            _reset_selection_state()
            lines = [
                f"- {result.files_removed} fichier(s) supprimé(s)",
                f"- {result.dirs_removed} dossier(s) supprimé(s)",
                f"- {format_bytes(result.bytes_freed)} libérés",
            ]
            for line in result.skipped[:10]:
                lines.append(f"- ignoré : {line}")
            st.session_state["storage_flash"] = {
                "title": "Suppression terminée",
                "lines": lines,
                "errors": result.errors[:20],
            }
            st.rerun()
