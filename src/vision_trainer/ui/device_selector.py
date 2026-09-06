"""Shared Streamlit widgets for device selection (training + inference)."""

from __future__ import annotations

import streamlit as st

from vision_trainer.training.device import (
    DeviceChoice,
    DeviceError,
    describe_device,
    format_hardware_summary,
    get_available_devices,
    get_hardware_info,
    resolve_device,
)


def render_device_selector(
    *,
    key_prefix: str,
    disabled: bool = False,
    default_choice: DeviceChoice = "auto",
) -> tuple[DeviceChoice, str, str]:
    """
    Render device radio/select + resolved info + hardware expander.

    Returns ``(choice, resolved_ultralytics_device, human_label)``.
    """
    hardware = get_hardware_info()
    options = get_available_devices()
    labels = [option.label for option in options]
    label_to_choice = {option.label: option.choice for option in options}

    default_index = 0
    for index, option in enumerate(options):
        if option.choice == default_choice:
            default_index = index
            break

    # When no GPU: show a clear CPU-only control (still allow Auto which resolves to CPU).
    if not hardware.cuda_available:
        # Prefer a radio with Auto + CPU for consistency with the GPU UI.
        selected_label = st.radio(
            "Device",
            options=labels,
            index=default_index,
            horizontal=True,
            disabled=disabled,
            key=f"{key_prefix}_device_radio",
        )
    else:
        selected_label = st.radio(
            "Device",
            options=labels,
            index=default_index,
            horizontal=True,
            disabled=disabled,
            key=f"{key_prefix}_device_radio",
        )

    choice = label_to_choice[selected_label]
    try:
        resolved = resolve_device(choice)
    except DeviceError as exc:
        st.error(str(exc))
        st.stop()

    human = describe_device(resolved, hardware=hardware)
    st.info(f"Device réellement sélectionné : **{human}** (`{resolved}`)")

    with st.expander("Informations matériel", expanded=False):
        for line in format_hardware_summary(hardware):
            st.write(line)

    return choice, resolved, human
