from __future__ import annotations

from pathlib import Path

import streamlit as st

from app import format_summary_markdown, init_state, inject_styles, render_sidebar
from exporters import (
    EXPORT_FORMATS,
    MARKDOWN,
    build_export,
    export_file_name,
    missing_dependency,
)


def _selected_from_state_or_query(summaries: list[dict]) -> str:
    names = [item["file_name"] for item in summaries]
    requested_name = st.query_params.get("summary", "")

    if requested_name in names:
        return requested_name

    selected_name = st.session_state.get("selected_summary_name", "")
    if selected_name in names:
        return selected_name

    return names[0]


@st.cache_data(show_spinner=False)
def _cached_export(markdown_text: str, title: str, format_label: str) -> bytes:
    return build_export(markdown_text, title, format_label)


def _render_export_controls(markdown_text: str, source_name: str) -> None:
    st.markdown("#### Export")
    format_label = st.selectbox(
        "Export Format",
        list(EXPORT_FORMATS),
        index=list(EXPORT_FORMATS).index(MARKDOWN),
        key="export_format",
        help="Markdown is the source format; every other option is rendered from it.",
    )

    package = missing_dependency(format_label)
    if package:
        st.warning(
            f"{format_label} export needs the `{package}` package. "
            "Stop the app, run `uv sync --locked`, then restart it."
        )
        return

    title = Path(source_name).stem or source_name
    try:
        data = _cached_export(markdown_text, title, format_label)
    except Exception as error:  # rendering must never take the page down
        st.error(f"Could not build the {format_label} export: {error}")
        return

    st.download_button(
        f"Download {format_label}",
        data=data,
        file_name=export_file_name(source_name, format_label),
        mime=EXPORT_FORMATS[format_label].mime,
        type="primary",
        key="download_summary",
    )


def render_page() -> None:
    init_state()
    inject_styles()
    render_sidebar()

    st.title("Individual Summary")
    summaries = st.session_state.get("generated_summaries", [])

    if not summaries:
        st.info("No generated summaries yet. Use the PDF summary page to create one first.")
        return

    default_name = _selected_from_state_or_query(summaries)
    options = [item["file_name"] for item in summaries]
    default_index = options.index(default_name) if default_name in options else 0

    selected_name = st.selectbox("Select Summary", options, index=default_index)
    st.session_state["selected_summary_name"] = selected_name
    st.query_params["summary"] = selected_name

    item = next((entry for entry in summaries if entry["file_name"] == selected_name), summaries[0])

    st.caption(item.get("engine", ""))
    summary_markdown = format_summary_markdown(item["summary_text"], item["file_name"])
    st.markdown(summary_markdown)

    st.divider()
    _render_export_controls(summary_markdown, selected_name)


if __name__ == "__main__":
    render_page()
