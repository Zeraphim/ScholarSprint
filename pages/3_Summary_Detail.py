from __future__ import annotations

import streamlit as st

from app import format_summary_markdown, init_state, inject_styles, render_sidebar


def _selected_from_state_or_query(summaries: list[dict]) -> str:
    names = [item["file_name"] for item in summaries]
    requested_name = st.query_params.get("summary", "")

    if requested_name in names:
        return requested_name

    selected_name = st.session_state.get("selected_summary_name", "")
    if selected_name in names:
        return selected_name

    return names[0]


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
    st.markdown(format_summary_markdown(item["summary_text"], item["file_name"]))

    st.download_button(
        "Export This Summary",
        data=item["summary_text"],
        file_name=f"{selected_name}.txt",
        use_container_width=False,
    )


if __name__ == "__main__":
    render_page()
