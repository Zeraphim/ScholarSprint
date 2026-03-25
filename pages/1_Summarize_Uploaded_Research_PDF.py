from __future__ import annotations

import datetime as dt

import streamlit as st

from app import (
    DEFAULT_MODEL,
    extract_text_from_pdf,
    format_summary_markdown,
    generate_summary_text,
    init_state,
    inject_styles,
    persist_generated_summaries,
    render_sidebar,
)


def _open_summary_detail(summary_name: str) -> None:
    st.session_state["selected_summary_name"] = summary_name
    st.query_params["summary"] = summary_name
    if hasattr(st, "switch_page"):
        st.switch_page("pages/3_Summary_Detail.py")
    else:
        st.info("Summary selected. Open the 'Summary Detail' page from the sidebar.")


def render_page() -> None:
    init_state()
    inject_styles()
    render_sidebar()

    st.title("Summarize Uploaded Research PDF")
    st.caption("Upload papers, tune summary options, and generate focused outputs.")

    with st.form("pdf_summary_form"):
        uploaded_files = st.file_uploader(
            "Upload Research Studies (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF studies for summary generation.",
        )

        output_format = st.selectbox(
            "Output Format",
            ["Executive Brief", "Structured Sections", "Bulleted Notes"],
        )
        audience = st.selectbox("Audience", ["Researcher", "Student", "Product Team", "General"])
        citation_mode = st.selectbox("Citation Mode", ["Inline", "Footnotes", "No Citations"])
        guidance = st.text_area(
            "Optional Guidance",
            placeholder="Example: Focus on methodology robustness and practical limitations for deployment.",
            height=90,
        )

        submitted = st.form_submit_button("Generate Summary", use_container_width=True)

    if submitted:
        summary_styles = st.session_state.get("Summary Style", [])
        max_summary_len = int(st.session_state.get("Max Summary Length (words)", 350))
        selected_model = st.session_state.get("selected_model", DEFAULT_MODEL)

        if not uploaded_files:
            st.warning("Please upload at least one PDF file.")
        else:
            generated: list[dict] = []
            for uploaded_file in uploaded_files:
                pdf_text = extract_text_from_pdf(uploaded_file)
                if not pdf_text:
                    st.warning(f"Could not extract text from {uploaded_file.name}.")
                    continue

                summary_text, engine_info = generate_summary_text(
                    text=pdf_text,
                    selected_model=selected_model,
                    max_words=max_summary_len,
                    output_format=output_format,
                    audience=audience,
                    citation_mode=citation_mode,
                    guidance=guidance,
                    styles=summary_styles,
                )

                generated.append(
                    {
                        "file_name": uploaded_file.name,
                        "summary_text": summary_text,
                        "word_count": len(summary_text.split()),
                        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "engine": engine_info,
                    }
                )

            if generated:
                st.session_state["generated_summaries"] = generated
                persist_generated_summaries(generated)
                st.success(f"Generated summaries for {len(generated)} file(s).")

    generated_summaries = st.session_state.get("generated_summaries", [])
    if not generated_summaries:
        st.info("No summaries generated yet.")
        return

    st.subheader("Generated Summaries")
    export_payload = "\n\n".join(
        [
            f"{item['file_name']}\n{'=' * len(item['file_name'])}\n{item['summary_text']}"
            for item in generated_summaries
        ]
    )
    st.download_button(
        "Export All Summaries",
        data=export_payload,
        file_name="summary_export.txt",
        use_container_width=False,
    )

    tabs = st.tabs([item["file_name"] for item in generated_summaries])
    for index, tab in enumerate(tabs):
        item = generated_summaries[index]
        with tab:
            st.caption(item.get("engine", ""))
            st.markdown(format_summary_markdown(item["summary_text"], item["file_name"]))
            c1, c2 = st.columns([0.35, 0.65])
            with c1:
                if st.button("Open Detail Page", key=f"open_detail_{index}"):
                    _open_summary_detail(item["file_name"])
            with c2:
                st.caption(
                    f"Words: {item.get('word_count', 0)} | Generated: {item.get('generated_at', '')}"
                )


if __name__ == "__main__":
    render_page()
