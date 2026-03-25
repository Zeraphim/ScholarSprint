from __future__ import annotations

import datetime as dt
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st

from app import (
    DEFAULT_MODEL,
    Study,
    build_study_cache_key,
    date_window_to_start,
    extractive_summarize,
    fetch_arxiv_for_topic,
    init_state,
    inject_styles,
    load_summary_from_disk_cache,
    parse_topics,
    quick_relevance_check,
    relevance_threshold_to_score,
    render_sidebar,
    save_summary_to_disk_cache,
    score_study_relevance,
    summarize_topic_result_with_timing,
)


def render_page() -> None:
    init_state()
    inject_styles()
    render_sidebar()

    st.title("Fetch Studies by Topic with Summarized Input")
    st.caption("Search arXiv by topic, filter for relevance, and summarize findings.")

    with st.form("topic_fetch_form"):
        topics_raw = st.text_input(
            "Topics (comma-separated)",
            value="explainable ai, medical imaging, carbon forecasting",
            placeholder="example: federated learning, oncology, bioinformatics",
        )
        c1, c2 = st.columns(2)
        with c1:
            max_studies = st.slider("Max Studies", min_value=5, max_value=100, value=25, step=5)
        with c2:
            date_window = st.selectbox(
                "Date Window",
                ["Last 30 days", "Last 6 months", "Last 2 years", "Any time"],
            )

        relevance_threshold = st.select_slider(
            "Relevance Threshold",
            options=["Low", "Medium", "High", "Very High"],
            value="High",
        )

        submitted = st.form_submit_button("Fetch and Summarize", use_container_width=True)

    if submitted:
        topics = parse_topics(topics_raw)
        if not topics:
            st.warning("Please enter at least one topic.")
        else:
            selected_model = st.session_state.get("selected_model", DEFAULT_MODEL)
            start_date = date_window_to_start(date_window)
            min_relevance = relevance_threshold_to_score(relevance_threshold)
            per_topic_limit = max(1, max_studies // len(topics))
            max_workers = int(st.session_state.get("parallel_workers", 4))
            use_persistent_cache = bool(st.session_state.get("use_persistent_cache", True))
            show_timing_logs = bool(st.session_state.get("show_timing_logs", False))
            summary_cache: dict[str, str] = st.session_state.get("study_summary_cache", {})

            all_results: list[Study] = []
            total_cache_hits = 0
            total_api_calls = 0
            total_summary_seconds = 0.0
            run_started_at = time.perf_counter()

            progress = st.progress(0.0)
            status = st.empty()

            topic_count = len(topics)
            for topic_index, topic in enumerate(topics, start=1):
                status.info(f"Fetching studies for topic {topic_index}/{topic_count}: {topic}")
                try:
                    raw_studies = fetch_arxiv_for_topic(topic, per_topic_limit * 4)
                except Exception as exc:
                    st.warning(f"Failed to fetch studies for topic '{topic}': {exc}")
                    progress.progress(topic_index / topic_count)
                    continue

                qualified: list[tuple[dict[str, str], float]] = []
                for item in raw_studies:
                    try:
                        published_date = dt.datetime.fromisoformat(item["published"].replace("Z", "+00:00")).date()
                    except ValueError:
                        continue

                    if published_date < start_date:
                        continue

                    combined_text = f"{item['title']} {item['summary']}"
                    if not quick_relevance_check(combined_text, topic):
                        continue
                    relevance = score_study_relevance(combined_text, topic)
                    if relevance < min_relevance:
                        continue

                    qualified.append((item, relevance))

                if not qualified:
                    progress.progress(topic_index / topic_count)
                    continue

                status.info(
                    f"Summarizing {len(qualified)} study/studies for topic {topic_index}/{topic_count}: {topic}"
                )

                ordered_studies: list[Study | None] = [None] * len(qualified)
                futures = {}
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for idx, (item, relevance) in enumerate(qualified):
                        cache_key = build_study_cache_key(
                            title=item["title"],
                            url=item.get("url", ""),
                            topic=topic,
                            selected_model=selected_model,
                        )
                        cached_summary = summary_cache.get(cache_key)
                        if not cached_summary and use_persistent_cache:
                            cached_summary = load_summary_from_disk_cache(cache_key)
                            if cached_summary:
                                summary_cache[cache_key] = cached_summary

                        try:
                            published_date = dt.datetime.fromisoformat(item["published"].replace("Z", "+00:00")).date()
                        except ValueError:
                            continue

                        if cached_summary:
                            total_cache_hits += 1
                            ordered_studies[idx] = Study(
                                title=item["title"],
                                summary=cached_summary,
                                published=published_date,
                                source="arXiv",
                                url=item["url"],
                                matched_topic=topic,
                                relevance_score=relevance,
                            )
                            continue

                        future = executor.submit(
                            summarize_topic_result_with_timing,
                            item["summary"],
                            topic,
                            selected_model,
                        )
                        futures[future] = (idx, item, relevance, cache_key, published_date)

                    for completed_count, future in enumerate(as_completed(futures), start=1):
                        idx, item, relevance, cache_key, published_date = futures[future]
                        total_api_calls += 1
                        try:
                            concise_summary, elapsed = future.result()
                        except Exception:
                            concise_summary = extractive_summarize(item["summary"], max_words=85, focus_hint=topic)
                            elapsed = 0.0

                        total_summary_seconds += elapsed
                        summary_cache[cache_key] = concise_summary
                        if use_persistent_cache:
                            save_summary_to_disk_cache(cache_key, concise_summary)

                        ordered_studies[idx] = Study(
                            title=item["title"],
                            summary=concise_summary,
                            published=published_date,
                            source="arXiv",
                            url=item["url"],
                            matched_topic=topic,
                            relevance_score=relevance,
                        )

                        status.info(
                            " / ".join(
                                [
                                    f"Topic {topic_index}/{topic_count}: {topic}",
                                    f"Completed summaries: {completed_count}/{len(futures)}",
                                ]
                            )
                        )

                all_results.extend([study for study in ordered_studies if study is not None])
                progress.progress(topic_index / topic_count)

            st.session_state["study_summary_cache"] = summary_cache

            all_results.sort(key=lambda study: (study.relevance_score, study.published), reverse=True)
            st.session_state["topic_results"] = all_results[:max_studies]
            progress.empty()
            status.empty()
            st.success(f"Fetched {len(st.session_state['topic_results'])} study result(s).")

            if show_timing_logs:
                total_elapsed = time.perf_counter() - run_started_at
                st.caption(
                    " | ".join(
                        [
                            f"Total elapsed: {total_elapsed:.1f}s",
                            f"API summaries: {total_api_calls}",
                            f"Summary time (aggregate): {total_summary_seconds:.1f}s",
                            f"Cache hits: {total_cache_hits}",
                        ]
                    )
                )

    topic_results = st.session_state.get("topic_results", [])
    if not topic_results:
        st.info("No fetched studies yet.")
        return

    st.subheader("Retrieval Output")
    for idx, study in enumerate(topic_results[:12], start=1):
        with st.expander(f"{idx}. {study.title}"):
            st.write(f"Matched Topic: {study.matched_topic}")
            st.write(f"Published: {study.published.isoformat()}")
            st.write(f"Relevance Score: {study.relevance_score:.2f}")
            st.write(study.summary)
            if study.url:
                st.link_button("Read Full Paper", study.url)


if __name__ == "__main__":
    render_page()
