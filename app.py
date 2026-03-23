import streamlit as st


st.set_page_config(
    page_title="Research Summarizer Dashboard",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

            :root {
                --bg: #f5f1e8;
                --paper: #fffcf6;
                --ink: #19242d;
                --muted: #5d6974;
                --accent: #16706b;
                --accent-soft: #d9f0ee;
                --line: #d8d6cc;
                --card-shadow: 0 10px 35px rgba(25, 36, 45, 0.08);
            }

            html, body, [class*="css"] {
                font-family: 'Space Grotesk', sans-serif;
                color: var(--ink);
            }

            .stApp {
                background:
                    radial-gradient(circle at 0% 0%, #fff4df 0%, transparent 35%),
                    radial-gradient(circle at 100% 0%, #d9f0ee 0%, transparent 30%),
                    linear-gradient(180deg, #f7f3ea 0%, #f2efe7 100%);
            }

            [data-testid="stSidebar"] {
                border-right: 1px solid var(--line);
                background: #f0ece2;
            }

            .hero {
                background: var(--paper);
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 1.2rem 1.4rem;
                box-shadow: var(--card-shadow);
            }

            .hero h1 {
                margin: 0;
                font-size: 2rem;
                letter-spacing: -0.02em;
            }

            .hero p {
                margin: 0.35rem 0 0;
                color: var(--muted);
                font-size: 0.98rem;
            }

            .feature-card {
                background: var(--paper);
                border: 1px solid var(--line);
                border-radius: 16px;
                padding: 1rem 1rem 0.5rem;
                box-shadow: var(--card-shadow);
                height: 100%;
            }

            .section-title {
                font-size: 1.05rem;
                font-weight: 700;
                margin: 0 0 0.35rem;
            }

            .section-subtitle {
                color: var(--muted);
                font-size: 0.92rem;
                margin: 0 0 1rem;
            }

            .pill-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
                margin-top: 0.25rem;
                margin-bottom: 0.8rem;
            }

            .pill {
                background: var(--accent-soft);
                border: 1px solid #b5e1dc;
                color: #0d4f4a;
                padding: 0.2rem 0.6rem;
                border-radius: 999px;
                font-size: 0.78rem;
                font-weight: 600;
            }

            .summary-card {
                background: #ffffff;
                border: 1px solid var(--line);
                border-left: 6px solid var(--accent);
                border-radius: 14px;
                padding: 0.9rem 0.95rem;
                margin-bottom: 0.8rem;
            }

            .summary-title {
                margin: 0;
                font-size: 0.95rem;
                font-weight: 700;
            }

            .summary-meta {
                margin: 0.2rem 0 0.35rem;
                color: var(--muted);
                font-size: 0.8rem;
            }

            .summary-snippet {
                margin: 0;
                font-size: 0.88rem;
                color: #2b3944;
                line-height: 1.5;
            }

            .placeholder-box {
                border: 1px dashed #b8c1c8;
                border-radius: 12px;
                padding: 0.75rem;
                background: #f8fafb;
                color: var(--muted);
                font-size: 0.86rem;
            }

            .kpi {
                background: #ffffff;
                border: 1px solid var(--line);
                border-radius: 12px;
                padding: 0.7rem 0.8rem;
                text-align: center;
            }

            .kpi .value {
                font-size: 1.2rem;
                font-weight: 700;
                color: var(--ink);
            }

            .kpi .label {
                font-size: 0.78rem;
                color: var(--muted);
            }

            [data-testid="stFileUploader"] {
                background: #fcfaf5;
                border: 1px dashed #a5b2be;
                border-radius: 12px;
                padding: 0.2rem;
            }

            .stButton > button,
            .stDownloadButton > button {
                border-radius: 10px;
                border: 1px solid #0f5f5a;
                background: var(--accent);
                color: white;
                font-weight: 600;
                padding: 0.42rem 0.9rem;
            }

            .stButton > button:hover,
            .stDownloadButton > button:hover {
                background: #0f5f5a;
                border-color: #0f5f5a;
            }

            @media (max-width: 900px) {
                .hero h1 {
                    font-size: 1.5rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Research Summarizer Dashboard</h1>
            <p>
                Upload research PDFs, generate structured summaries, and fetch topic-based studies
                with concise insights. This demo focuses on the interface only.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Workspace Controls")
        st.selectbox(
            "Domain",
            options=["All", "AI/ML", "Healthcare", "Climate", "Finance", "Education"],
            index=0,
        )
        st.multiselect(
            "Summary Style",
            options=["Abstract", "Bullet Highlights", "Methods", "Results", "Limitations"],
            default=["Bullet Highlights", "Results"],
        )
        st.slider("Max Summary Length (words)", min_value=100, max_value=1000, value=350, step=50)
        st.divider()
        st.caption("UI Preview: controls are visual and not wired to backend logic.")


def render_kpis() -> None:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="kpi"><div class="value">24</div><div class="label">PDFs Uploaded</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="kpi"><div class="value">18</div><div class="label">Summaries Ready</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="kpi"><div class="value">11</div><div class="label">Topics Tracked</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="kpi"><div class="value">92%</div><div class="label">Coverage Score</div></div>', unsafe_allow_html=True)


def render_uploaded_summary_preview() -> None:
    st.markdown('<p class="section-title">Recent PDF Summaries</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Generated summaries from uploaded studies</p>', unsafe_allow_html=True)

    cards = [
        {
            "title": "Transformer Approaches for Low-Resource Languages",
            "meta": "Uploaded 2h ago  |  18 pages",
            "snippet": "Compares multilingual transformer variants and reports stronger transfer learning gains when domain-specific pretraining is combined with targeted lexical augmentation.",
        },
        {
            "title": "Meta-analysis: AI-Assisted Clinical Triaging",
            "meta": "Uploaded yesterday  |  24 pages",
            "snippet": "Finds consistent improvements in triage speed, while outcome quality remains sensitive to dataset drift and calibration quality across hospitals.",
        },
        {
            "title": "Adaptive Energy Modeling in Smart Grids",
            "meta": "Uploaded 3 days ago  |  15 pages",
            "snippet": "Highlights that hybrid statistical + neural methods reduce short-term forecasting error and improve resilience under peak demand volatility.",
        },
    ]

    for card in cards:
        st.markdown(
            f"""
            <div class="summary-card">
                <p class="summary-title">{card['title']}</p>
                <p class="summary-meta">{card['meta']}</p>
                <p class="summary-snippet">{card['snippet']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_topic_fetch_preview() -> None:
    st.markdown('<p class="section-title">Fetched by Topic</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Discovery feed with summarized inputs</p>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="pill-row">
            <span class="pill">Topic: Explainable AI</span>
            <span class="pill">Topic: Medical Imaging</span>
            <span class="pill">Topic: Carbon Forecasting</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="summary-card">
            <p class="summary-title">Cross-domain Explainability Benchmark 2026</p>
            <p class="summary-meta">Source: arXiv mirror  |  Relevance: High</p>
            <p class="summary-snippet">
                Proposes a unified benchmark that evaluates explanation consistency across text,
                image, and tabular models; reports fidelity gains for ensemble explanation methods.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="summary-card">
            <p class="summary-title">Foundation Models for Radiology Prioritization</p>
            <p class="summary-meta">Source: PubMed sync  |  Relevance: Medium</p>
            <p class="summary-snippet">
                Reviews triage-oriented workflows and indicates that prompt-constrained
                vision-language pipelines outperform traditional CNN triage baselines.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard() -> None:
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">1) Summarize Uploaded Research PDFs</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-subtitle">Upload one or more studies, set summary options, and generate structured output.</p>',
            unsafe_allow_html=True,
        )
        st.file_uploader(
            "Upload Research Studies (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
            help="UI placeholder: ingestion pipeline is not connected yet.",
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.selectbox("Output Format", ["Executive Brief", "Structured Sections", "Bulleted Notes"]) 
        with c2:
            st.selectbox("Audience", ["Researcher", "Student", "Product Team", "General"])
        with c3:
            st.selectbox("Citation Mode", ["Inline", "Footnotes", "No Citations"])

        st.text_area(
            "Optional Guidance for the Summarizer",
            placeholder="Example: Focus on methodology robustness and practical limitations for deployment.",
            height=90,
        )

        b1, b2 = st.columns([0.38, 0.62])
        with b1:
            st.button("Generate Summary", use_container_width=True)
        with b2:
            st.download_button(
                "Export Summary (Preview)",
                data="UI placeholder",
                file_name="summary_preview.txt",
                use_container_width=True,
            )

        st.markdown('<div class="placeholder-box">Summary output panel placeholder: this area can show generated abstract, key findings, limitations, and action points.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        render_uploaded_summary_preview()
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">2) Fetch Studies by Topic with Summarized Input</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-subtitle">Enter one or more topics, define retrieval range, and preview summarized results.</p>',
            unsafe_allow_html=True,
        )

        st.text_input(
            "Topics (comma-separated)",
            value="explainable ai, medical imaging, carbon forecasting",
            placeholder="example: federated learning, oncology, bioinformatics",
        )
        r1, r2 = st.columns(2)
        with r1:
            st.slider("Max Studies", min_value=5, max_value=100, value=25, step=5)
        with r2:
            st.selectbox("Date Window", ["Last 30 days", "Last 6 months", "Last 2 years", "Any time"])

        st.select_slider("Relevance Threshold", options=["Low", "Medium", "High", "Very High"], value="High")

        st.button("Fetch and Summarize", use_container_width=True)
        st.markdown('<div class="placeholder-box">Retrieval output placeholder: this section can show ranked studies, source metadata, and concise summaries by topic.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        render_topic_fetch_preview()
        st.markdown('</div>', unsafe_allow_html=True)


def main() -> None:
    inject_styles()
    render_sidebar()
    render_hero()
    st.write("")
    render_kpis()
    st.write("")
    render_dashboard()


if __name__ == "__main__":
    main()
