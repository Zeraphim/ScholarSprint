from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import os
import re
import textwrap
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
from pypdf import PdfReader

from pydantic_ai import Agent
try:
    from pydantic_ai.models.openai import OpenAISettings
except ImportError:
    from pydantic_ai.models.openai import OpenAIModelSettings as OpenAISettings


st.set_page_config(
    page_title="Scholar Sprint",
    page_icon="assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)


ARXIV_API_URL = "https://export.arxiv.org/api/query"
SUMMARY_CACHE_DIR = Path(".cache/study_summaries")
GENERATED_SUMMARIES_FILE = Path(".cache/generated_summaries.json")
OPENROUTER_MAX_RETRIES = 2
OPENROUTER_RETRY_BASE_DELAY = 1.2
MODEL_OPTIONS = [
    "openrouter:minimax/minimax-m2.5:free",
    "openrouter:minimax/minimax-m2.5",
]
DEFAULT_MODEL = "openrouter:minimax/minimax-m2.5"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
SIDEBAR_LOGO_PATH = ASSETS_DIR / "logo.png"
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
    "this",
    "these",
    "those",
    "their",
    "we",
    "our",
    "or",
    "if",
    "into",
    "using",
    "used",
    "use",
}


@dataclass
class Study:
    title: str
    summary: str
    published: dt.date
    source: str
    url: str
    matched_topic: str
    relevance_score: float


def clean_text(raw_text: str) -> str:
    cleaned = re.sub(r"\s+", " ", raw_text or "").strip()
    return cleaned


def normalize_llm_output_markdown(raw_text: str) -> str:
    text = (raw_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""

    headings = [
        "Executive Summary",
        "Methods and Evidence",
        "Limitations and Risks",
        "Executive Brief",
        "Citation",
    ]
    for heading in headings:
        pattern = rf"(#{'{'}1,6{'}'}\s+(?:\d+\.\s*)?{re.escape(heading)})\s+"
        text = re.sub(pattern, r"\1\n\n", text, flags=re.IGNORECASE)

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    chunks = re.split(r"(?<=[.!?])\s+", text)
    return [chunk.strip() for chunk in chunks if len(chunk.strip()) > 30]


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]{1,}", text.lower())
    return [w for w in words if w not in STOP_WORDS]


def extractive_summarize(text: str, max_words: int, focus_hint: str = "") -> str:
    sentences = split_sentences(text)
    if not sentences:
        return "No text could be extracted for summarization."

    token_freq = Counter(tokenize(text + " " + focus_hint))
    if not token_freq:
        return " ".join(sentences[:3])

    sentence_scores: list[tuple[float, str]] = []
    for sentence in sentences:
        sentence_tokens = tokenize(sentence)
        if not sentence_tokens:
            continue
        score = sum(token_freq[token] for token in sentence_tokens) / max(len(sentence_tokens), 1)
        sentence_scores.append((score, sentence))

    sentence_scores.sort(key=lambda item: item[0], reverse=True)
    selected: list[str] = []
    total_words = 0
    for _, sentence in sentence_scores:
        sentence_word_count = len(sentence.split())
        if total_words + sentence_word_count > max_words and selected:
            continue
        selected.append(sentence)
        total_words += sentence_word_count
        if total_words >= max_words:
            break

    if not selected:
        selected = [sentences[0]]

    return " ".join(selected)


def build_structured_summary(
    title: str,
    text: str,
    max_words: int,
    output_format: str,
    audience: str,
    citation_mode: str,
    guidance: str,
    styles: list[str],
) -> str:
    highlight_words = max(120, int(max_words * 0.45))
    method_words = max(80, int(max_words * 0.28))
    limit_words = max(60, int(max_words * 0.22))

    key_points = extractive_summarize(text, highlight_words, focus_hint=guidance)
    methods = extractive_summarize(text, method_words, focus_hint="method experiment dataset")
    limitations = extractive_summarize(text, limit_words, focus_hint="limitation bias future work")

    header = [
        f"Title: {title}",
        f"Audience: {audience}",
        f"Format: {output_format}",
        f"Citation mode: {citation_mode}",
        f"Summary styles: {', '.join(styles) if styles else 'Default'}",
    ]

    sections = [
        "\n".join(header),
        "",
        "Executive Summary:",
        textwrap.fill(key_points, width=100),
        "",
        "Methods and Evidence:",
        textwrap.fill(methods, width=100),
        "",
        "Limitations and Risks:",
        textwrap.fill(limitations, width=100),
    ]

    if guidance.strip():
        sections.extend(["", f"Custom Guidance Applied: {guidance.strip()}"])

    if citation_mode != "No Citations":
        sections.extend(["", f"Citation Placeholder ({citation_mode}): [1]"])

    return "\n".join(sections)


def build_llm_prompt(
    text: str,
    max_words: int,
    output_format: str,
    audience: str,
    citation_mode: str,
    guidance: str,
    styles: list[str],
) -> str:
    style_value = ", ".join(styles) if styles else "Default"
    guidance_value = guidance.strip() if guidance.strip() else "No extra guidance"

    return (
        "You are a research summarization assistant.\n\n"
        "Generate a concise, structured summary based on the provided paper content.\n\n"
        "Constraints:\n"
        f"- Keep total output around {max_words} words.\n"
        f"- Output format preference: {output_format}\n"
        f"- Audience: {audience}\n"
        f"- Citation mode: {citation_mode}\n"
        f"- Style preferences: {style_value}\n"
        f"- Guidance: {guidance_value}\n\n"
        "Formatting rules (strict):\n"
        "- Return valid Markdown only.\n"
        "- Each heading must be on its own line, with no body text on the same line.\n"
        "- Add one blank line after each heading before paragraph text.\n"
        "- Never write this pattern: '## 1. Executive Summary This ...'.\n"
        "- Use normal paragraphs for body text; do not make whole paragraphs bold.\n"
        "- Use bold or italics sparingly for short phrases only.\n\n"
        "Required sections:\n"
        "1) Executive Summary\n"
        "2) Methods and Evidence\n"
        "3) Limitations and Risks\n\n"
        "Output template (follow exactly):\n"
        "## Executive Summary\n\n"
        "<paragraphs here>\n\n"
        "## Methods and Evidence\n\n"
        "<paragraphs here>\n\n"
        "## Limitations and Risks\n\n"
        "<paragraphs here>\n\n"
        "Input paper content:\n"
        f"\"\"\"\n{text[:24000]}\n\"\"\""
    )


def summarize_with_openrouter(
    text: str,
    selected_model: str,
    max_words: int,
    output_format: str,
    audience: str,
    citation_mode: str,
    guidance: str,
    styles: list[str],
) -> str:
    if Agent is None or OpenAISettings is None:
        raise RuntimeError("pydantic_ai is not available.")

    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    prompt = build_llm_prompt(
        text=text,
        max_words=max_words,
        output_format=output_format,
        audience=audience,
        citation_mode=citation_mode,
        guidance=guidance,
        styles=styles,
    )
    agent = Agent(
        selected_model,
        instructions=(
            "You summarize research papers clearly and accurately. "
            "Do not invent citations. If details are missing, say so briefly."
        ),
        model_settings=OpenAISettings(temperature=0.2),
    )
    last_error: Exception | None = None
    for attempt in range(OPENROUTER_MAX_RETRIES + 1):
        try:
            result = agent.run_sync(prompt)
            return normalize_llm_output_markdown(result.output)
        except Exception as exc:
            last_error = exc
            if attempt >= OPENROUTER_MAX_RETRIES:
                break
            delay = OPENROUTER_RETRY_BASE_DELAY * (2 ** attempt)
            time.sleep(delay)

    raise RuntimeError(f"OpenRouter request failed after retries: {last_error}")


def generate_summary_text(
    text: str,
    selected_model: str,
    max_words: int,
    output_format: str,
    audience: str,
    citation_mode: str,
    guidance: str,
    styles: list[str],
) -> tuple[str, str]:
    use_openrouter = selected_model.startswith("openrouter:")
    if use_openrouter:
        try:
            summary = summarize_with_openrouter(
                text=text,
                selected_model=selected_model,
                max_words=max_words,
                output_format=output_format,
                audience=audience,
                citation_mode=citation_mode,
                guidance=guidance,
                styles=styles,
            )
            return summary, f"Model: {selected_model}"
        except Exception as exc:
            fallback = build_structured_summary(
                title="Fallback Summary",
                text=text,
                max_words=max_words,
                output_format=output_format,
                audience=audience,
                citation_mode=citation_mode,
                guidance=guidance,
                styles=styles,
            )
            return fallback, f"Fallback (extractive) because OpenRouter failed: {exc}"

    fallback = build_structured_summary(
        title="Extractive Summary",
        text=text,
        max_words=max_words,
        output_format=output_format,
        audience=audience,
        citation_mode=citation_mode,
        guidance=guidance,
        styles=styles,
    )
    return fallback, "Fallback (extractive local summarizer)"


def summarize_topic_result(text: str, topic: str, selected_model: str) -> str:
    if selected_model.startswith("openrouter:"):
        try:
            return summarize_with_openrouter(
                text=text,
                selected_model=selected_model,
                max_words=90,
                output_format="Bulleted Notes",
                audience="Researcher",
                citation_mode="No Citations",
                guidance=f"Focus on relevance to topic: {topic}",
                styles=["Bullet Highlights"],
            )
        except Exception:
            pass
    return extractive_summarize(text, max_words=85, focus_hint=topic)


def quick_relevance_check(study_text: str, topic: str) -> bool:
    topic_tokens = set(tokenize(topic))
    if not topic_tokens:
        return True
    text_tokens = set(tokenize(study_text))
    if not text_tokens:
        return False
    return len(topic_tokens.intersection(text_tokens)) >= 1


def build_study_cache_key(title: str, url: str, topic: str, selected_model: str) -> str:
    payload = f"{clean_text(title)}|{clean_text(url)}|{clean_text(topic)}|{selected_model}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_summary_from_disk_cache(cache_key: str) -> str | None:
    cache_file = SUMMARY_CACHE_DIR / f"{cache_key}.json"
    if not cache_file.exists():
        return None
    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    summary = payload.get("summary", "")
    if not isinstance(summary, str):
        return None
    return summary or None


def save_summary_to_disk_cache(cache_key: str, summary: str) -> None:
    try:
        SUMMARY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = SUMMARY_CACHE_DIR / f"{cache_key}.json"
        payload = {"summary": summary, "cached_at": dt.datetime.now().isoformat()}
        cache_file.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        return


def summarize_topic_result_with_timing(text: str, topic: str, selected_model: str) -> tuple[str, float]:
    started_at = time.perf_counter()
    summary = summarize_topic_result(text=text, topic=topic, selected_model=selected_model)
    return summary, time.perf_counter() - started_at


def load_persisted_generated_summaries() -> list[dict]:
    if not GENERATED_SUMMARIES_FILE.exists():
        return []
    try:
        payload = json.loads(GENERATED_SUMMARIES_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    valid_items: list[dict] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        if "file_name" not in item or "summary_text" not in item:
            continue
        valid_items.append(item)
    return valid_items


def persist_generated_summaries(items: list[dict]) -> None:
    try:
        GENERATED_SUMMARIES_FILE.parent.mkdir(parents=True, exist_ok=True)
        GENERATED_SUMMARIES_FILE.write_text(json.dumps(items), encoding="utf-8")
    except OSError:
        return


def extract_text_from_pdf(uploaded_file) -> str:
    try:
        file_bytes = uploaded_file.getvalue()
        reader = PdfReader(io.BytesIO(file_bytes))
        pages_text = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages_text.append(page_text)
        return clean_text("\n".join(pages_text))
    except Exception:
        return ""


def date_window_to_start(date_window: str) -> dt.date:
    today = dt.date.today()
    if date_window == "Last 30 days":
        return today - dt.timedelta(days=30)
    if date_window == "Last 6 months":
        return today - dt.timedelta(days=180)
    if date_window == "Last 2 years":
        return today - dt.timedelta(days=730)
    return dt.date(1900, 1, 1)


def relevance_threshold_to_score(value: str) -> float:
    return {
        "Low": 0.2,
        "Medium": 0.35,
        "High": 0.5,
        "Very High": 0.65,
    }.get(value, 0.5)


def score_study_relevance(study_text: str, topic: str) -> float:
    topic_tokens = tokenize(topic)
    if not topic_tokens:
        return 0.0
    text_tokens = tokenize(study_text)
    if not text_tokens:
        return 0.0
    token_counts = Counter(text_tokens)
    match_count = sum(token_counts[token] for token in topic_tokens)
    normalized = match_count / max(len(text_tokens) * 0.02, 1)
    return min(normalized, 1.0)


@st.cache_data(ttl=3600)
def fetch_arxiv_for_topic(topic: str, max_results: int) -> list[dict[str, str]]:
    encoded_topic = urllib.parse.quote(topic)
    url = (
        f"{ARXIV_API_URL}?search_query=all:{encoded_topic}"
        f"&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "ResearchSummarizer/1.0"})
    with urllib.request.urlopen(req, timeout=15) as response:
        data = response.read()

    root = ET.fromstring(data)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    studies: list[dict[str, str]] = []

    for entry in root.findall("atom:entry", namespace):
        title = clean_text(entry.findtext("atom:title", default="", namespaces=namespace))
        summary = clean_text(entry.findtext("atom:summary", default="", namespaces=namespace))
        published = clean_text(entry.findtext("atom:published", default="", namespaces=namespace))
        link = ""
        for node in entry.findall("atom:link", namespace):
            if node.attrib.get("rel") == "alternate":
                link = node.attrib.get("href", "")
                break

        studies.append(
            {
                "title": title,
                "summary": summary,
                "published": published,
                "url": link,
            }
        )

    return studies


def parse_topics(raw_topics: str) -> list[str]:
    return [topic.strip() for topic in raw_topics.split(",") if topic.strip()]


def init_state() -> None:
    st.session_state.setdefault("generated_summaries", load_persisted_generated_summaries())
    st.session_state.setdefault("topic_results", [])
    st.session_state.setdefault("last_error", "")
    st.session_state.setdefault("selected_model", DEFAULT_MODEL)
    if st.session_state.get("selected_model") not in MODEL_OPTIONS:
        st.session_state["selected_model"] = DEFAULT_MODEL
    st.session_state.setdefault("study_summary_cache", {})
    st.session_state.setdefault("parallel_workers", 4)
    st.session_state.setdefault("use_persistent_cache", True)
    st.session_state.setdefault("show_timing_logs", False)


def resolve_theme_palette() -> dict[str, str]:
    theme_base = (st.get_option("theme.base") or "light").lower()

    light_defaults = {
        "bg": "#f5f1e8",
        "bg_alt": "#f2efe7",
        "paper": "#fffcf6",
        "ink": "#19242d",
        "muted": "#5d6974",
        "accent": "#16706b",
        "accent_soft": "#d9f0ee",
        "accent_soft_border": "#b5e1dc",
        "accent_text_on_soft": "#0d4f4a",
        "line": "#d8d6cc",
        "card_shadow": "0 10px 35px rgba(25, 36, 45, 0.08)",
        "snippet_text": "#2b3944",
        "strong_text": "#0f1f2b",
        "em_text": "#334452",
        "file_uploader_bg": "#fcfaf5",
        "file_uploader_border": "#a5b2be",
        "placeholder_bg": "#f8fafb",
        "placeholder_border": "#b8c1c8",
        "button_bg": "#d9f0ee",
        "button_text": "#0d4f4a",
        "button_border": "#b5e1dc",
        "button_hover": "#c7e8e4",
        "sidebar_bg": "#f0ece2",
        "summary_markdown_bg": "#fffdf8",
        "markdown_gradient_1": "#fff4df",
        "markdown_gradient_2": "#d9f0ee",
    }
    dark_defaults = {
        "bg": "#0f1117",
        "bg_alt": "#11151f",
        "paper": "#161a24",
        "ink": "#fafafa",
        "muted": "#b6b9c2",
        "accent": "#4a9d9a",
        "accent_soft": "#1f4d4b",
        "accent_soft_border": "#2a6563",
        "accent_text_on_soft": "#a0d9d5",
        "line": "#2a3042",
        "card_shadow": "0 10px 35px rgba(0, 0, 0, 0.4)",
        "snippet_text": "#d0d0d0",
        "strong_text": "#f0f0f0",
        "em_text": "#c8c8c8",
        "file_uploader_bg": "#171b27",
        "file_uploader_border": "#3e4761",
        "placeholder_bg": "#171b27",
        "placeholder_border": "#3e4761",
        "button_bg": "#2a6563",
        "button_text": "#d7f0ee",
        "button_border": "#3a7b78",
        "button_hover": "#347673",
        "sidebar_bg": "#111621",
        "summary_markdown_bg": "#151a26",
        "markdown_gradient_1": "#1a2233",
        "markdown_gradient_2": "#0f1a2d",
    }

    defaults = dark_defaults if theme_base == "dark" else light_defaults

    return {
        **defaults,
        "bg": st.get_option("theme.backgroundColor") or defaults["bg"],
        "paper": st.get_option("theme.secondaryBackgroundColor") or defaults["paper"],
        "ink": st.get_option("theme.textColor") or defaults["ink"],
        "accent": st.get_option("theme.primaryColor") or defaults["accent"],
    }


def inject_styles() -> None:
    palette = resolve_theme_palette()
    theme_css_vars = "\n".join(
        [
            ":root {",
            f"    --bg: {palette['bg']};",
            f"    --bg-alt: {palette['bg_alt']};",
            f"    --paper: {palette['paper']};",
            f"    --ink: {palette['ink']};",
            f"    --muted: {palette['muted']};",
            f"    --accent: {palette['accent']};",
            f"    --accent-soft: {palette['accent_soft']};",
            f"    --accent-soft-border: {palette['accent_soft_border']};",
            f"    --accent-text-on-soft: {palette['accent_text_on_soft']};",
            f"    --line: {palette['line']};",
            f"    --card-shadow: {palette['card_shadow']};",
            f"    --snippet-text: {palette['snippet_text']};",
            f"    --strong-text: {palette['strong_text']};",
            f"    --em-text: {palette['em_text']};",
            f"    --file-uploader-bg: {palette['file_uploader_bg']};",
            f"    --file-uploader-border: {palette['file_uploader_border']};",
            f"    --placeholder-bg: {palette['placeholder_bg']};",
            f"    --placeholder-border: {palette['placeholder_border']};",
            f"    --button-bg: {palette['button_bg']};",
            f"    --button-text: {palette['button_text']};",
            f"    --button-border: {palette['button_border']};",
            f"    --button-hover: {palette['button_hover']};",
            f"    --sidebar-bg: {palette['sidebar_bg']};",
            f"    --summary-markdown-bg: {palette['summary_markdown_bg']};",
            f"    --markdown-gradient-1: {palette['markdown_gradient_1']};",
            f"    --markdown-gradient-2: {palette['markdown_gradient_2']};",
            "}",
        ]
    )

    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

            __THEME_CSS_VARS__

            html, body, [class*="css"] {
                font-family: 'Space Grotesk', sans-serif;
                color: var(--ink);
            }

            .stApp {
                background:
                    radial-gradient(circle at 0% 0%, var(--markdown-gradient-1) 0%, transparent 35%),
                    radial-gradient(circle at 100% 0%, var(--markdown-gradient-2) 0%, transparent 30%),
                    linear-gradient(180deg, var(--bg) 0%, var(--bg-alt) 100%);
            }

            [data-testid="stSidebar"] {
                border-right: 1px solid var(--line);
                background: var(--sidebar-bg);
            }

            .sidebar-brand-title {
                margin: 0.1rem 0 0;
                font-size: 1.18rem;
                font-weight: 700;
                color: var(--ink);
                line-height: 1.2;
                letter-spacing: -0.01em;
            }

            .sidebar-brand-subtitle {
                margin: 0.2rem 0 0.75rem;
                color: var(--muted);
                font-size: 0.83rem;
                line-height: 1.45;
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
                color: var(--ink);
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
                color: var(--ink);
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
                border: 1px solid var(--accent-soft-border);
                color: var(--accent-text-on-soft);
                padding: 0.2rem 0.6rem;
                border-radius: 999px;
                font-size: 0.78rem;
                font-weight: 600;
            }

            .summary-card {
                background: var(--paper);
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
                color: var(--ink);
            }

            .summary-meta {
                margin: 0.2rem 0 0.35rem;
                color: var(--muted);
                font-size: 0.8rem;
            }

            .summary-snippet {
                margin: 0;
                font-size: 0.88rem;
                color: var(--snippet-text);
                line-height: 1.5;
            }

            .summary-markdown {
                background: var(--summary-markdown-bg);
                border: 1px solid var(--line);
                border-radius: 14px;
                padding: 1rem 1.1rem;
                margin-top: 0.35rem;
            }

            .summary-markdown h2 {
                margin-top: 0.3rem;
                margin-bottom: 0.55rem;
                font-size: 1.15rem;
                letter-spacing: -0.01em;
                color: var(--ink);
            }

            .summary-markdown h3 {
                margin-top: 0.95rem;
                margin-bottom: 0.35rem;
                font-size: 0.98rem;
                color: var(--ink);
            }

            .summary-markdown p {
                line-height: 1.62;
                margin-bottom: 0.6rem;
                color: var(--ink);
            }

            .summary-markdown ul {
                margin-top: 0.2rem;
                margin-bottom: 0.6rem;
            }

            .summary-markdown li {
                margin-bottom: 0.3rem;
                color: var(--ink);
            }

            [data-testid="stMarkdownContainer"] h1,
            [data-testid="stMarkdownContainer"] h2,
            [data-testid="stMarkdownContainer"] h3,
            [data-testid="stMarkdownContainer"] h4 {
                color: var(--ink);
                letter-spacing: -0.01em;
                line-height: 1.35;
            }

            [data-testid="stMarkdownContainer"] p,
            [data-testid="stMarkdownContainer"] li {
                line-height: 1.68;
                color: var(--ink);
            }

            [data-testid="stMarkdownContainer"] strong {
                font-weight: 700;
                color: var(--strong-text);
            }

            [data-testid="stMarkdownContainer"] em {
                font-style: italic;
                color: var(--em-text);
            }

            [data-testid="stMarkdownContainer"] hr {
                border: 0;
                border-top: 1px solid var(--line);
                margin: 1rem 0;
            }

            .summary-only-wrap {
                max-width: 980px;
                margin: 0 auto;
            }

            .placeholder-box {
                border: 1px dashed var(--placeholder-border);
                border-radius: 12px;
                padding: 0.75rem;
                background: var(--placeholder-bg);
                color: var(--muted);
                font-size: 0.86rem;
            }

            .kpi {
                background: var(--paper);
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
                background: var(--file-uploader-bg);
                border: 1px dashed var(--file-uploader-border);
                border-radius: 12px;
                padding: 0.2rem;
            }

            .stButton > button,
            .stDownloadButton > button {
                border-radius: 10px;
                border: 1px solid var(--button-border);
                background: var(--button-bg) !important;
                color: var(--button-text) !important;
                font-weight: 600;
                padding: 0.42rem 0.9rem;
            }

            .stButton > button *,
            .stDownloadButton > button * {
                color: var(--button-text) !important;
            }

            .stButton > button:hover,
            .stDownloadButton > button:hover {
                background: var(--button-hover) !important;
                border-color: var(--button-border) !important;
            }

            /* Input fields and text areas */
            input, textarea, select {
                background: var(--paper) !important;
                color: var(--ink) !important;
                border-color: var(--line) !important;
            }

            /* Ensure all text is readable */
            [data-testid="stMetricLabel"] {
                color: var(--muted) !important;
            }

            /* Selectboxes and multiselects */
            [data-baseweb="select__core"] {
                color: var(--ink) !important;
            }

            [data-testid="stSidebar"] [data-baseweb="select"] > div,
            [data-testid="stSelectbox"] [data-baseweb="select"] > div,
            [data-testid="stMultiSelect"] [data-baseweb="select"] > div {
                background: var(--paper) !important;
                border-color: var(--line) !important;
                box-shadow: none !important;
            }

            [data-testid="stSidebar"] [data-baseweb="select"]:hover > div,
            [data-testid="stSelectbox"] [data-baseweb="select"]:hover > div,
            [data-testid="stMultiSelect"] [data-baseweb="select"]:hover > div,
            [data-testid="stSidebar"] [data-baseweb="select"]:focus-within > div,
            [data-testid="stSelectbox"] [data-baseweb="select"]:focus-within > div,
            [data-testid="stMultiSelect"] [data-baseweb="select"]:focus-within > div {
                background: var(--paper) !important;
                border-color: var(--accent) !important;
                box-shadow: 0 0 0 1px var(--accent-soft-border) !important;
            }

            [data-testid="stSidebar"] [data-baseweb="select"] input,
            [data-testid="stMultiSelect"] [data-baseweb="select"] input {
                background: transparent !important;
                color: var(--ink) !important;
            }

            [data-testid="stMultiSelect"] [data-baseweb="tag"] {
                background: var(--accent-soft) !important;
                color: var(--accent-text-on-soft) !important;
                border: 1px solid var(--accent-soft-border) !important;
            }

            [data-testid="stSelectbox"] {
                color: var(--ink) !important;
            }

            /* Links */
            a {
                color: var(--accent) !important;
            }

            a:hover {
                color: var(--button-hover) !important;
            }

            @media (max-width: 900px) {
                .hero h1 {
                    font-size: 1.5rem;
                }
            }
        </style>
        """.replace("__THEME_CSS_VARS__", theme_css_vars),
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Scholar Sprint</h1>
            <p>
                Choose one workflow at a time: summarize uploaded PDFs or fetch studies by topic.
                The app now separates tasks into focused pages to reduce clutter and cognitive load.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_home_hub() -> None:
    st.markdown("### Start with a Focused Workflow")
    st.caption("Use the sidebar page navigation or the quick links below.")

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">1) Summarize Uploaded Research PDF</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-subtitle">Upload one or more papers and generate structured summaries.</p>',
            unsafe_allow_html=True,
        )
        if hasattr(st, "page_link"):
            st.page_link(
                "pages/1_Summarize_Uploaded_Research_PDF.py",
                label="Open PDF Summarization Page",
            )
        else:
            st.info("Open the 'Summarize Uploaded Research PDF' page from the sidebar.")
        render_uploaded_summary_preview()
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">2) Fetch Studies by Topic with Summarized Input</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-subtitle">Search arXiv topics, filter by relevance, and review concise summaries.</p>',
            unsafe_allow_html=True,
        )
        if hasattr(st, "page_link"):
            st.page_link(
                "pages/2_Fetch_Studies_By_Topic.py",
                label="Open Topic Fetch Page",
            )
        else:
            st.info("Open the 'Fetch Studies by Topic with Summarized Input' page from the sidebar.")
        render_topic_fetch_preview()
        st.markdown('</div>', unsafe_allow_html=True)


def render_sidebar() -> None:
    with st.sidebar:
        if SIDEBAR_LOGO_PATH.exists():
            st.image(str(SIDEBAR_LOGO_PATH), use_container_width=True)
        st.markdown('<p class="sidebar-brand-title">Scholar Sprint</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="sidebar-brand-subtitle">Your AI-powered research summarizer</p>',
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown("### Workspace Controls")
        st.selectbox(
            "Summarization Model",
            options=MODEL_OPTIONS,
            key="selected_model",
            help="Edit MODEL_OPTIONS in app.py to add/remove models.",
        )
        st.selectbox(
            "Domain",
            options=["All", "AI/ML", "Healthcare", "Climate", "Finance", "Education"],
            index=0,
        )
        st.slider(
            "Parallel Requests",
            min_value=1,
            max_value=5,
            step=1,
            key="parallel_workers",
            help="Controls concurrent study summarization requests during topic fetch.",
        )
        st.checkbox(
            "Enable persistent summary cache",
            key="use_persistent_cache",
            help="Reuse previous study summaries across app restarts.",
        )
        st.checkbox(
            "Show timing logs",
            key="show_timing_logs",
            help="Display per-run performance breakdown for tuning.",
        )
        st.multiselect(
            "Summary Style",
            options=["Abstract", "Bullet Highlights", "Methods", "Results", "Limitations"],
            default=["Bullet Highlights", "Results"],
        )
        st.slider("Max Summary Length (words)", min_value=100, max_value=1000, value=350, step=50)
        st.divider()
        st.caption("Set OPENROUTER_API_KEY to enable OpenRouter models. Local fallback is used otherwise.")


def render_kpis() -> None:
    generated_summaries: list[dict] = st.session_state.get("generated_summaries", [])
    topic_results: list[Study] = st.session_state.get("topic_results", [])

    uploaded_count = len(generated_summaries)
    summaries_ready = uploaded_count + len(topic_results)
    topics_tracked = len({study.matched_topic for study in topic_results})
    avg_relevance = sum(study.relevance_score for study in topic_results) / len(topic_results) if topic_results else 0.0
    coverage_score = f"{avg_relevance * 100:.0f}%"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="kpi"><div class="value">{uploaded_count}</div><div class="label">PDFs Uploaded</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="kpi"><div class="value">{summaries_ready}</div><div class="label">Summaries Ready</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="kpi"><div class="value">{topics_tracked}</div><div class="label">Topics Tracked</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f'<div class="kpi"><div class="value">{coverage_score}</div><div class="label">Coverage Score</div></div>',
            unsafe_allow_html=True,
        )


def render_uploaded_summary_preview() -> None:
    st.markdown('<p class="section-title">Recent PDF Summaries</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Generated summaries from uploaded studies</p>', unsafe_allow_html=True)

    summaries = st.session_state.get("generated_summaries", [])
    if not summaries:
        st.markdown(
            '<div class="placeholder-box">No generated summaries yet. Upload PDFs and click Generate Summary.</div>',
            unsafe_allow_html=True,
        )
        return

    for summary in summaries[:5]:
        preview = textwrap.shorten(summary["summary_text"].replace("\n", " "), width=220, placeholder="...")
        st.markdown(
            f"""
            <div class="summary-card">
                <p class="summary-title">{summary['file_name']}</p>
                <p class="summary-meta">Words: {summary['word_count']}  |  Generated: {summary['generated_at']}</p>
                <p class="summary-snippet">{preview}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_topic_fetch_preview() -> None:
    st.markdown('<p class="section-title">Fetched by Topic</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Discovery feed with summarized inputs</p>', unsafe_allow_html=True)

    topic_results: list[Study] = st.session_state.get("topic_results", [])
    if not topic_results:
        st.markdown(
            '<div class="placeholder-box">No fetched studies yet. Add topics and click Fetch and Summarize.</div>',
            unsafe_allow_html=True,
        )
        return

    topics = sorted({study.matched_topic for study in topic_results})
    pills = "".join([f'<span class="pill">Topic: {topic}</span>' for topic in topics])
    st.markdown(f'<div class="pill-row">{pills}</div>', unsafe_allow_html=True)

    for study in topic_results[:8]:
        relative = f"Relevance: {study.relevance_score:.2f}"
        snippet = textwrap.shorten(study.summary, width=240, placeholder="...")
        st.markdown(
            f"""
            <div class="summary-card">
                <p class="summary-title">{study.title}</p>
                <p class="summary-meta">Source: {study.source}  |  {relative}  |  {study.published.isoformat()}</p>
                <p class="summary-snippet">{snippet}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if study.url:
            st.link_button("Open Study", study.url)


def render_dashboard() -> None:
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">1) Summarize Uploaded Research PDFs</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-subtitle">Upload one or more studies, set summary options, and generate structured output.</p>',
            unsafe_allow_html=True,
        )
        uploaded_files = st.file_uploader(
            "Upload Research Studies (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload PDF studies to generate summaries.",
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            output_format = st.selectbox("Output Format", ["Executive Brief", "Structured Sections", "Bulleted Notes"])
        with c2:
            audience = st.selectbox("Audience", ["Researcher", "Student", "Product Team", "General"])
        with c3:
            citation_mode = st.selectbox("Citation Mode", ["Inline", "Footnotes", "No Citations"])

        guidance = st.text_area(
            "Optional Guidance for the Summarizer",
            placeholder="Example: Focus on methodology robustness and practical limitations for deployment.",
            height=90,
        )

        b1, b2 = st.columns([0.38, 0.62])
        with b1:
            generate_summary_clicked = st.button("Generate Summary", use_container_width=True)
        with b2:
            export_payload = "\n\n".join(
                [
                    f"{item['file_name']}\n{'=' * len(item['file_name'])}\n{item['summary_text']}"
                    for item in st.session_state.get("generated_summaries", [])
                ]
            ) or "No summaries generated yet."
            st.download_button(
                "Export Summary (Preview)",
                data=export_payload,
                file_name="summary_preview.txt",
                use_container_width=True,
            )

        if generate_summary_clicked:
            summary_styles = st.session_state.get("Summary Style", [])
            max_summary_len = int(st.session_state.get("Max Summary Length (words)", 350))
            selected_model = st.session_state.get("selected_model", DEFAULT_MODEL)

            if not uploaded_files:
                st.warning("Please upload at least one PDF file.")
            else:
                generated = []
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
        if generated_summaries:
            st.markdown("### Generated Summary Output")
            current_url = st.query_params.to_dict()
            current_url["view"] = "summary"
            qs = urllib.parse.urlencode(current_url)
            st.markdown(
                f'<a href="?{qs}" target="_blank" style="font-size:0.9rem; font-weight:600;">Open Summary-Only Page</a>',
                unsafe_allow_html=True,
            )
            tabs = st.tabs([item["file_name"] for item in generated_summaries])
            for index, tab in enumerate(tabs):
                with tab:
                    item = generated_summaries[index]
                    st.caption(item.get("engine", ""))
                    rendered = format_summary_markdown(item["summary_text"], item["file_name"])
                    st.markdown(rendered)
        else:
            st.markdown(
                '<div class="placeholder-box">Summary output will appear here after you generate it from uploaded PDFs.</div>',
                unsafe_allow_html=True,
            )
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

        topics_raw = st.text_input(
            "Topics (comma-separated)",
            value="explainable ai, medical imaging, carbon forecasting",
            placeholder="example: federated learning, oncology, bioinformatics",
        )
        r1, r2 = st.columns(2)
        with r1:
            max_studies = st.slider("Max Studies", min_value=5, max_value=100, value=25, step=5)
        with r2:
            date_window = st.selectbox("Date Window", ["Last 30 days", "Last 6 months", "Last 2 years", "Any time"])

        relevance_threshold = st.select_slider("Relevance Threshold", options=["Low", "Medium", "High", "Very High"], value="High")

        fetch_clicked = st.button("Fetch and Summarize", use_container_width=True)

        if fetch_clicked:
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

        if st.session_state.get("topic_results"):
            st.markdown("### Retrieval Output")
            for idx, study in enumerate(st.session_state["topic_results"][:6], start=1):
                with st.expander(f"{idx}. {study.title}"):
                    st.write(f"Matched Topic: {study.matched_topic}")
                    st.write(f"Published: {study.published.isoformat()}")
                    st.write(f"Relevance Score: {study.relevance_score:.2f}")
                    st.write(study.summary)
                    if study.url:
                        st.link_button("Read Full Paper", study.url)
        else:
            st.markdown(
                '<div class="placeholder-box">Retrieval output will appear here after fetching by topic.</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        render_topic_fetch_preview()
        st.markdown('</div>', unsafe_allow_html=True)


def main() -> None:
    init_state()
    inject_styles()

    render_sidebar()
    render_hero()
    st.write("")
    render_kpis()
    st.write("")
    render_home_hub()


def summary_lines_to_markdown(lines: list[str]) -> list[str]:
    sections = {
        "title:": "## Paper",
        "executive brief:": "## Executive Brief",
        "citation:": "### Citation",
        "audience:": "### Audience",
        "format:": "### Format",
        "citation mode:": "### Citation Mode",
        "summary styles:": "### Styles",
        "executive summary:": "### Executive Summary",
        "methods and evidence:": "### Methods and Evidence",
        "limitations and risks:": "### Limitations and Risks",
        "custom guidance applied:": "### Applied Guidance",
        "citation placeholder": "### Citation Placeholder",
    }
    md: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if line == "---":
            md.append(line)
            continue

        if re.match(r"^#{1,6}\s+", line):
            md.append(line)
            continue

        lower = line.lower()
        matched = False
        for prefix, heading in sections.items():
            if lower.startswith(prefix):
                body = line.split(":", 1)[1].strip() if ":" in line else ""
                md.append(heading)
                if body:
                    md.append(body)
                matched = True
                break
        if matched:
            continue

        if line.startswith(("- ", "* ", "1. ")):
            md.append(line)
            continue

        md.append(line)

    return md


def normalize_summary_raw_text(summary_text: str) -> str:
    summary_text = normalize_llm_output_markdown(summary_text)
    text = (summary_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""

    # If a model returns everything in one line, split around common markdown tokens.
    text = re.sub(r"\s*---\s*", "\n\n---\n\n", text)
    text = re.sub(r"\s+(#{1,6}\s+)", r"\n\n\1", text)

    section_markers = [
        "Executive Brief:",
        "Citation:",
        "Executive Summary:",
        "Methods and Evidence:",
        "Limitations and Risks:",
        "Custom Guidance Applied:",
        "Citation Placeholder:",
    ]
    for marker in section_markers:
        pattern = rf"\s+({re.escape(marker)})"
        text = re.sub(pattern, r"\n\n\1", text, flags=re.IGNORECASE)

    # Collapse excessive blank lines while preserving section spacing.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def format_summary_markdown(summary_text: str, fallback_title: str) -> str:
    normalized_text = normalize_summary_raw_text(summary_text)
    lines = [f"Title: {fallback_title}"] + normalized_text.splitlines()
    formatted_lines = summary_lines_to_markdown(lines)
    if not formatted_lines:
        return "_No summary text available._"
    return "\n\n".join(formatted_lines)


def render_summary_only_page() -> None:
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"], header, footer {
                display: none !important;
            }
            .block-container {
                padding-top: 1.4rem;
                padding-bottom: 1.2rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="summary-only-wrap">', unsafe_allow_html=True)
    st.title("Summary-Only View")
    summaries = st.session_state.get("generated_summaries", [])

    if not summaries:
        st.info("No generated summaries yet. Return to the dashboard to create one.")
        st.markdown('<a href="?view=dashboard">Back to Dashboard</a>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    selected_name = st.selectbox("Choose Summary", [item["file_name"] for item in summaries], index=0)
    item = next((entry for entry in summaries if entry["file_name"] == selected_name), summaries[0])

    st.caption(item.get("engine", ""))
    rendered = format_summary_markdown(item["summary_text"], item["file_name"])
    st.markdown(rendered)
    st.markdown('<a href="?view=dashboard">Back to Dashboard</a>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
