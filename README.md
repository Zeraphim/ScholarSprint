# ResearchSummarizer

Streamlit dashboard UI for research study summarization workflows.

## Run

1. Create and sync environment:
	`uv venv && uv pip install -r requirements.txt`
2. Start the app:
	`uv run streamlit run app.py`

## Model Setup

1. Set your OpenRouter key:
	`export OPENROUTER_API_KEY="your_key_here"`
2. Choose model from the sidebar in the app.
3. Edit model list in `app.py` under `MODEL_OPTIONS` to add/remove models.

## Scope

- UI-only dashboard for uploading PDF studies and previewing summary workflow
- UI-only dashboard for fetching studies by topic with summarized output preview
- No backend processing or retrieval logic implemented yet
