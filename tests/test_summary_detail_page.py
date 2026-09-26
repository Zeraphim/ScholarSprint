"""Page-level checks for the Summary Detail export controls."""

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from exporters import EXPORT_FORMATS, MARKDOWN

PAGE = str(Path(__file__).resolve().parent.parent / "pages" / "3_Summary_Detail.py")

SUMMARY = {
    "file_name": "attention.pdf",
    "summary_text": (
        "Executive Brief: The paper introduces the Transformer.\n\n"
        "Methods and Evidence:\n- Trained on WMT 2014\n- BLEU \u2265 28.4"
    ),
    "word_count": 20,
    "generated_at": "2026-09-26 19:00",
    "engine": "local extractive",
}


class SummaryDetailExportTests(unittest.TestCase):
    def _run_page(self, summaries=(SUMMARY,)):
        app = AppTest.from_file(PAGE, default_timeout=60)
        app.session_state["generated_summaries"] = list(summaries)
        return app.run()

    def test_default_offers_a_markdown_download(self):
        app = self._run_page()
        self.assertEqual(app.exception.values, [])
        self.assertEqual(app.selectbox(key="export_format").value, MARKDOWN)
        self.assertEqual(
            app.get("download_button")[0].label, f"Download {MARKDOWN}"
        )

    def test_every_format_renders_without_error(self):
        app = self._run_page()
        for label in EXPORT_FORMATS:
            with self.subTest(format=label):
                app.selectbox(key="export_format").set_value(label).run()
                self.assertEqual(app.exception.values, [])
                self.assertEqual(app.error.values, [])
                self.assertEqual(app.warning.values, [])
                self.assertEqual(
                    app.get("download_button")[0].label, f"Download {label}"
                )

    def test_no_summaries_shows_guidance_instead_of_export(self):
        app = self._run_page(summaries=())
        self.assertEqual(app.exception.values, [])
        self.assertEqual(app.get("download_button"), [])
        self.assertTrue(any("No generated summaries" in value for value in app.info.values))


if __name__ == "__main__":
    unittest.main()
