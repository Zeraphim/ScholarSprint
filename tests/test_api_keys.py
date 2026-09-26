import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from api_keys import get_openai_api_key


class OpenAIKeyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.profile = Path(self.directory.name) / ".zprofile"
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def test_literal_assignments(self):
        for assignment in (
            'export OPENAI_API_KEY="sk-test_123"',
            "OPENAI_API_KEY='sk-test_123' # local key",
            "  export OPENAI_API_KEY=sk-test_123",
        ):
            with self.subTest(assignment=assignment):
                self.profile.write_text(assignment)
                self.assertEqual(get_openai_api_key(self.profile), "sk-test_123")

    def test_environment_takes_precedence(self):
        self.profile.write_text('export OPENAI_API_KEY="sk-profile"')
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-environment"}):
            self.assertEqual(get_openai_api_key(self.profile), "sk-environment")

    def test_default_profile_path(self):
        self.profile.write_text('export OPENAI_API_KEY="sk-profile"')
        with patch("api_keys.Path.home", return_value=self.profile.parent):
            self.assertEqual(get_openai_api_key(), "sk-profile")

    def test_missing_or_unreadable_profile(self):
        self.assertIsNone(get_openai_api_key(self.profile))
        with patch.object(Path, "read_text", side_effect=PermissionError):
            self.assertIsNone(get_openai_api_key(self.profile))

    def test_shell_commands_and_other_variables_are_not_evaluated(self):
        marker = self.profile.parent / "executed"
        self.profile.write_text(
            f'touch {marker}\n'
            f'export OPENAI_API_KEY="$(touch {marker})"\n'
            'export OPENAI_API_KEY="$OTHER_KEY"\n'
            '# export OPENAI_API_KEY="sk-comment"\n'
            'export OPENROUTER_API_KEY="sk-other"\n'
        )
        self.assertIsNone(get_openai_api_key(self.profile))
        self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
