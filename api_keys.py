import os
import re
from pathlib import Path


def get_openai_api_key(profile_path: Path | None = None) -> str | None:
    """Read an environment key or a literal zprofile assignment without running shell code."""
    if key := os.getenv("OPENAI_API_KEY"):
        return key

    profile_path = profile_path if profile_path is not None else Path.home() / ".zprofile"
    try:
        profile = profile_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None

    key = None
    for line in profile.splitlines():
        match = re.fullmatch(
            r"\s*(?:export\s+)?OPENAI_API_KEY="
            r"(?:\"([A-Za-z0-9_-]+)\"|'([A-Za-z0-9_-]+)'|([A-Za-z0-9_-]+))"
            r"\s*(?:#.*)?",
            line,
        )
        if match:
            key = next(value for value in match.groups() if value is not None)
    return key
