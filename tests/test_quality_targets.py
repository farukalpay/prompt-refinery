import os
from pathlib import Path

from prompt_refinery.core import (
    DEFAULT_QUALITY_TARGETS,
    load_env_file,
    resolve_quality_targets,
)


def test_cli_targets_have_highest_priority(tmp_path: Path) -> None:
    profile = tmp_path / "profile.json"
    profile.write_text('{"quality_targets": ["Profile value"]}', encoding="utf-8")

    resolved = resolve_quality_targets(
        cli_targets=["CLI value", "No placeholders"],
        profile_path=profile,
        env_targets=["Env value"],
    )

    assert resolved == ["CLI value", "No placeholders"]


def test_profile_used_when_cli_missing(tmp_path: Path) -> None:
    profile = tmp_path / "profile.json"
    profile.write_text('{"quality_targets": ["Profile value", "Actionable"]}', encoding="utf-8")

    resolved = resolve_quality_targets(
        cli_targets=None,
        profile_path=profile,
        env_targets=["Env value"],
    )

    assert resolved == ["Profile value", "Actionable"]


def test_env_used_when_profile_missing(tmp_path: Path) -> None:
    profile = tmp_path / "missing.json"

    resolved = resolve_quality_targets(
        cli_targets=None,
        profile_path=profile,
        env_targets=["Env value", "Concise"],
    )

    assert resolved == ["Env value", "Concise"]


def test_defaults_used_when_no_overrides(tmp_path: Path) -> None:
    profile = tmp_path / "missing.json"

    resolved = resolve_quality_targets(
        cli_targets=None,
        profile_path=profile,
        env_targets=None,
    )

    assert resolved == DEFAULT_QUALITY_TARGETS


def test_load_env_file_overrides_blank_existing_env(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=from_env_file\n", encoding="utf-8")

    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    load_env_file(env_file)

    assert os.getenv("OPENROUTER_API_KEY") == "from_env_file"
