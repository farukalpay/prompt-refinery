#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from prompt_refinery import DEFAULT_QUALITY_TARGETS, RefineryEngine, RuntimePaths, RuntimeSettings


def ask_non_empty(prompt: str) -> str:
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("Input cannot be empty. Try again.")


def ask_quality_targets() -> list[str]:
    print("\nQuality targets (press Enter to keep defaults):")
    targets: list[str] = []
    for idx, default in enumerate(DEFAULT_QUALITY_TARGETS, start=1):
        user_value = input(f"  {idx}. [{default}] -> ").strip()
        targets.append(user_value or default)
    return targets


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    paths = RuntimePaths.from_project_dir(project_dir)
    settings = RuntimeSettings.from_env(project_dir)

    print("Prompt Refinery Quickstart")
    print("-" * 28)

    user_text = ask_non_empty("Enter your prompt request: ")
    quality_targets = ask_quality_targets()

    engine = RefineryEngine(settings=settings, paths=paths)
    try:
        result = engine.run(user_text=user_text, quality_targets=quality_targets, export_outputs=True)
    finally:
        engine.close()

    print("\n=== REPAIRED PROMPT ===\n")
    print(result["repaired_prompt"])
    print("\nSaved artifacts:")
    print(f"- {paths.last_result_txt}")
    print(f"- {paths.last_result_json}")
    print(f"- {paths.db_path}")


if __name__ == "__main__":
    main()
