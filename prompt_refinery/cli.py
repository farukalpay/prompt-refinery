from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

from .core import RefineryEngine, RuntimePaths, RuntimeSettings, clean_text


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="prompt-refinery",
        description="Turn short requests into production-ready prompts.",
        add_help=True,
    )
    parser.add_argument(
        "user_text",
        nargs="*",
        default=[],
        help="Free-form request text (positional).",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        metavar="T",
        help='Output standards the final prompt must satisfy, e.g. --targets "Concise" "No jargon"',
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        metavar="FILE",
        help="Path to a JSON profile file with quality_targets (default: ./refinery_profile.json).",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path.cwd(),
        metavar="DIR",
        help="Project root (default: current working directory).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON result to stdout instead of prompt-only output.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable GUI fallback and fail fast when no input is provided.",
    )
    return parser.parse_args(argv)


def _get_user_input_gui() -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import scrolledtext
    except Exception:
        return None

    result = {"text": None}
    root = tk.Tk()
    root.title("Prompt Refinery Input")
    root.geometry("900x500")

    tk.Label(
        root,
        text="Kullanici girdisini yapistir ve Gonder'e bas:",
        font=("Arial", 12),
    ).pack(pady=10)

    text_widget = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Consolas", 11))
    text_widget.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

    def submit() -> None:
        result["text"] = text_widget.get("1.0", tk.END).strip()
        root.destroy()

    tk.Button(root, text="Gonder", command=submit, font=("Arial", 12)).pack(pady=10)
    root.mainloop()

    return result["text"]


def _show_output_gui(title: str, content: str) -> None:
    try:
        import tkinter as tk
        from tkinter import scrolledtext
    except Exception:
        return

    root = tk.Tk()
    root.title(title)
    root.geometry("1000x700")

    text_widget = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Consolas", 11))
    text_widget.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
    text_widget.insert("1.0", content)
    root.mainloop()


def get_user_input(pre_parsed: Optional[str], allow_gui: bool) -> Tuple[str, bool]:
    if pre_parsed:
        return pre_parsed, False

    if sys.stdin and not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            return piped, False

    if sys.stdin and sys.stdin.isatty():
        typed = input("Kullanici girdisini yaz:\n> ").strip()
        if typed:
            return typed, False

    if allow_gui:
        gui_text = _get_user_input_gui()
        if gui_text:
            return gui_text, True

    raise RuntimeError("Kullanici girdisi alinamadi. Arguman veya stdin ile metin ver.")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    project_dir = args.project_dir.resolve()
    paths = RuntimePaths.from_project_dir(project_dir)
    settings = RuntimeSettings.from_env(project_dir)

    engine = RefineryEngine(settings=settings, paths=paths)

    try:
        profile_path = args.profile.resolve() if args.profile else None
        quality_targets = engine.resolve_targets(cli_targets=args.targets, profile_path=profile_path)

        pre_parsed = " ".join(args.user_text).strip() if args.user_text else None
        user_text, used_gui = get_user_input(pre_parsed=pre_parsed, allow_gui=not args.no_gui)

        result = engine.run(user_text=user_text, quality_targets=quality_targets, export_outputs=True)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print("\n=== REPAIRED PROMPT ===\n")
            print(clean_text(result["repaired_prompt"]))
            print(
                "\nKaydedildi:\n"
                f"- {paths.last_result_txt}\n"
                f"- {paths.last_result_json}\n"
                f"- {paths.db_path}"
            )

        if used_gui:
            _show_output_gui("Prompt Refinery Result", clean_text(result["repaired_prompt"]))
    finally:
        engine.close()


if __name__ == "__main__":
    main()
