from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .core import RefineryEngine, RuntimePaths, RuntimeSettings, clean_list, clean_text


PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "prompt-refinery"
SERVER_VERSION = "0.2.0"


class MCPServer:
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self._engine: Optional[RefineryEngine] = None

    @property
    def engine(self) -> RefineryEngine:
        if self._engine is None:
            paths = RuntimePaths.from_project_dir(self.project_dir)
            settings = RuntimeSettings.from_env(self.project_dir)
            self._engine = RefineryEngine(settings=settings, paths=paths)
        return self._engine

    def close(self) -> None:
        if self._engine is not None:
            self._engine.close()
            self._engine = None

    def handle(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        method = request.get("method")
        params = request.get("params") or {}
        request_id = request.get("id")

        if method == "notifications/initialized":
            return None

        try:
            if method == "initialize":
                result = self._handle_initialize(params)
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = self._handle_tools_list()
            elif method == "tools/call":
                result = self._handle_tools_call(params)
            else:
                raise ValueError(f"Unsupported method: {method}")

            if request_id is None:
                return None
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        except Exception as exc:
            if request_id is None:
                return None
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32000,
                    "message": str(exc),
                    "data": {"traceback": traceback.format_exc(limit=4)},
                },
            }

    def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        _ = params
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {
                "tools": {},
            },
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
            },
        }

    def _handle_tools_list(self) -> Dict[str, Any]:
        return {
            "tools": [
                {
                    "name": "refine_prompt",
                    "description": (
                        "Refine a short request into a retrieval-grounded, production-ready prompt."
                    ),
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "user_text": {
                                "type": "string",
                                "description": "Raw user request to refine.",
                            },
                            "quality_targets": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional explicit output standards.",
                            },
                            "export_outputs": {
                                "type": "boolean",
                                "description": "Persist runtime artifacts under runtime_db/exports.",
                            },
                        },
                        "required": ["user_text"],
                    },
                }
            ]
        }

    def _handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        name = clean_text(params.get("name"))
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise ValueError("tools/call arguments must be an object")

        if name != "refine_prompt":
            raise ValueError(f"Unknown tool: {name}")

        user_text = clean_text(arguments.get("user_text"))
        if not user_text:
            raise ValueError("user_text is required")

        quality_targets_raw = arguments.get("quality_targets")
        quality_targets = clean_list(quality_targets_raw) if isinstance(quality_targets_raw, list) else None

        export_outputs = bool(arguments.get("export_outputs", True))

        resolved_targets = (
            quality_targets
            if quality_targets
            else self.engine.resolve_targets(cli_targets=None)
        )
        result = self.engine.run(
            user_text=user_text,
            quality_targets=resolved_targets,
            export_outputs=export_outputs,
        )

        response_payload = {
            "chosen_act": result["chosen_act"],
            "repaired_prompt": result["repaired_prompt"],
            "intent_spec": result.get("intent_spec", {}),
            "quality_targets": result.get("meta", {}).get("quality_targets", resolved_targets),
            "artifacts": {
                "result_json": str(self.engine.paths.last_result_json),
                "result_text": str(self.engine.paths.last_result_txt),
                "runtime_db": str(self.engine.paths.db_path),
            },
        }

        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(response_payload, ensure_ascii=False, indent=2),
                }
            ]
        }


def read_message(stream) -> Optional[Dict[str, Any]]:
    headers: Dict[str, str] = {}

    while True:
        line = stream.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        decoded = line.decode("utf-8").strip()
        if not decoded or ":" not in decoded:
            continue
        key, value = decoded.split(":", 1)
        headers[key.strip().lower()] = value.strip()

    length_raw = headers.get("content-length")
    if not length_raw:
        raise ValueError("Missing Content-Length header")

    try:
        length = int(length_raw)
    except ValueError as exc:
        raise ValueError("Invalid Content-Length header") from exc

    payload = stream.read(length)
    if len(payload) < length:
        return None
    return json.loads(payload.decode("utf-8"))


def write_message(stream, message: Dict[str, Any]) -> None:
    data = json.dumps(message, ensure_ascii=False).encode("utf-8")
    header = f"Content-Length: {len(data)}\r\n\r\n".encode("ascii")
    stream.write(header)
    stream.write(data)
    stream.flush()


def run_stdio(project_dir: Path) -> None:
    server = MCPServer(project_dir=project_dir)
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    try:
        while True:
            message = read_message(stdin)
            if message is None:
                break
            response = server.handle(message)
            if response is not None:
                write_message(stdout, response)
    finally:
        server.close()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="prompt-refinery-mcp",
        description="Run Prompt Refinery as a stdio MCP server.",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path.cwd(),
        help="Project root containing .env and refinery_profile.json",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run_stdio(project_dir=args.project_dir.resolve())


if __name__ == "__main__":
    main()
