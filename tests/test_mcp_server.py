import json
from pathlib import Path

from prompt_refinery.core import RuntimePaths
from prompt_refinery.mcp_server import MCPServer


class FakeEngine:
    def __init__(self, project_dir: Path) -> None:
        self.paths = RuntimePaths.from_project_dir(project_dir)
        self.last_call = None

    def resolve_targets(self, cli_targets=None, profile_path=None):
        _ = cli_targets, profile_path
        return ["Fully specified output", "No unresolved placeholders"]

    def run(self, user_text, quality_targets=None, export_outputs=True):
        self.last_call = {
            "user_text": user_text,
            "quality_targets": quality_targets,
            "export_outputs": export_outputs,
        }
        return {
            "chosen_act": "Write concise executive summary",
            "repaired_prompt": "ROLE: Senior analyst...",
            "intent_spec": {"language": "en-US"},
            "meta": {"quality_targets": quality_targets or []},
        }

    def close(self):
        return None


def test_tools_list_exposes_refine_prompt(tmp_path: Path) -> None:
    server = MCPServer(project_dir=tmp_path)
    server._engine = FakeEngine(tmp_path)

    response = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
    assert response is not None
    tools = response["result"]["tools"]
    assert any(tool["name"] == "refine_prompt" for tool in tools)


def test_tools_call_returns_serialized_result(tmp_path: Path) -> None:
    server = MCPServer(project_dir=tmp_path)
    fake = FakeEngine(tmp_path)
    server._engine = fake

    request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "refine_prompt",
            "arguments": {
                "user_text": "prepare an incident postmortem template",
                "quality_targets": ["Clear actionable wording"],
                "export_outputs": False,
            },
        },
    }

    response = server.handle(request)
    assert response is not None
    content = response["result"]["content"]
    payload = json.loads(content[0]["text"])

    assert payload["chosen_act"] == "Write concise executive summary"
    assert payload["quality_targets"] == ["Clear actionable wording"]
    assert fake.last_call == {
        "user_text": "prepare an incident postmortem template",
        "quality_targets": ["Clear actionable wording"],
        "export_outputs": False,
    }
