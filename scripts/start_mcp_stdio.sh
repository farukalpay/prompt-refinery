#!/usr/bin/env bash
set -euo pipefail

exec python -m prompt_refinery.mcp_server "$@"
