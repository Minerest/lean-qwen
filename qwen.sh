#!/usr/bin/env bash
# qwen.sh — minimal one-shot wrapper around the Qwen Code CLI.
# All settings come from env vars (populated by loop_fix.py from config.toml).
set -euo pipefail

exec qwen \
  --auth-type       "${QWEN_AUTH_TYPE:-openai}" \
  --openai-base-url "${QWEN_BASE_URL:?QWEN_BASE_URL not set (see config.toml)}" \
  --openai-api-key  "${QWEN_API_KEY:?QWEN_API_KEY not set (see config.toml)}" \
  --model           "${QWEN_MODEL:?QWEN_MODEL not set (see config.toml)}" \
  --approval-mode   "${QWEN_APPROVAL_MODE:-yolo}" \
  "$@"
