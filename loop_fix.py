#!/usr/bin/env python3
"""loop_fix.py — task-driven auto-fix loop with local LLM.

Reads loopfix.toml. Supports two modes:

  1. Task mode ([[tasks]] in config) — iterate over bite-sized work items.
     Each task runs the model with a fresh context, applies the output,
     runs all tests, and enters an auto-fix loop if tests fail.

  2. Direct mode (no [[tasks]]) — legacy auto-fix loop that starts by
     running tests and fixing whatever breaks first.

The script assumes the LLM server (whatever serves the qwen-cli backend)
is already running and reachable at the configured health URL. It will
not start, stop, or escalate between servers. This keeps the runtime
fully cross-platform (no process-group / SIGKILL games).

Each model call is self-contained — messages array built fresh per call,
no history carryover between tasks or iterations.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

# Default locations of the static config (qwen runtime, health, base defaults).
# Per-project loopfix.toml deep-merges on top of this. Checked in order:
#   1. config.toml next to loop_fix.py (works for clone-and-run)
#   2. ~/.config/lean-qwen/config.toml  (works for pipx install)
STATIC_CONFIG_DEFAULTS = (
    Path(__file__).resolve().parent / "config.toml",
    Path.home() / ".config" / "lean-qwen" / "config.toml",
)


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Deep-merge overlay into a copy of base.

    Tables (dicts) merge recursively. Arrays and scalars in overlay replace
    whatever is in base — we do not concatenate lists, since that would make
    it impossible to *override* a list value.
    """
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_config(task_path: str = "loopfix.toml", static_path: str | None = None) -> dict:
    """Load the static config + task config and deep-merge them.

    Static config discovery (first hit wins):
      1. `static_path` argument (from --global-config)
      2. $LOOPFIX_CONFIG env var
      3. config.toml alongside loop_fix.py

    Static config is optional — if none is found we just return the task
    config. Task config is required.
    """
    task_p = Path(task_path)
    if not task_p.exists():
        print(f"❌ Task config not found: {task_path}")
        sys.exit(1)

    static_p: Path | None = None
    if static_path:
        static_p = Path(static_path)
        if not static_p.exists():
            print(f"❌ Static config not found: {static_path}")
            sys.exit(1)
    elif os.environ.get("LOOPFIX_CONFIG"):
        static_p = Path(os.environ["LOOPFIX_CONFIG"])
        if not static_p.exists():
            print(f"❌ $LOOPFIX_CONFIG points at missing file: {static_p}")
            sys.exit(1)
    else:
        for candidate in STATIC_CONFIG_DEFAULTS:
            if candidate.exists():
                static_p = candidate
                break

    task_cfg = _load_toml(task_p)
    if static_p is None:
        return task_cfg

    static_cfg = _load_toml(static_p)
    merged = _deep_merge(static_cfg, task_cfg)

    # Resolve qwen.binary relative to the static config's directory (the
    # natural anchor — qwen.sh ships alongside config.toml).
    qwen_bin = merged.get("qwen", {}).get("binary")
    if qwen_bin and not os.path.isabs(qwen_bin):
        merged.setdefault("qwen", {})["binary"] = str(
            (static_p.parent / qwen_bin).resolve()
        )

    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# Server health (cross-platform: uses urllib, no curl dependency)
# ═══════════════════════════════════════════════════════════════════════════════

def health_check(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            return 200 <= resp.status < 500
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False


def wait_for_server(url: str, max_wait: int) -> bool:
    print(f"⏳ Waiting for server (up to {max_wait}s)...", flush=True)
    for _ in range(max_wait // 2):
        if health_check(url):
            print("✅ Server up")
            return True
        time.sleep(2)
    print("❌ Server not reachable — aborting")
    return False


def ensure_server(health_url: str, max_wait: int) -> bool:
    """Check server health; briefly wait if it's not up yet."""
    if health_check(health_url):
        return True
    return wait_for_server(health_url, max_wait)


# ═══════════════════════════════════════════════════════════════════════════════
# Qwen CLI harness (handles model calls + file writes natively)
# ═══════════════════════════════════════════════════════════════════════════════

# Fallback if config has no [qwen] section at all.
QWEN_DEFAULT_BINARY = str(Path(__file__).resolve().parent / "qwen.sh")
QWEN_MAX_EMPTY_RETRIES = 5


def preflight_qwen(config: dict) -> bool:
    """Verify the qwen wrapper exists, is executable, and has a model set.

    Catches the common misconfigurations once at startup instead of letting
    them surface as 5x retry warnings per fix iteration.
    """
    qcfg = config.get("qwen", {})
    qwen_bin = qcfg.get("binary", QWEN_DEFAULT_BINARY)

    if not os.path.isfile(qwen_bin):
        print(f"❌ qwen.binary not found: {qwen_bin}")
        print("   Set [qwen] binary in config.toml (or alongside loop_fix.py as qwen.sh).")
        return False
    if not os.access(qwen_bin, os.X_OK):
        print(f"❌ qwen.binary is not executable: {qwen_bin}")
        print(f"   Try: chmod +x {qwen_bin}")
        return False
    if not qcfg.get("model"):
        print("❌ [qwen] model is not set (config.toml or loopfix.toml).")
        return False
    return True


def _qwen_env(config: dict) -> dict:
    """Build the env vars that qwen.sh expects, from the merged [qwen] config."""
    qcfg = config.get("qwen", {})
    env = os.environ.copy()
    if "base_url" in qcfg:
        env["QWEN_BASE_URL"] = str(qcfg["base_url"])
    if "api_key" in qcfg:
        env["QWEN_API_KEY"] = str(qcfg["api_key"])
    if "model" in qcfg:
        env["QWEN_MODEL"] = str(qcfg["model"])
    if "approval_mode" in qcfg:
        env["QWEN_APPROVAL_MODE"] = str(qcfg["approval_mode"])
    if "auth_type" in qcfg:
        env["QWEN_AUTH_TYPE"] = str(qcfg["auth_type"])
    return env


def qwen_call(prompt: str, config: dict) -> str | None:
    """Run the qwen wrapper with `-p prompt` and return stdout.

    The Qwen CLI handles model calls AND file writes natively — the model
    outputs edit_file/write_file operations and the CLI executes them.

    Known model quirk: qwen-cli occasionally exits 0 with empty stdout
    even though it ran tool calls successfully. Retry up to
    QWEN_MAX_EMPTY_RETRIES times on empty before giving up.
    Timeout and missing-binary failures are NOT retried.
    """
    qcfg = config.get("qwen", {})
    qwen_bin = qcfg.get("binary", QWEN_DEFAULT_BINARY)
    timeout = qcfg.get("timeout", 600)
    env = _qwen_env(config)

    for attempt in range(1, QWEN_MAX_EMPTY_RETRIES + 1):
        try:
            result = subprocess.run(
                [qwen_bin, "-p", prompt],
                capture_output=True, text=True, timeout=timeout, env=env,
            )
        except subprocess.TimeoutExpired:
            print("  ⚠️  qwen timed out")
            return None
        except FileNotFoundError:
            print(f"  ⚠️  qwen binary not found: {qwen_bin}")
            return None

        output = result.stdout.strip() if result.stdout else ""
        if output:
            return output

        if attempt < QWEN_MAX_EMPTY_RETRIES:
            print(f"  ↻ qwen returned empty (attempt {attempt}/{QWEN_MAX_EMPTY_RETRIES}), retrying...")

    return None


def qwen_oneshot(prompt: str, config: dict, label: str = "") -> str | None:
    """Call qwen, print a status line, return the output."""
    if label:
        print(f"  🤖 {label}...", end=" ", flush=True)
    result = qwen_call(prompt, config)
    if label:
        if result:
            print(f"✓ ({len(result)} chars)")
        else:
            print("⚠️  empty")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Git helpers
# ═══════════════════════════════════════════════════════════════════════════════

def git_diff_short(filepath: str | None, root: str) -> str:
    """First 30 lines of git diff for a file, or whole project if None."""
    try:
        cmd = ["git", "diff"]
        if filepath:
            cmd.append(filepath)
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=5, cwd=root,
        )
        lines = result.stdout.splitlines()[:30]
        return "\n".join(lines) if lines else "(no diff)"
    except Exception:
        return "(no git diff available)"


# ═══════════════════════════════════════════════════════════════════════════════
# (Model API removed — all model calls go through `qwen -p` now)
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# Traceback parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_project_frames(
    traceback_text: str, project_root: str
) -> list[dict]:
    """Extract all project-relative frames from a traceback.

    Returns list of {"file", "line", "function", "relative"}, outermost first.
    """
    project_root = os.path.abspath(project_root)
    frames = []
    for line in traceback_text.splitlines():
        m = re.match(r'  File "([^"]+)", line (\d+)(?:, in (\S+))?', line)
        if not m:
            continue
        filepath = os.path.abspath(m.group(1))
        lineno = int(m.group(2))
        funcname = m.group(3) or "?"

        rel = os.path.relpath(filepath, project_root)
        if rel.startswith(".."):
            continue
        if "venv" in rel.split(os.sep) or "site-packages" in rel.split(os.sep):
            continue

        frames.append({
            "file": filepath,
            "line": lineno,
            "function": funcname,
            "relative": rel,
        })
    return frames


def _as_prefix_list(value) -> list[str]:
    """Normalize a string-or-list config value into a list of strings."""
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def pick_target_frame(frames: list[dict], cfg: dict) -> dict | None:
    """Pick the best frame to target for a fix.

    Preference order:
      1. deepest frame under any path in [defaults] source_prefix
      2. deepest frame NOT under any path in [defaults] test_prefix
      3. first frame
    """
    if not frames:
        return None
    defaults = cfg.get("defaults", {})
    source_prefixes = _as_prefix_list(defaults.get("source_prefix"))
    test_prefixes = _as_prefix_list(defaults.get("test_prefix")) or ["tests/", "test_", "scripts/"]

    if source_prefixes:
        for f in reversed(frames):
            if any(f["relative"].startswith(p) for p in source_prefixes):
                return f

    for f in reversed(frames):
        rel = f["relative"]
        if not any(p in rel for p in test_prefixes):
            return f
    return frames[0]


def resolve_called_function(
    frame: dict, project_root: str, cfg: dict
) -> dict | None:
    """If frame is a test file, find the actual function being called.

    Reads the failing line, extracts function calls, searches for definitions
    inside the configured source_prefix (or project_root if unset).
    """
    defaults = cfg.get("defaults", {})
    test_prefixes = _as_prefix_list(defaults.get("test_prefix")) or ["test_", "scripts/"]
    if not any(p in frame["relative"] for p in test_prefixes):
        return None

    try:
        with open(frame["file"]) as f:
            lines = f.readlines()
    except OSError:
        return None

    line_idx = frame["line"] - 1
    if line_idx < 0 or line_idx >= len(lines):
        return None

    line_text = lines[line_idx]
    calls = re.findall(r"(?:\.)?(\w+)\s*\(", line_text)
    if not calls:
        return None

    search_roots = _as_prefix_list(defaults.get("source_prefix")) or ["."]

    for func_name in reversed(calls):
        for search_root in search_roots:
            search_abs = os.path.join(project_root, search_root)
            if not os.path.isdir(search_abs):
                continue
            match = _find_def(func_name, search_abs)
            if not match:
                continue
            source_abs, source_line = match
            source_rel = os.path.relpath(source_abs, project_root)
            return {
                "file": source_abs,
                "line": source_line,
                "function": func_name,
                "relative": source_rel,
            }
    return None


def _find_def(func_name: str, root: str) -> tuple[str, int] | None:
    """Pure-Python `grep -rn "def func_name"` — portable to Windows.

    Walks `root`, scans .py files, returns (abs_path, line_number) of the
    first match.
    """
    needle = f"def {func_name}"
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip the usual suspects.
        dirnames[:] = [d for d in dirnames if d not in {".git", "venv", ".venv", "__pycache__", "node_modules", ".tox"}]
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                with open(fpath, encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f, 1):
                        if needle in line:
                            return fpath, i
            except OSError:
                continue
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Source context
# ═══════════════════════════════════════════════════════════════════════════════

def read_source_window(filepath: str, line: int, window: int = 30) -> str:
    """Read a window of lines around the given line number.
    Auto-shrinks if >12k chars. Marks the failing line with >>>.
    """
    try:
        with open(filepath) as f:
            lines = f.readlines()
    except FileNotFoundError:
        return f"(file not found: {filepath})"

    start = max(0, line - window - 1)
    end = min(len(lines), line + window)
    result = []
    for i in range(start, end):
        prefix = ">>>" if i == line - 1 else "   "
        result.append(f"{prefix} {i + 1:5d} | {lines[i]}")

    out = "".join(result)
    if len(out) > 12000:
        mid = line - 1
        half = window // 2
        start = max(0, mid - half)
        end = min(len(lines), mid + half)
        result = []
        for i in range(start, end):
            prefix = ">>>" if i == line - 1 else "   "
            result.append(f"{prefix} {i + 1:5d} | {lines[i]}")
        out = "".join(result)
    return out


_NOISE_SUBSTRINGS = (
    "site-packages",   # Python venv
    "venv/", "venv\\", # Python venv
    "node_modules",    # JS / TS
    "/vendor/",        # Go / PHP / Rust vendored deps
    ".cargo/registry", # Rust deps
    "For further",     # pytest "For further information" tail
)


def compress_traceback(tb_text: str, cfg: dict | None = None) -> str:
    """Strip well-known dependency-noise lines, keep the tail.

    Tail size is `defaults.error_tail_lines` (default 40). This is the
    language-agnostic fallback — most test runners (pytest, go test, jest,
    cargo test, rspec) put the failing assertion at the bottom, so a tail
    captures the meat regardless of language.
    """
    tail = 40
    if cfg is not None:
        tail = cfg.get("defaults", {}).get("error_tail_lines", 40)
    lines = tb_text.splitlines()
    filtered = [l for l in lines if not any(s in l for s in _NOISE_SUBSTRINGS)]
    return "\n".join(filtered[-tail:])


# ═══════════════════════════════════════════════════════════════════════════════
# Test runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_tests(config: dict) -> tuple[int, str]:
    """Run the test command from config. Returns (returncode, output).

    Config:
      [runner]
      command = "./venv/bin/pytest"     # any executable or shell command
      args = ["tests/", "-x", "-q"]     # optional list of args
      timeout = 30                      # optional, seconds
      shell = false                     # optional, run via shell (default false)
      env = { FOO = "bar" }             # optional, extra env vars
    """
    runner = config["runner"]
    cmd = runner["command"]
    args = runner.get("args", [])
    use_shell = bool(runner.get("shell", False))

    env = None
    if runner.get("env"):
        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in runner["env"].items()})

    if use_shell:
        # Treat command as a shell line; ignore args.
        full_cmd = cmd
        printable = cmd
    else:
        full_cmd = [cmd] + list(args)
        printable = " ".join(full_cmd)

    print(f"  🏃 {printable}")
    try:
        result = subprocess.run(
            full_cmd,
            shell=use_shell,
            capture_output=True, text=True,
            timeout=runner.get("timeout", 30),
            cwd=config.get("defaults", {}).get("project_root", "."),
            env=env,
        )
        return result.returncode, result.stdout + "\n" + result.stderr
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"
    except FileNotFoundError as e:
        return -2, f"Command not found: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# Model prompt stages
# ═══════════════════════════════════════════════════════════════════════════════

def check_quality(diagnosis: str) -> bool:
    """Reject empty, too-short, or vagued-out diagnoses."""
    d = diagnosis.strip()
    if not d or len(d) < 20:
        return False
    if not re.search(r"(line \d+|function|\.py|error|exception)", d, re.I):
        return False
    return True


def summarize_error(tb_compressed: str, config: dict) -> str | None:
    """Stage 1: summarize the error output in one sentence via qwen."""
    prompt = config.get("prompts", {}).get(
        "summary",
        "You are a debugger. Summarize this test failure in ONE sentence. "
        "Name the file, line number, and root cause when discoverable. "
        "Output ONLY the summary, no code, no markdown.",
    )
    full_prompt = f"{prompt}\n\n```\n{tb_compressed}\n```"
    result = qwen_call(full_prompt, config)
    return result.strip() if result else None


def generate_fix(
    diagnosis: str,
    source_context: str,
    git_diff: str,
    target_file: str,
    config: dict,
) -> bool:
    """Stage 2: fix the bug using Qwen CLI.

    The prompt includes the diagnosis and source context. Qwen runs the
    model and applies any file writes directly. Returns True if the
    output was non-empty (file writes handled by qwen).
    """
    prompt = (
        f"Bug in {target_file}:\n{diagnosis}\n\n"
        f"Git diff:\n{git_diff}\n\n"
        f"Source code (>>> marks the failing line):\n{source_context}\n\n"
        f"Fix the root cause. Apply the fix to the file."
    )
    result = qwen_oneshot(prompt, config, label="fix")
    return bool(result)





# ═══════════════════════════════════════════════════════════════════════════════
# Fix loop (one per task or direct mode)
# ═══════════════════════════════════════════════════════════════════════════════

def run_fix_loop(cfg: dict, label: str = "fix loop", task: dict | None = None) -> bool:
    """Run the auto-fix loop: test → diagnose → fix → retry.

    Two paths per iteration:
      Deluxe (Python frames parse): extract failing file:line, dump source
        window around it, plus the called function definition. Sharp context.
      Fallback (no frames): use the compressed error tail + git diff + the
        task's `files` (if any). Language-agnostic, coarser context.

    Returns True if tests pass within the iteration budget, False otherwise.
    """
    defaults = cfg.get("defaults", {})
    max_iters = defaults.get("max_iters_per_task", 5)
    source_window = defaults.get("source_window", 30)
    project_root = os.path.abspath(defaults.get("project_root", "."))
    health_url = cfg.get("health", {}).get("check_url", "http://127.0.0.1:8080/health")
    max_wait = cfg.get("health", {}).get("max_wait", 60)

    print(f"  🔄 {label}: {max_iters} max iterations")

    for i in range(1, max_iters + 1):
        print(f"  --- iteration {i}/{max_iters} ---")

        if not ensure_server(health_url, max_wait):
            return False

        retcode, output = run_tests(cfg)
        if retcode == 0:
            print("  ✅ Tests passed")
            return True
        if retcode == -1:
            print("  ⚠️  Timeout")
            if i >= max_iters:
                return False
            continue
        if retcode == -2:
            print(f"  ⚠️  Runner error:\n{output}")
            return False

        compressed = compress_traceback(output, cfg)
        if not compressed.strip():
            print("  ⚠️  No output to diagnose")
            if i >= max_iters:
                return False
            continue

        # ── Try the Python frame-based deluxe path ───────────────────
        frames = parse_project_frames(output, project_root)
        frame = pick_target_frame(frames, cfg) if frames else None

        source = ""
        diff = ""
        target = "(unknown)"

        if frame:
            source_frame = resolve_called_function(frame, project_root, cfg)
            if source_frame:
                print(f"  🐛 {frame['relative']}:{frame['line']} → {source_frame['relative']}:{source_frame['line']}")
                test_src = read_source_window(frame["file"], frame["line"], source_window)
                func_src = read_source_window(source_frame["file"], source_frame["line"], source_window)
                source = (
                    f"--- Test code ({frame['relative']}) ---\n{test_src}\n"
                    f"--- Source function ({source_frame['relative']}) ---\n{func_src}"
                )
                diff = git_diff_short(None, project_root)
                target = source_frame["relative"]
            else:
                print(f"  🐛 {frame['relative']}:{frame['line']} in {frame['function'] or '?'}")
                window = read_source_window(frame["file"], frame["line"], source_window)
                if not window.startswith("(file not found"):
                    source = window
                    diff = git_diff_short(frame["relative"], project_root)
                    target = frame["relative"]
                else:
                    frame = None  # force fallback below

        if not frame:
            # ── Fallback: language-agnostic, no source window ─────────
            print("  🔁 Fallback: no parseable frame — using error tail + diff")
            diff = git_diff_short(None, project_root)
            if task and task.get("files"):
                source = build_task_context(task, cfg)
                target = ", ".join(task["files"])
            else:
                source = "(no source context available)"

        # ── Stage 1: summarize ────────────────────────────────────────
        print("  📤 Stage 1: summarize...", end=" ", flush=True)
        diagnosis = summarize_error(compressed, cfg)
        if not diagnosis:
            print("⚠️  Empty")
            if ensure_server(health_url, max_wait):
                diagnosis = summarize_error(compressed, cfg)
            if not diagnosis:
                if not ensure_server(health_url, max_wait):
                    return False
                continue
        if not check_quality(diagnosis):
            print(f"⚠️  Poor: {diagnosis[:80]}")
            if i >= max_iters:
                return False
            continue
        print(f"✓ {diagnosis}")

        # ── Stage 2: fix via qwen (handles file writes natively) ─────
        ok = generate_fix(diagnosis, source, diff, target, cfg)
        if not ok:
            if not ensure_server(health_url, max_wait):
                return False
            continue

    print(f"  ❌ {label}: iterations exhausted")
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# Task system
# ═══════════════════════════════════════════════════════════════════════════════

def build_task_context(task: dict, cfg: dict) -> str:
    """Read the task's files and combine into a context string.

    Each file is annotated with its path. The total stays within
    the task's context_budget, or a default of 24k chars.
    """
    project_root = os.path.abspath(
        cfg.get("defaults", {}).get("project_root", ".")
    )
    budget = task.get("context_budget", 24000)
    files = task.get("files", [])
    parts = []
    remaining = budget

    for fpath in files:
        abs_path = os.path.join(project_root, fpath) if not os.path.isabs(fpath) else fpath
        try:
            with open(abs_path) as f:
                content = f.read()
        except OSError as e:
            parts.append(f"# {fpath} — ERROR: {e}")
            continue

        header = f"# ── {fpath} ──\n"
        total = len(header) + len(content)
        if remaining - total < 0 and parts:
            # Truncate to fit budget
            max_content = max(0, remaining - len(header) - 200)
            if max_content > 100:
                content = content[:max_content] + "\n# ... (truncated)"
                total = len(header) + len(content)

        parts.append(header + content)
        remaining -= total
        if remaining < 200:
            break

    return "\n".join(parts)


def _run_after_commands(commands: list[str], cwd: str, timeout: int = 600) -> bool:
    """Run a sequence of shell commands after a task succeeds.

    Each command runs via bash -c with cwd=project_root. Stdout is echoed.
    Returns False on the first non-zero exit, True if all succeed.

    `timeout` is per-command (seconds). Overridable via the toml's
    `[defaults] after_command_timeout`.
    """
    for cmd in commands:
        print(f"  ▶ {cmd}")
        try:
            result = subprocess.run(
                cmd, shell=True, cwd=cwd,
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            print("    ⚠️  timed out")
            return False
        for line in (result.stdout or "").splitlines():
            print(f"    {line}")
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            if stderr:
                print(f"    ❌ stderr: {stderr[:300]}")
            print(f"  ❌ after-command failed (exit {result.returncode})")
            return False
    return True


def process_task(task: dict, cfg: dict) -> bool:
    """Process one task: model call → tests → fix loop → after-commands."""

    # ── Build context ─────────────────────────────────────────────────
    print(f"\n  📚 Reading files...", flush=True)
    context = build_task_context(task, cfg)
    ctx_len = len(context)
    print(f"     Context: {ctx_len} chars across {len(task.get('files', []))} file(s)")

    task_prompt = task["prompt"]
    project_root = os.path.abspath(
        cfg.get("defaults", {}).get("project_root", ".")
    )

    # ── Fresh qwen call (handles file writes natively) ─────────────────
    qwen_prompt = (
        f"Task: {task_prompt}\n\n"
        f"Project files:\n{context}\n\n"
        f"Current git diff:\n```diff\n{git_diff_short(None, project_root)}\n```"
    )
    output = qwen_oneshot(qwen_prompt, cfg, label="generating")
    if not output:
        print("  ⚠️  Model returned nothing — skipping task")
        return False
    print(f"  📥 {output[:200]}")

    # ── Run tests ──────────────────────────────────────────────────────
    print(f"  🧪 Running tests...", flush=True)
    retcode, test_output = run_tests(cfg)
    if retcode == 0:
        print(f"  ✅ All tests pass")
        passed = True
    else:
        print(f"  ❌ Tests failed — entering fix loop")
        passed = run_fix_loop(cfg, label=f"fix loop for '{task['name']}'", task=task)

    if not passed:
        return False

    # ── Post-task hooks ───────────────────────────────────────────────
    after = task.get("after", [])
    if after:
        print(f"  🔁 Running {len(after)} after-command(s)")
        after_timeout = cfg.get("defaults", {}).get("after_command_timeout", 600)
        if not _run_after_commands(after, project_root, timeout=after_timeout):
            return False

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Sample config generator
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_CONFIG = """\
# loopfix.toml — per-project task config.
# Generated by loop_fix.py --init
#
# This file deep-merges on top of the static config.toml that ships with
# loop_fix.py (qwen runtime, health URL, base defaults). Only put
# project-specific stuff here; override static defaults as needed.
#
# Two modes:
#   Task mode: define one or more [[tasks]] — each is a bite-sized work
#              item with a fresh model context. After each task, tests run.
#              If they fail, an auto-fix loop kicks in for that task.
#   Direct mode: no [[tasks]] → runs the fix loop straight against tests.
#
# The LLM server (e.g. llama-server) MUST be running and reachable at the
# configured health URL before launching loop_fix.py.

[runner]
# Fully generic. `command` is run with `args` appended. Whatever your
# language's test runner is — pytest, go test, npm test, cargo test,
# make check — point it here.
command = "./venv/bin/pytest"          # any executable or absolute path
args    = ["tests/", "-x", "--tb=native", "-q"]
timeout = 30                           # seconds
# shell = false                        # set true to run `command` as a shell line (ignores args)
# env = { PYTHONDONTWRITEBYTECODE = "1" }  # optional extra env vars

[defaults]
# Where your *source* code lives. Optional. If set, the (Python) traceback
# parser prefers frames under these paths and searches them for function
# defs. Accepts a single string or a list.
# source_prefix = "src/"
# source_prefix = ["src/", "lib/"]

# Path fragments that mark *test* files. Used to fall back from test frames
# to the underlying source. Defaults to ["tests/", "test_", "scripts/"].
# test_prefix = ["tests/", "test_"]

# Override any static default here:
# project_root      = "."
# source_window     = 30
# error_tail_lines  = 40
# max_iters_per_task = 5

# ── Qwen overrides (optional) ────────────────────────────────────────────────
# Anything not set here falls back to config.toml.
# [qwen]
# model   = "some-other-model.gguf"
# timeout = 900

# ── Prompt overrides (optional, advanced) ────────────────────────────────────
# Replace the stock summarization prompt fed to the model on Stage 1 of the
# fix loop. Useful if you want domain-specific framing (e.g. "you are a Rust
# debugger" instead of the generic default).
# [prompts]
# summary = "You are a Go debugger. Summarize this test failure in ONE sentence. Name the failing test, the assertion, and the likely cause. Output ONLY the summary."

# ── Tasks ────────────────────────────────────────────────────────────────────
# Each [[tasks]] block defines one work item. Uncomment and edit to use.

# [[tasks]]
# name = "my-task"
# prompt = "Describe what to do here. Be specific about the file and function."
# files = ["path/to/file.py"]
# # context_budget = 8000  # optional: max chars of file content sent to the model
# # after = [                  # optional: shell commands run after tests pass
# #   "wc -l data/seed.yaml",
# # ]
"""


def _write_sample_config(dest: str) -> None:
    """Write the sample config to stdout or a file."""
    if dest == "-":
        print(SAMPLE_CONFIG)
    else:
        path = Path(dest)
        path.write_text(SAMPLE_CONFIG)
        print(f"✅ Wrote sample config to {path.resolve()}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def _run_workload(cfg: dict, args) -> bool:
    """Run the configured workload (tasks or direct fix loop)."""
    tasks = cfg.get("tasks", [])

    if not tasks:
        print("=== loop_fix.py — direct fix loop ===")
        return run_fix_loop(cfg, label="direct loop")

    task_list = tasks
    if args.task:
        task_list = [t for t in tasks if t.get("name") == args.task]
        if not task_list:
            print(f"❌ No task named '{args.task}'")
            return False

    print(f"=== loop_fix.py — {len(task_list)} task(s) ===")

    for idx, task in enumerate(task_list):
        name = task.get("name", f"unnamed-{idx + 1}")
        print(f"\n{'═' * 60}")
        print(f"  Task {idx + 1}/{len(task_list)}: {name}")
        print(f"{'═' * 60}")

        if not process_task(task, cfg):
            print(f"\n❌ Task '{name}' failed")
            return False

    print(f"\n{'═' * 60}")
    print("✅ All tasks completed")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Task-driven auto-fix loop using a local LLM. "
                    "Requires the LLM server to already be running at the "
                    "configured health URL.",
    )
    parser.add_argument(
        "-c", "--config", default="loopfix.toml",
        help="Path to per-project task config (default: loopfix.toml)",
    )
    parser.add_argument(
        "-g", "--global-config", default=None, dest="global_config",
        help="Path to static config (default: $LOOPFIX_CONFIG or "
             "config.toml alongside this script)",
    )
    parser.add_argument(
        "--task", help="Run only this task name (skips others)",
    )
    parser.add_argument(
        "--init", nargs="?", const="-", metavar="FILE",
        help="Write a sample task config FILE and exit (default: stdout)",
    )
    args = parser.parse_args()

    if args.init:
        _write_sample_config(args.init)
        return 0

    cfg = load_config(args.config, args.global_config)

    if not preflight_qwen(cfg):
        return 1

    health_url = cfg.get("health", {}).get("check_url", "http://127.0.0.1:8080/health")
    max_wait = cfg.get("health", {}).get("max_wait", 60)
    if not ensure_server(health_url, max_wait):
        print(
            f"❌ LLM server not reachable at {health_url}. "
            f"Start it (e.g. `./qwen.sh` or your llama-server launcher) and rerun."
        )
        return 1

    return 0 if _run_workload(cfg, args) else 1


if __name__ == "__main__":
    sys.exit(main())
