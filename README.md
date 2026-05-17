# lean-qwen

A small auto-fix loop that drives the [Qwen Code CLI](https://github.com/QwenLM/qwen-code)
against a local LLM, with a deliberately **lean per-task context** — fresh
state per call, no history bloat, only the files you list. That's what
makes it usable with a tiny local model that would otherwise choke on the
chat history a typical agent framework piles up.

Define bite-sized tasks in TOML, point it at your test runner, and let it
iterate until the tests pass.

## The Purpose
This is my answer to subsidizing cloud API costs with my local LLM with a workflow I can trust. The workflow I got nailed down is that I would talk with the godfather cloud AI about a feature, say something like full stack notifications for a web app. Cloud AI will read through and create the task toml file and it would kick off the lean-qwen script. The local AI would do the bulk of the work autonomously leading to a 10x cloud saving expenditure according to Claude code. The savings scale linearly with feature and task length.

## The loop

```
for each task in loopfix.toml:
    1. read the listed files + git diff
    2. send task + context to qwen-cli (fresh context, no history)
    3. qwen-cli edits files directly
    4. run the test command
    5. if tests pass → next task
       if tests fail → enter fix loop (up to max_iters):
           a. compress the error output (tail N lines, strip dep noise)
           b. ask qwen for a one-sentence diagnosis
           c. ask qwen to fix it
           d. re-run tests; loop
```

Each model call is self-contained — no chat history is carried between tasks
or iterations. The Qwen CLI does the actual file writes; `loop_fix.py` only
orchestrates.

## Requirements

- Python 3.11+ (stdlib only — no `pip install` needed)
- [Qwen Code CLI](https://github.com/QwenLM/qwen-code) on your `PATH`
- A running OpenAI-compatible LLM server (e.g. `llama-server` from llama.cpp
  serving a Qwen GGUF).

## Install

Two options:

**Clone and run**
```
git clone <this repo> lean-qwen && cd lean-qwen
python3 loop_fix.py --help
```

**`pipx install`** (gets you a `lean-qwen` command on PATH):
```
pipx install /path/to/lean-qwen
mkdir -p ~/.config/lean-qwen && cp /path/to/lean-qwen/config.toml ~/.config/lean-qwen/
lean-qwen --help
```

## Setup

1. Start your LLM server (e.g. `llama-server -m Qwen3.6-35B-A3B.gguf --host 127.0.0.1 --port 8080`).
2. Edit `config.toml` (next to `loop_fix.py`, or `~/.config/lean-qwen/config.toml`
   if you installed via pipx) — set `[qwen] model`, `base_url`,
   `[health] check_url` to match your server.
3. In your project, run `lean-qwen --init loopfix.toml` (or
   `python /path/to/loop_fix.py --init loopfix.toml`) to drop a sample task
   config, then edit it: point `[runner]` at your test command and add
   `[[tasks]]` blocks. See [`examples/`](examples/) for samples.
4. `lean-qwen -c loopfix.toml`

## Config

Two files, deep-merged at load time:

- **`config.toml`** (ships with the tool, alongside `loop_fix.py`) — static,
  per-user defaults: qwen binary path, model, server URL, base behavior
  knobs. Customize once for your environment.
- **`loopfix.toml`** (per-project) — `[runner]`, `[[tasks]]`, and any
  per-project overrides (e.g. `[defaults] source_prefix = "src/"`). Anything
  set here wins over `config.toml`.

Static config discovery: `--global-config` flag → `$LOOPFIX_CONFIG` env var →
`config.toml` next to `loop_fix.py` → `~/.config/lean-qwen/config.toml`.

## `qwen.sh`

A six-line wrapper that execs the `qwen` binary with `QWEN_*` env vars.
`loop_fix.py` populates those vars from the merged `[qwen]` table before
each call, so you don't normally touch this file. If you want a different
wrapper, set `[qwen] binary = "/path/to/your/wrapper.sh"`.

## Modes

- **Task mode** — at least one `[[tasks]]` block in `loopfix.toml`. Tasks run
  sequentially; tests run after each task.
- **Direct mode** — no `[[tasks]]`. Runs the fix loop straight against
  whatever the test command spits out, fixing whatever breaks first.

## Language support

- **Test runner**: any command. Set `[runner] command` and `args` to whatever
  your project uses (pytest, `go test`, `npm test`, `cargo test`, `make
  check`, ...). Optional `shell = true` runs `command` as a shell line.
- **Traceback parsing**: tuned for Python (`File "x.py", line N`). For other
  languages the script falls back to "compressed error tail + git diff +
  task files" — fixes are coarser but it still works.

## Platform

Pure stdlib, no POSIX-only syscalls — runs on Linux, macOS, and Windows
(provided you have the Qwen Code CLI and an LLM server reachable there).
`qwen.sh` is bash, so on Windows either run it through Git Bash / WSL or
point `[qwen] binary` at a `.bat`/`.ps1` equivalent.

## CLI

```
lean-qwen [-c CONFIG] [-g GLOBAL_CONFIG] [--task NAME] [--init [FILE]]
```

- `-c, --config` — per-project task config (default `loopfix.toml`)
- `-g, --global-config` — static config override
- `--task NAME` — run only the named task
- `--init [FILE]` — write a sample task config (default: stdout)

## Examples

Sample `loopfix.toml` files for different test runners live in
[`examples/`](examples/) (pytest, `go test`, jest).

## Development

```
pip install -e ".[dev]"
pytest
```

Self-tests live in [`tests/`](tests/) and cover the pure-function bits
(config merge, traceback parsing, frame picking, preflight). End-to-end
runs against a real LLM are exercised by the BANDEEZY playground in
`playground/BANDEEZY/`.

## License

MIT — see [LICENSE](LICENSE).
