"""Self-tests for loop_fix.py — the pure-function bits.

We don't test anything that needs a live qwen binary or LLM server; those
are exercised end-to-end by the playground in playground/BANDEEZY/.
"""
from pathlib import Path

import pytest

import loop_fix


# ── _deep_merge ───────────────────────────────────────────────────────────────

class TestDeepMerge:
    def test_nested_tables_merge_recursively(self):
        base = {"a": {"x": 1, "y": 2}, "b": 10}
        overlay = {"a": {"y": 99, "z": 3}}
        out = loop_fix._deep_merge(base, overlay)
        assert out == {"a": {"x": 1, "y": 99, "z": 3}, "b": 10}

    def test_scalar_in_overlay_replaces_base(self):
        out = loop_fix._deep_merge({"a": 1}, {"a": 2})
        assert out == {"a": 2}

    def test_list_in_overlay_replaces_not_concatenates(self):
        # Critical: if lists concatenated, users couldn't override list values.
        out = loop_fix._deep_merge({"args": ["a", "b"]}, {"args": ["c"]})
        assert out == {"args": ["c"]}

    def test_overlay_can_add_new_keys(self):
        out = loop_fix._deep_merge({"a": 1}, {"b": 2})
        assert out == {"a": 1, "b": 2}

    def test_does_not_mutate_inputs(self):
        base = {"a": {"x": 1}}
        overlay = {"a": {"y": 2}}
        loop_fix._deep_merge(base, overlay)
        assert base == {"a": {"x": 1}}
        assert overlay == {"a": {"y": 2}}

    def test_overlay_table_over_base_scalar_replaces(self):
        # If types disagree (scalar vs dict), overlay wins without recursing.
        out = loop_fix._deep_merge({"a": 1}, {"a": {"x": 2}})
        assert out == {"a": {"x": 2}}


# ── _as_prefix_list ───────────────────────────────────────────────────────────

class TestAsPrefixList:
    def test_none_returns_empty(self):
        assert loop_fix._as_prefix_list(None) == []

    def test_empty_string_returns_empty(self):
        assert loop_fix._as_prefix_list("") == []

    def test_string_wrapped_into_list(self):
        assert loop_fix._as_prefix_list("src/") == ["src/"]

    def test_list_passed_through(self):
        assert loop_fix._as_prefix_list(["a/", "b/"]) == ["a/", "b/"]


# ── compress_traceback ────────────────────────────────────────────────────────

class TestCompressTraceback:
    def test_keeps_last_n_lines(self):
        text = "\n".join(f"line {i}" for i in range(1, 101))
        out = loop_fix.compress_traceback(text, {"defaults": {"error_tail_lines": 5}})
        assert out.splitlines() == ["line 96", "line 97", "line 98", "line 99", "line 100"]

    def test_default_tail_is_40(self):
        text = "\n".join(f"line {i}" for i in range(1, 101))
        out = loop_fix.compress_traceback(text)
        assert len(out.splitlines()) == 40

    def test_strips_python_venv_noise(self):
        text = (
            "real error 1\n"
            'File "/proj/venv/lib/python3.11/site-packages/pluggy/_callers.py", line 1, in foo\n'
            "real error 2"
        )
        out = loop_fix.compress_traceback(text)
        assert "site-packages" not in out
        assert "real error 1" in out
        assert "real error 2" in out

    def test_strips_node_modules_noise(self):
        text = "real error\nat noisy (/proj/node_modules/jest/foo.js:1:1)\nfinal"
        out = loop_fix.compress_traceback(text)
        assert "node_modules" not in out
        assert "real error" in out
        assert "final" in out


# ── parse_project_frames ──────────────────────────────────────────────────────

class TestParseProjectFrames:
    def test_extracts_python_frames(self, tmp_path):
        # Create real files in tmp_path so abspath/relpath produce stable output.
        (tmp_path / "foo.py").write_text("x = 1\n")
        (tmp_path / "bar.py").write_text("y = 2\n")
        tb = (
            'Traceback (most recent call last):\n'
            f'  File "{tmp_path}/foo.py", line 5, in caller\n'
            "    something()\n"
            f'  File "{tmp_path}/bar.py", line 12, in inner\n'
            "    boom()\n"
            "RuntimeError: boom"
        )
        frames = loop_fix.parse_project_frames(tb, str(tmp_path))
        assert len(frames) == 2
        assert frames[0]["relative"] == "foo.py"
        assert frames[0]["line"] == 5
        assert frames[0]["function"] == "caller"
        assert frames[1]["relative"] == "bar.py"
        assert frames[1]["line"] == 12

    def test_skips_venv_frames(self, tmp_path):
        # venv-pathed files outside project shouldn't appear.
        (tmp_path / "real.py").write_text("\n")
        tb = (
            f'  File "{tmp_path}/real.py", line 1, in foo\n'
            f'  File "{tmp_path}/venv/site-packages/x.py", line 99, in bar\n'
        )
        frames = loop_fix.parse_project_frames(tb, str(tmp_path))
        assert len(frames) == 1
        assert frames[0]["relative"] == "real.py"

    def test_skips_frames_outside_project_root(self, tmp_path):
        (tmp_path / "real.py").write_text("\n")
        tb = (
            f'  File "{tmp_path}/real.py", line 1, in foo\n'
            f'  File "/totally/elsewhere.py", line 1, in bar\n'
        )
        frames = loop_fix.parse_project_frames(tb, str(tmp_path))
        assert len(frames) == 1

    def test_empty_text_returns_empty(self):
        assert loop_fix.parse_project_frames("", "/tmp") == []

    def test_non_python_text_returns_empty(self):
        # Go panic, jest output, etc. — no `File "x.py", line N` shape.
        tb = "panic: runtime error\n\tgoroutine 1 [running]:\n\tmain.foo()"
        assert loop_fix.parse_project_frames(tb, "/tmp") == []


# ── pick_target_frame ─────────────────────────────────────────────────────────

class TestPickTargetFrame:
    def _frame(self, rel, line=10, func="f"):
        return {"file": f"/abs/{rel}", "line": line, "function": func, "relative": rel}

    def test_returns_none_for_empty(self):
        assert loop_fix.pick_target_frame([], {}) is None

    def test_prefers_source_prefix_when_set(self):
        frames = [
            self._frame("tests/test_foo.py"),
            self._frame("src/app/bar.py"),
            self._frame("scripts/run.py"),
        ]
        cfg = {"defaults": {"source_prefix": "src/"}}
        chosen = loop_fix.pick_target_frame(frames, cfg)
        assert chosen["relative"] == "src/app/bar.py"

    def test_picks_deepest_source_prefix_match(self):
        frames = [
            self._frame("src/outer.py"),
            self._frame("src/middle.py"),
            self._frame("src/inner.py"),
        ]
        cfg = {"defaults": {"source_prefix": "src/"}}
        chosen = loop_fix.pick_target_frame(frames, cfg)
        # Deepest = last in frames list.
        assert chosen["relative"] == "src/inner.py"

    def test_falls_back_to_non_test_frame(self):
        frames = [
            self._frame("tests/test_a.py"),
            self._frame("lib/util.py"),
        ]
        chosen = loop_fix.pick_target_frame(frames, {})
        assert chosen["relative"] == "lib/util.py"

    def test_falls_back_to_first_when_all_match_test_prefix(self):
        frames = [
            self._frame("tests/test_a.py"),
            self._frame("tests/test_b.py"),
        ]
        chosen = loop_fix.pick_target_frame(frames, {})
        assert chosen["relative"] == "tests/test_a.py"

    def test_source_prefix_accepts_list(self):
        frames = [
            self._frame("other/foo.py"),
            self._frame("lib/bar.py"),
        ]
        cfg = {"defaults": {"source_prefix": ["src/", "lib/"]}}
        chosen = loop_fix.pick_target_frame(frames, cfg)
        assert chosen["relative"] == "lib/bar.py"


# ── _qwen_env ────────────────────────────────────────────────────────────────

class TestQwenEnv:
    def test_exports_set_keys(self, monkeypatch):
        monkeypatch.delenv("QWEN_BASE_URL", raising=False)
        cfg = {"qwen": {
            "base_url": "http://x:1/v1",
            "api_key": "k",
            "model": "M",
            "approval_mode": "yolo",
        }}
        env = loop_fix._qwen_env(cfg)
        assert env["QWEN_BASE_URL"] == "http://x:1/v1"
        assert env["QWEN_API_KEY"] == "k"
        assert env["QWEN_MODEL"] == "M"
        assert env["QWEN_APPROVAL_MODE"] == "yolo"

    def test_skips_unset_keys(self, monkeypatch):
        # If a key isn't in the config, don't export it — let qwen.sh's :? default fire.
        monkeypatch.delenv("QWEN_API_KEY", raising=False)
        cfg = {"qwen": {"model": "M"}}
        env = loop_fix._qwen_env(cfg)
        assert "QWEN_API_KEY" not in env
        assert env["QWEN_MODEL"] == "M"

    def test_preserves_existing_environ(self, monkeypatch):
        monkeypatch.setenv("PATH", "/custom/path")
        env = loop_fix._qwen_env({"qwen": {"model": "M"}})
        assert env["PATH"] == "/custom/path"


# ── preflight_qwen ────────────────────────────────────────────────────────────

class TestPreflightQwen:
    def test_passes_for_real_executable(self, tmp_path):
        bin_path = tmp_path / "fake_qwen.sh"
        bin_path.write_text("#!/bin/sh\nexit 0\n")
        bin_path.chmod(0o755)
        cfg = {"qwen": {"binary": str(bin_path), "model": "M"}}
        assert loop_fix.preflight_qwen(cfg) is True

    def test_fails_when_binary_missing(self, capsys):
        cfg = {"qwen": {"binary": "/nope/nope.sh", "model": "M"}}
        assert loop_fix.preflight_qwen(cfg) is False
        out = capsys.readouterr().out
        assert "not found" in out.lower()

    def test_fails_when_binary_not_executable(self, tmp_path, capsys):
        bin_path = tmp_path / "notexec.sh"
        bin_path.write_text("#!/bin/sh\n")
        bin_path.chmod(0o644)
        cfg = {"qwen": {"binary": str(bin_path), "model": "M"}}
        assert loop_fix.preflight_qwen(cfg) is False
        assert "executable" in capsys.readouterr().out.lower()

    def test_fails_when_model_missing(self, tmp_path, capsys):
        bin_path = tmp_path / "fake_qwen.sh"
        bin_path.write_text("#!/bin/sh\n")
        bin_path.chmod(0o755)
        cfg = {"qwen": {"binary": str(bin_path)}}
        assert loop_fix.preflight_qwen(cfg) is False
        assert "model" in capsys.readouterr().out.lower()


# ── check_quality ─────────────────────────────────────────────────────────────

class TestCheckQuality:
    def test_rejects_empty(self):
        assert loop_fix.check_quality("") is False

    def test_rejects_too_short(self):
        assert loop_fix.check_quality("nope") is False

    def test_rejects_vague(self):
        # No keyword like line/error/exception/.py/function — gets rejected.
        assert loop_fix.check_quality("Something is wrong somewhere in the code." * 2) is False

    def test_accepts_specific_diagnosis(self):
        good = "Test fails because parse_zip returns 9021 instead of 90210 — root cause on line 22."
        assert loop_fix.check_quality(good) is True


# ── load_config (with merge) ──────────────────────────────────────────────────

class TestLoadConfig:
    def test_static_and_task_merge(self, tmp_path):
        static = tmp_path / "config.toml"
        static.write_text(
            '[qwen]\n'
            'binary = "./qwen.sh"\n'
            'model = "static-model"\n'
            'timeout = 600\n'
            '[defaults]\n'
            'source_window = 30\n'
        )
        task = tmp_path / "loopfix.toml"
        task.write_text(
            '[runner]\n'
            'command = "pytest"\n'
            '[defaults]\n'
            'source_prefix = "src/"\n'
            '[qwen]\n'
            'timeout = 900\n'
        )
        cfg = loop_fix.load_config(str(task), str(static))
        # Task overrides static.qwen.timeout
        assert cfg["qwen"]["timeout"] == 900
        # Static.qwen.model preserved (not in task)
        assert cfg["qwen"]["model"] == "static-model"
        # Defaults merged — both survive
        assert cfg["defaults"]["source_window"] == 30
        assert cfg["defaults"]["source_prefix"] == "src/"
        # qwen.binary resolved relative to static config dir
        assert cfg["qwen"]["binary"] == str(tmp_path / "qwen.sh")

    def test_task_only_works_without_static(self, tmp_path, monkeypatch):
        monkeypatch.delenv("LOOPFIX_CONFIG", raising=False)
        # Point STATIC_CONFIG_DEFAULTS at non-existent paths so nothing is picked up.
        monkeypatch.setattr(loop_fix, "STATIC_CONFIG_DEFAULTS",
                            (tmp_path / "missing.toml",))
        task = tmp_path / "loopfix.toml"
        task.write_text('[runner]\ncommand = "pytest"\n')
        cfg = loop_fix.load_config(str(task))
        assert cfg == {"runner": {"command": "pytest"}}

    def test_missing_task_config_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            loop_fix.load_config(str(tmp_path / "nonexistent.toml"))
