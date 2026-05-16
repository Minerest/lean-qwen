# Examples

Sample `loopfix.toml` files showing how to wire `lean-qwen` to different test
runners. Copy one into your project, edit the `[runner]` command and add
your own `[[tasks]]`, then run:

```
lean-qwen -c loopfix.toml
```

| File                                    | Stack                              |
| --------------------------------------- | ---------------------------------- |
| [`python-pytest/loopfix.toml`](python-pytest/loopfix.toml) | Python / pytest (deluxe traceback path) |
| [`go-test/loopfix.toml`](go-test/loopfix.toml)             | Go / `go test` (fallback path)          |
| [`js-jest/loopfix.toml`](js-jest/loopfix.toml)             | JS-TS / jest (fallback path)            |

The Python example exercises the full deluxe path — `lean-qwen` parses the
pytest traceback, extracts the failing frame, and feeds the model a source
window around the failing line. Go and JS use the language-agnostic
fallback: compressed error tail + git diff + the files listed in the task.
