# Contributing to DiodeMCP

Thank you for contributing to **DiodeMCP (`diodemcp`)**! Our core engineering priorities are **fail-closed security**, **near-zero false positives on real developer workflows**, and **lightweight local CPU execution (zero PyTorch in the runtime package)**.

## Development Setup (`uv`)

Prerequisites:
- Python `3.11`–`3.13`
- [`uv`](https://docs.astral.sh/uv/) (`>= 0.5`)
- Node.js `>= 18` (optional, for `packages/npm-diodemcp` launcher tests)

Clone the repository and sync the runtime environment:

```bash
git clone https://github.com/PhilippFrenzel2005/diodemcp.git
cd diodemcp
uv sync --dev
```

## Architectural Guardrails

1. **Zero-PyTorch Runtime Contract:** `pyproject.toml` and `packages/diodemcp/pyproject.toml` must **never** depend on `torch`, `transformers`, `scipy`, `triton`, or platform-locked `fcntl`. All neural inference in `packages/diodemcp` runs through `onnxruntime` + `tokenizers`. PyTorch training and quantization live strictly inside `model-lab/`.
2. **No Binary Model Bloat in Git:** Never commit `.onnx` weights or `bundles/` directories to git. Model bundles (`DiodeModelBundle v1.0`) are distributed via GitHub Releases / Hugging Face Hub and verified via SHA-256 digests in `manifest.json`.
3. **Controlling-TTY Guard (`C4`):** Privileged CLI transitions (`diodemcp allow`, `diodemcp unwrap`, `diodemcp shutdown`) must verify TTY streams and prompt via `/dev/tty` (`CONIN$`/`CONOUT$` on Windows) in `cli.py` so non-interactive subprocess pipes are rejected, while keeping documentation honest that same-UID PTY allocation requires OS container/UID isolation.

## Running the Test Suites

Run all runtime, packaging, benchmark, and launcher tests before opening a pull request:

```bash
# 1. Core runtime, ONNX predictor, hub, and root uv wheel build tests
uv run pytest packages/diodemcp/tests -q

# 2. Benchmark harness & scoring unit tests
PYTHONPATH=.:packages/diodemcp/src uv run pytest benchmarks/tests -q

# 3. Node.js npx launcher tests
npm --prefix packages/npm-diodemcp test

# 4. Verify root wheel packaging cleanly includes registry_data.toml
uv build --wheel
```

If you modify training, export, or calibration code in `model-lab/`:

```bash
cd model-lab && uv run pytest tests -q
```

## Pull Request Checklist

- [ ] Tests added/updated for any bugfix or new behavior (`pytest` passes cleanly).
- [ ] Zero new heavy dependencies added to root `pyproject.toml` or `packages/diodemcp/pyproject.toml`.
- [ ] Relative links and SVG assets in `README.md` pass `test_repo_launch.py`.

By submitting a contribution, you agree that your work will be licensed under the [Apache License 2.0](LICENSE).
