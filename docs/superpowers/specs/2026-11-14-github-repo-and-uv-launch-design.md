# DiodeMCP GitHub Repository Launch & `uv` Install — Short Design

**Date:** 2026-11-14  
**Repository:** `PhilippFrenzel2005/diodemcp` (kept **private** for maintainer review before flipping public)

## Intent & Why
Give `PhilippFrenzel2005/diodemcp` a developer-first, star-worthy GitHub repository landing experience modeled on `astral-sh/uv` and `rtk-ai/rtk` where visitors grasp the security problem and v2 benchmark proof in 5 seconds, and install a working neural MCP firewall via `uv` in two copy-paste commands.

## Goals, Non-Goals & Anti-Goals
- **Goals:**
  - Unified `main` branch containing `packages/diodemcp`, `packages/npm-diodemcp`, `model-lab`, and `benchmarks`.
  - Root `pyproject.toml` so `uv tool install git+https://github.com/PhilippFrenzel2005/diodemcp` and `uvx --from git+https://github.com/PhilippFrenzel2005/diodemcp diodemcp ...` work out of the box (and `uv build` / `uv publish` works for PyPI).
  - Hosted v2 INT8 ONNX bundles on GitHub Releases (`models-v2.0-base` and `models-v2.0-nano`) with transparent SHA-256-verified fallback in `diodemcp.hub` (including authenticated GitHub Release asset download via `GITHUB_TOKEN`/`GH_TOKEN` while the repo is private).
  - Custom dark/light `<picture>` SVGs (`.github/assets/`), flagship `README.md`, `CONTRIBUTING.md`, `SECURITY.md`, and green GitHub Actions CI workflow (`.github/workflows/ci.yml`).
- **Non-Goals:** Separate hosted `github.io` web app or PyPI credential configuration in this pass.
- **Anti-Goals:** Marketing buzzwords, wall-of-text prose above the install snippet, badge clutter, or committing `.onnx` weights into git history.

## Constraints
- Keep `PhilippFrenzel2005/diodemcp` **private** until maintainer review completes.
- Zero `.onnx` binaries in git; runtime Python dependencies strictly zero-PyTorch (`numpy`, `onnxruntime`, `tokenizers`, `httpx`, `pydantic`, `typer`, `filelock`).
- Preserve all existing `MODEL_ALIASES` / `DEFAULT_MODEL_REPOS` Hugging Face contracts and all `409` existing tests.

## Approaches
- **Option 1 (Selected):** Full self-contained GitHub launch — merge `PR #1–#3` into `main`, add root `pyproject.toml`, host both v2 ONNX bundles on GitHub Releases with `hub.py` fallback, and build SVG-rich `README.md` + CI + community docs.
- **Option 2 (Rejected):** Repo polish only without hosted ONNX release bundles — rejected because `diodemcp init` after `uv tool install` would fail without `--tier0-only`.
