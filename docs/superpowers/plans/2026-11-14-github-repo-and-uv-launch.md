# DiodeMCP GitHub Repo Launch & `uv` Install Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship root `uv` installation (`uv tool install` / `uvx`), GitHub Releases v2 INT8 model bundle fallback (public + private token support), custom dark/light SVGs, flagship `README.md`, `CONTRIBUTING.md`, `SECURITY.md`, and GitHub Actions CI on `PhilippFrenzel2005/diodemcp` (kept private for review).

**Architecture:** Add a root Hatchling `pyproject.toml` mapping `packages/diodemcp/src/diodemcp` without duplicating source code; extend `diodemcp.hub` so canonical `base`/`nano` aliases transparently fall back to GitHub Releases (`models-v2.0-base` / `models-v2.0-nano`) on Hugging Face `404` (with GitHub REST API asset resolution when `GITHUB_TOKEN`/`GH_TOKEN` is set on a private repo); author pure-SVG dark/light hero, benchmark, and terminal assets plus CI and documentation.

**Tech Stack:** Python `>=3.11,<3.14`, Hatchling, `uv`, `httpx`, `pytest`, GitHub Actions, SVG 1.1.

**Spec:** `docs/superpowers/specs/2026-11-14-github-repo-and-uv-launch-design.md`

## Global Constraints
- Keep `PhilippFrenzel2005/diodemcp` private (`private: true`).
- Zero `torch`, `transformers`, `scipy`, `triton`, or `fcntl` runtime dependencies in root `pyproject.toml`.
- Zero `.onnx` binaries tracked in git.
- Preserve existing `DEFAULT_MODEL_REPOS` and `MODEL_ALIASES` values for `base`, `large`, `nano`.

## Review Focus
1. Root `pyproject.toml` wheel build omits `registry_data.toml` non-py package data -> test building wheel with `uv build --wheel` and running `diodemcp status --json` from an isolated `uv` environment.
2. Private GitHub Release asset download forwards `Authorization` header onto cross-origin S3/Azure redirect (`400 InvalidArgument`) -> test that `Authorization` is sent only to `api.github.com` and stripped on cross-host `302` redirect.
3. Tampered ONNX or HTTP 500 on primary Hugging Face URL erroneously triggers GitHub fallback -> test that fallback triggers ONLY on initial `manifest.json` HTTP 404 for `base`/`nano` at `revision == "main"`.
4. Relative links or `<picture>` SVG paths in `README.md` point to missing files or contain `<script>`/`<foreignObject>` stripped by GitHub sanitizer -> test XML validity, forbidden tags, and relative file existence.
5. CI workflow misses `packages/diodemcp` `PYTHONPATH` in `benchmarks/tests` -> test exact CI commands locally before pushing.

---

### Task 1: Root `uv` Packaging (`pyproject.toml`)

**Files:**
- Create: `pyproject.toml`
- Test: `packages/diodemcp/tests/test_repo_launch.py`

**Interfaces:**
- Produces: Root buildable distribution `diodemcp==0.2.0` (`[project.scripts] diodemcp = "diodemcp.cli:main"`), synced with `packages/diodemcp/pyproject.toml` (`version = "0.2.0"`).

- [ ] **Step 1: Write failing test `test_root_pyproject_matches_zero_pytorch_contract_and_builds_wheel` in `packages/diodemcp/tests/test_repo_launch.py`**
- [ ] **Step 2: Run pytest to verify failure (`pyproject.toml` missing)**
- [ ] **Step 3: Create root `pyproject.toml` and bump `packages/diodemcp/pyproject.toml` + `packages/npm-diodemcp/package.json` to `0.2.0`**
- [ ] **Step 4: Run pytest + `uv build --wheel` to verify pass**
- [ ] **Step 5: Commit**

---

### Task 2: GitHub Releases v2 Bundle Fallback & Private-Repo Token Support in `hub.py`

**Files:**
- Modify: `packages/diodemcp/src/diodemcp/hub.py`
- Test: `packages/diodemcp/tests/test_ort_predictor.py`

**Interfaces:**
- Produces: `GITHUB_RELEASE_FALLBACK_URLS: Final[dict[str, str]]` (`"base"`, `"nano"`), `MODEL_ALIASES["gh-base"]` & `MODEL_ALIASES["gh-nano"]`, transparent HF-404 -> GitHub Releases fallback in `download_and_verify_bundle`, and `GITHUB_TOKEN` / `GH_TOKEN` private release asset resolution via `https://api.github.com/repos/<owner>/<repo>/releases/tags/<tag>` without leaking `Authorization` across S3 redirects.

- [ ] **Step 1: Write failing tests in `packages/diodemcp/tests/test_ort_predictor.py` for HF-404 fallback, non-404 non-fallback, and private GitHub API asset redirect auth stripping**
- [ ] **Step 2: Run pytest to verify failure**
- [ ] **Step 3: Implement fallback & private GitHub Release asset streaming in `packages/diodemcp/src/diodemcp/hub.py`**
- [ ] **Step 4: Run `pytest packages/diodemcp/tests/test_ort_predictor.py` to verify pass**
- [ ] **Step 5: Commit**

---

### Task 3: Custom Dark/Light SVGs, Flagship `README.md`, `CONTRIBUTING.md`, `SECURITY.md`, & `.github/workflows/ci.yml`

**Files:**
- Create: `.github/assets/hero-dark.svg`, `.github/assets/hero-light.svg`, `.github/assets/benchmark-dark.svg`, `.github/assets/benchmark-light.svg`, `.github/assets/terminal-demo.svg`
- Modify: `README.md`
- Create: `CONTRIBUTING.md`, `SECURITY.md`, `.github/workflows/ci.yml`
- Test: `packages/diodemcp/tests/test_repo_launch.py`

**Interfaces:**
- Produces: Sanitizer-safe SVGs, complete `README.md` with `<picture>` dark/light switching and `uv` quickstart, community health files, and automated GitHub Actions CI.

- [ ] **Step 1: Add failing tests in `packages/diodemcp/tests/test_repo_launch.py` asserting SVG XML validity (no `<script>`/`<foreignObject>`), `README.md` relative link integrity, and presence of `CONTRIBUTING.md`, `SECURITY.md`, and `.github/workflows/ci.yml`**
- [ ] **Step 2: Run pytest to verify failure**
- [ ] **Step 3: Create the 5 SVGs, `README.md`, `CONTRIBUTING.md`, `SECURITY.md`, and `.github/workflows/ci.yml`**
- [ ] **Step 4: Run full test suites (`packages/diodemcp`, `model-lab`, `benchmarks/tests`, `npm-diodemcp`)**
- [ ] **Step 5: Commit, push PR, verify green CI on GitHub, merge to `main`, and run live `uv tool install` + `diodemcp init` smoke test**
