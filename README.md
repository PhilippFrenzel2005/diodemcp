<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/hero-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset=".github/assets/hero-light.svg">
  <img alt="DiodeMCP — System-One Jev MCP Security Firewall" src=".github/assets/hero-dark.svg" width="100%">
</picture>

<br>

### A local, provenance-aware ONNX taint firewall for MCP servers & AI coding agents.

[![CI](https://github.com/PhilippFrenzel2005/diodemcp/actions/workflows/ci.yml/badge.svg)](https://github.com/PhilippFrenzel2005/diodemcp/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-0ea5e9.svg)](LICENSE)
[![Python 3.11–3.13](https://img.shields.io/badge/Python-3.11%E2%80%933.13-10b981.svg)](pyproject.toml)
[![Runtime: Zero-PyTorch ONNX](https://img.shields.io/badge/Runtime-INT8_ONNX_(Zero_PyTorch)-22d3ee.svg)](packages/diodemcp/pyproject.toml)

[**Quickstart (`uv`)**](#-quickstart-with-uv) · [**Live Wire Catch**](#-what-it-catches-on-the-wire) · [**Benchmarks**](#-v2-competitive-benchmarks) · [**How Jev Works**](#-how-system-one-jev-works) · [**Model Tiers**](#-model-tiers) · [**CLI Reference**](#-cli-reference)

</div>

---

AI coding agents (**Claude Code**, **Cursor**, **Antigravity**) connect high-privilege tools (`filesystem`, `postgres`, `exec`, `.env` readers) directly to untrusted external surfaces (GitHub issues, PR descriptions, web pages, third-party `tools/list` catalogs). A single poisoned issue or rug-pulled MCP tool description can silently hijack your agent into exfiltrating secrets or running destructive SQL/shell commands.

**DiodeMCP (`diodemcp`)** wraps your MCP servers in one command and enforces **provenance-aware information-flow control** on every JSON-RPC call — **100% locally on CPU** via quantized INT8 `onnxruntime` with **zero cloud telemetry** and **zero PyTorch/CUDA dependencies**.

- **Intent-conditioned provenance cuts false alarms on matched sinks:** On our `600`-trace counterfactual replay suite (**Suite A**), `NanoJev-0.6B` reaches **`95.6%` recall at `0.4%` benign FPR** (`1/240`) and `Base-149M` reaches **`94.4%` recall at `1.2%` FPR** (`3/240`), compared to `55.8%` FPR (`134/240`) when running single-turn stateless scanners (`LLM-Guard` adapter) without user-intent context.
- **Tracks multi-turn & `tools/list` wire taint:** Evaluates outbound tool calls against `[TRUSTED_INTENT]`, `[PRIVATE_DATA]`, and `[UNTRUSTED_DATA]`, catching encoded/chunked exfiltration, `tools/list` description poisoning, catalog rug-pulls, and JSON-RPC error injections (`19/20` live `stdio` attacks blocked in **Suite B**).
- **Zero-PyTorch CPU ONNX runtime:** Pure `onnxruntime` + `tokenizers` install (`~65 MB` wheels, no `torch` or `transformers`). Fast-path wire overhead is **`7.8–11.8 ms` p50** (`81–246 ms` p95 on live sessions), while full CPU INT8 ONNX sink evaluations take **`~154 ms` p50** on `Base-149M` (`~515 ms` on `NanoJev-0.6B`) on 4 desktop CPU threads.
- **Controlling-TTY approval prompt (`C4`) & clear OS trust boundary:** `diodemcp allow` and `diodemcp unwrap` prompt on `/dev/tty` (`CONIN$`/`CONOUT$` on Windows) to reject non-interactive pipes and MCP-only tool servers. *(Note: unsandboxed host-shell agents running as the same OS UID can allocate PTYs; run untrusted shell agents in a container or separate OS user — see [SECURITY.md](SECURITY.md).)*

---

## ⚡ Quickstart with `uv`

Install globally as an isolated CLI tool with [`uv`](https://docs.astral.sh/uv/) and wrap your IDE's MCP servers in two commands:

```bash
# 1. Install the zero-PyTorch CLI via uv
uv tool install git+https://github.com/PhilippFrenzel2005/diodemcp

# 2. Download & SHA-256 verify the INT8 ONNX model and wrap Claude Code, Cursor & Antigravity
diodemcp init
```

Or run ephemerally with **`uvx`** without installing globally:

```bash
uvx --from git+https://github.com/PhilippFrenzel2005/diodemcp diodemcp init
```

<details>
<summary><b>Alternative: Flagship <code>NanoJev-0.6B</code> install or <code>npx</code> launcher</b></summary>

**Install with the flagship `NanoJev-0.6B` bundle (`0.999` AUROC, `0.4%` FPR):**
```bash
uv tool install git+https://github.com/PhilippFrenzel2005/diodemcp
diodemcp init --model nano
```

**Node / `npx` wrapper** (bootstraps an isolated `~/.diodemcp/venv` via `uv` or `python3`):
```bash
npx --yes @diodemcp/cli init
```
</details>

---

## 🛡️ What It Catches on the Wire

When an untrusted MCP tool output or poisoned `tools/list` catalog hijacks your agent into sending secrets or running unauthorized mutations, DiodeMCP blocks the outbound `tools/call` **before** it reaches the upstream server and attributes the exact untrusted source chunk and sink argument:

<div align="center">
  <img alt="DiodeMCP Live Terminal Interception Demo" src=".github/assets/terminal-demo.svg" width="100%">
</div>

---

## 📊 v2 Competitive Benchmarks

Evaluated inside network-isolated namespaces (`unshare -rn`, air-gapped, `4` CPU threads on an `Intel Core i3-10100`, zero GPU) across two reproducible suites:
- **Suite A (`600` Multi-Turn Trajectories):** `360` attacks across 6 vectors (`V1` Indirect Injection → Exfil, `V2` Unicode `Cf`/NFKC Obfuscation, `V3` `tools/list` Poisoning & Rug-Pull, `V4` Base64/Chunked Cross-Turn Exfil, `V5` Confused-Deputy SQL/Shell Sabotage, `V6` Cross-Server Escalation) paired against **`240` matched-sink benign developer twins** invoking the *exact same* `http.post`, `github.create_issue`, `postgres.query`, `filesystem.write_file`, and `exec.run_cmd` sinks.
- **Suite B (`30` Live `stdio` MCP Wire Sessions):** Real child-process MCP servers testing `tools/list` catalog poisoning, mid-session rug-pulls, cross-server secret exfiltration, and JSON-RPC error injections.

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/benchmark-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset=".github/assets/benchmark-light.svg">
  <img alt="DiodeMCP Suite A Benchmark Chart" src=".github/assets/benchmark-dark.svg" width="100%">
</picture>
</div>

<br>

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/plot-roc-pareto-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset=".github/assets/plot-roc-pareto-light.svg">
  <img alt="DiodeMCP Empirical ROC Curves and Security-Usability Pareto Frontier" src=".github/assets/plot-roc-pareto-dark.svg" width="100%">
</picture>
</div>

<br>

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/plot-vector-heatmap-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset=".github/assets/plot-vector-heatmap-light.svg">
  <img alt="DiodeMCP Per-Vector Attack Recall (V1-V6) and Matched-Sink Benign FPR (B1-B6) Heatmap" src=".github/assets/plot-vector-heatmap-dark.svg" width="100%">
</picture>
</div>

<br>

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/plot-live-wire-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset=".github/assets/plot-live-wire-light.svg">
  <img alt="DiodeMCP Suite B Live MCP stdio Wire Recall and Latency" src=".github/assets/plot-live-wire-dark.svg" width="100%">
</picture>
</div>

| System (Autonomous Hard-Block Gate) | Suite A Recall (`360`) | Suite A Benign FPR (`240`) | Suite A AUROC | Suite A Eval p50 | Suite B Wire Recall (`20`) | Suite B Benign FP (`10`) | Suite B Wire p50 / p95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`DiodeMCP NanoJev-0.6B` (v2)** | **`95.6%`** (`344/360`) | **`0.4%`** (`1/240`) | **`0.999`** | `514.7 ms` | **`95.0%`** (`19/20`) | `1/10` | `7.8 / 246.0 ms` |
| **`DiodeMCP Base-149M` (v2 Default)** | **`94.4%`** (`340/360`) | **`1.2%`** (`3/240`) | **`0.990`** | **`153.7 ms`** | **`95.0%`** (`19/20`) | `2/10` | `11.8 / 81.2 ms` |
| `DiodeMCP Tier-0 Only` *(rules ablation)* | `94.4%` (`340/360`) | `15.4%` (`37/240`) | `0.979` | `27.6 ms` | `45.0%` (`9/20`) | `0/10` | `7.7 / 19.3 ms` |
| `ProtectAI LLM-Guard` *(stateless adapter)* | `92.8%` (`334/360`) | `55.8%` (`134/240`) | `0.742` | `1,025.9 ms` | `75.0%` (`15/20`) | `3/10` | `317.2 / 344.2 ms` |
| `Stateless Regex Policy` | `48.6%` (`175/360`) | `5.4%` (`13/240`) | `0.716` | `0.35 ms` | `30.0%` (`6/20`) | `0/10` | `0.35 / 0.55 ms` |
| `Cisco mcp-scanner` *(offline static/YARA)* | `12.5%` (`45/360`) | `0.0%` (`0/240`) | `0.562` | `0.71 ms` | `40.0%` (`8/20`) | `0/10` | `0.27 / 0.48 ms` |

> [!NOTE]
> **Benchmark Scope, Adapter Fairness & Limitations**
> - **Templated counterfactual coverage:** Suite A (`600` traces) and Suite B (`30` live `stdio` sessions) are procedurally generated from structured templates with varied entities, secrets, and sinks (disjoint from training augmentations). They measure controlled architectural separation between stateful provenance and stateless scanning — **not** unseen adaptive jailbreaks against live frontier agents in the wild.
> - **Small live-wire benign sample (`n=10`):** Suite B is an end-to-end protocol smoke suite; `NanoJev-0.6B` falsely blocked `1/10` benign live sessions and `Base-149M` blocked `2/10`. Larger real-world workflow evaluations are needed to bound everyday live false-alarm rates.
> - **Baseline adapter context:** `Cisco mcp-scanner` (`v4.8.7`) was run in its local offline YARA/schema mode (without cloud LLM API keys), and `ProtectAI LLM-Guard` (`v0.3.16`) combines single-turn `PromptInjection`, `Secrets`, and `Sensitive` scanners via our benchmark adapter. Because neither tool is designed as a stateful multi-turn taint tracker, these numbers reflect **task-fit differences** on multi-turn MCP traces rather than general defects in those tools.
> - Full methodology, confidence intervals, and v1-to-v2 ablations: [COMPETITIVE_BENCHMARK_REPORT.md](benchmarks/results/COMPETITIVE_BENCHMARK_REPORT.md).

---

## 🧠 How System-One Jev Works

Prompt-only classifiers (`LLM-Guard`, regexes) inspect tool inputs or outputs in isolation without knowing **what the developer actually asked for** or **where each piece of data originated**. When a developer legitimately posts a build config to a webhook or writes security unit tests containing attack strings, stateless scanners trigger constant false alarms (`55.8%` FPR).

DiodeMCP models MCP security as **structured provenance & taint verification**:

1. **Four-Lane Provenance Serialization (`14` Special Jev Tokens):** Every session event is tagged into structured spans before neural evaluation:
   - `[TRUSTED_INTENT]` — The user's explicit task and authorized target scope.
   - `[PRIVATE_DATA]` (`pchunk#k`) — Sensitive local reads (`.env`, SSH/cloud credentials, private DB rows).
   - `[UNTRUSTED_DATA]` (`uchunk#k`) — External content (GitHub issues/PRs, web fetches, `tools/list` catalog descriptions, upstream JSON-RPC errors).
   - `[CANDIDATE_ACTION]` (`arg=…`) — The outbound `tools/call` tool name and structured arguments under inspection.
2. **Tier-0 Cryptographic Gate + Catalog Pinning:** Blocks raw private keys/API tokens (`sk-…`, `AKIA…`, GitHub PATs, PEM blocks) leaving untrusted sinks in `<0.2 ms`, and hashes `tools/list` schemas on first sight to detect mid-session rug-pull mutations.
3. **System-One Jev Neural Heads + Span Pointers:** A single quantized INT8 ONNX forward pass scores three calibrated heads — **`p_enc`** (encoded/transformed secret leakage), **`p_sem`** (semantic exfiltration), and **`p_hijack`** (control-flow / confused-deputy hijack) — plus **bipartite span-pointer heads** that pinpoint which `uchunk`/`pchunk` tainted which outbound argument.
4. **Fail-Closed & Controlling-TTY Guarded (`C4`):** Unverified model hashes fail closed; interactive overrides (`diodemcp allow`, `unwrap`, `shutdown`) verify TTY `stdin`/`stdout` and prompt directly on `/dev/tty` (`CONIN$`/`CONOUT$` on Windows) to block non-interactive pipes and MCP-only tool servers. *(Same-UID host-shell agents with PTY/filesystem access require OS container or UID isolation.)*

---

## 🧩 Model Tiers

Switch models anytime with `diodemcp model install --model <alias>`. Bundles are verified against pinned SHA-256 digests (`manifest.json`) and cached in `~/.diodemcp/models/`.

| Alias | Model Bundle | Backbone | INT8 Size | CPU Eval p50 | Suite A AUROC / FPR | Best For |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **`base`** *(default)* | `DiodeMCP-Jev-Base-149M` | `ModernBERT-base` | `266 MiB` | `154 ms` | `0.990` / `1.2%` | Fast everyday laptop CPU protection (`diodemcp init`) |
| **`nano`** *(flagship)* | `DiodeMCP-NanoJev-0.6B` | `Qwen3-0.6B-Base` | `1.29 GiB` | `515 ms` | **`0.999` / `0.4%`** | Highest Suite A multi-turn accuracy (`diodemcp init --model nano`) |
| `--tier0-only` | Deterministic Gate Only | Regex + Catalog Pins | `0 MiB` | `<1 ms` | `0.979` / `15.4%` | Zero-download cryptographic secret & catalog pin guard |

---

## 🛠️ CLI Reference

```bash
# Initialize model bundle & wrap Claude Code, Cursor, and Antigravity MCP configs
diodemcp init                          # Default: Base-149M (266 MiB INT8 ONNX, ~154ms CPU eval)
diodemcp init --model nano             # Flagship: NanoJev-0.6B (1.29 GiB, ~515ms CPU eval)
diodemcp init --target claude          # Target a single IDE: claude | cursor | agy | all
diodemcp init --tier0-only             # Zero-download cryptographic secret & pin mode

# Upgrade or switch the active neural model bundle
diodemcp model install --model nano

# Inspect wrapped MCP servers, daemon socket, and active model health
diodemcp status

# View recent redacted incidents and span attributions
diodemcp audit --limit 20
diodemcp audit --json

# Approve a single blocked tool call on the controlling terminal (/dev/tty)
diodemcp allow inc_01J9X7K2M8

# Cleanly restore original IDE MCP configs from backup (/dev/tty prompt)
diodemcp unwrap
```

---

## 📂 Repository Layout

| Path | Description |
| :--- | :--- |
| [`pyproject.toml`](pyproject.toml) | Root Hatchling build config for one-line `uv tool install` / `uvx` installation |
| [`packages/diodemcp/`](packages/diodemcp) | Zero-PyTorch Python runtime: MCP `stdio` shim, taint graph, Tier-0 gate, `onnxruntime` Jev predictor, daemon, and CLI |
| [`packages/npm-diodemcp/`](packages/npm-diodemcp) | `@diodemcp/cli` Node.js bootstrapper for `npx` workflows |
| [`model-lab/`](model-lab) | Training, `wire_augment` curriculum synthesis, temperature/per-head calibration, and INT8 ONNX export pipeline |
| [`benchmarks/`](benchmarks) | Reproducible Suite A (`600` multi-turn traces) & Suite B (`30` live `stdio` MCP wire scenarios) harness and results |

---

## 🤝 Contributing & Security

- **Contributing:** See [CONTRIBUTING.md](CONTRIBUTING.md) for `uv` workspace setup and running the unit, model-lab, and benchmark test suites.
- **Security Policy:** See [SECURITY.md](SECURITY.md) for the threat model, controlling-TTY & same-UID boundaries, and vulnerability reporting.
- **License:** Licensed under the [Apache License 2.0](LICENSE).
