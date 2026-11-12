# `diodemcp` (npm / `npx` Launcher)

Zero-dependency Node.js (`>=18`) ESM bootstrapper for **DiodeMCP** — the local taint-tracking firewall and Jev cross-encoder runtime for MCP servers (`stdio` + Streamable HTTP) and coding agents (`Claude Code`, `Cursor`, `Antigravity`).

## Quickstart

### 1. One-Command Setup via `npx`

```bash
# Downloads & SHA256-verifies the DiodeModelBundle v1 ONNX bundle upfront,
# wraps MCP servers, and installs fail-closed IDE hooks:
npx diodemcp init

# Offline / deterministic Tier-0 mode (no ONNX model download required):
npx diodemcp init --tier0-only
```

### 2. Persistent Global Install via `uv` (Recommended)

For the fastest cold-start hook latency (`<20ms`), install the Python CLI globally with `uv`:

```bash
# Install uv if not already installed:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install diodemcp globally on PATH:
uv tool install diodemcp

# Initialize IDE hooks & MCP shims:
diodemcp init
```

### 3. Inspecting Status, Redacted Audit Logs & Out-of-Band Approvals

```bash
# Check daemon health, active predictor bundle, and wrapped/OAuth MCP servers
diodemcp status
# (or: npx diodemcp doctor / npx diodemcp status)

# Inspect recent security incidents (secrets & code previews are redacted on disk)
diodemcp audit

# Grant a single-use, time-bounded human TTY approval for a blocked incident
diodemcp allow <incident_id>

# Losslessly unwrap IDE configs & restore original MCP servers (TTY-guarded)
diodemcp unwrap
```

## How `npx diodemcp` Resolves the Runtime

1. **`DIODEMCP_BIN`** environment variable override (when set to an executable path).
2. **`diodemcp`** binary on `PATH` (automatically skipping `node_modules/.bin/diodemcp` and self symlinks to prevent `npx` infinite recursion, with `PATHEXT` support on Windows).
3. **`uvx --quiet diodemcp ...`** fallback when `uvx` is available on `PATH`.
4. **`pipx run --quiet diodemcp ...`** fallback when `pipx` is available on `PATH`.
5. Exits with code `127` and prints an install hint to `stderr` (never polluting `stdout`).

All `stdin`, `stdout`, and `stderr` streams are inherited byte-for-byte with zero startup banners on `stdout`, making `npx -y diodemcp shim ...` 100% safe for `stdio` JSON-RPC MCP transports.

## License

Apache-2.0
