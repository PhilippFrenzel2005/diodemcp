# Security Policy

## Threat Model & Security Boundaries

DiodeMCP is a **local defense-in-depth MCP wire firewall** designed to protect developer workstations and AI coding agents (Claude Code, Cursor, Antigravity) against:

1. **Indirect Prompt Injection & Confused-Deputy Hijacks:** Untrusted external data (GitHub issues, PR comments, fetched web pages, database records) steering the agent into unauthorized tool calls (`http.post`, `exec.run_cmd`, `filesystem.write_file`, destructive SQL).
2. **Multi-Turn & Encoded Secret Exfiltration:** Sensitive local data (`.env` files, SSH keys, cloud tokens, internal rows) read in earlier turns and exfiltrated verbatim, chunked, or base64/hex/URL-encoded in later turns.
3. **MCP `tools/list` Poisoning & Mid-Session Rug-Pulls:** Malicious instructions embedded inside tool schema descriptions or silent schema mutations after initial catalog discovery.
4. **Agent Self-Approval Bypass (`C4`):** A hijacked coding agent attempting to run `diodemcp allow` or `diodemcp unwrap` via a shell tool. DiodeMCP mitigates this by opening `/dev/tty` (`CONIN$` on Windows) directly rather than reading `stdin`, rejecting non-interactive subprocess pipes.

### Non-Goals / Out of Scope

- **Root / Kernel Compromise on the Same Host:** If an attacker already has arbitrary local code execution outside the MCP client/agent boundary under your OS user account, they can modify user-owned files directly. Run untrusted code execution tools inside containers or VMs alongside DiodeMCP.
- **Model Bundle Integrity Bypass via Custom Unpinned Local Paths:** Official `base` and `nano` bundles enforce strict SHA-256 verification against `manifest.json` before `onnxruntime` loads any graph.

## Supported Versions

| Version | Supported |
| :--- | :---: |
| `0.1.x` (`main` / v2 ONNX bundles) | ✅ |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for suspected security vulnerabilities, firewall bypasses, `tty_guard` escapes, or hash-verification flaws.

Instead, please report vulnerabilities privately via [**GitHub Private Vulnerability Reporting**](https://github.com/PhilippFrenzel2005/diodemcp/security/advisories/new) on this repository, including:

1. A minimal reproduction trace (JSON-RPC sequence or `Suite A`/`Suite B` test case).
2. Which model tier (`nano`, `base`, or `--tier0-only`) and OS/Python version were tested.
3. Expected vs. observed firewall verdict (`ALLOW`, `WARN`, `CRITICAL`).

We aim to acknowledge reports within **48 hours** and ship regression-tested fixes prior to coordinated disclosure.
