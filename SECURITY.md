# Security Policy

## Threat Model & Security Boundaries

DiodeMCP is a **local defense-in-depth MCP wire firewall** designed to protect developer workstations and AI coding agents (Claude Code, Cursor, Antigravity) against:

1. **Indirect Prompt Injection & Confused-Deputy Hijacks:** Untrusted external data (GitHub issues, PR comments, fetched web pages, database records) steering the agent into unauthorized tool calls (`http.post`, `exec.run_cmd`, `filesystem.write_file`, destructive SQL).
2. **Multi-Turn & Encoded Secret Exfiltration:** Sensitive local data (`.env` files, SSH keys, cloud tokens, internal rows) read in earlier turns and exfiltrated verbatim, chunked, or base64/hex/URL-encoded in later turns.
3. **MCP `tools/list` Poisoning & Mid-Session Rug-Pulls:** Malicious instructions embedded inside tool schema descriptions or silent schema mutations after initial catalog discovery.
4. **Non-Interactive Subprocess Pipe Friction (`C4`):** Privileged CLI commands (`diodemcp allow`, `diodemcp unwrap`, `diodemcp shutdown`) require `stdin`/`stdout` TTYs and read interactive confirmation directly from `/dev/tty` (`CONIN$`/`CONOUT$` on Windows), preventing MCP-only tool servers and naive non-PTY subprocess pipes from auto-approving blocked calls.

### Non-Goals & Same-UID Limitations

- **Unsandboxed Same-UID Host Shell Agents:** Terminal checks (`/dev/tty`, `isatty()`) and `0600` files (`~/.diodemcp/admin.secret`) are **not** an OS privilege boundary against a process already executing arbitrary shell/Python code as your own OS user: a same-UID process can allocate a pseudo-terminal (`pty.openpty()`), pass `--yes` inside that PTY, or read `~/.diodemcp/admin.secret` directly. DiodeMCP is an **MCP wire & cooperative IDE hook guard** — if an agent has unrestricted shell execution, run the agent inside a container, VM, or separate OS UID while keeping DiodeMCP outside that sandbox.
- **Classifier Generalization Limits:** Neural taint scoring reduces false positives substantially on structured MCP flows, but like any ML classifier it can produce false positives on unusual benign payloads or miss novel adaptive encodings/obfuscations outside its training distribution. Treat DiodeMCP as a defense-in-depth layer alongside least-privilege credentials and network egress controls.
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
