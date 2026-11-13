"""Typer CLI (`diodemcp init|unwrap|model|status|audit|allow|shutdown|shim|hook`) with `C4` TTY guard."""

from __future__ import annotations

from collections.abc import Sequence
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Optional

from diodemcp.constants import DEFAULT_STATE_DIR

_CACHED_APP: Any = None


def _resolve_state_dir(
    state_dir: Path | None,
    *,
    home_dir: Path | None = None,
) -> Path:
    if state_dir is not None:
        return Path(state_dir).expanduser().resolve()
    env_val = os.environ.get("DIODEMCP_STATE_DIR")
    if env_val and env_val.strip():
        return Path(env_val.strip()).expanduser().resolve()
    if home_dir is not None:
        return (Path(home_dir).expanduser().resolve() / ".diodemcp").resolve()
    return DEFAULT_STATE_DIR.expanduser().resolve()


def _resolve_home_dir(home_dir: Path | None) -> Path:
    if home_dir is not None:
        return Path(home_dir).expanduser().resolve()
    env_val = os.environ.get("DIODEMCP_HOME_DIR")
    if env_val and env_val.strip():
        return Path(env_val.strip()).expanduser().resolve()
    return Path.home().resolve()


def _sync_config_toml_tier0_flag(
    state_dir: Path,
    *,
    tier0_only: bool,
) -> Path:
    """Persist `tier0_only = true|false` in `0600` `<state_dir>/config.toml` while preserving user tables."""
    from diodemcp.transport import atomic_write_private_bytes, ensure_private_dir

    ensure_private_dir(state_dir)
    cfg_toml = state_dir / "config.toml"
    preserved_lines: list[str] = []
    if cfg_toml.is_file() and not cfg_toml.is_symlink():
        try:
            for line in cfg_toml.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith("tier0_only") and "=" in stripped:
                    continue
                preserved_lines.append(line)
        except OSError:
            preserved_lines = []

    flag_line = f"tier0_only = {'true' if tier0_only else 'false'}"
    body = "\n".join([flag_line, *preserved_lines]).strip() + "\n"
    atomic_write_private_bytes(cfg_toml, body.encode("utf-8"))
    return cfg_toml


def _restart_daemon_if_running(state_dir: Path) -> None:
    """Stop any live daemon on `state_dir` and wait for endpoint teardown so the next RPC loads updated config."""
    import time

    from diodemcp.engine import EngineClient
    from diodemcp.transport import (
        TransportError,
        default_transport,
        is_pid_alive,
        read_admin_secret,
    )

    cfg = default_transport(state_dir)
    sock_path = getattr(cfg, "socket_path", None)
    has_endpoint = cfg.addr_file.exists() or (
        sock_path is not None and sock_path.exists()
    )
    if not has_endpoint or not cfg.admin_secret_file.exists():
        return
    try:
        secret = read_admin_secret(cfg)
        res = EngineClient(cfg, auto_spawn=False).shutdown(admin_secret=secret)
    except (TransportError, ValueError, OSError):
        return

    old_pid = res.get("pid") if isinstance(res, dict) else None
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        still_bound = cfg.addr_file.exists() or (
            sock_path is not None and sock_path.exists()
        )
        pid_alive = isinstance(old_pid, int) and is_pid_alive(old_pid)
        if not still_bound and not pid_alive:
            break
        time.sleep(0.01)


def _ensure_interactive_tty() -> None:
    """Verify both `sys.stdin` and `sys.stdout` are real interactive TTYs (`C4`)."""
    stdin_ok = bool(
        sys.stdin is not None
        and hasattr(sys.stdin, "isatty")
        and sys.stdin.isatty()
    )
    stdout_ok = bool(
        sys.stdout is not None
        and hasattr(sys.stdout, "isatty")
        and sys.stdout.isatty()
    )
    if not (stdin_ok and stdout_ok):
        sys.stderr.write(
            "Error [C4]: This privileged DiodeMCP command requires an interactive human TTY "
            "(both stdin and stdout must be a TTY). Non-interactive agent subprocesses are forbidden.\n"
        )
        sys.stderr.flush()
        raise SystemExit(2)


def require_interactive_tty(
    *,
    yes: bool = False,
    action_description: str = "perform privileged action",
    accepted_tokens: Sequence[str] = (),
) -> None:
    """Enforce `C4` interactive human TTY requirement (`sys.stdin.isatty() and sys.stdout.isatty()`)."""
    _ensure_interactive_tty()

    if not yes:
        sys.stdout.write(
            f"Confirm DiodeMCP privileged action ({action_description})? [y/N]: "
        )
        sys.stdout.flush()
        try:
            answer = sys.stdin.readline()
        except BaseException as exc:
            sys.stderr.write("\nAborted.\n")
            sys.stderr.flush()
            raise SystemExit(2) from exc
        normalized = answer.strip().lower()
        allowed = {"y", "yes", *(tok.strip().lower() for tok in accepted_tokens if tok and tok.strip())}
        if not normalized or normalized not in allowed:
            sys.stderr.write("Aborted by user.\n")
            sys.stderr.flush()
            raise SystemExit(2)


def _get_typer_app() -> Any:
    global _CACHED_APP
    if _CACHED_APP is not None:
        return _CACHED_APP

    import click
    import typer

    globals()["click"] = click
    globals()["typer"] = typer

    typer_app = typer.Typer(
        name="diodemcp",
        help="DiodeMCP local taint-tracking firewall and Jev cross-encoder runtime for MCP & coding agents.",
        add_completion=False,
        no_args_is_help=True,
    )
    model_app = typer.Typer(
        name="model",
        help="Verify, install, or switch DiodeModelBundle v1 ONNX bundles.",
        no_args_is_help=True,
    )
    typer_app.add_typer(model_app, name="model")

    @typer_app.command("init")
    def init_cmd(
        model: str = typer.Option(
            "base",
            "--model",
            "-m",
            help="Model bundle alias (base|large|nano) or Hugging Face repo ID",
        ),
        tier0_only: bool = typer.Option(
            False,
            "--tier0-only",
            help="Use deterministic Tier-0 floor only without loading an ONNX model",
        ),
        bundle_dir: Optional[Path] = typer.Option(
            None,
            "--bundle-dir",
            help="Local DiodeModelBundle v1 directory to validate and activate upfront",
        ),
        bundle_url: Optional[str] = typer.Option(
            None,
            "--bundle-url",
            help="HTTPS URL or repo spec to download DiodeModelBundle v1 from",
        ),
        sha256: Optional[str] = typer.Option(
            None,
            "--sha256",
            help="Optional expected SHA256 hex digest for model_int8.onnx or manifest.json",
        ),
        target: str = typer.Option(
            "all",
            "--target",
            "-t",
            help="IDE target(s) to wrap: claude, cursor, agy, or all",
        ),
        include_workspace: bool = typer.Option(
            False,
            "--include-workspace",
            help="Also wrap git-tracked workspace .mcp.json (off by default per I15)",
        ),
        state_dir: Optional[Path] = typer.Option(
            None,
            "--state-dir",
            help="DiodeMCP private state directory",
        ),
        home_dir: Optional[Path] = typer.Option(
            None,
            "--home-dir",
            help="Home directory override for IDE config discovery",
        ),
        workspace_root: Optional[Path] = typer.Option(
            None,
            "--workspace-root",
            help="Workspace root directory",
        ),
        no_daemon: bool = typer.Option(
            False,
            "--no-daemon",
            help="Skip background daemon autospawn/ping after configuring",
        ),
    ) -> None:
        """Initialize DiodeMCP credentials, verify/install model bundle upfront, and wrap IDE configs."""
        from diodemcp.bundle import (
            BundleValidationError,
            compute_file_sha256,
            validate_bundle_dir,
        )
        from diodemcp.config_patcher import PROTECTED_OAUTH_WARNING, wrap_ide_configs
        from diodemcp.engine import EngineClient, write_active_bundle_config
        from diodemcp.hub import BundleDownloadError, download_and_verify_bundle
        from diodemcp.transport import (
            default_transport,
            ensure_admin_secret,
            ensure_private_dir,
        )

        resolved_home = _resolve_home_dir(home_dir)
        resolved_state = _resolve_state_dir(state_dir, home_dir=home_dir)
        resolved_ws = (
            Path(workspace_root).expanduser().resolve()
            if workspace_root is not None
            else Path.cwd().resolve()
        )

        try:
            ensure_private_dir(resolved_state)
            cfg = default_transport(resolved_state)
            ensure_admin_secret(cfg)

            active_bundle_path: Path | None = None
            if tier0_only:
                write_active_bundle_config(
                    resolved_state, bundle_dir=None, tier0_only=True
                )
                _sync_config_toml_tier0_flag(resolved_state, tier0_only=True)
            elif bundle_dir is not None:
                resolved_bundle = Path(bundle_dir).expanduser().resolve()
                manifest = validate_bundle_dir(resolved_bundle, verify_sha256=True)
                if sha256 is not None:
                    expected_hex = sha256.strip().lower()
                    manifest_sha = compute_file_sha256(
                        resolved_bundle / "manifest.json"
                    )
                    if expected_hex not in (manifest.sha256_model_onnx, manifest_sha):
                        raise BundleValidationError(
                            f"Bundle SHA256 mismatch: expected {expected_hex}, got model={manifest.sha256_model_onnx}"
                        )
                write_active_bundle_config(
                    resolved_state, bundle_dir=resolved_bundle, tier0_only=False
                )
                _sync_config_toml_tier0_flag(resolved_state, tier0_only=False)
                active_bundle_path = resolved_bundle
            elif bundle_url is not None:
                downloaded = download_and_verify_bundle(
                    bundle_url,
                    cache_dir=resolved_state / "models",
                    progress=False,
                )
                manifest = validate_bundle_dir(downloaded, verify_sha256=True)
                if sha256 is not None:
                    expected_hex = sha256.strip().lower()
                    manifest_sha = compute_file_sha256(downloaded / "manifest.json")
                    if expected_hex not in (manifest.sha256_model_onnx, manifest_sha):
                        raise BundleValidationError(
                            f"Downloaded bundle SHA256 mismatch: expected {expected_hex}"
                        )
                write_active_bundle_config(
                    resolved_state, bundle_dir=downloaded, tier0_only=False
                )
                _sync_config_toml_tier0_flag(resolved_state, tier0_only=False)
                active_bundle_path = downloaded
            else:
                downloaded = download_and_verify_bundle(
                    model,
                    cache_dir=resolved_state / "models",
                    progress=False,
                )
                write_active_bundle_config(
                    resolved_state, bundle_dir=downloaded, tier0_only=False
                )
                _sync_config_toml_tier0_flag(resolved_state, tier0_only=False)
                active_bundle_path = downloaded

            patch_report = wrap_ide_configs(
                target=target,
                state_dir=resolved_state,
                home_dir=resolved_home,
                workspace_root=resolved_ws,
                include_workspace=include_workspace,
            )

            _restart_daemon_if_running(resolved_state)
            daemon_status: dict[str, Any] | None = None
            if not no_daemon:
                client = EngineClient(cfg, bundle_dir=active_bundle_path)
                daemon_status = client.status()

        except (BundleValidationError, BundleDownloadError, ValueError, OSError) as exc:
            sys.stderr.write(f"[diodemcp init] Error: {exc}\n")
            sys.stderr.flush()
            raise typer.Exit(code=1) from exc

        mode_str = (
            "tier0"
            if tier0_only
            else (
                daemon_status.get("predictor", f"bundle:{active_bundle_path}")
                if daemon_status
                else f"bundle:{active_bundle_path}"
            )
        )
        typer.echo(
            f"DiodeMCP initialized (state_dir={resolved_state}, predictor={mode_str})."
        )
        typer.echo(
            f"Wrapped {len(patch_report['wrapped_files'])} IDE config file(s) "
            f"and {len(patch_report['wrapped_servers'])} MCP server(s)."
        )
        for skipped_srv in patch_report["skipped_oauth_servers"]:
            typer.echo(f"WARNING [{skipped_srv}]: {PROTECTED_OAUTH_WARNING}")

    @typer_app.command("unwrap")
    def unwrap_cmd(
        target: str = typer.Option(
            "all",
            "--target",
            "-t",
            help="IDE target(s) to unwrap: claude, cursor, agy, or all",
        ),
        include_workspace: bool = typer.Option(
            False,
            "--include-workspace",
            help="Also unwrap workspace .mcp.json",
        ),
        state_dir: Optional[Path] = typer.Option(
            None,
            "--state-dir",
            help="DiodeMCP private state directory",
        ),
        home_dir: Optional[Path] = typer.Option(
            None,
            "--home-dir",
            help="Home directory override",
        ),
        workspace_root: Optional[Path] = typer.Option(
            None,
            "--workspace-root",
            help="Workspace root directory",
        ),
        yes: bool = typer.Option(
            False,
            "--yes",
            "-y",
            help="Skip interactive confirmation prompt on a verified human TTY",
        ),
    ) -> None:
        """Losslessly unwrap DiodeMCP shims and hooks (TTY-guarded per `C4`)."""
        require_interactive_tty(
            yes=yes,
            action_description=f"unwrap DiodeMCP IDE configs for target={target}",
        )

        from diodemcp.config_patcher import unwrap_ide_configs
        from diodemcp.transport import (
            TransportError,
            default_transport,
            ensure_admin_secret,
            read_admin_secret,
        )

        resolved_home = _resolve_home_dir(home_dir)
        resolved_state = _resolve_state_dir(state_dir, home_dir=home_dir)
        resolved_ws = (
            Path(workspace_root).expanduser().resolve()
            if workspace_root is not None
            else Path.cwd().resolve()
        )

        try:
            cfg = default_transport(resolved_state)
            if cfg.admin_secret_file.exists():
                read_admin_secret(cfg)
            else:
                ensure_admin_secret(cfg)

            report = unwrap_ide_configs(
                target=target,
                state_dir=resolved_state,
                home_dir=resolved_home,
                workspace_root=resolved_ws,
                include_workspace=include_workspace,
            )
        except (TransportError, ValueError, OSError) as exc:
            sys.stderr.write(f"[diodemcp unwrap] Error: {exc}\n")
            sys.stderr.flush()
            raise typer.Exit(code=1) from exc

        typer.echo(
            f"Unwrapped {len(report['unwrapped_files'])} config file(s) "
            f"and restored {len(report['restored_servers'])} MCP server(s)."
        )

    @model_app.command("verify")
    def model_verify_cmd(
        bundle_dir: Path = typer.Argument(
            ..., help="Path to DiodeModelBundle v1 directory"
        ),
    ) -> None:
        """Validate a DiodeModelBundle v1 directory, artifact SHA256s, and 17 special tokens."""
        from diodemcp.bundle import BundleValidationError, validate_bundle_dir

        try:
            manifest = validate_bundle_dir(
                bundle_dir.expanduser().resolve(), verify_sha256=True
            )
        except (BundleValidationError, ValueError, OSError) as exc:
            sys.stderr.write(f"[diodemcp model verify] Invalid bundle: {exc}\n")
            sys.stderr.flush()
            raise typer.Exit(code=1) from exc

        typer.echo(
            json.dumps(
                {
                    "ok": True,
                    "model_id": manifest.model_id,
                    "bundle_version": manifest.bundle_version,
                    "ece_val": manifest.ece_val,
                    "thresholds": manifest.thresholds,
                }
            )
        )

    @model_app.command("use")
    def model_use_cmd(
        bundle_dir: Path = typer.Argument(
            ..., help="Path to DiodeModelBundle v1 directory"
        ),
        state_dir: Optional[Path] = typer.Option(None, "--state-dir"),
    ) -> None:
        """Validate and activate a local DiodeModelBundle v1 directory."""
        from diodemcp.bundle import BundleValidationError, validate_bundle_dir
        from diodemcp.engine import write_active_bundle_config

        resolved_state = _resolve_state_dir(state_dir)
        resolved_bundle = bundle_dir.expanduser().resolve()
        try:
            manifest = validate_bundle_dir(resolved_bundle, verify_sha256=True)
            write_active_bundle_config(
                resolved_state, bundle_dir=resolved_bundle, tier0_only=False
            )
            _sync_config_toml_tier0_flag(resolved_state, tier0_only=False)
            _restart_daemon_if_running(resolved_state)
        except (BundleValidationError, ValueError, OSError) as exc:
            sys.stderr.write(f"[diodemcp model use] Error: {exc}\n")
            sys.stderr.flush()
            raise typer.Exit(code=1) from exc

        typer.echo(f"Activated bundle '{manifest.model_id}' ({resolved_bundle}).")

    @model_app.command("install")
    def model_install_cmd(
        url: Optional[str] = typer.Option(
            None,
            "--url",
            help="HTTPS URL or Hugging Face repo / alias to install",
        ),
        model: str = typer.Option(
            "base",
            "--model",
            "-m",
            help="Model alias (base|large|nano) if --url is omitted",
        ),
        sha256: Optional[str] = typer.Option(
            None,
            "--sha256",
            help="Expected SHA256 hex digest of model_int8.onnx or manifest.json",
        ),
        name: Optional[str] = typer.Option(
            None,
            "--name",
            help="Optional local bundle directory name under <state_dir>/models/",
        ),
        state_dir: Optional[Path] = typer.Option(None, "--state-dir"),
    ) -> None:
        """Download, verify SHA256 digests, and activate a DiodeModelBundle v1 bundle."""
        from diodemcp.bundle import (
            BundleValidationError,
            compute_file_sha256,
            validate_bundle_dir,
        )
        from diodemcp.engine import write_active_bundle_config
        from diodemcp.hub import BundleDownloadError, download_and_verify_bundle

        resolved_state = _resolve_state_dir(state_dir)
        spec = url if (url is not None and url.strip()) else model
        models_root = resolved_state / "models"

        try:
            installed_dir = download_and_verify_bundle(
                spec,
                cache_dir=models_root,
                progress=False,
            )
            manifest = validate_bundle_dir(installed_dir, verify_sha256=True)
            if sha256 is not None:
                expected_hex = sha256.strip().lower()
                manifest_sha = compute_file_sha256(installed_dir / "manifest.json")
                if expected_hex not in (manifest.sha256_model_onnx, manifest_sha):
                    raise BundleValidationError(
                        f"Installed bundle SHA256 mismatch: expected {expected_hex}, "
                        f"got model_int8.onnx={manifest.sha256_model_onnx}"
                    )
            if name is not None and name.strip():
                safe_name = os.path.basename(name.strip())
                named_dir = models_root / safe_name
                if named_dir.resolve() != installed_dir.resolve():
                    if named_dir.exists():
                        shutil.rmtree(named_dir)
                    shutil.copytree(installed_dir, named_dir)
                    installed_dir = named_dir

            write_active_bundle_config(
                resolved_state, bundle_dir=installed_dir, tier0_only=False
            )
            _sync_config_toml_tier0_flag(resolved_state, tier0_only=False)
            _restart_daemon_if_running(resolved_state)
        except (
            BundleValidationError,
            BundleDownloadError,
            ValueError,
            OSError,
        ) as exc:
            sys.stderr.write(f"[diodemcp model install] Error: {exc}\n")
            sys.stderr.flush()
            raise typer.Exit(code=1) from exc

        typer.echo(
            f"Installed and activated model '{manifest.model_id}' at {installed_dir}."
        )

    @typer_app.command("status")
    def status_cmd(
        state_dir: Optional[Path] = typer.Option(None, "--state-dir"),
        as_json: bool = typer.Option(
            False, "--json", help="Emit machine-readable JSON status"
        ),
    ) -> None:
        """Query DiodeMCP engine daemon status and report wrapped / skipped OAuth servers."""
        from diodemcp.config_patcher import PROTECTED_OAUTH_WARNING, read_wrap_manifest
        from diodemcp.engine import EngineClient
        from diodemcp.transport import TransportError, default_transport

        resolved_state = _resolve_state_dir(state_dir)
        try:
            cfg = default_transport(resolved_state)
            client = EngineClient(cfg)
            st = client.status()
            wrap_info = read_wrap_manifest(resolved_state)
        except (TransportError, ValueError, OSError) as exc:
            sys.stderr.write(f"[diodemcp status] Error: {exc}\n")
            sys.stderr.flush()
            raise typer.Exit(code=1) from exc

        skipped = wrap_info.get("skipped_oauth_servers", [])
        payload = {
            **st,
            "state_dir": str(resolved_state),
            "skipped_oauth_servers": skipped,
            "oauth_protection_note": PROTECTED_OAUTH_WARNING if skipped else None,
        }
        if as_json:
            typer.echo(json.dumps(payload))
            return

        typer.echo(
            f"DiodeMCP Engine: ok={st.get('ok')} pid={st.get('pid')} "
            f"mode={st.get('mode')} predictor={st.get('predictor')} "
            f"workspaces={st.get('active_workspaces')} incidents={st.get('incident_count')}"
        )
        for srv in skipped:
            typer.echo(f"WARNING [{srv}]: {PROTECTED_OAUTH_WARNING}")

    @typer_app.command("audit")
    def audit_cmd(
        limit: int = typer.Option(
            50, "--limit", "-n", help="Maximum incidents to return"
        ),
        workspace: Optional[Path] = typer.Option(
            None, "--workspace", "-w", help="Filter incidents by workspace root"
        ),
        state_dir: Optional[Path] = typer.Option(None, "--state-dir"),
        as_json: bool = typer.Option(
            False, "--json", help="Emit JSON array of incidents"
        ),
    ) -> None:
        """List recent redacted security incidents (`C4`: never exposes raw secrets or tokens)."""
        from diodemcp.engine import EngineClient
        from diodemcp.transport import (
            TransportError,
            default_transport,
            ensure_admin_secret,
            read_admin_secret,
        )

        resolved_state = _resolve_state_dir(state_dir)
        try:
            cfg = default_transport(resolved_state)
            admin_secret = (
                read_admin_secret(cfg)
                if cfg.admin_secret_file.exists()
                else ensure_admin_secret(cfg)
            )
            client = EngineClient(cfg)
            incidents = client.list_incidents(
                limit=limit,
                workspace_root=workspace,
                admin_secret=admin_secret,
            )
        except (TransportError, ValueError, OSError) as exc:
            sys.stderr.write(f"[diodemcp audit] Error: {exc}\n")
            sys.stderr.flush()
            raise typer.Exit(code=1) from exc

        if as_json:
            typer.echo(json.dumps({"incidents": incidents}))
            return

        if not incidents:
            typer.echo("No DiodeMCP incidents recorded.")
            return

        for inc in incidents:
            typer.echo(
                f"[{inc.get('incident_id')}] band={inc.get('band')} "
                f"action={inc.get('action_emitted')} tool={inc.get('server')}.{inc.get('tool')} "
                f"p_max={inc.get('p_max')} consumed={inc.get('consumed')} — {inc.get('reason')}"
            )

    @typer_app.command("allow")
    def allow_cmd(
        incident_id: str = typer.Argument(
            ..., help="Incident ID (`inc_...`) from `diodemcp audit`"
        ),
        ttl: float = typer.Option(
            120.0,
            "--ttl",
            help="Single-use approval TTL in seconds (max 600s)",
        ),
        yes: bool = typer.Option(
            False,
            "--yes",
            "-y",
            help="Skip interactive confirmation prompt on a verified human TTY",
        ),
        state_dir: Optional[Path] = typer.Option(None, "--state-dir"),
    ) -> None:
        """Grant a single-use, time-bounded approval for an exact blocked call (TTY-guarded per `C4`)."""
        _ensure_interactive_tty()

        from diodemcp.engine import EngineClient
        from diodemcp.transport import (
            TransportError,
            default_transport,
            read_admin_secret,
        )

        resolved_state = _resolve_state_dir(state_dir)
        try:
            cfg = default_transport(resolved_state)
            admin_secret = read_admin_secret(cfg)
            client = EngineClient(cfg)

            matched_inc: dict[str, Any] | None = None
            try:
                incidents = client.list_incidents(
                    limit=200,
                    admin_secret=admin_secret,
                )
                for inc in incidents:
                    if isinstance(inc, dict) and inc.get("incident_id") == incident_id:
                        matched_inc = inc
                        break
            except (TransportError, ValueError, OSError):
                matched_inc = None

            if matched_inc is not None:
                srv = matched_inc.get("server")
                tl = matched_inc.get("tool")
                band = matched_inc.get("band")
                reason = matched_inc.get("reason")
                typer.echo(
                    f"Incident Summary\n"
                    f"  incident_id={incident_id}\n"
                    f"  server={srv}\n"
                    f"  tool={tl}\n"
                    f"  band={band}\n"
                    f"  reason={reason}"
                )
                action_desc = (
                    f"approve single-use replay for {srv}.{tl} "
                    f"(incident {incident_id}, ttl={ttl}s)"
                )
            else:
                typer.echo(f"Incident Summary\n  incident_id={incident_id}")
                action_desc = f"approve single-use replay for incident {incident_id} (ttl={ttl}s)"

            short_prefix = incident_id[:8] if len(incident_id) >= 8 else incident_id
            require_interactive_tty(
                yes=yes,
                action_description=action_desc,
                accepted_tokens=(incident_id, short_prefix),
            )

            res = client.approve_incident(
                incident_id,
                ttl_seconds=ttl,
                admin_secret=admin_secret,
            )
        except (TransportError, ValueError, OSError) as exc:
            sys.stderr.write(f"[diodemcp allow] Error: {exc}\n")
            sys.stderr.flush()
            raise typer.Exit(code=1) from exc

        typer.echo(
            f"Approved incident {res.get('incident_id')} for single-use replay "
            f"({res.get('server')}.{res.get('tool')}, expires_at={res.get('expires_at')})."
        )

    @typer_app.command("shutdown")
    def shutdown_cmd(
        yes: bool = typer.Option(
            False,
            "--yes",
            "-y",
            help="Skip interactive confirmation prompt on a verified human TTY",
        ),
        state_dir: Optional[Path] = typer.Option(None, "--state-dir"),
    ) -> None:
        """Stop the background DiodeMCP engine daemon (TTY-guarded per `C4`)."""
        require_interactive_tty(
            yes=yes,
            action_description="shutdown DiodeMCP background engine daemon",
        )

        from diodemcp.engine import EngineClient
        from diodemcp.transport import (
            TransportError,
            default_transport,
            ensure_admin_secret,
            read_admin_secret,
        )

        resolved_state = _resolve_state_dir(state_dir)
        try:
            cfg = default_transport(resolved_state)
            admin_secret = (
                read_admin_secret(cfg)
                if cfg.admin_secret_file.exists()
                else ensure_admin_secret(cfg)
            )
            client = EngineClient(cfg, auto_spawn=False)
            try:
                client.shutdown(admin_secret=admin_secret)
            except TransportError:
                pass
        except (ValueError, OSError) as exc:
            sys.stderr.write(f"[diodemcp shutdown] Error: {exc}\n")
            sys.stderr.flush()
            raise typer.Exit(code=1) from exc

        typer.echo("DiodeMCP engine daemon stopped.")

    @typer_app.command(
        "shim",
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    )
    def shim_cmd(ctx: typer.Context) -> None:
        """Run MCP stdio or Streamable-HTTP bridge shim."""
        from diodemcp.shim import _main_shim_cli

        rc = _main_shim_cli(list(ctx.args))
        raise typer.Exit(code=rc)

    @typer_app.command(
        "hook",
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    )
    def hook_cmd(ctx: typer.Context) -> None:
        """Run fail-closed native IDE hook (`claude`, `cursor`, `agy`)."""
        from diodemcp.hooks import main_hook_cli

        rc = main_hook_cli(list(ctx.args))
        raise typer.Exit(code=rc)

    _CACHED_APP = typer_app
    return typer_app


def __getattr__(name: str) -> Any:
    if name == "app":
        return _get_typer_app()
    raise AttributeError(f"module 'diodemcp.cli' has no attribute {name!r}")


def main(argv: Sequence[str] | None = None) -> int:
    """Main CLI entrypoint with `<20ms` fast-path for `diodemcp hook` and `diodemcp shim`."""
    eff_argv = list(sys.argv[1:] if argv is None else argv)

    # Fast-path `diodemcp hook ...` before importing Typer/Click (<20ms requirement)
    if (
        eff_argv
        and eff_argv[0] == "hook"
        and "--help" not in eff_argv
        and "-h" not in eff_argv
    ):
        from diodemcp.hooks import main_hook_cli

        return int(main_hook_cli(eff_argv[1:]))

    if (
        eff_argv
        and eff_argv[0] == "shim"
        and "--help" not in eff_argv
        and "-h" not in eff_argv
    ):
        from diodemcp.shim import _main_shim_cli

        return int(_main_shim_cli(eff_argv[1:]))

    import click
    import typer

    typer_app = _get_typer_app()
    try:
        res = typer_app(args=eff_argv, prog_name="diodemcp", standalone_mode=False)
        return int(res) if isinstance(res, int) else 0
    except (typer.Exit, click.exceptions.Exit) as exc:
        return int(getattr(exc, "exit_code", getattr(exc, "code", 1)))
    except click.exceptions.ClickException as exc:
        exc.show()
        return int(exc.exit_code)
    except SystemExit as exc:
        return (
            int(exc.code)
            if isinstance(exc.code, int)
            else (0 if exc.code is None else 1)
        )


if __name__ == "__main__":
    raise SystemExit(main())
