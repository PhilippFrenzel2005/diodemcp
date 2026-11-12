"""Cross-platform Unix/TCP daemon transport, mutual-HMAC wire auth, and lock-safe autospawn (`C4`, `I13`)."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import socket
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterator, Literal

from filelock import FileLock

from diodemcp.constants import DEFAULT_IDLE_TIMEOUT_SECONDS, DEFAULT_STATE_DIR

MAX_IPC_LINE_BYTES: int = 16 * 1024 * 1024  # 16 MiB frame ceiling on daemon IPC
_SUN_PATH_SAFE_BYTES: int = 96

PRIVILEGED_METHODS: frozenset[str] = frozenset(
    {
        "approve_incident",
        "approve_allow_token",
        "trust_tool_shape",
        "shutdown",
    }
)


class TransportError(RuntimeError):
    """Base error for DiodeMCP daemon IPC transport failures."""


class TransportAuthError(TransportError):
    """Raised on missing/invalid `client_token` or failed mutual-HMAC verification (`I13`)."""


class TransportPermissionError(TransportError):
    """Raised when a privileged RPC lacks a valid `admin_secret` (`C4`)."""


class _TransientSocketError(TransportError):
    """Internal marker for stale/closing listener races eligible for lock-guarded retry."""


def _current_uid_tag() -> str:
    if hasattr(os, "getuid"):
        return str(os.getuid())
    return os.environ.get("USERNAME") or os.environ.get("USER") or "0"


def ensure_private_dir(path: Path) -> Path:
    """Create `path` with `0700` permissions, rejecting symlinks and foreign UIDs on POSIX."""
    expanded = Path(path).expanduser()
    if expanded.is_symlink():
        raise TransportAuthError(f"Refusing symlinked DiodeMCP directory: {expanded}")
    expanded.mkdir(parents=True, exist_ok=True, mode=0o700)
    if expanded.is_symlink():
        raise TransportAuthError(f"Refusing symlinked DiodeMCP directory: {expanded}")
    if os.name == "posix":
        st = expanded.lstat()
        if not stat.S_ISDIR(st.st_mode):
            raise TransportAuthError(f"Expected directory at {expanded}")
        if hasattr(os, "getuid") and st.st_uid != os.getuid():
            raise TransportAuthError(
                f"Directory {expanded} is owned by uid {st.st_uid}, expected {os.getuid()}"
            )
        if stat.S_IMODE(st.st_mode) != 0o700:
            os.chmod(expanded, 0o700)
    return expanded


def atomic_write_private_bytes(target_path: Path, data: bytes) -> None:
    """Atomically write `target_path` at mode `0600` inside a `0700` directory."""
    parent = ensure_private_dir(target_path.parent)
    if target_path.is_symlink():
        raise TransportAuthError(f"Refusing to overwrite symlink: {target_path}")

    tmp_name = f".{target_path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    tmp_path = parent / tmp_name
    fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(data)
        written = 0
        while written < len(view):
            written += os.write(fd, view[written:])
        os.fsync(fd)
        if os.name == "posix" and hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
    except BaseException:
        os.close(fd)
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    else:
        os.close(fd)

    try:
        if os.name == "posix":
            os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, target_path)
        if os.name == "posix":
            os.chmod(target_path, 0o600)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def atomic_write_private_json(target_path: Path, payload: dict[str, Any]) -> None:
    """Atomically serialize `payload` as UTF-8 JSON with `0600` permissions."""
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    atomic_write_private_bytes(target_path, serialized)


def _short_user_socket_path(state_dir: Path) -> Path:
    real_state = os.path.realpath(os.fspath(state_dir))
    digest = hashlib.sha256(real_state.encode("utf-8")).hexdigest()[:16]
    short_root = Path(tempfile.gettempdir()) / f"dmcp-{_current_uid_tag()}"
    return short_root / f"{digest}.sock"


@dataclass(frozen=True)
class TransportConfig:
    """Cross-platform IPC endpoint and split-credential paths (`R4`, `C4`, `I13`)."""

    mode: Literal["unix", "tcp"]
    socket_path: Path
    addr_file: Path
    admin_secret_file: Path
    state_dir: Path = field(default_factory=Path)
    lock_file: Path = field(default_factory=Path)

    def __post_init__(self) -> None:
        if self.mode not in ("unix", "tcp"):
            raise ValueError(f"TransportConfig.mode must be 'unix' or 'tcp', got {self.mode!r}")

        sock = Path(self.socket_path).expanduser()
        addr = Path(self.addr_file).expanduser()
        admin = Path(self.admin_secret_file).expanduser()

        raw_state = self.state_dir
        if raw_state is None or raw_state == Path():
            resolved_state = addr.parent
        else:
            resolved_state = Path(raw_state).expanduser()

        raw_lock = self.lock_file
        if raw_lock is None or raw_lock == Path():
            resolved_lock = resolved_state / "daemon.lock"
        else:
            resolved_lock = Path(raw_lock).expanduser()

        object.__setattr__(self, "socket_path", sock)
        object.__setattr__(self, "addr_file", addr)
        object.__setattr__(self, "admin_secret_file", admin)
        object.__setattr__(self, "state_dir", resolved_state)
        object.__setattr__(self, "lock_file", resolved_lock)


def default_transport(
    state_dir: Path | str | None = None,
    *,
    mode: Literal["unix", "tcp"] | None = None,
) -> TransportConfig:
    """Construct platform-appropriate `TransportConfig` (`unix` on POSIX, `tcp` on `win32`)."""
    resolved_state = (
        Path(state_dir).expanduser() if state_dir is not None else DEFAULT_STATE_DIR
    )
    selected_mode: Literal["unix", "tcp"] = (
        mode if mode is not None else ("tcp" if sys.platform == "win32" else "unix")
    )
    if selected_mode not in ("unix", "tcp"):
        raise ValueError(f"Unsupported transport mode: {selected_mode!r}")

    candidate_sock = resolved_state / "engine.sock"
    if selected_mode == "unix" and len(os.fsencode(candidate_sock)) > _SUN_PATH_SAFE_BYTES:
        socket_path = _short_user_socket_path(resolved_state)
    else:
        socket_path = candidate_sock

    return TransportConfig(
        mode=selected_mode,
        state_dir=resolved_state,
        socket_path=socket_path,
        addr_file=resolved_state / "addr.json",
        admin_secret_file=resolved_state / "admin.secret",
        lock_file=resolved_state / "daemon.lock",
    )


@contextmanager
def unix_path_handle(socket_path: Path) -> Iterator[str]:
    """Yield an `AF_UNIX`-bindable/connectable path, handling >104-byte paths on Linux via `/proc/self/fd`."""
    raw = os.fspath(socket_path)
    if len(os.fsencode(raw)) <= _SUN_PATH_SAFE_BYTES:
        yield raw
        return

    if sys.platform.startswith("linux") and hasattr(os, "O_DIRECTORY"):
        dir_fd = os.open(os.fspath(socket_path.parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            short_ref = f"/proc/self/fd/{dir_fd}/{socket_path.name}"
            if len(os.fsencode(short_ref)) <= 104:
                yield short_ref
                return
        finally:
            os.close(dir_fd)

    yield raw


def _verify_private_regular_file(path: Path) -> None:
    if path.is_symlink():
        raise TransportAuthError(f"Refusing symlinked credential file: {path}")
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode):
        raise TransportAuthError(f"Expected regular credential file at {path}")
    if os.name == "posix":
        if hasattr(os, "getuid") and st.st_uid != os.getuid():
            raise TransportAuthError(
                f"Credential file {path} owned by uid {st.st_uid}, expected {os.getuid()}"
            )
        if (stat.S_IMODE(st.st_mode) & 0o077) != 0:
            raise TransportAuthError(
                f"Credential file {path} has insecure permissions {oct(stat.S_IMODE(st.st_mode))}"
            )


def ensure_admin_secret(config: TransportConfig) -> str:
    """Ensure `admin.secret` exists at `0600` without ever overwriting an existing valid secret."""
    ensure_private_dir(config.state_dir)
    ensure_private_dir(config.admin_secret_file.parent)
    secret_path = config.admin_secret_file

    if secret_path.is_symlink():
        raise TransportAuthError(f"Refusing symlinked admin.secret: {secret_path}")

    if secret_path.exists():
        _verify_private_regular_file(secret_path)
        existing = secret_path.read_text(encoding="utf-8").strip()
        if len(existing) >= 32:
            return existing

    fresh = secrets.token_hex(32)
    try:
        fd = os.open(str(secret_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        _verify_private_regular_file(secret_path)
        existing = secret_path.read_text(encoding="utf-8").strip()
        if len(existing) >= 32:
            return existing
        atomic_write_private_bytes(secret_path, f"{fresh}\n".encode("utf-8"))
        return fresh
    else:
        try:
            os.write(fd, f"{fresh}\n".encode("utf-8"))
            os.fsync(fd)
            if os.name == "posix" and hasattr(os, "fchmod"):
                os.fchmod(fd, 0o600)
        finally:
            os.close(fd)
        if os.name == "posix":
            os.chmod(secret_path, 0o600)
        return fresh


def read_admin_secret(config: TransportConfig) -> str:
    """Read and validate the `0600` privileged `admin.secret` for human CLI/test operations."""
    if not config.admin_secret_file.exists():
        return ensure_admin_secret(config)
    _verify_private_regular_file(config.admin_secret_file)
    secret = config.admin_secret_file.read_text(encoding="utf-8").strip()
    if len(secret) < 16:
        raise TransportAuthError(f"Invalid or empty admin.secret at {config.admin_secret_file}")
    return secret


def read_addr_info(config: TransportConfig) -> dict[str, Any] | None:
    """Read and validate `addr.json` (`0600`), returning `None` if absent."""
    addr_path = config.addr_file
    if not addr_path.exists():
        return None
    _verify_private_regular_file(addr_path)
    try:
        raw = addr_path.read_text(encoding="utf-8")
        if not raw.strip():
            return None
        data = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None

    if not isinstance(data, dict):
        return None
    mode = data.get("mode")
    token = data.get("client_token")
    if mode != config.mode or not isinstance(token, str) or len(token) < 16:
        raise TransportAuthError("Malformed or mode-mismatched addr.json")

    if mode == "tcp":
        host = data.get("host")
        port = data.get("port")
        if host != "127.0.0.1":
            raise TransportAuthError(
                f"Refusing non-loopback TCP daemon host in addr.json: {host!r}"
            )
        if isinstance(port, bool) or not isinstance(port, int) or not (1 <= port <= 65535):
            raise TransportAuthError(f"Invalid TCP port in addr.json: {port!r}")
    else:
        sock_p = data.get("socket_path")
        if not isinstance(sock_p, str) or not sock_p.strip():
            raise TransportAuthError("Missing socket_path in Unix addr.json")

    return data


def canonical_json_sha256(value: Any) -> str:
    """Compute deterministic SHA-256 over canonical JSON representation."""
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def compute_client_hmac(
    client_token: str,
    client_nonce: str,
    request_id: str,
    method: str,
    params_sha256: str,
) -> str:
    msg = f"client:v1:{client_nonce}:{request_id}:{method}:{params_sha256}".encode("utf-8")
    return hmac.new(client_token.encode("utf-8"), msg, hashlib.sha256).hexdigest()


def compute_admin_hmac(
    admin_secret: str,
    client_nonce: str,
    request_id: str,
    method: str,
    params_sha256: str,
) -> str:
    msg = f"admin:v1:{client_nonce}:{request_id}:{method}:{params_sha256}".encode("utf-8")
    return hmac.new(admin_secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()


def compute_server_hmac(
    client_token: str,
    client_nonce: str,
    request_id: str,
    ok: bool,
    payload_sha256: str,
) -> str:
    flag = "1" if ok else "0"
    msg = f"server:v1:{client_nonce}:{request_id}:{flag}:{payload_sha256}".encode("utf-8")
    return hmac.new(client_token.encode("utf-8"), msg, hashlib.sha256).hexdigest()


def build_signed_request_frame(
    *,
    method: str,
    params: dict[str, Any],
    client_token: str,
    admin_secret: str | None = None,
) -> tuple[bytes, str, str]:
    """Build `\n`-delimited JSON request frame without exposing raw secrets on the wire (`I13`)."""
    request_id = f"req_{secrets.token_hex(16)}"
    client_nonce = secrets.token_hex(16)
    params_sha = canonical_json_sha256(params)

    envelope: dict[str, Any] = {
        "request_id": request_id,
        "method": method,
        "params": params,
        "client_nonce": client_nonce,
        "client_hmac": compute_client_hmac(
            client_token, client_nonce, request_id, method, params_sha
        ),
    }
    if admin_secret is not None:
        envelope["admin_hmac"] = compute_admin_hmac(
            admin_secret, client_nonce, request_id, method, params_sha
        )

    raw_line = (json.dumps(envelope, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    if len(raw_line) > MAX_IPC_LINE_BYTES:
        raise TransportError(f"IPC request exceeds {MAX_IPC_LINE_BYTES} bytes")
    return raw_line, request_id, client_nonce


def verify_and_unwrap_response(
    raw_line: bytes,
    *,
    expected_request_id: str,
    client_nonce: str,
    client_token: str,
) -> dict[str, Any]:
    """Verify server mutual-HMAC before trusting any response (`I13`)."""
    if not raw_line:
        raise _TransientSocketError("Empty EOF from daemon socket")

    try:
        resp = json.loads(raw_line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TransportAuthError(f"Malformed JSON response from daemon: {exc}") from exc

    if not isinstance(resp, dict):
        raise TransportAuthError("Daemon response root must be a JSON object")

    if resp.get("code") == "UNAUTHORIZED":
        raise TransportAuthError(str(resp.get("error") or "Unauthorized client_token"))

    resp_req_id = resp.get("request_id")
    server_mac = resp.get("server_hmac")
    ok_val = resp.get("ok")
    if (
        resp_req_id != expected_request_id
        or not isinstance(server_mac, str)
        or not isinstance(ok_val, bool)
    ):
        raise TransportAuthError(
            "Daemon response missing valid request_id or mutual server_hmac"
        )

    if ok_val:
        signed_payload: Any = resp.get("result")
    else:
        signed_payload = {
            "code": resp.get("code", "ERROR"),
            "error": resp.get("error", ""),
        }

    expected_mac = compute_server_hmac(
        client_token,
        client_nonce,
        expected_request_id,
        ok_val,
        canonical_json_sha256(signed_payload),
    )
    if not hmac.compare_digest(server_mac, expected_mac):
        raise TransportAuthError(
            "Mutual HMAC verification failed: refusing untrusted daemon response"
        )

    if not ok_val:
        code = resp.get("code")
        err_msg = str(resp.get("error") or "Daemon RPC error")
        if code == "FORBIDDEN":
            raise TransportPermissionError(err_msg)
        raise TransportError(err_msg)

    result = resp.get("result")
    if not isinstance(result, dict):
        raise TransportError("Daemon RPC result must be a dict")
    return result


def _read_bounded_line_sync(sock: socket.socket) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        buf = sock.recv(65536)
        if not buf:
            break
        nl_idx = buf.find(b"\n")
        if nl_idx != -1:
            part = buf[: nl_idx + 1]
            total += len(part)
            if total > MAX_IPC_LINE_BYTES:
                raise TransportError("Daemon IPC response exceeded max line size")
            chunks.append(part)
            break
        total += len(buf)
        if total > MAX_IPC_LINE_BYTES:
            raise TransportError("Daemon IPC response exceeded max line size")
        chunks.append(buf)
    return b"".join(chunks)


def _rpc_once_sync(
    addr: dict[str, Any],
    method: str,
    params: dict[str, Any],
    *,
    admin_secret: str | None = None,
    timeout_s: float = 15.0,
) -> dict[str, Any]:
    client_token = str(addr["client_token"])
    frame, req_id, nonce = build_signed_request_frame(
        method=method,
        params=params,
        client_token=client_token,
        admin_secret=admin_secret,
    )

    try:
        if addr["mode"] == "unix":
            if not hasattr(socket, "AF_UNIX"):
                raise TransportError("AF_UNIX sockets are not supported on this platform")
            sock_path = Path(str(addr["socket_path"]))
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.settimeout(timeout_s)
                with unix_path_handle(sock_path) as connect_target:
                    sock.connect(connect_target)
                sock.sendall(frame)
                raw_resp = _read_bounded_line_sync(sock)
            finally:
                sock.close()
        else:
            host = str(addr["host"])
            port = int(addr["port"])
            if host != "127.0.0.1":
                raise TransportAuthError(f"Refusing non-loopback TCP host: {host!r}")
            with socket.create_connection((host, port), timeout=timeout_s) as sock:
                sock.settimeout(timeout_s)
                sock.sendall(frame)
                raw_resp = _read_bounded_line_sync(sock)
    except (
        FileNotFoundError,
        ConnectionRefusedError,
        ConnectionResetError,
        BrokenPipeError,
        TimeoutError,
        socket.timeout,
    ) as exc:
        raise _TransientSocketError(str(exc)) from exc
    except OSError as exc:
        raise _TransientSocketError(str(exc)) from exc

    return verify_and_unwrap_response(
        raw_resp,
        expected_request_id=req_id,
        client_nonce=nonce,
        client_token=client_token,
    )


async def _rpc_once_async(
    addr: dict[str, Any],
    method: str,
    params: dict[str, Any],
    *,
    admin_secret: str | None = None,
    timeout_s: float = 15.0,
) -> dict[str, Any]:
    client_token = str(addr["client_token"])
    frame, req_id, nonce = build_signed_request_frame(
        method=method,
        params=params,
        client_token=client_token,
        admin_secret=admin_secret,
    )

    writer: asyncio.StreamWriter | None = None
    try:
        if addr["mode"] == "unix":
            sock_path = Path(str(addr["socket_path"]))
            with unix_path_handle(sock_path) as connect_target:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_unix_connection(
                        connect_target, limit=MAX_IPC_LINE_BYTES
                    ),
                    timeout=timeout_s,
                )
        else:
            host = str(addr["host"])
            port = int(addr["port"])
            if host != "127.0.0.1":
                raise TransportAuthError(f"Refusing non-loopback TCP host: {host!r}")
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port, limit=MAX_IPC_LINE_BYTES),
                timeout=timeout_s,
            )

        writer.write(frame)
        await asyncio.wait_for(writer.drain(), timeout=timeout_s)
        raw_resp = await asyncio.wait_for(reader.readline(), timeout=timeout_s)
    except asyncio.LimitOverrunError as exc:
        raise TransportError("Daemon IPC response exceeded max line size") from exc
    except (
        FileNotFoundError,
        ConnectionRefusedError,
        ConnectionResetError,
        BrokenPipeError,
        TimeoutError,
        asyncio.TimeoutError,
        OSError,
    ) as exc:
        raise _TransientSocketError(str(exc)) from exc
    finally:
        if writer is not None:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    return verify_and_unwrap_response(
        raw_resp,
        expected_request_id=req_id,
        client_nonce=nonce,
        client_token=client_token,
    )


def _open_daemon_log_fd(state_dir: Path) -> int:
    ensure_private_dir(state_dir)
    log_path = state_dir / "daemon.log"
    if log_path.is_symlink():
        raise TransportAuthError(f"Refusing symlinked daemon.log: {log_path}")
    fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    if os.name == "posix" and hasattr(os, "fchmod"):
        os.fchmod(fd, 0o600)
    return fd


def spawn_daemon_if_needed(
    config: TransportConfig,
    *,
    idle_timeout_s: float = float(DEFAULT_IDLE_TIMEOUT_SECONDS),
    bundle_dir: Path | str | None = None,
    startup_timeout_s: float = 12.0,
) -> dict[str, Any]:
    """Acquire `daemon.lock` (`filelock`) and ensure exactly 1 live OS daemon subprocess exists (`I13`)."""
    ensure_private_dir(config.state_dir)
    ensure_private_dir(config.lock_file.parent)
    if config.mode == "unix":
        ensure_private_dir(config.socket_path.parent)

    lock = FileLock(str(config.lock_file), timeout=startup_timeout_s)
    with lock:
        if os.name == "posix" and config.lock_file.exists() and not config.lock_file.is_symlink():
            try:
                os.chmod(config.lock_file, 0o600)
            except OSError:
                pass

        # 1. Re-verify inside lock whether another process already started the daemon
        existing_addr = read_addr_info(config)
        if existing_addr is not None:
            try:
                ping_res = _rpc_once_sync(existing_addr, "ping", {}, timeout_s=2.0)
                if ping_res.get("pong") is True:
                    return existing_addr
            except _TransientSocketError:
                pass

        # 2. Clean stale endpoint artifacts while holding daemon.lock
        try:
            config.addr_file.unlink(missing_ok=True)
        except OSError:
            pass
        if config.mode == "unix":
            try:
                config.socket_path.unlink(missing_ok=True)
            except OSError:
                pass

        ensure_admin_secret(config)

        cmd = [
            sys.executable,
            "-m",
            "diodemcp.engine",
            "--state-dir",
            str(config.state_dir),
            "--mode",
            config.mode,
            "--socket-path",
            str(config.socket_path),
            "--addr-file",
            str(config.addr_file),
            "--admin-secret-file",
            str(config.admin_secret_file),
            "--lock-file",
            str(config.lock_file),
            "--idle-timeout",
            str(float(idle_timeout_s)),
        ]
        if bundle_dir is not None:
            cmd.extend(["--bundle-dir", str(bundle_dir)])

        # Strip any accidental admin secret env vars before spawning child process
        child_env = {
            k: v
            for k, v in os.environ.items()
            if "ADMIN_SECRET" not in k.upper()
        }

        popen_kwargs: dict[str, Any] = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "env": child_env,
        }
        if sys.platform == "win32":
            flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
                subprocess, "CREATE_NO_WINDOW", 0
            )
            popen_kwargs["creationflags"] = flags
        else:
            popen_kwargs["start_new_session"] = True
            popen_kwargs["close_fds"] = True

        log_fd = _open_daemon_log_fd(config.state_dir)
        try:
            popen_kwargs["stderr"] = log_fd
            proc = subprocess.Popen(cmd, **popen_kwargs)
        finally:
            os.close(log_fd)

        deadline = time.monotonic() + startup_timeout_s
        sleep_step = 0.01
        while time.monotonic() < deadline:
            addr = read_addr_info(config)
            if addr is not None:
                try:
                    ping_res = _rpc_once_sync(addr, "ping", {}, timeout_s=2.0)
                    if ping_res.get("pong") is True:
                        return addr
                except _TransientSocketError:
                    pass

            rc = proc.poll()
            if rc is not None and not config.addr_file.exists():
                log_tail = ""
                log_file = config.state_dir / "daemon.log"
                if log_file.is_file():
                    try:
                        log_tail = log_file.read_text(encoding="utf-8", errors="replace")[
                            -1024:
                        ].strip()
                    except OSError:
                        log_tail = ""
                raise TransportError(
                    f"DiodeMCP daemon subprocess exited prematurely with code {rc}: {log_tail}"
                )

            time.sleep(sleep_step)
            sleep_step = min(0.04, sleep_step * 1.3)

        try:
            proc.terminate()
        except OSError:
            pass
        raise TransportError(
            f"Timed out after {startup_timeout_s:.1f}s waiting for DiodeMCP daemon on {config.mode}"
        )


def send_rpc_sync(
    config: TransportConfig,
    method: str,
    params: dict[str, Any],
    *,
    admin_secret: str | None = None,
    auto_spawn: bool = True,
    idle_timeout_s: float = float(DEFAULT_IDLE_TIMEOUT_SECONDS),
    bundle_dir: Path | str | None = None,
    timeout_s: float = 15.0,
) -> dict[str, Any]:
    """Execute a mutually authenticated synchronous RPC, autospawning on cold/idle socket if enabled."""
    addr = read_addr_info(config)
    if addr is not None:
        try:
            return _rpc_once_sync(
                addr,
                method,
                params,
                admin_secret=admin_secret,
                timeout_s=timeout_s,
            )
        except _TransientSocketError as exc:
            if not auto_spawn:
                raise TransportError(f"Failed connecting to DiodeMCP daemon: {exc}") from exc

    if not auto_spawn:
        raise TransportError(
            f"DiodeMCP daemon is not running (missing {config.addr_file})"
        )

    last_err: Exception | None = None
    for _attempt in range(2):
        spawned_addr = spawn_daemon_if_needed(
            config,
            idle_timeout_s=idle_timeout_s,
            bundle_dir=bundle_dir,
        )
        try:
            return _rpc_once_sync(
                spawned_addr,
                method,
                params,
                admin_secret=admin_secret,
                timeout_s=timeout_s,
            )
        except _TransientSocketError as exc:
            last_err = exc

    raise TransportError(
        f"Failed communicating with DiodeMCP daemon after autospawn: {last_err}"
    )


async def send_rpc_async(
    config: TransportConfig,
    method: str,
    params: dict[str, Any],
    *,
    admin_secret: str | None = None,
    auto_spawn: bool = True,
    idle_timeout_s: float = float(DEFAULT_IDLE_TIMEOUT_SECONDS),
    bundle_dir: Path | str | None = None,
    timeout_s: float = 15.0,
) -> dict[str, Any]:
    """Execute a mutually authenticated async RPC without blocking the caller's event loop."""
    addr = read_addr_info(config)
    if addr is not None:
        try:
            return await _rpc_once_async(
                addr,
                method,
                params,
                admin_secret=admin_secret,
                timeout_s=timeout_s,
            )
        except _TransientSocketError as exc:
            if not auto_spawn:
                raise TransportError(f"Failed connecting to DiodeMCP daemon: {exc}") from exc

    if not auto_spawn:
        raise TransportError(
            f"DiodeMCP daemon is not running (missing {config.addr_file})"
        )

    last_err: Exception | None = None
    for _attempt in range(2):
        spawned_addr = await asyncio.to_thread(
            spawn_daemon_if_needed,
            config,
            idle_timeout_s=idle_timeout_s,
            bundle_dir=bundle_dir,
        )
        try:
            return await _rpc_once_async(
                spawned_addr,
                method,
                params,
                admin_secret=admin_secret,
                timeout_s=timeout_s,
            )
        except _TransientSocketError as exc:
            last_err = exc

    raise TransportError(
        f"Failed communicating with DiodeMCP daemon after autospawn: {last_err}"
    )
