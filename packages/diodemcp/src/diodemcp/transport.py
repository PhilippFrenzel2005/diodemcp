"""Cross-platform Unix/TCP daemon transport, mutual-HMAC wire auth, and lock-safe autospawn (`C4`, `I13`)."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass, field
import errno
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import secrets
import socket
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterator, Literal

from filelock import FileLock, Timeout as FileLockTimeout

from diodemcp.constants import DEFAULT_IDLE_TIMEOUT_SECONDS, DEFAULT_STATE_DIR

MAX_IPC_LINE_BYTES: int = 16 * 1024 * 1024  # 16 MiB frame ceiling on daemon IPC
PRE_AUTH_TIMEOUT_SECONDS: float = 5.0
REPLAY_WINDOW_SECONDS: float = 60.0
MIN_CREDENTIAL_CHARS: int = 32
_SUN_PATH_SAFE_BYTES: int = 96

_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

PRIVILEGED_METHODS: frozenset[str] = frozenset(
    {
        "approve_incident",
        "approve_allow_token",
        "trust_tool_shape",
        "list_incidents",
        "shutdown",
    }
)


class TransportError(RuntimeError):
    """Base error for DiodeMCP daemon IPC transport failures."""


class TransportAuthError(TransportError):
    """Raised on missing/invalid `client_token` or failed mutual-HMAC verification (`I13`)."""


class TransportPermissionError(TransportError):
    """Raised when a privileged RPC lacks a valid `admin_secret` or `shim_secret` (`C4`)."""


class _DeadEndpointError(TransportError):
    """Internal marker for dead/refused/missing socket endpoints eligible for lock-guarded respawn."""


class _StaleServerHelloError(TransportAuthError):
    """Raised when TCP 1-RTT server hello fails HMAC (squatter or stale recycled port)."""


def sanitize_error_message(message: Any, *, max_len: int = 240) -> str:
    """Sanitize error text to single-line bounded printable string without control chars."""
    raw = str(message or "Daemon transport error")
    cleaned = _SURROGATE_RE.sub("\ufffd", raw)
    cleaned = cleaned.replace("\r", " ").replace("\n", " ")
    cleaned = _CONTROL_CHAR_RE.sub("", cleaned).strip()
    if not cleaned:
        return "Daemon transport error"
    if len(cleaned) > max_len:
        return cleaned[: max_len - 3] + "..."
    return cleaned


def constant_time_ascii_equals(candidate: Any, expected_ascii: str) -> bool:
    """Compare `candidate` against `expected_ascii` as ASCII bytes in constant time without TypeError."""
    exp_bytes = expected_ascii.encode("ascii")
    if not isinstance(candidate, str):
        hmac.compare_digest(exp_bytes, exp_bytes)
        return False
    try:
        cand_bytes = candidate.encode("ascii")
    except UnicodeEncodeError:
        hmac.compare_digest(exp_bytes, exp_bytes)
        return False
    return hmac.compare_digest(cand_bytes, exp_bytes)


def compute_popup_binding_sha256(shim_secret: str) -> str:
    """Derive one-way lowercase SHA-256 popup binding digest from an in-memory `shim_secret`."""
    if not isinstance(shim_secret, str) or not (MIN_CREDENTIAL_CHARS <= len(shim_secret) <= 512):
        raise ValueError(
            f"shim_secret must be an ASCII string of {MIN_CREDENTIAL_CHARS}..512 characters"
        )
    try:
        raw = shim_secret.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("shim_secret must contain ASCII characters only") from exc
    return hashlib.sha256(raw).hexdigest()


def _is_ascii_credential(value: Any, *, min_len: int = MIN_CREDENTIAL_CHARS, max_len: int = 512) -> bool:
    if not isinstance(value, str) or not (min_len <= len(value) <= max_len):
        return False
    try:
        value.encode("ascii")
    except UnicodeEncodeError:
        return False
    return True


def is_pid_alive(pid: int) -> bool:
    """Return True if `pid` refers to a live, non-zombie same-UID OS process."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "posix":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            # A process owned by a different UID cannot be our same-UID daemon
            return False
        proc_stat = Path(f"/proc/{pid}/stat")
        if proc_stat.is_file():
            try:
                content = proc_stat.read_text(encoding="utf-8", errors="ignore")
                state_char = content.rsplit(")", 1)[1].strip().split(" ", 1)[0]
                if state_char == "Z":
                    return False
            except (OSError, IndexError):
                return False
        return True
    return True


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


def _sanitize_surrogates_in_obj(value: Any) -> Any:
    if isinstance(value, str):
        return _SURROGATE_RE.sub("\ufffd", value) if _SURROGATE_RE.search(value) else value
    if isinstance(value, dict):
        return {
            _sanitize_surrogates_in_obj(str(k)): _sanitize_surrogates_in_obj(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_surrogates_in_obj(item) for item in value]
    return value


def atomic_write_private_json(target_path: Path, payload: dict[str, Any]) -> None:
    """Atomically serialize `payload` as UTF-8 JSON with `0600` permissions."""
    safe_payload = _sanitize_surrogates_in_obj(payload)
    serialized = json.dumps(safe_payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
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


def _read_verified_secret_file(secret_path: Path) -> str:
    if secret_path.is_symlink():
        raise TransportAuthError(f"Refusing symlinked admin.secret: {secret_path}")
    if not secret_path.exists():
        raise TransportAuthError(f"Missing admin.secret at {secret_path}")
    _verify_private_regular_file(secret_path)
    secret = secret_path.read_text(encoding="utf-8").strip()
    if not _is_ascii_credential(secret, min_len=MIN_CREDENTIAL_CHARS):
        raise TransportAuthError(
            f"Invalid or truncated admin.secret at {secret_path} (expected >= {MIN_CREDENTIAL_CHARS} ASCII chars)"
        )
    return secret


def ensure_admin_secret(config: TransportConfig) -> str:
    """Atomically initialize `admin.secret` at `0600` via temp file + `os.link` (zero 0-byte window, never overwrites)."""
    ensure_private_dir(config.state_dir)
    parent = ensure_private_dir(config.admin_secret_file.parent)
    secret_path = config.admin_secret_file

    if secret_path.is_symlink():
        raise TransportAuthError(f"Refusing symlinked admin.secret: {secret_path}")
    if secret_path.exists():
        return _read_verified_secret_file(secret_path)

    fresh = secrets.token_hex(32)
    payload = f"{fresh}\n".encode("ascii")
    tmp_path = parent / f".admin.secret.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        written = 0
        view = memoryview(payload)
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
            try:
                os.link(tmp_path, secret_path)
            except FileExistsError:
                return _read_verified_secret_file(secret_path)
            return fresh

        # Windows fallback guarded by filelock without overwriting existing secret
        ensure_private_dir(config.lock_file.parent)
        try:
            with FileLock(str(config.lock_file), timeout=5.0):
                if secret_path.exists():
                    return _read_verified_secret_file(secret_path)
                os.replace(tmp_path, secret_path)
                return fresh
        except FileLockTimeout as exc:
            raise TransportError("Timed out acquiring lock to initialize admin.secret") from exc
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def read_admin_secret(config: TransportConfig) -> str:
    """Read and validate existing `0600` `admin.secret` strictly read-only (never creating)."""
    return _read_verified_secret_file(config.admin_secret_file)


def read_addr_info(config: TransportConfig) -> dict[str, Any] | None:
    """Read and validate `addr.json` (`0600`), returning `None` if absent."""
    addr_path = config.addr_file
    if addr_path.is_symlink():
        raise TransportAuthError(f"Refusing symlinked addr.json: {addr_path}")
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
    if mode != config.mode or not _is_ascii_credential(token, min_len=MIN_CREDENTIAL_CHARS):
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


def _validate_strict_json_value(value: Any) -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite float values (NaN/Infinity) are not valid JSON")
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_strict_json_value(item)
        return
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError(f"Canonical JSON object keys must be str, got {type(k).__name__}")
            _validate_strict_json_value(v)
        return
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def canonical_json_sha256(value: Any) -> str:
    """Compute deterministic SHA-256 over strict canonical JSON (surrogate-safe, type-distinct)."""
    _validate_strict_json_value(value)
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(serialized).hexdigest()


def compute_hello_hmac(
    client_token: str,
    hello_nonce: str,
    server_nonce: str,
) -> str:
    """Compute 1-RTT TCP server-first challenge proof `HMAC(client_token, 'hello:' + hello_nonce + ':' + server_nonce)`."""
    msg = f"hello:{hello_nonce}:{server_nonce}".encode("ascii")
    return hmac.new(client_token.encode("ascii"), msg, hashlib.sha256).hexdigest()


def compute_client_hmac(
    client_token: str,
    client_nonce: str,
    request_id: str,
    method: str,
    params_sha256: str,
    *,
    timestamp: int | None = None,
) -> str:
    if timestamp is not None:
        msg = f"client:v1:{timestamp}:{client_nonce}:{request_id}:{method}:{params_sha256}".encode(
            "utf-8"
        )
    else:
        msg = f"client:v1:{client_nonce}:{request_id}:{method}:{params_sha256}".encode("utf-8")
    return hmac.new(client_token.encode("utf-8"), msg, hashlib.sha256).hexdigest()


def compute_admin_hmac(
    admin_secret: str,
    client_nonce: str,
    request_id: str,
    method: str,
    params_sha256: str,
    *,
    timestamp: int | None = None,
) -> str:
    if timestamp is not None:
        msg = f"admin:v1:{timestamp}:{client_nonce}:{request_id}:{method}:{params_sha256}".encode(
            "utf-8"
        )
    else:
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
    timestamp: int | None = None,
) -> tuple[bytes, str, str]:
    """Build `\\n`-delimited JSON request frame with replay timestamp and zero raw secrets (`I13`)."""
    request_id = f"req_{secrets.token_hex(16)}"
    client_nonce = secrets.token_hex(16)
    ts = int(time.time()) if timestamp is None else int(timestamp)
    params_sha = canonical_json_sha256(params)

    envelope: dict[str, Any] = {
        "request_id": request_id,
        "method": method,
        "params": params,
        "timestamp": ts,
        "client_nonce": client_nonce,
        "client_hmac": compute_client_hmac(
            client_token,
            client_nonce,
            request_id,
            method,
            params_sha,
            timestamp=ts,
        ),
    }
    if admin_secret is not None:
        envelope["admin_hmac"] = compute_admin_hmac(
            admin_secret,
            client_nonce,
            request_id,
            method,
            params_sha,
            timestamp=ts,
        )

    raw_line = (
        json.dumps(envelope, ensure_ascii=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("ascii")
    if len(raw_line) > MAX_IPC_LINE_BYTES:
        raise TransportError(f"IPC request exceeds {MAX_IPC_LINE_BYTES} bytes")
    return raw_line, request_id, client_nonce


def _verify_tcp_hello_response(
    raw_hello_resp: bytes,
    *,
    hello_nonce: str,
    client_token: str,
) -> None:
    if not raw_hello_resp:
        raise _DeadEndpointError("Empty EOF from TCP endpoint during server hello")
    try:
        hello_obj = json.loads(raw_hello_resp.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _StaleServerHelloError("Malformed JSON during TCP server hello") from exc
    if not isinstance(hello_obj, dict):
        raise _StaleServerHelloError("TCP server hello response must be a JSON object")

    srv_nonce = hello_obj.get("server_nonce")
    srv_mac = hello_obj.get("server_hmac")
    if (
        not _is_ascii_credential(srv_nonce, min_len=16, max_len=128)
        or not isinstance(srv_mac, str)
    ):
        raise _StaleServerHelloError("Missing or invalid TCP server_nonce / server_hmac")

    expected = compute_hello_hmac(client_token, hello_nonce, str(srv_nonce))
    if not constant_time_ascii_equals(srv_mac, expected):
        raise _StaleServerHelloError(
            "TCP server hello HMAC verification failed: refusing to send RPC payload"
        )


def verify_and_unwrap_response(
    raw_line: bytes,
    *,
    expected_request_id: str,
    client_nonce: str,
    client_token: str,
) -> dict[str, Any]:
    """Verify server mutual-HMAC before trusting any response (`I13`)."""
    if not raw_line:
        raise _DeadEndpointError("Empty EOF from daemon socket")

    try:
        resp = json.loads(raw_line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TransportAuthError("Malformed JSON response from daemon") from exc

    if not isinstance(resp, dict):
        raise TransportAuthError("Daemon response root must be a JSON object")

    if resp.get("code") == "UNAUTHORIZED":
        raise TransportAuthError(sanitize_error_message(resp.get("error") or "Unauthorized"))

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
    if not constant_time_ascii_equals(server_mac, expected_mac):
        raise TransportAuthError(
            "Mutual HMAC verification failed: refusing untrusted daemon response"
        )

    if not ok_val:
        code = resp.get("code")
        err_msg = sanitize_error_message(resp.get("error") or "Daemon RPC error")
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


def _classify_connect_os_error(exc: BaseException) -> Exception:
    if isinstance(exc, (FileNotFoundError, ConnectionRefusedError)):
        return _DeadEndpointError(sanitize_error_message(exc))
    if isinstance(exc, OSError) and exc.errno in (
        errno.ENOENT,
        errno.ECONNREFUSED,
        errno.ENOTSOCK,
        errno.ECONNRESET,
        errno.EPIPE,
    ):
        return _DeadEndpointError(sanitize_error_message(exc))
    return TransportError(sanitize_error_message(exc))


def _rpc_once_sync(
    addr: dict[str, Any],
    method: str,
    params: dict[str, Any],
    *,
    admin_secret: str | None = None,
    timeout_s: float = 15.0,
) -> dict[str, Any]:
    client_token = str(addr["client_token"])
    hello_timeout = min(float(timeout_s), PRE_AUTH_TIMEOUT_SECONDS)

    if addr["mode"] == "unix":
        if not hasattr(socket, "AF_UNIX"):
            raise TransportError("AF_UNIX sockets are not supported on this platform")
        frame, req_id, nonce = build_signed_request_frame(
            method=method,
            params=params,
            client_token=client_token,
            admin_secret=admin_secret,
        )
        sock_path = Path(str(addr["socket_path"]))
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(hello_timeout)
            try:
                with unix_path_handle(sock_path) as connect_target:
                    sock.connect(connect_target)
            except BaseException as exc:
                raise _classify_connect_os_error(exc) from exc

            sock.settimeout(timeout_s)
            try:
                sock.sendall(frame)
                raw_resp = _read_bounded_line_sync(sock)
            except (TimeoutError, socket.timeout) as exc:
                raise TransportError(
                    f"Daemon RPC {method!r} timed out after {timeout_s:.2f}s"
                ) from exc
            except (ConnectionResetError, BrokenPipeError) as exc:
                raise _DeadEndpointError(sanitize_error_message(exc)) from exc
            except OSError as exc:
                raise _classify_connect_os_error(exc) from exc
        finally:
            sock.close()
    else:
        host = str(addr["host"])
        port = int(addr["port"])
        if host != "127.0.0.1":
            raise TransportAuthError(f"Refusing non-loopback TCP host: {host!r}")

        try:
            sock = socket.create_connection((host, port), timeout=hello_timeout)
        except BaseException as exc:
            raise _classify_connect_os_error(exc) from exc

        try:
            # Step 1: 1-RTT server-first challenge BEFORE serializing/sending any params!
            hello_nonce = secrets.token_hex(16)
            hello_wire = (
                json.dumps({"hello_nonce": hello_nonce}, separators=(",", ":")) + "\n"
            ).encode("ascii")
            sock.settimeout(hello_timeout)
            try:
                sock.sendall(hello_wire)
                raw_hello = _read_bounded_line_sync(sock)
            except (TimeoutError, socket.timeout) as exc:
                raise TransportError("TCP daemon hello timed out") from exc
            except (ConnectionResetError, BrokenPipeError) as exc:
                raise _DeadEndpointError(sanitize_error_message(exc)) from exc
            except OSError as exc:
                raise _classify_connect_os_error(exc) from exc

            _verify_tcp_hello_response(
                raw_hello,
                hello_nonce=hello_nonce,
                client_token=client_token,
            )

            # Step 2: Server proved possession of client_token -> now send RPC frame
            frame, req_id, nonce = build_signed_request_frame(
                method=method,
                params=params,
                client_token=client_token,
                admin_secret=admin_secret,
            )
            sock.settimeout(timeout_s)
            try:
                sock.sendall(frame)
                raw_resp = _read_bounded_line_sync(sock)
            except (TimeoutError, socket.timeout) as exc:
                raise TransportError(
                    f"Daemon RPC {method!r} timed out after {timeout_s:.2f}s"
                ) from exc
            except (ConnectionResetError, BrokenPipeError) as exc:
                raise _DeadEndpointError(sanitize_error_message(exc)) from exc
            except OSError as exc:
                raise _classify_connect_os_error(exc) from exc
        finally:
            sock.close()

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
    hello_timeout = min(float(timeout_s), PRE_AUTH_TIMEOUT_SECONDS)

    writer: asyncio.StreamWriter | None = None
    try:
        if addr["mode"] == "unix":
            frame, req_id, nonce = build_signed_request_frame(
                method=method,
                params=params,
                client_token=client_token,
                admin_secret=admin_secret,
            )
            sock_path = Path(str(addr["socket_path"]))
            try:
                with unix_path_handle(sock_path) as connect_target:
                    reader, writer = await asyncio.wait_for(
                        asyncio.open_unix_connection(
                            connect_target, limit=MAX_IPC_LINE_BYTES
                        ),
                        timeout=hello_timeout,
                    )
            except (TimeoutError, asyncio.TimeoutError) as exc:
                raise TransportError("Timed out connecting to Unix daemon socket") from exc
            except BaseException as exc:
                raise _classify_connect_os_error(exc) from exc

            try:
                writer.write(frame)
                await asyncio.wait_for(writer.drain(), timeout=timeout_s)
                raw_resp = await asyncio.wait_for(reader.readline(), timeout=timeout_s)
            except (asyncio.LimitOverrunError, ValueError) as exc:
                raise TransportError("Daemon IPC response exceeded max line size") from exc
            except (TimeoutError, asyncio.TimeoutError) as exc:
                raise TransportError(
                    f"Daemon RPC {method!r} timed out after {timeout_s:.2f}s"
                ) from exc
            except (ConnectionResetError, BrokenPipeError) as exc:
                raise _DeadEndpointError(sanitize_error_message(exc)) from exc
            except OSError as exc:
                raise _classify_connect_os_error(exc) from exc
        else:
            host = str(addr["host"])
            port = int(addr["port"])
            if host != "127.0.0.1":
                raise TransportAuthError(f"Refusing non-loopback TCP host: {host!r}")
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port, limit=MAX_IPC_LINE_BYTES),
                    timeout=hello_timeout,
                )
            except (TimeoutError, asyncio.TimeoutError) as exc:
                raise TransportError("Timed out connecting to TCP daemon socket") from exc
            except BaseException as exc:
                raise _classify_connect_os_error(exc) from exc

            # Step 1: 1-RTT TCP server-first hello challenge before transmitting params
            hello_nonce = secrets.token_hex(16)
            hello_wire = (
                json.dumps({"hello_nonce": hello_nonce}, separators=(",", ":")) + "\n"
            ).encode("ascii")
            try:
                writer.write(hello_wire)
                await asyncio.wait_for(writer.drain(), timeout=hello_timeout)
                raw_hello = await asyncio.wait_for(reader.readline(), timeout=hello_timeout)
            except (asyncio.LimitOverrunError, ValueError) as exc:
                raise _StaleServerHelloError("Oversized TCP server hello line") from exc
            except (TimeoutError, asyncio.TimeoutError) as exc:
                raise TransportError("TCP daemon hello timed out") from exc
            except (ConnectionResetError, BrokenPipeError) as exc:
                raise _DeadEndpointError(sanitize_error_message(exc)) from exc
            except OSError as exc:
                raise _classify_connect_os_error(exc) from exc

            _verify_tcp_hello_response(
                raw_hello,
                hello_nonce=hello_nonce,
                client_token=client_token,
            )

            # Step 2: Server verified -> transmit signed RPC frame
            frame, req_id, nonce = build_signed_request_frame(
                method=method,
                params=params,
                client_token=client_token,
                admin_secret=admin_secret,
            )
            try:
                writer.write(frame)
                await asyncio.wait_for(writer.drain(), timeout=timeout_s)
                raw_resp = await asyncio.wait_for(reader.readline(), timeout=timeout_s)
            except (asyncio.LimitOverrunError, ValueError) as exc:
                raise TransportError("Daemon IPC response exceeded max line size") from exc
            except (TimeoutError, asyncio.TimeoutError) as exc:
                raise TransportError(
                    f"Daemon RPC {method!r} timed out after {timeout_s:.2f}s"
                ) from exc
            except (ConnectionResetError, BrokenPipeError) as exc:
                raise _DeadEndpointError(sanitize_error_message(exc)) from exc
            except OSError as exc:
                raise _classify_connect_os_error(exc) from exc
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


def _unlink_stale_addr_if_matching(addr_file: Path, expected_token: str | None) -> None:
    if not addr_file.exists() or addr_file.is_symlink():
        return
    if expected_token is None:
        try:
            addr_file.unlink(missing_ok=True)
        except OSError:
            pass
        return
    try:
        cur = json.loads(addr_file.read_text(encoding="utf-8"))
        if isinstance(cur, dict) and cur.get("client_token") == expected_token:
            addr_file.unlink(missing_ok=True)
    except OSError:
        pass
    except (UnicodeDecodeError, json.JSONDecodeError):
        try:
            addr_file.unlink(missing_ok=True)
        except OSError:
            pass


def _unlink_stale_unix_socket_if_dead(socket_path: Path) -> None:
    if not socket_path.exists() or socket_path.is_symlink():
        return
    try:
        st = socket_path.lstat()
    except OSError:
        return
    if not stat.S_ISSOCK(st.st_mode):
        return
    # Probe socket before unlinking: NEVER unlink an active listening Unix socket!
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.settimeout(0.5)
        with unix_path_handle(socket_path) as target:
            probe.connect(target)
        # Connect succeeded -> live daemon is listening! Do NOT unlink!
        return
    except (FileNotFoundError, ConnectionRefusedError, OSError):
        pass
    finally:
        probe.close()

    try:
        socket_path.unlink(missing_ok=True)
    except OSError:
        pass


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
    try:
        lock_ctx = lock.acquire(timeout=startup_timeout_s)
    except FileLockTimeout as exc:
        raise TransportError(
            f"Timed out after {startup_timeout_s:.2f}s acquiring DiodeMCP daemon lock"
        ) from exc

    with lock_ctx:
        if os.name == "posix" and config.lock_file.exists() and not config.lock_file.is_symlink():
            try:
                os.chmod(config.lock_file, 0o600)
            except OSError:
                pass

        # 1. Re-verify inside lock whether a live daemon is already serving
        existing_addr = read_addr_info(config)
        stale_token: str | None = None
        if existing_addr is not None:
            stale_token = str(existing_addr.get("client_token", "")) or None
            existing_pid = int(existing_addr.get("pid", 0) or 0)
            try:
                ping_res = _rpc_once_sync(existing_addr, "ping", {}, timeout_s=2.5)
                if ping_res.get("pong") is True:
                    return existing_addr
            except _StaleServerHelloError:
                # Stale TCP port squatter failed hello challenge -> unlink stale addr.json & respawn on fresh port
                _unlink_stale_addr_if_matching(config.addr_file, stale_token)
            except _DeadEndpointError:
                # Connection refused / missing socket -> dead daemon, safe to clean & respawn
                _unlink_stale_addr_if_matching(config.addr_file, stale_token)
                if config.mode == "unix":
                    _unlink_stale_unix_socket_if_dead(config.socket_path)
            except TransportError:
                # Mere read timeout: if PID is still alive, NEVER unlink or spawn duplicate daemon!
                if is_pid_alive(existing_pid):
                    return existing_addr
                _unlink_stale_addr_if_matching(config.addr_file, stale_token)
                if config.mode == "unix":
                    _unlink_stale_unix_socket_if_dead(config.socket_path)
        else:
            _unlink_stale_addr_if_matching(config.addr_file, None)
            if config.mode == "unix":
                _unlink_stale_unix_socket_if_dead(config.socket_path)

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

        # Strip any accidental admin or shim secret env vars before spawning child process
        child_env = {
            k: v
            for k, v in os.environ.items()
            if "ADMIN_SECRET" not in k.upper() and "SHIM_SECRET" not in k.upper()
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
            if addr is not None and int(addr.get("pid", 0) or 0) == proc.pid:
                try:
                    ping_res = _rpc_once_sync(addr, "ping", {}, timeout_s=2.0)
                    if ping_res.get("pong") is True:
                        return addr
                except (_DeadEndpointError, _StaleServerHelloError, TransportError):
                    pass

            rc = proc.poll()
            if rc is not None and not config.addr_file.exists():
                raise TransportError(
                    f"DiodeMCP daemon subprocess exited prematurely with code {rc}"
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
    """Execute a mutually authenticated synchronous RPC, autospawning on dead/stale socket if enabled."""
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
        except _StaleServerHelloError:
            if not auto_spawn:
                raise
        except _DeadEndpointError as exc:
            if not auto_spawn:
                raise TransportError(
                    f"Failed connecting to DiodeMCP daemon: {sanitize_error_message(exc)}"
                ) from exc

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
        except (_DeadEndpointError, _StaleServerHelloError) as exc:
            last_err = exc

    raise TransportError(
        f"Failed communicating with DiodeMCP daemon after autospawn: {sanitize_error_message(last_err)}"
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
        except _StaleServerHelloError:
            if not auto_spawn:
                raise
        except _DeadEndpointError as exc:
            if not auto_spawn:
                raise TransportError(
                    f"Failed connecting to DiodeMCP daemon: {sanitize_error_message(exc)}"
                ) from exc

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
        except (_DeadEndpointError, _StaleServerHelloError) as exc:
            last_err = exc

    raise TransportError(
        f"Failed communicating with DiodeMCP daemon after autospawn: {sanitize_error_message(last_err)}"
    )
