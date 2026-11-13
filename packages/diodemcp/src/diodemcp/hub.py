"""Upfront Hugging Face / HTTPS DiodeModelBundle v1 downloader with streaming SHA256 verification."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import sys
import tempfile
from typing import Final
import urllib.parse

from filelock import FileLock
import httpx
from pydantic import ValidationError

from diodemcp.bundle import BundleManifest, BundleValidationError, validate_bundle_dir
from diodemcp.constants import (
    DEFAULT_HF_NAMESPACE,
    DEFAULT_MODEL_REPOS,
    DEFAULT_STATE_DIR,
)

MODEL_ALIASES: Final[dict[str, str]] = {
    **DEFAULT_MODEL_REPOS,
    "nano": f"{DEFAULT_HF_NAMESPACE}/diodemcp-nanojev-0.6b-int8",
}

DEFAULT_HF_ENDPOINT: Final[str] = "https://huggingface.co"
DEFAULT_MAX_FILE_BYTES: Final[int] = 2 * 1024 * 1024 * 1024  # 2 GiB safety cap
_MAX_MANIFEST_BYTES: Final[int] = 1024 * 1024  # 1 MiB cap for manifest.json
_STREAM_CHUNK_SIZE: Final[int] = 65536
_MAX_REDIRECTS: Final[int] = 5

_HF_REPO_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$"
)
_REVISION_RE = re.compile(r"^[A-Za-z0-9._/-]{1,128}$")


class BundleDownloadError(BundleValidationError):
    """Raised when downloading or verifying a remote DiodeModelBundle v1 fails."""


def _validate_revision(revision: str) -> str:
    rev = revision.strip()
    if (
        not rev
        or not _REVISION_RE.match(rev)
        or ".." in rev.split("/")
        or ".." in rev
        or rev.startswith("/")
        or rev.endswith("/")
        or "//" in rev
    ):
        raise BundleDownloadError(f"Invalid bundle revision: {revision!r}")
    return rev


def _looks_like_local_path(spec: str) -> bool:
    if spec.startswith(("/", "./", "../", "~", "\\", ".\\", "..\\")):
        return True
    if len(spec) >= 2 and spec[0].isalpha() and spec[1] == ":":
        return True
    return False


def _resolve_remote_spec(spec: str, revision: str) -> tuple[str, str]:
    """Return `(base_https_url, cache_slug)` for an alias, `org/repo`, or `https://` URL."""
    rev = _validate_revision(revision)

    if spec.lower() in MODEL_ALIASES:
        repo_id = MODEL_ALIASES[spec.lower()]
    elif "://" in spec:
        parsed = urllib.parse.urlsplit(spec)
        if parsed.scheme.lower() != "https" or not parsed.netloc:
            raise BundleDownloadError(
                f"Only https:// bundle URLs are allowed, got: {spec!r}"
            )
        if parsed.query or parsed.fragment or ".." in parsed.path.split("/"):
            raise BundleDownloadError(f"Invalid bundle base URL: {spec!r}")
        base_url = spec.rstrip("/")
        url_hash = hashlib.sha256(base_url.encode("utf-8")).hexdigest()[:16]
        return base_url, f"url--{url_hash}"
    else:
        repo_id = spec

    if not _HF_REPO_RE.match(repo_id) or ".." in repo_id:
        raise BundleDownloadError(
            f"Invalid model alias, Hugging Face repo ID, or path: {spec!r}"
        )

    hf_endpoint = (
        os.environ.get("DIODEMCP_HF_ENDPOINT", DEFAULT_HF_ENDPOINT).strip().rstrip("/")
    )
    parsed_ep = urllib.parse.urlsplit(hf_endpoint)
    if parsed_ep.scheme.lower() != "https" or not parsed_ep.netloc:
        raise BundleDownloadError(
            f"DIODEMCP_HF_ENDPOINT must be an https:// URL, got: {hf_endpoint!r}"
        )

    encoded_rev = "/".join(urllib.parse.quote(part, safe="") for part in rev.split("/"))
    base_url = f"{hf_endpoint}/{repo_id}/resolve/{encoded_rev}"

    org, repo_name = repo_id.split("/", 1)
    if org == DEFAULT_HF_NAMESPACE and rev == "main":
        slug = repo_name
    else:
        safe_rev = re.sub(r"[^A-Za-z0-9._-]", "_", rev)
        slug = f"{org}--{repo_name}--{safe_rev}"
    return base_url, slug


def _stream_https_file(
    client: httpx.Client,
    url: str,
    dest_path: Path,
    *,
    label: str,
    max_bytes: int,
    expected_sha256: str | None = None,
    progress: bool = False,
) -> str:
    """Stream an HTTPS resource with strict HTTPS-only redirect validation and SHA256 hashing."""
    current_url = url
    for _hop in range(_MAX_REDIRECTS + 1):
        parsed_curr = urllib.parse.urlsplit(current_url)
        if parsed_curr.scheme.lower() != "https" or not parsed_curr.netloc:
            raise BundleDownloadError(
                f"Refusing non-HTTPS download or redirect target for {label}: {current_url!r}"
            )

        try:
            with client.stream("GET", current_url, follow_redirects=False) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("Location")
                    if not location:
                        raise BundleDownloadError(
                            f"Redirect missing Location header for {label}"
                        )
                    current_url = urllib.parse.urljoin(current_url, location)
                    continue

                if response.status_code != 200:
                    raise BundleDownloadError(
                        f"HTTP {response.status_code} while downloading {label} from {current_url}"
                    )

                content_length_hdr = response.headers.get("Content-Length")
                total_expected: int | None = None
                if content_length_hdr is not None:
                    try:
                        total_expected = int(content_length_hdr)
                    except ValueError:
                        total_expected = None
                    if total_expected is not None and total_expected > max_bytes:
                        raise BundleDownloadError(
                            f"Artifact {label} Content-Length ({total_expected}) exceeds "
                            f"safety cap ({max_bytes} bytes)"
                        )

                hasher = hashlib.sha256()
                bytes_written = 0
                with dest_path.open("wb") as out_f:
                    for chunk in response.iter_bytes(chunk_size=_STREAM_CHUNK_SIZE):
                        if not chunk:
                            continue
                        bytes_written += len(chunk)
                        if bytes_written > max_bytes:
                            raise BundleDownloadError(
                                f"Artifact {label} exceeded max size ({max_bytes} bytes)"
                            )
                        hasher.update(chunk)
                        out_f.write(chunk)
                    out_f.flush()

                if progress:
                    if total_expected and total_expected > 0:
                        sys.stderr.write(
                            f"[diodemcp] Downloaded {label} ({bytes_written}/{total_expected} bytes)\n"
                        )
                    else:
                        sys.stderr.write(
                            f"[diodemcp] Downloaded {label} ({bytes_written} bytes)\n"
                        )
                    sys.stderr.flush()

                actual_sha = hasher.hexdigest()
                if expected_sha256 is not None and actual_sha != expected_sha256:
                    raise BundleDownloadError(
                        f"SHA256 mismatch for {label}: expected {expected_sha256}, "
                        f"got {actual_sha}"
                    )
                return actual_sha
        except httpx.HTTPError as exc:
            raise BundleDownloadError(
                f"Network error while downloading {label}: {exc}"
            ) from exc

    raise BundleDownloadError(f"Exceeded {_MAX_REDIRECTS} redirects while fetching {label}")


def download_and_verify_bundle(
    model_alias_or_repo: str,
    cache_dir: Path | str | None = None,
    progress: bool = True,
    *,
    revision: str = "main",
    force: bool = False,
    client: httpx.Client | None = None,
    timeout: float = 60.0,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> Path:
    """Download, stream-hash all 3 bundle artifacts, validate, and atomically promote into `cache_dir`."""
    if not isinstance(model_alias_or_repo, str) or not model_alias_or_repo.strip():
        raise BundleDownloadError("model_alias_or_repo must be a non-empty string")
    if max_file_bytes < 1:
        raise ValueError("max_file_bytes must be >= 1")

    raw_spec = model_alias_or_repo.strip()

    # Check existing local directory path (unless raw_spec is a canonical alias without path separators)
    if raw_spec.lower() not in MODEL_ALIASES and "://" not in raw_spec:
        expanded = Path(raw_spec).expanduser()
        if expanded.exists():
            if not expanded.is_dir():
                raise BundleValidationError(
                    f"Local bundle path is not a directory: {expanded}"
                )
            validate_bundle_dir(expanded, verify_sha256=True)
            return expanded.resolve()
        if _looks_like_local_path(raw_spec):
            raise BundleValidationError(f"Bundle directory does not exist: {expanded}")

    base_url, slug = _resolve_remote_spec(raw_spec, revision)
    cache_root = (
        Path(cache_dir).expanduser()
        if cache_dir is not None
        else (DEFAULT_STATE_DIR / "models")
    )
    cache_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    resolved_cache_root = cache_root.resolve()

    target_dir = cache_root / slug
    if not target_dir.resolve().is_relative_to(resolved_cache_root):
        raise BundleDownloadError(f"Refusing unsafe cache target path: {target_dir}")

    lock_path = cache_root / f".{slug}.lock"
    with FileLock(str(lock_path)):
        if not force and target_dir.is_dir():
            try:
                validate_bundle_dir(target_dir, verify_sha256=True)
                return target_dir
            except BundleValidationError:
                shutil.rmtree(target_dir, ignore_errors=True)

        staging_dir = Path(
            tempfile.mkdtemp(prefix=f".staging-{slug}-", dir=str(cache_root))
        )
        os.chmod(staging_dir, 0o700)

        owns_client = client is None
        active_client: httpx.Client = (
            httpx.Client(timeout=timeout, follow_redirects=False)
            if client is None
            else client
        )

        try:
            manifest_path = staging_dir / "manifest.json"
            _stream_https_file(
                active_client,
                f"{base_url}/manifest.json",
                manifest_path,
                label="manifest.json",
                max_bytes=min(max_file_bytes, _MAX_MANIFEST_BYTES),
                expected_sha256=None,
                progress=progress,
            )

            try:
                raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest = BundleManifest.model_validate(raw_manifest)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
                raise BundleDownloadError(
                    f"Invalid remote manifest.json from {base_url}: {exc}"
                ) from exc

            artifact_specs = (
                ("model_int8.onnx", manifest.sha256_model_onnx),
                ("tokenizer.json", manifest.sha256_tokenizer_json),
                ("calibration.json", manifest.sha256_calibration_json),
            )
            for filename, expected_sha in artifact_specs:
                _stream_https_file(
                    active_client,
                    f"{base_url}/{filename}",
                    staging_dir / filename,
                    label=filename,
                    max_bytes=max_file_bytes,
                    expected_sha256=expected_sha,
                    progress=progress,
                )

            # Full structural + SHA256 + tokenizer special-token cross-check BEFORE promotion
            validate_bundle_dir(staging_dir, verify_sha256=True)

            backup_dir: Path | None = None
            if target_dir.exists():
                backup_dir = cache_root / f".old-{slug}-{secrets.token_hex(6)}"
                os.replace(target_dir, backup_dir)

            try:
                os.replace(staging_dir, target_dir)
            except BaseException:
                if backup_dir is not None and backup_dir.exists() and not target_dir.exists():
                    try:
                        os.replace(backup_dir, target_dir)
                    except OSError:
                        pass
                raise
            else:
                if backup_dir is not None and backup_dir.exists():
                    shutil.rmtree(backup_dir, ignore_errors=True)

            return target_dir
        finally:
            if owns_client:
                active_client.close()
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
