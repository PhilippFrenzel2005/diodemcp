"""Minimal hardened MCP JSON-RPC 2.0 stdio proxy for competitor Suite B contenders."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping, Sequence
import json
import os
import sys
from typing import Any

from benchmarks.contenders.stateless_policy_worker import (
    StatelessPolicyEvaluator,
    setup_pristine_ipc_stdout,
)

MAX_JSONRPC_LINE_BYTES: int = 64 * 1024 * 1024  # 64 MiB frame ceiling
_READ_CHUNK_SIZE: int = 65536


class _ChunkedLineReader:
    """Chunked newline reader enforcing `MAX_JSONRPC_LINE_BYTES` without LimitOverrunError."""

    def __init__(self, stream: asyncio.StreamReader, max_bytes: int = MAX_JSONRPC_LINE_BYTES) -> None:
        self._stream = stream
        self._max_bytes = max_bytes
        self._buf = bytearray()
        self._discarding = False

    async def read_line(self) -> tuple[bytes | None, bool]:
        while True:
            nl = self._buf.find(b"\n")
            if nl != -1:
                if self._discarding:
                    del self._buf[: nl + 1]
                    self._discarding = False
                    return b"", True
                line = bytes(self._buf[:nl])
                del self._buf[: nl + 1]
                if line.endswith(b"\r"):
                    line = line[:-1]
                if len(line) > self._max_bytes:
                    return line[:4096], True
                return line, False

            if not self._discarding and len(self._buf) > self._max_bytes:
                self._buf.clear()
                self._discarding = True

            chunk = await self._stream.read(_READ_CHUNK_SIZE)
            if not chunk:
                if self._discarding:
                    self._discarding = False
                    self._buf.clear()
                    return b"", True
                if self._buf:
                    tail = bytes(self._buf)
                    self._buf.clear()
                    if tail.endswith(b"\r"):
                        tail = tail[:-1]
                    if len(tail) > self._max_bytes:
                        return tail[:4096], True
                    return tail, False
                return None, False

            if self._discarding:
                nl_pos = chunk.find(b"\n")
                if nl_pos != -1:
                    self._buf.extend(chunk[nl_pos:])
            else:
                self._buf.extend(chunk)


def _extract_response_text(method: str, msg: Mapping[str, Any]) -> str:
    err = msg.get("error")
    if isinstance(err, Mapping):
        parts = [str(err.get("message") or "")]
        if err.get("data") is not None:
            parts.append(json.dumps(err["data"], ensure_ascii=False))
        return "\n".join(p for p in parts if p)

    res = msg.get("result")
    if not isinstance(res, Mapping):
        return ""

    parts: list[str] = []
    if method == "tools/call":
        for blk in res.get("content") or []:
            if isinstance(blk, Mapping) and isinstance(blk.get("text"), str):
                parts.append(blk["text"])
    elif method == "resources/read":
        for item in res.get("contents") or []:
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                parts.append(item["text"])
    elif method == "prompts/get":
        if isinstance(res.get("description"), str):
            parts.append(res["description"])
        for m in res.get("messages") or []:
            if isinstance(m, Mapping):
                content = m.get("content")
                if isinstance(content, Mapping) and isinstance(content.get("text"), str):
                    parts.append(content["text"])
    return "\n".join(p for p in parts if p)


def _blocked_jsonrpc(req_id: Any, contender_id: str, reason: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {
            "code": -32001,
            "message": f"Blocked by {contender_id}: {reason}",
        },
    }


async def _run_proxy(
    mode: str,
    child_cmd: list[str],
    proto_out: Any,
    *,
    threads: int = 4,
) -> int:
    stateless_eval: StatelessPolicyEvaluator | None = None
    llm_guard_eval: Any = None

    if mode == "stateless_policy":
        stateless_eval = StatelessPolicyEvaluator(contender_id="stateless-regex-policy")
    elif mode == "llm_guard":
        from benchmarks.contenders.llm_guard_worker import LlmGuardEvaluator

        llm_guard_eval = LlmGuardEvaluator(
            contender_id="protectai-llm-guard",
            threads=threads,
        )

    child = await asyncio.create_subprocess_exec(
        *child_cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=sys.stderr,
        limit=MAX_JSONRPC_LINE_BYTES,
    )
    assert child.stdin is not None
    assert child.stdout is not None

    loop = asyncio.get_running_loop()
    host_reader = asyncio.StreamReader(limit=MAX_JSONRPC_LINE_BYTES)
    host_protocol = asyncio.StreamReaderProtocol(host_reader)
    await loop.connect_read_pipe(lambda: host_protocol, sys.stdin.buffer)

    write_lock = asyncio.Lock()
    child_stdin_lock = asyncio.Lock()
    pending_requests: dict[Any, str] = {}

    async def _write_host(payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        async with write_lock:
            proto_out.write(line)
            proto_out.flush()

    async def _write_child(payload: dict[str, Any]) -> None:
        raw = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        async with child_stdin_lock:
            if child.stdin is not None and not child.stdin.is_closing():
                child.stdin.write(raw)
                await child.stdin.drain()

    def _screen_request_sync(method: str, params: Mapping[str, Any]) -> str | None:
        if mode == "raw_passthrough" or method != "tools/call":
            return None
        tool_args = params.get("arguments")
        args_dict = dict(tool_args) if isinstance(tool_args, Mapping) else {}
        if stateless_eval is not None:
            return stateless_eval.check_candidate_args(args_dict)
        if llm_guard_eval is not None:
            ok, _score, reason = llm_guard_eval.scan_candidate_args(args_dict)
            if not ok:
                return reason or "Blocked by protectai-llm-guard"
        return None

    def _screen_response_sync(method: str, msg: Mapping[str, Any]) -> str | None:
        if mode == "raw_passthrough":
            return None
        res = msg.get("result")
        if method == "tools/list" and isinstance(res, Mapping):
            tools = res.get("tools")
            if isinstance(tools, list):
                if stateless_eval is not None:
                    return stateless_eval.check_tools_list(tools)
                if llm_guard_eval is not None:
                    ok, _score, reason = llm_guard_eval.scan_tools_list(tools)
                    if not ok:
                        return reason or "Blocked by protectai-llm-guard"
        if llm_guard_eval is not None and method in ("tools/call", "resources/read", "prompts/get"):
            text = _extract_response_text(method, msg)
            if text:
                ok, _score, reason = llm_guard_eval.scan_tool_output(text, method)
                if not ok:
                    return reason or "Blocked by protectai-llm-guard"
        return None

    async def _host_to_child() -> None:
        reader = _ChunkedLineReader(host_reader)
        try:
            while True:
                raw_line, oversized = await reader.read_line()
                if raw_line is None:
                    break
                if oversized or not raw_line.strip():
                    continue
                try:
                    msg = json.loads(raw_line.decode("utf-8"))
                except Exception as exc:
                    sys.stderr.write(f"[competitor_proxy] Ignoring invalid host JSON: {exc}\n")
                    sys.stderr.flush()
                    continue
                if not isinstance(msg, dict):
                    continue

                method = str(msg.get("method") or "")
                params = msg.get("params") if isinstance(msg.get("params"), Mapping) else {}
                req_id = msg.get("id")

                if "id" in msg and method:
                    contender_name = (
                        "stateless-regex-policy"
                        if mode == "stateless_policy"
                        else "protectai-llm-guard"
                    )
                    block_reason = await asyncio.to_thread(_screen_request_sync, method, params)
                    if block_reason is not None:
                        await _write_host(_blocked_jsonrpc(req_id, contender_name, block_reason))
                        continue
                    pending_requests[req_id] = method

                await _write_child(msg)
        finally:
            try:
                if child.stdin is not None and not child.stdin.is_closing():
                    child.stdin.close()
            except Exception:
                pass

    async def _child_to_host() -> None:
        assert child.stdout is not None
        reader = _ChunkedLineReader(child.stdout)
        while True:
            raw_line, oversized = await reader.read_line()
            if raw_line is None:
                break
            if oversized:
                sys.stderr.write("[competitor_proxy] Dropped oversized child stdout frame\n")
                sys.stderr.flush()
                continue
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                msg = json.loads(stripped.decode("utf-8"))
            except Exception:
                # Review Focus #5: route non-JSON debug lines from child stdout safely to stderr
                sys.stderr.write(stripped.decode("utf-8", errors="replace") + "\n")
                sys.stderr.flush()
                continue
            if not isinstance(msg, dict):
                sys.stderr.write(stripped.decode("utf-8", errors="replace") + "\n")
                sys.stderr.flush()
                continue

            if "method" not in msg and "id" in msg:
                method = pending_requests.pop(msg["id"], "")
                contender_name = (
                    "stateless-regex-policy"
                    if mode == "stateless_policy"
                    else "protectai-llm-guard"
                )
                block_reason = await asyncio.to_thread(_screen_response_sync, method, msg)
                if block_reason is not None:
                    await _write_host(_blocked_jsonrpc(msg["id"], contender_name, block_reason))
                    continue

            await _write_host(msg)

    await asyncio.gather(_host_to_child(), _child_to_host())
    return await child.wait()


def main(argv: Sequence[str] | None = None) -> int:
    proto_out = setup_pristine_ipc_stdout()
    try:
        parser = argparse.ArgumentParser(description="Competitor MCP stdio proxy")
        parser.add_argument(
            "--mode",
            choices=("raw_passthrough", "stateless_policy", "llm_guard"),
            required=True,
        )
        parser.add_argument("--threads", type=int, default=4)
        parser.add_argument("child_cmd", nargs=argparse.REMAINDER)
        args = parser.parse_args(list(argv) if argv is not None else None)

        cmd = list(args.child_cmd)
        if cmd and cmd[0] == "--":
            cmd = cmd[1:]
        if not cmd:
            parser.error("Child MCP server command is required after `--`")

        return asyncio.run(
            _run_proxy(args.mode, cmd, proto_out, threads=args.threads)
        )
    finally:
        proto_out.close()


if __name__ == "__main__":
    raise SystemExit(main())
