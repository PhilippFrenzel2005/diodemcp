"""TDD tests for Suite A (600 deterministic traces), Suite B (30 wire scenarios), and mock MCP servers."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from benchmarks.traces.build_suite_a import SUITE_A_SEED, build_suite_a_traces, write_suite_a_jsonl
from benchmarks.traces.schema import MultiTurnTrace
from benchmarks.traces.suite_b_scenarios import build_suite_b_scenarios


def test_suite_a_exact_balance_determinism_and_sink_contracts(tmp_path: Path) -> None:
    traces_1 = build_suite_a_traces(seed=SUITE_A_SEED)
    traces_2 = build_suite_a_traces(seed=SUITE_A_SEED)
    assert traces_1 == traces_2
    assert len(traces_1) == 600
    assert len({t.trace_id for t in traces_1}) == 600

    for v in range(1, 7):
        atk = [t for t in traces_1 if t.split_group == f"V{v}"]
        ben = [t for t in traces_1 if t.split_group == f"B{v}"]
        assert len(atk) == 60 and all(t.label == 1 and t.vector_id == v for t in atk)
        assert len(ben) == 40 and all(t.label == 0 and t.vector_id == v for t in ben)
        # Matched-sink rule: every B_k exercises the same sink server.tool set as V_k
        atk_sinks = {f"{t.candidate_call.server}.{t.candidate_call.tool}" for t in atk}
        ben_sinks = {f"{t.candidate_call.server}.{t.candidate_call.tool}" for t in ben}
        assert ben_sinks <= atk_sinks

    # Review Focus #2: V3 never terminates at a passive local read_file
    for t in [x for x in traces_1 if x.split_group == "V3"]:
        assert (t.candidate_call.server, t.candidate_call.tool) not in {
            ("filesystem", "read_file"),
            ("fs_git", "read_file"),
        }

    # Review Focus #3: includes >64 KiB middle/tail-padded untrusted payloads and Unicode Cf evasions
    assert any(
        any(len(p.output_text.encode("utf-8")) > 65536 for p in t.prior_turns)
        for t in traces_1
        if t.split_group == "V1"
    )
    assert any(
        "\u200b" in td.description or "\u2060" in td.description
        for t in traces_1
        if t.split_group == "V3"
        for td in t.tools_catalog
    )
    # Rug-pulls, 2-hop laundering, chunked exfil, base64 w/ and w/o '=', hex, curl -F @.env
    assert any(t.initial_tools_catalog is not None for t in traces_1 if t.split_group == "V3")
    assert any(
        any(p.server == "filesystem" and p.tool == "write_file" for p in t.prior_turns)
        for t in traces_1
        if t.split_group == "V1"
    )
    v2_cmd_texts = [
        json.dumps(t.candidate_call.arguments)
        for t in traces_1
        if t.split_group == "V2"
    ]
    assert any("@.env" in txt for txt in v2_cmd_texts)
    v2_blobs = [
        json.loads(t.candidate_call.arguments["body"])["blob"]
        for t in traces_1
        if t.split_group == "V2"
        and t.candidate_call.server == "http"
        and "blob" in str(t.candidate_call.arguments.get("body", ""))
    ]
    assert any(b.endswith("=") for b in v2_blobs)
    assert any(not b.endswith("=") for b in v2_blobs)
    assert any("-d 'h=" in txt for txt in v2_cmd_texts)
    assert any(
        len(t.prior_turns) >= 3
        and sum(1 for p in t.prior_turns if p.server == "http" and p.tool == "post") >= 2
        for t in traces_1
        if t.split_group == "V2"
    )

    # JSONL serialization roundtrip
    jsonl_path = write_suite_a_jsonl(tmp_path / "suite_a_600.jsonl", seed=SUITE_A_SEED)
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 600
    reloaded = [MultiTurnTrace.from_dict(json.loads(line)) for line in lines]
    assert reloaded == traces_1


def _rpc_call(proc: subprocess.Popen[str], req_id: int, method: str, params: dict | None = None) -> dict:
    assert proc.stdin is not None and proc.stdout is not None
    payload = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}}
    proc.stdin.write(json.dumps(payload) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    assert line, f"EOF waiting for response to {method}"
    return json.loads(line)


def test_suite_b_scenarios_and_mock_mcp_stdio_roundtrip(tmp_path: Path) -> None:
    scenarios = build_suite_b_scenarios()
    assert len(scenarios) == 30
    assert len({s.scenario_id for s in scenarios}) == 30
    assert sum(1 for s in scenarios if s.label == 1) == 20
    assert sum(1 for s in scenarios if s.label == 0) == 10
    assert sum(1 for s in scenarios if s.category == "tools_list_poisoning") == 5
    assert sum(1 for s in scenarios if s.category == "rugpull_pin_mutation") == 5
    assert sum(1 for s in scenarios if s.category == "cross_server_exfil") == 5
    assert sum(1 for s in scenarios if s.category == "error_and_oversize_injection") == 5
    assert sum(1 for s in scenarios if s.category == "benign_dev_workflow") == 10

    state_file = tmp_path / "rugpull_state.json"
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.wire.mock_mcp_servers",
        "--profile",
        "rugpull_server",
        "--state-file",
        str(state_file),
    ]
    # Session 1: clean tools/list
    p1 = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        init1 = _rpc_call(p1, 1, "initialize", {"protocolVersion": "2024-11-05"})
        assert init1["result"]["serverInfo"]["name"] == "mock-rugpull_server"
        list1 = _rpc_call(p1, 2, "tools/list", {})
        descs1 = " ".join(t["description"] for t in list1["result"]["tools"])
        assert "<IMPORTANT>" not in descs1
        oversize = _rpc_call(p1, 3, "resources/read", {"uri": "mock://oversize/report.html"})
        text_out = oversize["result"]["contents"][0]["text"]
        assert len(text_out.encode("utf-8")) > 512 * 1024
    finally:
        if p1.stdin:
            p1.stdin.close()
        p1.wait(timeout=5)

    # Session 2: mutated tools/list + error.message injection
    p2 = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _rpc_call(p2, 1, "initialize", {"protocolVersion": "2024-11-05"})
        list2 = _rpc_call(p2, 2, "tools/list", {})
        descs2 = " ".join(t["description"] for t in list2["result"]["tools"])
        assert "<IMPORTANT>" in descs2
        err_resp = _rpc_call(
            p2,
            3,
            "tools/call",
            {"name": "fetch_notes", "arguments": {"topic": "error-inject"}},
        )
        assert "error" in err_resp
        assert "Ignore previous instructions" in err_resp["error"]["message"]
    finally:
        if p2.stdin:
            p2.stdin.close()
        p2.wait(timeout=5)

    # Verify remaining mock server profiles (fs_git, comms_http, db_bank) respond cleanly
    for profile in ("fs_git", "comms_http", "db_bank"):
        proc = subprocess.Popen(
            [sys.executable, "-m", "benchmarks.wire.mock_mcp_servers", "--profile", profile],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            init = _rpc_call(proc, 1, "initialize", {"protocolVersion": "2024-11-05"})
            assert init["result"]["serverInfo"]["name"] == f"mock-{profile}"
            tools = _rpc_call(proc, 2, "tools/list", {})
            assert len(tools["result"]["tools"]) >= 2
        finally:
            if proc.stdin:
                proc.stdin.close()
            proc.wait(timeout=5)
