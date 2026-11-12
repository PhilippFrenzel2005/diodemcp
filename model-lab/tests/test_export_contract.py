"""Task A6 tests: opset-17 ONNX export, MatMul/Gemm-only INT8 quantization, post-INT8 temperature scaling,
DiodeModelBundle v1 packaging, benchmark suite, and cross-track Track B OrtJevPredictor smoke test.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
import pytest
import torch
from tokenizers import Tokenizer
from transformers import ModernBertConfig, Qwen3Config

from diodemodel.contract import (
    ONNX_INPUT_SPECS,
    ONNX_OPSET_VERSION,
    ONNX_OUTPUT_SPECS,
    REQUIRED_BUNDLE_FILES,
    REQUIRED_SPECIAL_TOKENS,
    ROLE_PAD,
    compute_sha256,
    validate_bundle_dir,
    validate_onnx_contract,
)
from diodemodel.losses_rlcd import fit_temperature_scaling, probs_to_logits
from diodemodel.modeling_jev import (
    HEAD_SCOPE_NAMES,
    ModernBERTJevModel,
    NanoJevCausalModel,
)
from diodemodel.train import (
    JevTrainerConfig,
    collate_jev_batch,
    make_synthetic_curriculum_dataset,
    run_curriculum_training,
)

GOLDEN_TOKENIZER_PATH = Path(__file__).parent / "fixtures" / "golden_tokenizer.json"
TRACK_B_SRC = Path("/root/diodemcp/packages/diodemcp/src")
INPUT_NAMES = tuple(ONNX_INPUT_SPECS.keys())
OUTPUT_NAMES = tuple(ONNX_OUTPUT_SPECS.keys())


def _golden_tokenizer_info() -> tuple[Tokenizer, dict[str, int], int, int]:
    tok = Tokenizer.from_file(str(GOLDEN_TOKENIZER_PATH))
    special_ids = {t: int(tok.token_to_id(t)) for t in REQUIRED_SPECIAL_TOKENS}
    vocab_size = max(int(tok.get_vocab_size()), max(special_ids.values()) + 1, 64)
    pad_id = int(special_ids["[PAD]"])
    return tok, special_ids, vocab_size, pad_id


def _tiny_modernbert(vocab_size: int = 64, pad_id: int = 0) -> ModernBERTJevModel:
    torch.manual_seed(2026)
    cfg = ModernBertConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        local_attention=8,
        max_position_embeddings=4096,
        pad_token_id=pad_id,
        bos_token_id=1,
        eos_token_id=2,
        cls_token_id=1,
        sep_token_id=2,
    )
    return ModernBERTJevModel.from_config(cfg).eval()


def _tiny_nanojev(vocab_size: int = 64, pad_id: int = 0) -> NanoJevCausalModel:
    torch.manual_seed(2027)
    cfg = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=4096,
        pad_token_id=pad_id,
    )
    return NanoJevCausalModel.from_config(cfg).eval()


MODEL_BUILDERS = {
    "modernbert": ("DiodeMCP-Jev-Base-149M", _tiny_modernbert),
    "nanojev": ("DiodeMCP-NanoJev-0.6B", _tiny_nanojev),
}


def _repad_batch(batch: dict[str, Any], new_len: int, pad_id: int = 0) -> dict[str, Any]:
    out = dict(batch)
    cur_len = int(batch["input_ids"].shape[1])
    assert new_len >= cur_len
    pad_len = new_len - cur_len
    for key, fill_val in (("input_ids", pad_id), ("seg_ids", 0), ("token_roles", ROLE_PAD)):
        out[key] = torch.nn.functional.pad(batch[key], (0, pad_len), value=fill_val)
    return out


@pytest.mark.parametrize("arch", ["modernbert", "nanojev"])
def test_export_diode_bundle_contract_and_int8_quantization_invariants(tmp_path: Path, arch: str) -> None:
    from diodemodel.export_bundle import (
        collect_excluded_onnx_nodes,
        export_diode_bundle,
        verify_onnx_no_bool_where,
    )

    _, special_ids, vocab_size, pad_id = _golden_tokenizer_info()
    model_id, builder = MODEL_BUILDERS[arch]
    model = builder(vocab_size=vocab_size, pad_id=pad_id)
    val_rows = make_synthetic_curriculum_dataset(
        16, vocab_size=vocab_size, seed=101, pad_token_id=pad_id
    )

    bundle_dir = tmp_path / f"bundle_{arch}"
    result = export_diode_bundle(
        model,
        bundle_dir,
        tokenizer_source=GOLDEN_TOKENIZER_PATH,
        model_id=model_id,
        val_calib_rows=val_rows,
        special_token_ids=special_ids,
        calib_buckets=(128, 256, 512),
    )

    for req_file in REQUIRED_BUNDLE_FILES:
        assert (bundle_dir / req_file).is_file(), f"Missing required bundle file: {req_file}"

    # 1. Full DiodeModelBundle v1 validation (manifest, SHA256s, tokenizer special tokens, ONNX contract)
    manifest = validate_bundle_dir(bundle_dir, verify_sha256=True)
    assert manifest.model_id == model_id
    assert manifest.sha256_model_onnx == compute_sha256(bundle_dir / "model_int8.onnx")
    assert manifest.sha256_tokenizer_json == compute_sha256(bundle_dir / "tokenizer.json")
    assert manifest.sha256_calibration_json == compute_sha256(bundle_dir / "calibration.json")

    raw_manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "sha256_model_onnx" in raw_manifest
    assert "sha256_model_int8_onnx" not in raw_manifest

    # 2. ONNX opset=17, exact 8 inputs + 3 outputs, static widths pinned, zero bool Where ops
    onnx_path = bundle_dir / "model_int8.onnx"
    validate_onnx_contract(onnx_path)
    proto = onnx.load(str(onnx_path))
    default_opsets = [imp.version for imp in proto.opset_import if imp.domain in ("", "ai.onnx")]
    assert default_opsets == [ONNX_OPSET_VERSION]

    init_names = {i.name for i in proto.graph.initializer}
    graph_inputs = [inp.name for inp in proto.graph.input if inp.name not in init_names]
    graph_outputs = [out.name for out in proto.graph.output]
    assert tuple(graph_inputs) == INPUT_NAMES
    assert tuple(graph_outputs) == OUTPUT_NAMES

    verify_onnx_no_bool_where(proto)

    # 3. DynamicQuantizeLinear / MatMulInteger ONLY on backbone MatMul/Gemm weight projections;
    # LayerNorm, Softmax, 4D mask ops, AND all head nodes under HEAD_SCOPE_NAMES stay pure FP32!
    op_types = {n.op_type for n in proto.graph.node}
    assert "DynamicQuantizeLinear" in op_types
    assert "MatMulInteger" in op_types
    assert "QLinearConv" not in op_types
    assert "QLinearAdd" not in op_types

    # Verify INT8 weight initializers are present
    assert any(init.data_type == TensorProto.INT8 for init in proto.graph.initializer)

    matmul_int_nodes = [n for n in proto.graph.node if n.op_type == "MatMulInteger"]
    dql_nodes = [n for n in proto.graph.node if n.op_type == "DynamicQuantizeLinear"]
    assert len(matmul_int_nodes) >= 8
    assert len(dql_nodes) >= 1

    # Every MatMulInteger node must belong to /backbone/ and NEVER to HEAD_SCOPE_NAMES or logit_temperatures
    for q_node in matmul_int_nodes:
        assert q_node.name.startswith("/backbone/"), f"Non-backbone MatMul quantized: {q_node.name}"
        for scope in (*HEAD_SCOPE_NAMES, "logit_temperatures"):
            assert f"/{scope}" not in q_node.name and not q_node.name.startswith(scope)

    # Verify head scopes and Softmax remain FP32
    head_nodes = [
        n
        for n in proto.graph.node
        if any(f"/{s}" in n.name or n.name.startswith(f"{s}/") for s in HEAD_SCOPE_NAMES)
    ]
    assert len(head_nodes) > 0
    for hn in head_nodes:
        assert hn.op_type not in ("MatMulInteger", "DynamicQuantizeLinear", "QGemm", "QLinearMatMul"), (
            f"Head node {hn.name} was quantized ({hn.op_type})"
        )

    excluded = result["quantization"]["nodes_to_exclude"]
    for scope in HEAD_SCOPE_NAMES:
        assert any(f"/{scope}" in name for name in excluded), f"Scope {scope} missing from nodes_to_exclude"

    # Also verify collect_excluded_onnx_nodes is deterministic
    assert len(collect_excluded_onnx_nodes(proto)) > 0


def test_verify_onnx_no_bool_where_rejects_bool_where_graph() -> None:
    from diodemodel.export_bundle import verify_onnx_no_bool_where

    cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, ["B", 4])
    x = helper.make_tensor_value_info("x", TensorProto.BOOL, ["B", 4])
    y = helper.make_tensor_value_info("y", TensorProto.BOOL, ["B", 4])
    out = helper.make_tensor_value_info("out", TensorProto.BOOL, ["B", 4])
    node = helper.make_node("Where", ["cond", "x", "y"], ["out"], name="/bad_bool_where")
    graph = helper.make_graph([node], "BadBoolWhere", [cond, x, y], [out])
    bad_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    with pytest.raises(ValueError, match="Where.*BOOL|bool.*Where"):
        verify_onnx_no_bool_where(bad_model)


@pytest.mark.parametrize("arch", ["modernbert", "nanojev"])
def test_int8_probability_drift_and_padding_bucket_invariance(tmp_path: Path, arch: str) -> None:
    from diodemodel.export_bundle import export_diode_bundle

    _, special_ids, vocab_size, pad_id = _golden_tokenizer_info()
    model_id, builder = MODEL_BUILDERS[arch]
    model = builder(vocab_size=vocab_size, pad_id=pad_id)
    val_rows = make_synthetic_curriculum_dataset(
        16, vocab_size=vocab_size, seed=202, pad_token_id=pad_id
    )

    bundle_dir = tmp_path / f"drift_{arch}"
    export_diode_bundle(
        model,
        bundle_dir,
        tokenizer_source=GOLDEN_TOKENIZER_PATH,
        model_id=model_id,
        val_calib_rows=val_rows,
        special_token_ids=special_ids,
        calib_buckets=(128, 256, 512),
    )

    sess = ort.InferenceSession(
        str(bundle_dir / "model_int8.onnx"), providers=["CPUExecutionProvider"]
    )

    # 1. Probability drift between FP32 PyTorch (at matching logit_temperatures) and INT8 ORT < 0.015 (< 0.02 max)
    test_rows = make_synthetic_curriculum_dataset(
        12, vocab_size=vocab_size, seed=203, pad_token_id=pad_id
    )
    batch_128 = collate_jev_batch(test_rows[:6], bucket_len=128, pad_token_id=pad_id)
    batch_256_direct = collate_jev_batch(test_rows[6:], bucket_len=256, pad_token_id=pad_id)

    for b in (batch_128, batch_256_direct):
        with torch.no_grad():
            pt_outs = model(*[b[name] for name in INPUT_NAMES])
        ort_outs = sess.run(list(OUTPUT_NAMES), {name: b[name].numpy() for name in INPUT_NAMES})
        for out_name, pt_t, ort_arr in zip(OUTPUT_NAMES, pt_outs, ort_outs, strict=True):
            assert ort_arr.dtype == np.float32
            max_drift = float(np.abs(pt_t.numpy() - ort_arr).max())
            assert max_drift < 0.015, (
                f"{arch} {out_name} INT8 probability drift {max_drift:.5f} >= 0.015"
            )

    # 2. Padding-bucket invariance (128 vs 256 padded ROLE_PAD tail) < 1e-4 across all 3 outputs
    batch_128_repadded_256 = _repad_batch(batch_128, 256, pad_id=pad_id)
    ort_128 = sess.run(list(OUTPUT_NAMES), {name: batch_128[name].numpy() for name in INPUT_NAMES})
    ort_256 = sess.run(
        list(OUTPUT_NAMES), {name: batch_128_repadded_256[name].numpy() for name in INPUT_NAMES}
    )
    for out_name, arr_128, arr_256 in zip(OUTPUT_NAMES, ort_128, ort_256, strict=True):
        pad_diff = float(np.abs(arr_128 - arr_256).max())
        assert pad_diff < 1e-4, (
            f"{arch} {out_name} padding invariance 128 vs 256 diff {pad_diff:.6f} >= 1e-4"
        )


@pytest.mark.parametrize("arch", ["modernbert", "nanojev"])
def test_post_int8_temperature_scaling_bakes_temperatures_and_updates_manifest_sha256(
    tmp_path: Path, arch: str
) -> None:
    from diodemodel.export_bundle import (
        export_diode_bundle,
        read_onnx_logit_temperatures,
    )

    _, special_ids, vocab_size, pad_id = _golden_tokenizer_info()
    model_id, builder = MODEL_BUILDERS[arch]
    model = builder(vocab_size=vocab_size, pad_id=pad_id)

    # Amplify head weights slightly so pre-temperature logits have strong signal and start with non-unit T_prev
    with torch.no_grad():
        model.noul_head.out_weight.mul_(4.0)
        model.noul_head.out_bias.copy_(torch.tensor([1.2, -1.0, 0.8], dtype=torch.float32))
        t_prev = torch.tensor([2.4, 0.55, 1.85], dtype=torch.float32)
        model.set_logit_temperatures(t_prev)

    val_rows = make_synthetic_curriculum_dataset(
        24, vocab_size=vocab_size, seed=305, pad_token_id=pad_id
    )
    bundle_dir = tmp_path / f"calib_{arch}"
    res = export_diode_bundle(
        model,
        bundle_dir,
        tokenizer_source=GOLDEN_TOKENIZER_PATH,
        model_id=model_id,
        val_calib_rows=val_rows,
        special_token_ids=special_ids,
        thresholds={"warn_min": 0.18, "critical_min": 0.82},
        calib_buckets=(128, 256, 512),
    )

    onnx_path = bundle_dir / "model_int8.onnx"
    baked_temps = read_onnx_logit_temperatures(onnx_path)
    fitted_temps = res["fitted_temperatures"]
    np.testing.assert_allclose(baked_temps, fitted_temps.numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        model.logit_temperatures.detach().cpu().numpy(),
        fitted_temps.numpy(),
        rtol=1e-6,
        atol=1e-6,
    )

    # Verify int8_raw_logits came from the quantized INT8 ORT session and refitting matches fit_temperature_scaling
    int8_raw_logits = res["int8_raw_logits"]
    val_targets = res["val_targets"]
    expected_fitted = fit_temperature_scaling(int8_raw_logits, val_targets)
    torch.testing.assert_close(fitted_temps, expected_fitted, rtol=1e-5, atol=1e-5)

    # Re-run the baked INT8 ORT session on val_rows and verify noul_probs == sigmoid(int8_raw_logits / T_fitted) < 1e-5
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    post_probs_list: list[np.ndarray] = []
    for i in range(0, len(val_rows), 16):
        chunk = val_rows[i : i + 16]
        b_len = max(int(r["unpadded_length"]) for r in chunk)
        bucket = 128 if b_len <= 128 else (256 if b_len <= 256 else 512)
        batch = collate_jev_batch(chunk, bucket_len=bucket, pad_token_id=pad_id)
        noul_p, _, _ = sess.run(
            list(OUTPUT_NAMES), {name: batch[name].numpy() for name in INPUT_NAMES}
        )
        post_probs_list.append(noul_p)
    ort_post_probs = torch.from_numpy(np.concatenate(post_probs_list, axis=0))
    expected_post_probs = torch.sigmoid(int8_raw_logits / fitted_temps)
    max_prob_err = float((ort_post_probs - expected_post_probs).abs().max().item())
    assert max_prob_err < 1e-5, f"Post-baking ORT probs diverged from sigmoid(z_int8 / T_fitted) by {max_prob_err}"

    #Also verify probs_to_logits(ort_post_probs, fitted_temps) recovers int8_raw_logits
    recovered = probs_to_logits(ort_post_probs, fitted_temps)
    torch.testing.assert_close(recovered, int8_raw_logits, rtol=2e-3, atol=2e-3)

    # Verify calibration.json structure and contents
    calib_json = json.loads((bundle_dir / "calibration.json").read_text(encoding="utf-8"))
    assert calib_json["method"] == "temperature_scaling"
    assert calib_json["post_int8_calibrated"] is True
    assert calib_json["num_calibration_rows"] == len(val_rows)
    assert np.allclose(calib_json["previous_temperatures"], t_prev.tolist(), atol=1e-5)
    assert np.allclose(calib_json["temperatures"], fitted_temps.tolist(), atol=1e-5)
    assert set(calib_json["temperatures_by_head"].keys()) == {"T_enc", "T_sem", "T_hijack"}
    for metric_key in (
        "ece_before",
        "ece_after",
        "brier_before",
        "brier_after",
        "nll_before",
        "nll_after",
    ):
        assert len(calib_json[metric_key]) == 3
    assert 0.0 <= calib_json["ece_val"] <= 1.0
    assert 0.0 <= calib_json["brier_val"] <= 1.0
    assert calib_json["thresholds"] == {"warn_min": 0.18, "critical_min": 0.82}

    # Verify manifest.json SHA256s and ece_val were computed AFTER baking temperatures
    manifest = validate_bundle_dir(bundle_dir, verify_sha256=True)
    assert manifest.ece_val == pytest.approx(calib_json["ece_val"], abs=1e-6)
    assert manifest.sha256_model_onnx == compute_sha256(onnx_path)
    assert manifest.sha256_calibration_json == compute_sha256(bundle_dir / "calibration.json")


def test_export_from_saved_checkpoint_and_cli_smoke(tmp_path: Path) -> None:
    from diodemodel.export_bundle import export_diode_bundle, main as export_cli_main

    _, special_ids, vocab_size, pad_id = _golden_tokenizer_info()
    model = _tiny_modernbert(vocab_size=vocab_size, pad_id=pad_id)
    rows = make_synthetic_curriculum_dataset(12, vocab_size=vocab_size, seed=404, pad_token_id=pad_id)

    ckpt_dir = tmp_path / "ckpt_modernbert"
    cfg = JevTrainerConfig(
        curriculum_buckets=(128,),
        batch_size=6,
        stage1_epochs=1,
        stage2_epochs=1,
        max_steps_per_bucket=1,
        pad_token_id=pad_id,
        checkpoint_dir=ckpt_dir,
    )
    run_curriculum_training(model, rows, val_rows=rows, config=cfg)
    ckpt_file = ckpt_dir / "checkpoint_best.pt"
    assert ckpt_file.is_file()

    bundle_dir = tmp_path / "bundle_from_ckpt"
    export_diode_bundle(
        ckpt_file,
        bundle_dir,
        tokenizer_source=GOLDEN_TOKENIZER_PATH,
        model_id="DiodeMCP-Jev-Base-149M",
        val_calib_rows=rows,
        special_token_ids=special_ids,
        calib_buckets=(128, 256),
    )
    manifest = validate_bundle_dir(bundle_dir, verify_sha256=True)
    assert manifest.model_id == "DiodeMCP-Jev-Base-149M"

    # CLI smoke test
    cli_out_dir = tmp_path / "cli_bundles"
    rc = export_cli_main(["--smoke", "--output-dir", str(cli_out_dir)])
    assert rc == 0
    for sub in ("DiodeMCP-Jev-Base-149M", "DiodeMCP-NanoJev-0.6B"):
        validate_bundle_dir(cli_out_dir / sub, verify_sha256=True)


def test_benchmark_suite_metrics_latency_and_comparison_table(tmp_path: Path) -> None:
    from diodemodel.benchmark_suite import (
        format_benchmark_comparison_table,
        main as bench_cli_main,
        run_bundle_benchmark,
        run_multi_bundle_benchmark,
    )
    from diodemodel.export_bundle import export_diode_bundle

    _, special_ids, vocab_size, pad_id = _golden_tokenizer_info()
    val_rows = make_synthetic_curriculum_dataset(
        16, vocab_size=vocab_size, seed=501, pad_token_id=pad_id
    )
    trifecta_rows = make_synthetic_curriculum_dataset(
        16, vocab_size=vocab_size, seed=502, pad_token_id=pad_id
    )

    bundle_dirs: list[Path] = []
    for arch in ("modernbert", "nanojev"):
        model_id, builder = MODEL_BUILDERS[arch]
        m = builder(vocab_size=vocab_size, pad_id=pad_id)
        b_dir = tmp_path / model_id
        export_diode_bundle(
            m,
            b_dir,
            tokenizer_source=GOLDEN_TOKENIZER_PATH,
            model_id=model_id,
            val_calib_rows=val_rows,
            special_token_ids=special_ids,
            calib_buckets=(128, 256, 512),
        )
        bundle_dirs.append(b_dir)

    single_report_json = tmp_path / "base_bench.json"
    rep = run_bundle_benchmark(
        bundle_dirs[0],
        val_rows=val_rows,
        trifecta_rows=trifecta_rows,
        latency_buckets=(512, 2048),
        num_latency_runs=3,
        warmup_runs=1,
        output_json_path=single_report_json,
    )
    assert single_report_json.is_file()
    assert rep["model_id"] == "DiodeMCP-Jev-Base-149M"

    for split_key in ("val_calib", "trifecta_test"):
        split_metrics = rep[split_key]
        for cat in ("T1", "T2", "T3", "T4", "T5", "T6", "T7"):
            assert cat in split_metrics["per_threat_recall"]
            assert 0.0 <= split_metrics["per_threat_recall"][cat] <= 1.0
        assert 0.0 <= split_metrics["attack_recall"] <= 1.0
        assert 0.0 <= split_metrics["flipjev_attack_recall"] <= 1.0
        assert 0.0 <= split_metrics["flipjev_benign_dos_fpr"] <= 1.0
        assert 0.0 <= split_metrics["trifecta_fpr"] <= 1.0
        assert 0.0 <= split_metrics["trifecta_safe_fpr"] <= 1.0
        assert 0.0 <= split_metrics["src_pointer_top1_acc"] <= 1.0
        assert 0.0 <= split_metrics["arg_pointer_top1_acc"] <= 1.0
        assert 0.0 <= split_metrics["pointer_top1_acc"] <= 1.0
        assert 0.0 <= split_metrics["ece_val"] <= 1.0
        assert 0.0 <= split_metrics["brier"] <= 1.0

    # Latency at 512 and 2048 buckets
    lat = rep["cpu_latency"]
    for b_key in ("512", "2048"):
        assert b_key in lat
        assert lat[b_key]["p50_ms"] > 0.0
        assert lat[b_key]["p95_ms"] >= lat[b_key]["p50_ms"]
    assert rep["latency_512_p50_ms"] > 0.0
    assert rep["latency_512_p95_ms"] > 0.0
    assert rep["latency_2048_p50_ms"] > 0.0
    assert rep["latency_2048_p95_ms"] > 0.0

    # Multi-bundle benchmark + comparison table
    multi_out = run_multi_bundle_benchmark(
        bundle_dirs,
        val_rows=val_rows,
        trifecta_rows=trifecta_rows,
        latency_buckets=(512, 2048),
        num_latency_runs=2,
        warmup_runs=1,
    )
    assert len(multi_out["reports"]) == 2
    table_md = format_benchmark_comparison_table(multi_out["reports"])
    assert "DiodeMCP-Jev-Base-149M" in table_md
    assert "DiodeMCP-NanoJev-0.6B" in table_md
    assert "FlipJev" in table_md
    assert "Trifecta" in table_md
    assert "512" in table_md and "2048" in table_md

    # Benchmark CLI test
    cli_json = tmp_path / "cli_bench.json"
    cli_md = tmp_path / "cli_bench.md"
    rc = bench_cli_main(
        [
            "--smoke",
            "--output-json",
            str(cli_json),
            "--output-markdown",
            str(cli_md),
        ]
    )
    assert rc == 0
    assert cli_json.is_file()
    assert cli_md.is_file()


@pytest.mark.parametrize("arch", ["modernbert", "nanojev"])
def test_cross_track_ort_jev_predictor_end_to_end_on_exported_bundles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arch: str
) -> None:
    if not TRACK_B_SRC.is_dir():
        pytest.skip("Track B source directory not present in workspace")

    import sys
    import types

    from diodemodel.export_bundle import export_diode_bundle

    _, special_ids, vocab_size, pad_id = _golden_tokenizer_info()
    model_id, builder = MODEL_BUILDERS[arch]
    model = builder(vocab_size=vocab_size, pad_id=pad_id)
    val_rows = make_synthetic_curriculum_dataset(
        12, vocab_size=vocab_size, seed=601, pad_token_id=pad_id
    )

    bundle_dir = tmp_path / f"cross_track_{arch}"
    export_diode_bundle(
        model,
        bundle_dir,
        tokenizer_source=GOLDEN_TOKENIZER_PATH,
        model_id=model_id,
        val_calib_rows=val_rows,
        special_token_ids=special_ids,
        calib_buckets=(128, 256, 512),
    )

    monkeypatch.syspath_prepend(str(TRACK_B_SRC))
    if "diodemcp" not in sys.modules:
        pkg_stub = types.ModuleType("diodemcp")
        pkg_stub.__path__ = [str(TRACK_B_SRC / "diodemcp")]  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "diodemcp", pkg_stub)

    from diodemcp.ledger import LedgerSnapshot
    from diodemcp.ort_predictor import OrtJevPredictor
    from diodemcp.types import CandidateCall, CapabilityFlag, DecisionScores, TaintSpan

    predictor = OrtJevPredictor(bundle_dir, intra_op_threads=2)
    assert predictor.manifest.model_id == model_id

    snapshot = LedgerSnapshot(
        intent="Summarize repository status and open an issue if tests fail",
        private_spans=(
            TaintSpan(
                span_id="priv_0",
                source_label="vault.read",
                text="confidential acquisition valuation quarterly churn",
                flags=CapabilityFlag.READS_PRIVATE,
            ),
        ),
        untrusted_spans=(
            TaintSpan(
                span_id="untrust_0",
                source_label="web.fetch",
                text="external release notes and public changelog",
                flags=CapabilityFlag.READS_UNTRUSTED,
            ),
        ),
    )
    candidate_call = CandidateCall(
        server="github",
        tool="create_issue",
        arguments={"repo": "acme/monorepo", "title": "Status", "body": "unit tests summary"},
    )

    scores_snap = predictor.evaluate_snapshot(snapshot, candidate_call)
    scores_eval = predictor.evaluate(
        intent=snapshot.intent,
        private_spans=list(snapshot.private_spans),
        untrusted_spans=list(snapshot.untrusted_spans),
        call=candidate_call,
    )
    for scores in (scores_snap, scores_eval):
        assert isinstance(scores, DecisionScores)
        assert 0.0 <= scores.p_enc <= 1.0
        assert 0.0 <= scores.p_sem <= 1.0
        assert 0.0 <= scores.p_hijack <= 1.0
        assert 0.0 <= scores.p_max <= 1.0
    assert scores_snap == scores_eval


def test_notebook_03_export_and_bench_structure_and_cells() -> None:
    nb_path = Path(__file__).resolve().parents[1] / "notebooks" / "03_export_and_bench.ipynb"
    assert nb_path.is_file(), f"Missing Colab export & benchmark notebook: {nb_path}"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    cells = nb["cells"]
    assert len(cells) >= 6

    code_sources: list[str] = []
    for cell in cells:
        if cell["cell_type"] == "code":
            raw_lines = (
                cell["source"]
                if isinstance(cell["source"], list)
                else cell["source"].splitlines(True)
            )
            clean_lines = [line for line in raw_lines if not line.lstrip().startswith("!")]
            src = "".join(clean_lines)
            ast.parse(src)
            code_sources.append("".join(raw_lines))

    joined = "\n".join(code_sources)
    for expected in (
        "/content/drive",
        "model_lab_src.tar.gz",
        "/content/drive/MyDrive/diodemcp/bundles",
        "DiodeMCP-Jev-Base-149M",
        "DiodeMCP-Jev-Large-395M",
        "DiodeMCP-NanoJev-0.6B",
        "export_diode_bundle",
        "validate_bundle_dir",
        "run_bundle_benchmark",
        "format_benchmark_comparison_table",
        "val_calib",
        "trifecta",
    ):
        assert expected in joined, f"Expected {expected!r} in 03_export_and_bench.ipynb"
