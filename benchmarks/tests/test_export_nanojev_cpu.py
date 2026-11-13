"""Smoke tests for CPU INT8 ONNX export of NanoJev with >2 GiB staging and down_proj fallback."""

from __future__ import annotations

import json
from pathlib import Path

import onnx
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diodemodel")
from diodemcp.bundle import REQUIRED_BUNDLE_FILES, validate_bundle_dir
from diodemodel.export_bundle import (
    build_smoke_jev_tokenizer,
    export_fp32_onnx,
    read_onnx_logit_temperatures,
)
from diodemodel.train import (
    _build_tiny_smoke_nanojev,
    make_synthetic_curriculum_dataset,
    save_atomic_checkpoint,
)

from benchmarks.scripts.export_nanojev_cpu import (
    _quantize_fp32_to_single_int8,
    export_nanojev_with_outlier_fallback,
)


def _write_tiny_nanojev_checkpoint(
    tmp_path: Path,
    *,
    bfloat16: bool = False,
) -> tuple[Path, Path, Path, dict[str, int]]:
    tok, special_ids = build_smoke_jev_tokenizer()
    pad_id = int(special_ids["[PAD]"])
    model = _build_tiny_smoke_nanojev(vocab_size=64, pad_token_id=pad_id)
    if bfloat16:
        model = model.to(torch.bfloat16)
    ckpt_path = tmp_path / "checkpoint_best.pt"
    tok_path = tmp_path / "tokenizer.json"
    val_path = tmp_path / "val_calib.jsonl"

    tok.save(str(tok_path))
    rows = make_synthetic_curriculum_dataset(8, vocab_size=64, seed=13, pad_token_id=pad_id)
    val_path.write_text(
        "\n".join(json.dumps(r if isinstance(r, dict) else r.to_dict()) for r in rows) + "\n",
        encoding="utf-8",
    )

    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    save_atomic_checkpoint(
        ckpt_path,
        model=model,
        optimizer=opt,
        scheduler=None,
        epoch=1,
        stage=1,
        bucket_len=128,
        global_step=1,
        best_val_metric=0.0,
    )
    return ckpt_path, tok_path, val_path, special_ids


def test_export_nanojev_cpu_standard_and_down_proj_fallback(tmp_path: Path) -> None:
    ckpt_path, tok_path, val_path, _ = _write_tiny_nanojev_checkpoint(tmp_path, bfloat16=True)

    # (a) Standard per_channel=True export with no fallback triggered
    bundle_std = tmp_path / "bundle_std"
    res_std = export_nanojev_with_outlier_fallback(
        checkpoint_path=ckpt_path,
        tokenizer_path=tok_path,
        val_calib_path=val_path,
        output_bundle=bundle_std,
        model_id="DiodeMCP-NanoJev-0.6B-smoke",
        max_flip_rate=1.0,
        calib_max_rows=8,
        calib_buckets=(128, 256),
        intra_op_threads=2,
        staging_dir=tmp_path / "stage_std",
    )
    assert res_std["fallback_used"] is False
    assert sorted(p.name for p in bundle_std.iterdir()) == sorted(REQUIRED_BUNDLE_FILES)
    assert not list(bundle_std.glob("*.data"))
    manifest_std = validate_bundle_dir(bundle_std, verify_sha256=True)
    assert manifest_std.model_id == "DiodeMCP-NanoJev-0.6B-smoke"
    assert res_std["reused_fp32"] is False
    assert all(t >= 0.75 for t in res_std["temperatures"])
    baked_temps = read_onnx_logit_temperatures(bundle_std / "model_int8.onnx").tolist()
    assert all(abs(a - b) < 1e-5 for a, b in zip(baked_temps, res_std["temperatures"], strict=True))
    calib_std = json.loads((bundle_std / "calibration.json").read_text(encoding="utf-8"))
    assert calib_std["min_temp"] == 0.75
    assert calib_std["temperatures"] == res_std["temperatures"]

    # (a2) Reuse existing FP32 ONNX in staging_dir without touching checkpoint
    res_reuse = export_nanojev_with_outlier_fallback(
        checkpoint_path=None,
        tokenizer_path=tok_path,
        val_calib_path=val_path,
        output_bundle=bundle_std,
        model_id="DiodeMCP-NanoJev-0.6B-smoke-reuse",
        max_flip_rate=1.0,
        calib_max_rows=8,
        flip_max_rows=4,
        calib_buckets=(128, 256),
        intra_op_threads=2,
        staging_dir=tmp_path / "stage_std",
        reuse_fp32=True,
    )
    assert res_reuse["reused_fp32"] is True
    assert res_reuse["flip_eval_rows"] == 4
    assert validate_bundle_dir(bundle_std, verify_sha256=True).model_id == "DiodeMCP-NanoJev-0.6B-smoke-reuse"

    # (a3) Verify min_temp floor actively clamps, re-bakes ONNX, and passes bundle verification
    res_floor = export_nanojev_with_outlier_fallback(
        checkpoint_path=None,
        tokenizer_path=tok_path,
        val_calib_path=val_path,
        output_bundle=bundle_std,
        model_id="DiodeMCP-NanoJev-0.6B-smoke-floor",
        max_flip_rate=1.0,
        calib_max_rows=8,
        flip_max_rows=4,
        min_temp=2.0,
        calib_buckets=(128, 256),
        intra_op_threads=2,
        staging_dir=tmp_path / "stage_std",
        reuse_fp32=True,
    )
    assert all(t >= 2.0 for t in res_floor["temperatures"])
    baked_floor = read_onnx_logit_temperatures(bundle_std / "model_int8.onnx").tolist()
    assert all(abs(a - b) < 1e-5 for a, b in zip(baked_floor, res_floor["temperatures"], strict=True))
    calib_floor = json.loads((bundle_std / "calibration.json").read_text(encoding="utf-8"))
    assert calib_floor["min_temp"] == 2.0
    assert calib_floor["temperatures"] == res_floor["temperatures"]
    assert validate_bundle_dir(bundle_std, verify_sha256=True).model_id == "DiodeMCP-NanoJev-0.6B-smoke-floor"

    # (b) Force max_flip_rate=-1.0 to trigger mlp.down_proj exclusion fallback
    bundle_fb = tmp_path / "bundle_fb"
    res_fb = export_nanojev_with_outlier_fallback(
        checkpoint_path=ckpt_path,
        tokenizer_path=tok_path,
        val_calib_path=val_path,
        output_bundle=bundle_fb,
        model_id="DiodeMCP-NanoJev-0.6B-smoke-fb",
        max_flip_rate=-1.0,
        calib_max_rows=8,
        calib_buckets=(128, 256),
        intra_op_threads=2,
        staging_dir=tmp_path / "stage_fb",
    )
    assert res_fb["fallback_used"] is True
    assert any("down_proj" in name for name in res_fb["extra_excluded_nodes"])
    assert sorted(p.name for p in bundle_fb.iterdir()) == sorted(REQUIRED_BUNDLE_FILES)
    assert not list(bundle_fb.glob("*.data"))
    manifest_fb = validate_bundle_dir(bundle_fb, verify_sha256=True)
    assert manifest_fb.model_id == "DiodeMCP-NanoJev-0.6B-smoke-fb"


def test_quantize_fp32_to_single_int8_handles_external_data(tmp_path: Path) -> None:
    tok, special_ids = build_smoke_jev_tokenizer()
    pad_id = int(special_ids["[PAD]"])
    model = _build_tiny_smoke_nanojev(vocab_size=64, pad_token_id=pad_id)

    stage_dir = tmp_path / "ext_stage"
    fp32_path = stage_dir / "model_fp32.onnx"
    export_fp32_onnx(model, fp32_path)

    # Re-save FP32 ONNX using external tensor data to verify >2 GiB staging workflow.
    proto = onnx.load(str(fp32_path))
    onnx.save_model(
        proto,
        str(fp32_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.data",
        size_threshold=0,
    )
    assert (stage_dir / "weights.data").is_file()

    out_dir = tmp_path / "single_out"
    int8_path = out_dir / "model_int8.onnx"
    q_info, _ = _quantize_fp32_to_single_int8(fp32_path, int8_path, per_channel=True)
    assert q_info["quantized_matmul_count"] > 0
    assert sorted(p.name for p in out_dir.iterdir()) == ["model_int8.onnx"]
