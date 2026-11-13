"""Unit tests for zero-leakage wire-protocol augmentation and Colab INT8 export upgrades."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer
from transformers import Qwen3Config

from diodemodel.contract import REQUIRED_BUNDLE_FILES, REQUIRED_SPECIAL_TOKENS, validate_bundle_dir
from diodemodel.dataset_packer import PACKED_PARQUET_SCHEMA, JevDatasetPacker, _row_to_parquet_dict
from diodemodel.export_bundle import export_diode_bundle, read_onnx_logit_temperatures
from diodemodel.modeling_jev import NanoJevCausalModel
from diodemodel.train import load_packed_jev_dataset
from diodemodel.wire_augment import (
    build_curated_retrain_parquet_splits,
    generate_wire_augmentation_pairs,
)

GOLDEN_TOKENIZER_PATH = Path(__file__).parent / "fixtures" / "golden_tokenizer.json"

# Benchmark-specific domains/servers/entities that must NEVER appear in synthetic training data
FORBIDDEN_BENCHMARK_SUBSTRINGS = (
    "acmecorp.internal",
    "evil-drop.net",
    "attacker-c2",
    "exfil-collector",
    "db_bank",
    "custom_sql_srv",
    "cfo-private-deal",
    "project-chimera",
    "iban_de89",
)


def test_wire_augmentation_families_and_zero_benchmark_overlap() -> None:
    train_pairs = generate_wire_augmentation_pairs(120, split_tag="train", rng_seed=20261113)
    val_pairs = generate_wire_augmentation_pairs(40, split_tag="val_calib", rng_seed=20269999)

    assert len(train_pairs) == 120
    assert len(val_pairs) == 40

    # Verify disjoint seed_id and repo_or_source across splits
    train_seeds = {p.seed_id for p in train_pairs}
    val_seeds = {p.seed_id for p in val_pairs}
    assert train_seeds.isdisjoint(val_seeds)

    families = {p.attack_family for p in train_pairs}
    assert any("catalog_poisoned_tool" in f for f in families)
    assert any("catalog_rug_pull_poisoned_tool" in f for f in families)
    assert any("rpc_error_injection" in f for f in families)
    assert any("headless_clean" in f for f in families)
    assert any("directory_email_lookup" in f for f in families)
    assert any("approved_vendor_payment" in f for f in families)
    assert any("sql_file_migration" in f for f in families)
    assert any("ci_metrics_webhook" in f for f in families)
    assert any(p.flipjev_applied for p in train_pairs)

    # Verify both missing-intent and explicit-intent coverage
    assert any(p.intent is None and p.intent_masked for p in train_pairs)
    assert any(p.intent is not None and not p.intent_masked for p in train_pairs)

    # Verify zero overlap with competitive benchmark strings
    for p in (*train_pairs, *val_pairs):
        blob = json.dumps(p.model_dump()).lower()
        for forbidden in FORBIDDEN_BENCHMARK_SUBSTRINGS:
            assert forbidden not in blob, f"Forbidden benchmark token {forbidden!r} found in {p.pair_id}"


def test_build_curated_retrain_parquet_splits_passes_audit_and_loads(tmp_path: Path) -> None:
    tok = Tokenizer.from_file(str(GOLDEN_TOKENIZER_PATH))
    sp_ids = {t: int(tok.token_to_id(t)) for t in REQUIRED_SPECIAL_TOKENS}

    from diodemodel.contract import BundleManifest

    manifest = BundleManifest(
        model_id="test-base-seed",
        sha256_model_onnx="0" * 64,
        sha256_tokenizer_json="1" * 64,
        sha256_calibration_json="2" * 64,
        special_token_ids=sp_ids,
        ece_val=0.01,
    )
    packer = JevDatasetPacker(tok, manifest)

    # Build small synthetic base train & val parquets
    base_tr = generate_wire_augmentation_pairs(30, split_tag="base_tr", rng_seed=11)
    base_va = generate_wire_augmentation_pairs(12, split_tag="base_va", rng_seed=22)

    base_train_pq = tmp_path / "base_train.parquet"
    base_val_pq = tmp_path / "base_val.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                _row_to_parquet_dict(p, packer.pack_pair(p, max_bucket=1024, pad_to_max_bucket=False), "train")
                for p in base_tr
            ],
            schema=PACKED_PARQUET_SCHEMA,
        ),
        str(base_train_pq),
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                _row_to_parquet_dict(p, packer.pack_pair(p, max_bucket=1024, pad_to_max_bucket=False), "val_calib")
                for p in base_va
            ],
            schema=PACKED_PARQUET_SCHEMA,
        ),
        str(base_val_pq),
    )

    out_dir = tmp_path / "retrain_splits"
    summary = build_curated_retrain_parquet_splits(
        tokenizer_json_path=GOLDEN_TOKENIZER_PATH,
        base_train_parquet=base_train_pq,
        base_val_parquet=base_val_pq,
        out_dir=out_dir,
        base_train_sample=20,
        aug_train_count=40,
        base_val_sample=8,
        aug_val_count=16,
        max_bucket=2048,
        rng_seed=20261113,
    )

    assert summary["audit_leakage_count"] == 0
    assert summary["audit_real_secret_hits"] == 0
    assert summary["train_total"] == 60
    assert summary["val_calib_total"] == 24

    loaded_train = load_packed_jev_dataset(summary["train_parquet"])
    loaded_val = load_packed_jev_dataset(summary["val_calib_parquet"])
    assert len(loaded_train) == 60
    assert len(loaded_val) == 24


def test_export_diode_bundle_bf16_min_temp_and_down_proj_fallback(tmp_path: Path) -> None:
    tok = Tokenizer.from_file(str(GOLDEN_TOKENIZER_PATH))
    special_ids = {t: int(tok.token_to_id(t)) for t in REQUIRED_SPECIAL_TOKENS}
    vocab_size = max(int(tok.get_vocab_size()), max(special_ids.values()) + 1, 64)
    pad_id = int(special_ids["[PAD]"])

    torch.manual_seed(99)
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
    bf16_model = NanoJevCausalModel.from_config(cfg).to(torch.bfloat16).eval()

    from diodemodel.train import make_synthetic_curriculum_dataset

    val_rows = make_synthetic_curriculum_dataset(
        16, vocab_size=vocab_size, seed=77, pad_token_id=pad_id
    )
    bundle_dir = tmp_path / "bf16_fallback_bundle"
    res = export_diode_bundle(
        bf16_model,
        bundle_dir,
        tokenizer_source=GOLDEN_TOKENIZER_PATH,
        model_id="DiodeMCP-NanoJev-0.6B",
        val_calib_rows=val_rows,
        special_token_ids=special_ids,
        calib_buckets=(128, 256),
        calib_max_rows=10,
        flip_max_rows=6,
        min_temp=1.75,
        max_flip_rate=-1.0,  # Force down_proj fallback path
        reduce_range=True,
        intra_op_threads=2,
    )

    # Model dtype restored to BF16 after export
    assert next(bf16_model.parameters()).dtype == torch.bfloat16
    assert res["fallback_used"] is True
    assert any("down_proj" in name for name in res["quantization"]["extra_excluded_nodes"])
    assert sorted(p.name for p in bundle_dir.iterdir()) == sorted(REQUIRED_BUNDLE_FILES)

    manifest = validate_bundle_dir(bundle_dir, verify_sha256=True)
    assert manifest.model_id == "DiodeMCP-NanoJev-0.6B"

    baked = read_onnx_logit_temperatures(bundle_dir / "model_int8.onnx")
    assert all(float(t) >= 1.75 - 1e-5 for t in baked.tolist())
    calib_json = json.loads((bundle_dir / "calibration.json").read_text(encoding="utf-8"))
    assert calib_json["down_proj_fallback_used"] is True
    assert calib_json["reduce_range"] is True
    assert calib_json["num_calibration_rows"] == 10
