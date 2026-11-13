"""Tests for repository-root `uv` packaging, SVG assets, README links, and CI workflow."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tomllib
import zipfile

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_root_pyproject_matches_zero_pytorch_contract_and_builds_wheel(
    tmp_path: Path,
) -> None:
    root_pyproject = REPO_ROOT / "pyproject.toml"
    assert root_pyproject.is_file(), "Root pyproject.toml must exist for `uv tool install git+...`"

    data = tomllib.loads(root_pyproject.read_text(encoding="utf-8"))
    project = data["project"]
    assert project["name"] == "diodemcp"
    assert project["scripts"]["diodemcp"] == "diodemcp.cli:main"

    deps_joined = " ".join(project["dependencies"]).lower()
    for forbidden in ("torch", "transformers", "scipy", "triton", "fcntl"):
        assert forbidden not in deps_joined

    uv_bin = shutil.which("uv")
    if uv_bin is None:
        pytest.skip("uv binary not installed in environment")

    out_dir = tmp_path / "dist"
    proc = subprocess.run(
        [uv_bin, "build", "--wheel", "--out-dir", str(out_dir), str(REPO_ROOT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"uv build --wheel failed:\n{proc.stderr}\n{proc.stdout}"

    wheels = list(out_dir.glob("diodemcp-*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as zf:
        names = set(zf.namelist())
        assert "diodemcp/__init__.py" in names
        assert "diodemcp/cli.py" in names
        assert "diodemcp/ort_predictor.py" in names
        assert "diodemcp/registry_data.toml" in names
