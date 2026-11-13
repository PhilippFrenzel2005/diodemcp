"""Tests for repository-root `uv` packaging, SVG assets, README links, and CI workflow."""

from __future__ import annotations

import re
from pathlib import Path
import shutil
import subprocess
import tomllib
import xml.etree.ElementTree as ET
import zipfile

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_SVGS = (
    ".github/assets/hero-dark.svg",
    ".github/assets/hero-light.svg",
    ".github/assets/benchmark-dark.svg",
    ".github/assets/benchmark-light.svg",
    ".github/assets/plot-roc-pareto-dark.svg",
    ".github/assets/plot-roc-pareto-light.svg",
    ".github/assets/plot-vector-heatmap-dark.svg",
    ".github/assets/plot-vector-heatmap-light.svg",
    ".github/assets/plot-live-wire-dark.svg",
    ".github/assets/plot-live-wire-light.svg",
    ".github/assets/terminal-demo.svg",
)


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


def test_github_svg_assets_are_valid_xml_and_sanitizer_safe() -> None:
    for rel_path in EXPECTED_SVGS:
        svg_path = REPO_ROOT / rel_path
        assert svg_path.is_file(), f"Missing SVG asset: {rel_path}"
        tree = ET.parse(svg_path)
        root = tree.getroot()
        assert root.tag.endswith("svg"), f"Root tag is not <svg> in {rel_path}"
        assert "viewBox" in root.attrib, f"Missing viewBox on {rel_path}"
        for elem in root.iter():
            local_tag = elem.tag.split("}", 1)[-1].lower()
            assert local_tag not in {"script", "foreignobject"}, (
                f"GitHub-stripped tag <{local_tag}> found in {rel_path}"
            )


def test_readme_links_community_files_and_ci_workflow_are_complete() -> None:
    readme = REPO_ROOT / "README.md"
    assert readme.is_file()
    text = readme.read_text(encoding="utf-8")
    assert len(text.splitlines()) >= 140, "README.md should be a complete flagship repo landing page"
    assert "uv tool install" in text
    assert "<picture>" in text

    for req_file in ("CONTRIBUTING.md", "SECURITY.md", ".github/workflows/ci.yml"):
        fpath = REPO_ROOT / req_file
        assert fpath.is_file() and fpath.stat().st_size > 200, f"Missing or stub {req_file}"

    # Verify all relative links/assets in README.md resolve inside REPO_ROOT
    candidates: list[str] = []
    for match in re.finditer(r'(?:src|srcset|href)="([^"]+)"', text):
        candidates.append(match.group(1))
    for match in re.finditer(r"\]\(([^)]+)\)", text):
        candidates.append(match.group(1))

    for raw_ref in candidates:
        ref = raw_ref.strip().split()[0]
        if (
            not ref
            or ref.startswith(("#", "http://", "https://", "mailto:"))
        ):
            continue
        clean_ref = ref.split("#", 1)[0].split("?", 1)[0]
        if not clean_ref:
            continue
        target = (REPO_ROOT / clean_ref).resolve()
        assert target.exists(), f"Broken relative link/asset in README.md: {raw_ref!r}"

