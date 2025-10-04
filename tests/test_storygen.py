import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG_PY = ROOT / "story_app.py"
def _resolve_storygen_exe():
    exe = shutil.which("storygen")
    if exe:
        return exe
    candidate = Path(sys.executable).with_name("storygen")
    if candidate.exists():
        return str(candidate)
    return None

ENTRY = _resolve_storygen_exe()


def run_cli(args):
    exe = ENTRY or "storygen"
    cmd = [exe] + args
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(ROOT))


def test_offline_default_and_help():
    if ENTRY is None:
        pytest.skip("storygen entrypoint not found in PATH or venv bin")
    out = run_cli(["--help"])  # no crash
    assert out.returncode == 0
    assert "Generate a short story" in out.stdout


def test_deterministic_generation_and_markdown_export(tmp_path):
    if ENTRY is None:
        pytest.skip("storygen entrypoint not found in PATH or venv bin")
    out_md = tmp_path / "out.md"
    res = run_cli([
        "--seed", "7",
        "--length", "flash",
        "--themes", "redemption",
        "--elements", "lockbox",
        "--genre", "cozy mystery",
        "--tone", "whimsical",
        "--setting", "quaint English village",
        "--pov", "third",
        "--tense", "past",
        "--ending", "bittersweet",
        "--output", "story",
        "--yes",
        "--deterministic",
        "--arc-style", "fall-rise",
        "--protagonist-age", "teen",
        "--protagonist-role", "outsider",
        "--time-period", "medieval",
        "--tech-level", "industrial",
        "--magic-system", "forbidden",
        "--weather-mood", "stormy",
        "--cultural-conflict", "tradition vs progress",
        "--save", str(out_md),
    ])
    assert res.returncode == 0
    assert out_md.exists()
    text = out_md.read_text(encoding="utf-8")
    assert "Generated Story" in text


def test_outline_non_empty():
    import importlib.util
    import sys
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location('story_app', str(PKG_PY))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.StoryConfig(required_elements=["lockbox"], themes=["redemption"])  
    gen = mod.StoryGenerator(cfg, load_model=False, deterministic=True, dry_run=True)  
    outline = gen.generate_outline()
    assert outline["Hook"]
    assert outline["Resolution"]


def test_branch_override_midpoint():
    import importlib.util
    import sys
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location('story_app', str(PKG_PY))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.StoryConfig(required_elements=["lockbox"], themes=["redemption"])  
    gen = mod.StoryGenerator(cfg, load_model=False, deterministic=True, dry_run=True)  
    gen.set_branch_override('midpoint.choice', 'n')
    mp = gen.generate_midpoint()
    assert "Distrust" in mp
