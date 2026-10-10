"""The reproduction registry (tools/reproduce.py) covers every figure and every script: one command per figure."""
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _tool():
    spec = importlib.util.spec_from_file_location("reproduce", ROOT / "tools" / "reproduce.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve their annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


def test_every_figure_has_a_target():
    tool = _tool()
    outs = {o for t in tool.REGISTRY for o in tool._expand(t.outputs)}
    figs = {f"figures/{p.name}" for p in (ROOT / "figures").iterdir() if p.suffix in (".png", ".pdf")}
    assert figs <= outs, sorted(figs - outs)


def test_every_script_is_registered():
    tool = _tool()
    scripts = {t.script for t in tool.REGISTRY}
    own = {"tools/reproduce.py", "tools/compare_to_git.py",   # the reproduction utilities themselves
           "tools/export_public.py"}                          # writes the public copy (no figure or data product)
    files = {f"examples/{p.name}" for p in (ROOT / "examples").glob("*.py")}
    files |= {f"tools/{p.name}" for p in (ROOT / "tools").glob("*.py")} - own
    assert files <= scripts, sorted(files - scripts)


def test_registry_is_consistent():
    tool = _tool()
    names = [t.name for t in tool.REGISTRY]
    assert len(names) == len(set(names))
    for t in tool.REGISTRY:
        assert (ROOT / t.script).exists(), t.script
        assert set(t.after) <= set(names), (t.name, t.after)
        assert t.minutes > 0
    order = tool._order(names)
    pos = {n: i for i, n in enumerate(order)}
    assert all(pos[m] < pos[t.name] for t in tool.REGISTRY for m in t.after)


def test_every_data_product_has_a_target():
    tool = _tool()
    outs = {o for t in tool.REGISTRY for o in tool._expand(t.outputs)}
    files = set()
    for sub in ("validation", "digitized", "calibration", "atlas", "branches", "tls", "apt"):
        files |= {f"data/{sub}/{p.name}" for p in (ROOT / "data" / sub).iterdir() if p.is_file()}
    assert files <= outs, sorted(files - outs)
