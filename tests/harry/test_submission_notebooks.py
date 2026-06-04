"""Validation tests for the per-asset-class submission notebooks.

These guard the highest-risk failure mode of a figure-assembling notebook: a
``display(Image(path))`` call whose ``path`` does not exist renders *nothing*
and never raises, so a broken artifact reference is invisible until a human
opens the notebook. By asserting that every narrative-critical path the
generator will reference actually exists, this test catches that before the
notebooks are emitted or executed.

The generator (`src/stml/new_work/_gen_final_notebooks.py`) exposes the path
sets as pure functions so they can be checked here without importing the heavy
modelling stack or running any cell.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GEN = REPO / "src/stml/new_work/_gen_final_notebooks.py"
CLASSES = ("equity", "energy", "metals")


def _gen():
    spec = importlib.util.spec_from_file_location("genfinal", GEN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_required_artifacts_exist():
    """Every narrative-critical figure/CSV must be present for all classes."""
    m = _gen()
    for cls in CLASSES:
        missing = [str(p) for p in m.required_artifacts(cls) if not Path(p).exists()]
        assert missing == [], f"{cls}: {len(missing)} missing required artifacts: {missing}"


def test_build_is_valid_nbformat():
    """build(cls) must return a structurally valid notebook with a markdown lead."""
    import nbformat

    m = _gen()
    for cls in CLASSES:
        nb = m.build(cls)
        nbformat.validate(nb)
        assert nb["cells"], f"{cls}: notebook has no cells"
        assert nb["cells"][0]["cell_type"] == "markdown", f"{cls}: first cell not markdown"


def test_class_narratives_are_hand_authored_and_distinct():
    """The class-specific narrative passages must be genuinely different prose.

    Shared methodology blocks (triple-barrier mechanics, CPCV, costs) are
    legitimately identical across the three self-contained notebooks, so they
    are not tested here. What must differ is the hand-authored, class-specific
    analysis held in ``NARRATIVES`` (abstract, signal character, model and
    importance commentary, discussion). This catches accidental class-name
    string-templating where it actually matters.
    """
    import difflib

    m = _gen()
    sections = sorted(set().union(*(m.NARRATIVES[c].keys() for c in CLASSES)))
    for sec in sections:
        for a in CLASSES:
            for b in CLASSES:
                if a >= b:
                    continue
                sa = m.NARRATIVES[a].get(sec, "").replace(a, "X")
                sb = m.NARRATIVES[b].get(sec, "").replace(b, "X")
                if not sa or not sb:
                    continue
                ratio = difflib.SequenceMatcher(None, sa, sb).ratio()
                assert ratio < 0.85, f"{sec}: {a} vs {b} narrative {ratio:.2f} too similar"
