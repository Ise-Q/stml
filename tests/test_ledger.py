"""Tests for ``stml.replication.ledger`` (US-009).

All tests use ``tmp_path`` so the real ``results/jj/ledger.json`` is never
touched.  Assertions target invariants, not brittle magic numbers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from stml.replication.ledger import Ledger


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _make_trial(
    archetype: str = "mean_reversion",
    cell: str = "es1s",
    tier: str = "tpe",
    kappa: float = 0.25,
    os_val: float = 0.30,
    passed: bool = True,
    motivated_by: list | None = None,
) -> dict:
    return {
        "archetype": archetype,
        "cell": cell,
        "tier": tier,
        "params": {"L": 20, "deadband": 0.5},
        "val_metrics": {"kappa": kappa, "ordinal_skill": os_val},
        "gate_result": {"passed": passed},
        "motivated_by": motivated_by or [],
    }


# --------------------------------------------------------------------------- #
# Round-trip persistence                                                       #
# --------------------------------------------------------------------------- #
def test_round_trip_empty(tmp_path: Path) -> None:
    """An empty ledger round-trips cleanly (file is created, reload is empty)."""
    path = tmp_path / "ledger.json"
    led = Ledger(path)
    # No trials yet — no file on disk until record() is called.
    assert led.prior_trials("mean_reversion", "es1s") == []


def test_round_trip_n_trials(tmp_path: Path) -> None:
    """Record N trials, reload a fresh Ledger → identical trial list."""
    path = tmp_path / "ledger.json"
    led = Ledger(path)

    n = 5
    for i in range(n):
        led.record(_make_trial(kappa=0.1 * i, motivated_by=[i - 1] if i > 0 else []))

    # Reload from the same path.
    led2 = Ledger(path)
    assert len(led2._trials) == n

    for orig, reloaded in zip(led._trials, led2._trials):
        assert orig["id"] == reloaded["id"]
        assert orig["archetype"] == reloaded["archetype"]
        assert orig["cell"] == reloaded["cell"]
        assert orig["tier"] == reloaded["tier"]
        assert orig["params"] == reloaded["params"]
        assert orig["val_metrics"]["kappa"] == pytest.approx(
            reloaded["val_metrics"]["kappa"]
        )
        assert orig["gate_result"] == reloaded["gate_result"]
        assert orig["motivated_by"] == reloaded["motivated_by"]
        assert "ts" in reloaded


def test_record_auto_assigns_id_and_ts(tmp_path: Path) -> None:
    """``record`` auto-assigns monotone integer ids and ISO timestamps."""
    path = tmp_path / "ledger.json"
    led = Ledger(path)
    t0 = led.record(_make_trial())
    t1 = led.record(_make_trial())
    assert t0["id"] == 0
    assert t1["id"] == 1
    assert isinstance(t0["ts"], str) and "T" in t0["ts"]
    assert isinstance(t1["ts"], str) and "T" in t1["ts"]


def test_record_preserves_explicit_id(tmp_path: Path) -> None:
    """If the caller supplies an ``id`` it is preserved unchanged."""
    path = tmp_path / "ledger.json"
    led = Ledger(path)
    trial = _make_trial()
    trial["id"] = 42
    recorded = led.record(trial)
    assert recorded["id"] == 42


def test_record_returns_the_trial(tmp_path: Path) -> None:
    """``record`` returns the (possibly augmented) trial dict."""
    path = tmp_path / "ledger.json"
    led = Ledger(path)
    t = led.record(_make_trial())
    assert t["archetype"] == "mean_reversion"
    assert "id" in t
    assert "ts" in t


# --------------------------------------------------------------------------- #
# prior_trials: read-before-propose                                            #
# --------------------------------------------------------------------------- #
def test_prior_trials_returns_matching(tmp_path: Path) -> None:
    """``prior_trials`` returns exactly the trials for the requested archetype+cell."""
    path = tmp_path / "ledger.json"
    led = Ledger(path)

    led.record(_make_trial(archetype="mean_reversion", cell="es1s"))
    led.record(_make_trial(archetype="mean_reversion", cell="es1s", kappa=0.35))
    led.record(_make_trial(archetype="ts_momentum", cell="es1s"))
    led.record(_make_trial(archetype="mean_reversion", cell="nq1s"))

    matching = led.prior_trials("mean_reversion", "es1s")
    assert len(matching) == 2
    for t in matching:
        assert t["archetype"] == "mean_reversion"
        assert t["cell"] == "es1s"


def test_prior_trials_excludes_other_archetypes_and_cells(tmp_path: Path) -> None:
    """Trials for a different archetype or cell do NOT appear."""
    path = tmp_path / "ledger.json"
    led = Ledger(path)
    led.record(_make_trial(archetype="ts_momentum", cell="nq1s"))
    led.record(_make_trial(archetype="mean_reversion", cell="pool:energy"))

    # Neither of the above matches this query.
    assert led.prior_trials("mean_reversion", "es1s") == []


def test_prior_trials_empty_ledger(tmp_path: Path) -> None:
    """An empty ledger returns an empty list without raising."""
    led = Ledger(tmp_path / "ledger.json")
    assert led.prior_trials("mean_reversion", "es1s") == []


def test_prior_trials_pool_cell(tmp_path: Path) -> None:
    """Pool cells like ``pool:energy`` are matched correctly."""
    path = tmp_path / "ledger.json"
    led = Ledger(path)
    led.record(_make_trial(archetype="mean_reversion", cell="pool:energy"))
    led.record(_make_trial(archetype="mean_reversion", cell="es1s"))

    pool_trials = led.prior_trials("mean_reversion", "pool:energy")
    assert len(pool_trials) == 1
    assert pool_trials[0]["cell"] == "pool:energy"


# --------------------------------------------------------------------------- #
# best: maximise val_metrics[key]                                              #
# --------------------------------------------------------------------------- #
def test_best_returns_top_trial(tmp_path: Path) -> None:
    """``best`` returns the trial with the highest value of the requested metric."""
    path = tmp_path / "ledger.json"
    led = Ledger(path)
    led.record(_make_trial(kappa=0.10))
    led.record(_make_trial(kappa=0.40))  # <-- best
    led.record(_make_trial(kappa=0.25))

    winner = led.best("mean_reversion", "es1s", "kappa")
    assert winner is not None
    assert winner["val_metrics"]["kappa"] == pytest.approx(0.40)


def test_best_none_when_no_matching_trials(tmp_path: Path) -> None:
    """``best`` returns ``None`` when no trials match the archetype+cell."""
    led = Ledger(tmp_path / "ledger.json")
    led.record(_make_trial(archetype="ts_momentum", cell="nq1s"))
    assert led.best("mean_reversion", "es1s", "kappa") is None


def test_best_none_when_all_nonfinite(tmp_path: Path) -> None:
    """``best`` returns ``None`` when every matching trial has a non-finite metric."""
    import math

    path = tmp_path / "ledger.json"
    led = Ledger(path)
    t = _make_trial()
    t["val_metrics"]["kappa"] = math.nan
    led.record(t)
    assert led.best("mean_reversion", "es1s", "kappa") is None


def test_best_ignores_other_cells(tmp_path: Path) -> None:
    """``best`` scopes to the requested cell and ignores higher-metric other cells."""
    path = tmp_path / "ledger.json"
    led = Ledger(path)
    led.record(_make_trial(cell="es1s", kappa=0.10))
    led.record(_make_trial(cell="nq1s", kappa=0.99))  # different cell

    winner = led.best("mean_reversion", "es1s", "kappa")
    assert winner is not None
    assert winner["val_metrics"]["kappa"] == pytest.approx(0.10)


# --------------------------------------------------------------------------- #
# render_markdown                                                              #
# --------------------------------------------------------------------------- #
def test_render_markdown_writes_file(tmp_path: Path, monkeypatch) -> None:
    """``render_markdown`` writes ``reports/replication-ledger.md`` under the repo root."""
    # Patch _repo_root so it points to tmp_path — avoids touching real reports/.
    import stml.replication.ledger as ledger_mod

    monkeypatch.setattr(ledger_mod, "_repo_root", lambda: tmp_path)

    path = tmp_path / "ledger.json"
    led = Ledger(path)
    led.record(_make_trial(tier="tpe", motivated_by=[]))
    led.record(_make_trial(tier="grid", motivated_by=[0]))

    text = led.render_markdown()
    report = tmp_path / "reports" / "replication-ledger.md"
    assert report.is_file(), "render_markdown must write reports/replication-ledger.md"
    assert report.read_text(encoding="utf-8") == text


def test_render_markdown_lists_tiers(tmp_path: Path, monkeypatch) -> None:
    """The rendered report lists all tier values present in the ledger."""
    import stml.replication.ledger as ledger_mod

    monkeypatch.setattr(ledger_mod, "_repo_root", lambda: tmp_path)

    path = tmp_path / "ledger.json"
    led = Ledger(path)
    led.record(_make_trial(tier="tpe"))
    led.record(_make_trial(tier="grid"))

    text = led.render_markdown()
    assert "tpe" in text
    assert "grid" in text


def test_render_markdown_surfaces_motivated_by(tmp_path: Path, monkeypatch) -> None:
    """``motivated_by`` references appear in the rendered report."""
    import stml.replication.ledger as ledger_mod

    monkeypatch.setattr(ledger_mod, "_repo_root", lambda: tmp_path)

    path = tmp_path / "ledger.json"
    led = Ledger(path)
    led.record(_make_trial(motivated_by=[]))
    led.record(_make_trial(motivated_by=[0, "seed_note"]))

    text = led.render_markdown()
    # The second trial's motivated_by should appear somewhere in the narrative.
    assert "0" in text
    assert "seed_note" in text


def test_render_markdown_empty_ledger(tmp_path: Path, monkeypatch) -> None:
    """``render_markdown`` does not crash on an empty ledger."""
    import stml.replication.ledger as ledger_mod

    monkeypatch.setattr(ledger_mod, "_repo_root", lambda: tmp_path)

    led = Ledger(tmp_path / "ledger.json")
    text = led.render_markdown()
    assert "Replication Search Ledger" in text
    assert "0" in text  # total trials count


def test_render_markdown_returns_string(tmp_path: Path, monkeypatch) -> None:
    """``render_markdown`` always returns a string."""
    import stml.replication.ledger as ledger_mod

    monkeypatch.setattr(ledger_mod, "_repo_root", lambda: tmp_path)

    led = Ledger(tmp_path / "ledger.json")
    led.record(_make_trial())
    result = led.render_markdown()
    assert isinstance(result, str)
    assert len(result) > 0
