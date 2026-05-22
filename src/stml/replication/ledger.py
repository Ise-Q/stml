"""
ledger.py
=========
Auditable cross-iteration memory for the replication search (US-009).

Every search trial is appended here with its provenance (``motivated_by``),
making the learning loop transparent: each proposed configuration records
exactly which prior trials seeded it.  The ledger persists to disk so
restarts retain full history.

The default path mirrors the repo-root walk pattern used by
:func:`stml.io._find_repo_root` and :mod:`stml.replication.run_characterize`.

Trial dict shape
----------------
::

    {
        "id": int,            # auto-assigned, 0-based, globally unique
        "archetype": str,     # archetype family name (e.g. "mean_reversion")
        "cell": str,          # instrument or "pool:energy" / "pool:equity" / ...
        "tier": str,          # "tpe" | "grid"
        "params": dict,       # hyperparameters used
        "val_metrics": dict,  # output of metrics.panel on the val split
        "gate_result": dict,  # GateResult.details or a simple bool dict
        "motivated_by": list, # prior trial ids or free-text notes
        "ts": str,            # ISO-8601 timestamp (auto-assigned if absent)
    }
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Repo-root helper (mirrors stml.io._find_repo_root)                          #
# --------------------------------------------------------------------------- #
def _repo_root() -> Path:
    """Walk up from this file until ``data/`` + ``pyproject.toml`` are found."""
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "data").is_dir() and (p / "pyproject.toml").is_file():
            return p
    raise FileNotFoundError(
        f"could not locate stml repo root (data/ + pyproject.toml) from {here}"
    )


_LEDGER_REL = Path("results/jj/ledger.json")
_REPORT_REL = Path("reports/replication-ledger.md")


# --------------------------------------------------------------------------- #
# JSON serialisation helpers                                                   #
# --------------------------------------------------------------------------- #
def _json_default(obj: Any) -> Any:
    """Coerce numpy / pandas scalars to plain Python types for JSON."""
    # Avoid a hard numpy import — use duck-typing so the module is usable
    # even if numpy is not installed in a stripped test environment.
    t = type(obj).__name__
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    if hasattr(obj, "tolist"):  # numpy array / pandas Series
        return obj.tolist()
    if t in ("bool_",):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


def _dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=_json_default)


# --------------------------------------------------------------------------- #
# Ledger                                                                       #
# --------------------------------------------------------------------------- #
class Ledger:
    """Append-only, disk-backed store of search trials.

    Parameters
    ----------
    path :
        Where to persist the JSON file.  ``None`` uses the canonical repo-root
        path ``results/jj/ledger.json`` (mirrors the run_characterize pattern).
        Pass an explicit path in tests so the real artifact is never touched.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            self._path = _repo_root() / _LEDGER_REL
        else:
            self._path = Path(path)

        self._trials: list[dict] = []
        if self._path.is_file():
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._trials = raw if isinstance(raw, list) else []

    # ---------------------------------------------------------------------- #
    # Public API                                                               #
    # ---------------------------------------------------------------------- #
    def record(self, trial: dict) -> dict:
        """Append *trial* to the ledger, persist to disk, and return it.

        Auto-assigns ``id`` (next integer) and ``ts`` (UTC ISO-8601) when
        absent, so callers need not set them explicitly.
        """
        t = dict(trial)  # shallow copy — do not mutate the caller's dict
        if "id" not in t:
            t["id"] = len(self._trials)
        if "ts" not in t:
            t["ts"] = datetime.now(tz=timezone.utc).isoformat()
        # Ensure motivated_by is always a list.
        if "motivated_by" not in t:
            t["motivated_by"] = []

        self._trials.append(t)
        self._persist()
        return t

    def prior_trials(self, archetype: str, cell: str) -> list[dict]:
        """All recorded trials for *archetype* + *cell* (read-before-propose).

        Returns them in insertion order.  Callers use this to seed or avoid
        configurations that have already been evaluated.
        """
        return [
            t
            for t in self._trials
            if t.get("archetype") == archetype and t.get("cell") == cell
        ]

    def best(self, archetype: str, cell: str, key: str) -> dict | None:
        """The trial maximising ``val_metrics[key]`` for *archetype* + *cell*.

        Returns ``None`` when no matching trial exists or when all ``key``
        values are non-finite (e.g. NaN, inf).
        """
        candidates = self.prior_trials(archetype, cell)
        finite = [
            t
            for t in candidates
            if isinstance(t.get("val_metrics"), dict)
            and isinstance(t["val_metrics"].get(key), (int, float))
            and math.isfinite(float(t["val_metrics"][key]))
        ]
        if not finite:
            return None
        return max(finite, key=lambda t: float(t["val_metrics"][key]))

    def render_markdown(self) -> str:
        """Render a summary Markdown report and write it to ``reports/replication-ledger.md``.

        The narrative surfaces ``motivated_by`` to make the learning loop
        visible: later configurations explicitly reference the prior trials
        that seeded them.

        Returns
        -------
        str
            The full Markdown text (also written to disk).
        """
        lines: list[str] = []
        lines.append("# Replication Search Ledger")
        lines.append("")
        lines.append(
            "Auditable record of every search trial.  Each entry carries a "
            "``motivated_by`` list that links the configuration back to prior "
            "trials or notes — making the iterative learning loop transparent "
            "and reproducible."
        )
        lines.append("")
        lines.append(f"**Total trials recorded:** {len(self._trials)}")
        lines.append("")

        # --- Main table -------------------------------------------------------
        lines.append(
            "| id | archetype | cell | tier | kappa | ordinal_skill | passed | motivated_by |"
        )
        lines.append(
            "|----|-----------|------|------|------:|-------------:|:------:|--------------|"
        )
        for t in self._trials:
            vm = t.get("val_metrics") or {}
            gr = t.get("gate_result") or {}

            kappa = vm.get("kappa")
            os_val = vm.get("ordinal_skill")
            # ordinal_skill may be a nested dict {vs_flat, vs_random}
            if isinstance(os_val, dict):
                os_val = os_val.get("vs_flat")

            kappa_str = (
                f"{float(kappa):.3f}"
                if isinstance(kappa, (int, float)) and math.isfinite(float(kappa))
                else "n/a"
            )
            os_str = (
                f"{float(os_val):.3f}"
                if isinstance(os_val, (int, float)) and math.isfinite(float(os_val))
                else "n/a"
            )

            # gate_result.passed may be a top-level bool or nested
            passed = gr.get("passed")
            passed_str = (
                "yes" if passed is True else ("no" if passed is False else "n/a")
            )

            mb = t.get("motivated_by", [])
            mb_str = ", ".join(str(x) for x in mb) if mb else "—"

            lines.append(
                f"| {t.get('id', '?')} "
                f"| {t.get('archetype', 'n/a')} "
                f"| {t.get('cell', 'n/a')} "
                f"| {t.get('tier', 'n/a')} "
                f"| {kappa_str} "
                f"| {os_str} "
                f"| {passed_str} "
                f"| {mb_str} |"
            )
        lines.append("")

        # --- Tier breakdown ---------------------------------------------------
        tiers: dict[str, int] = {}
        for t in self._trials:
            tier = str(t.get("tier", "unknown"))
            tiers[tier] = tiers.get(tier, 0) + 1
        lines.append("## Tier breakdown")
        lines.append("")
        for tier, count in sorted(tiers.items()):
            lines.append(f"- **{tier}**: {count} trial(s)")
        lines.append("")

        # --- Read-before-propose narrative ------------------------------------
        lines.append("## Read-before-propose narrative")
        lines.append("")
        lines.append(
            "The search reads prior runs via ``ledger.prior_trials(archetype, cell)`` "
            "before proposing each new configuration.  The ``motivated_by`` field "
            "records which prior trial ids (or free-text notes) seeded the proposal, "
            "making the learning loop auditable across restarts."
        )
        lines.append("")

        # Summarise which trials mention earlier ids in motivated_by.
        has_motivation = [t for t in self._trials if t.get("motivated_by")]
        if has_motivation:
            lines.append(
                f"{len(has_motivation)} of {len(self._trials)} trial(s) reference "
                "prior trials in ``motivated_by``:"
            )
            lines.append("")
            for t in has_motivation:
                mb = t.get("motivated_by", [])
                lines.append(
                    f"- Trial {t.get('id')} ({t.get('archetype')}/{t.get('cell')}) "
                    f"← {', '.join(str(x) for x in mb)}"
                )
            lines.append("")
        else:
            lines.append(
                "No trials have been linked to prior runs yet "
                "(all entries are seed / first-iteration trials)."
            )
            lines.append("")

        text = "\n".join(lines) + "\n"

        # Write the report next to the canonical deliverables.
        try:
            root = _repo_root()
            report_path = root / _REPORT_REL
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(text, encoding="utf-8")
        except FileNotFoundError:
            # Silently skip if we cannot resolve the repo root (e.g. in a
            # stripped CI environment without data/ — the caller still gets
            # the string return value).
            pass

        return text

    # ---------------------------------------------------------------------- #
    # Internal                                                                #
    # ---------------------------------------------------------------------- #
    def _persist(self) -> None:
        """Write the current trial list to disk as JSON."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(_dumps(self._trials) + "\n", encoding="utf-8")
