"""
test_features_catalog.py
========================
Tests for the graded feature catalog (:mod:`stml.metamodel.catalog`,
US-FE-008, AC-1 / AC-2 / AC-5) against the **full 11-instrument** feature
matrix built once on REAL data via :class:`stml.metamodel.pipeline.FeaturePipeline`.

This is the heavy integration test of the layer: a single module-scoped
fixture loads the clean data, fits the pipeline on all 11 instruments and
transforms once (the ~60-90s build), and every assertion reuses that one
matrix. The tests prove:

* **AC-1 (1:1 coverage).** :func:`catalog.assert_coverage` passes on the live
  matrix columns -- every non-meta column has a :class:`catalog.FeatureSpec`
  and there are no orphan specs. The catalog's column set equals the produced
  feature-column set exactly.
* **AC-5 (row counts).** The matrix has exactly ``4984`` rows, equal to the
  per-instrument nonzero-signal-day counts recomputed from the raw signals
  (the source of truth); the verified literals (nq1s 604, ng1s 124, es1s 575,
  ho1s 63, ...) are asserted against that recomputation.
* **AC-2 / render.** :func:`catalog.render_catalog` writes a non-empty
  ``feature-catalog.md`` (to ``tmp_path`` -- the real file is written by the
  build CLI) that lists every feature column and carries the four required
  annotation keywords (``mean-reversion``, ``0.09``, ``run_length_p90``,
  ``Amihud``).
"""

from __future__ import annotations

import warnings

import pytest

from stml.io import load_clean_data
from stml.metamodel.catalog import (
    CATALOG,
    META_COLS,
    assert_coverage,
    render_catalog,
)
from stml.metamodel.pipeline import FeaturePipeline

# All 11 universe instruments (CONTRACT_FE Section 2 / D5).
ALL_INSTRUMENTS = [
    "es1s",
    "nq1s",
    "fesx1s",
    "cl1s",
    "ho1s",
    "rb1s",
    "ng1s",
    "gc1s",
    "si1s",
    "hg1s",
    "pl1s",
]

# CONTRACT_FE Section 2 verified ground truth: per-instrument nonzero-signal
# rows and the total. These are asserted against the raw-signal recomputation
# (the source of truth) AND against the produced matrix.
EXPECTED_NONZERO = {
    "es1s": 575,
    "nq1s": 604,
    "fesx1s": 637,
    "cl1s": 422,
    "ho1s": 63,
    "rb1s": 628,
    "ng1s": 124,
    "gc1s": 168,
    "si1s": 578,
    "hg1s": 628,
    "pl1s": 557,
}
EXPECTED_TOTAL = 4984


# --------------------------------------------------------------------------- #
# Fixtures — build the FULL 11-instrument matrix ONCE (heavy integration).    #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def clean_data():
    """Real clean OHLCV + the full wide signal panel (all 11 instruments)."""
    return load_clean_data()


@pytest.fixture(scope="module")
def matrix(clean_data):
    """The full 11-instrument tidy-long feature matrix, built once.

    Warnings (statsmodels EM convergence on the thin low-power instruments) are
    silenced so the slow real-data fit/transform stays quiet; the boundary and
    leakage behaviour are exercised by the other test modules.
    """
    ohlcv, signals = clean_data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = FeaturePipeline().fit(ohlcv, signals)
        return pipe.transform(ohlcv, signals)


# --------------------------------------------------------------------------- #
# AC-1 — exact 1:1 catalog coverage (no missing specs, no orphans).           #
# --------------------------------------------------------------------------- #
def test_assert_coverage_passes_on_full_matrix(matrix) -> None:
    """Every produced non-meta column has a spec and there are no orphans."""
    # Does not raise.
    assert_coverage(matrix.columns)


def test_catalog_column_set_equals_produced_features(matrix) -> None:
    """The CATALOG keys equal the produced feature-column set exactly."""
    produced = [c for c in matrix.columns if c not in META_COLS]
    assert set(CATALOG) == set(produced), (
        "CATALOG keys must match the produced feature columns exactly"
    )
    # 175 documented feature columns after the curated Harry/Sreeram union:
    # 120 base (F1-F11) + 31 new family columns + 24 expanding-z twins.
    assert len(produced) == 175
    assert len(CATALOG) == 175


def test_meta_columns_have_no_spec(matrix) -> None:
    """The four meta columns are present and carry no FeatureSpec."""
    assert META_COLS <= set(matrix.columns)
    for meta in META_COLS:
        assert meta not in CATALOG


def test_every_spec_name_matches_its_key() -> None:
    """Each FeatureSpec's recorded ``name`` equals its CATALOG dict key."""
    for name, spec in CATALOG.items():
        assert spec.name == name
        assert spec.family.startswith("F")
        assert spec.leakage_class in {"E", "TF", "LI"}
        assert spec.what_it_captures.strip()
        assert spec.reuse_pointer.strip()


def test_assert_coverage_detects_missing_spec(matrix) -> None:
    """A produced column with no spec triggers an AssertionError."""
    bad = list(matrix.columns) + ["fX_not_a_real_feature"]
    with pytest.raises(AssertionError, match="missing a CATALOG entry"):
        assert_coverage(bad)


def test_assert_coverage_detects_orphan_spec() -> None:
    """A CATALOG entry with no matching column triggers an AssertionError."""
    # Drop one real feature column -> its spec becomes an orphan.
    cols = list(META_COLS) + [c for c in CATALOG if c != "f1_mr_score_20"]
    with pytest.raises(AssertionError, match="orphan CATALOG"):
        assert_coverage(cols)


# --------------------------------------------------------------------------- #
# AC-1 — leakage-class composition (the contract's E / TF / LI split).        #
# --------------------------------------------------------------------------- #
def test_leakage_class_composition() -> None:
    """F3/F4/F11/F16/F17 are TF; f2_vol_20 + f5_trailing_run_length are LI; rest E."""
    tf = {n for n, s in CATALOG.items() if s.leakage_class == "TF"}
    li = {n for n, s in CATALOG.items() if s.leakage_class == "LI"}
    eng = {n for n, s in CATALOG.items() if s.leakage_class == "E"}

    # TF = exactly the fitted families: F3 + F4 + F11 + F16 + F17.
    tf_prefixes = ("f3_", "f4_", "f11_", "f16_", "f17_")
    assert all(n.startswith(tf_prefixes) for n in tf)
    assert tf == {n for n in CATALOG if n.startswith(tf_prefixes)}
    assert len(tf) == 65  # 4 F3 + 11 F4 + 45 F11 + 1 F16 + 4 F17

    # LI = exactly the label-interface subset.
    assert li == {"f2_vol_20", "f5_trailing_run_length"}

    # The three classes partition the 175 columns.
    assert len(eng) == 108
    assert tf | li | eng == set(CATALOG)
    assert not (tf & li) and not (tf & eng) and not (li & eng)


# --------------------------------------------------------------------------- #
# AC-5 — matrix row counts == Sigma_inst (s != 0), with verified literals.    #
# --------------------------------------------------------------------------- #
def test_matrix_total_row_count_is_4984(matrix) -> None:
    """The full matrix has exactly 4984 rows (CONTRACT_FE Section 2)."""
    assert len(matrix) == EXPECTED_TOTAL


def test_per_instrument_counts_match_raw_signals(matrix, clean_data) -> None:
    """Per-instrument row counts equal the raw nonzero-signal counts.

    The raw-signal recomputation is the *source of truth*; the verified
    ground-truth literals (nq1s 604, ng1s 124, es1s 575, ho1s 63, ...) are then
    asserted against that recomputation, and the matrix is asserted to match.
    """
    _, signals = clean_data
    sig_idx = signals.set_index("date")

    # Source of truth: recompute nonzero-signal day counts from raw signals.
    raw_counts = {
        inst: int((sig_idx[inst] != 0).sum()) for inst in ALL_INSTRUMENTS
    }

    # The verified literals match the raw recomputation.
    assert raw_counts == EXPECTED_NONZERO
    # Spot-check the four enumerated in the story brief.
    assert raw_counts["nq1s"] == 604
    assert raw_counts["ng1s"] == 124
    assert raw_counts["es1s"] == 575
    assert raw_counts["ho1s"] == 63

    # The produced matrix matches the raw counts per instrument.
    actual = matrix["instrument"].value_counts().to_dict()
    assert actual == raw_counts, f"matrix counts {actual} != raw {raw_counts}"
    assert sum(raw_counts.values()) == EXPECTED_TOTAL


def test_all_eleven_instruments_present(matrix) -> None:
    """Every one of the 11 universe instruments appears in the matrix."""
    assert set(matrix["instrument"].unique()) == set(ALL_INSTRUMENTS)


# --------------------------------------------------------------------------- #
# AC-2 — render_catalog writes a non-empty, fully-annotated markdown file.     #
# --------------------------------------------------------------------------- #
def test_render_catalog_writes_annotated_markdown(matrix, tmp_path) -> None:
    """render_catalog writes a non-empty md with the four required annotations
    and every feature column listed (rendered to tmp_path, not the real file)."""
    out = tmp_path / "feature-catalog.md"
    recon_mse = {
        "EQ": {"pca_k4": 0.512, "ae_k4": 0.487},
        "EN": {"pca_k4": 0.601, "ae_k4": 0.640},
        "ME": {"pca_k4": 0.550, "ae_k4": 0.548},
    }
    render_catalog(matrix.columns, recon_mse=recon_mse, path=str(out))

    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert text.strip(), "rendered catalog must be non-empty"

    # The four required annotation keywords (CONTRACT_FE Section 3).
    for keyword in ("mean-reversion", "0.09", "run_length_p90", "Amihud"):
        assert keyword in text, f"missing required annotation keyword {keyword!r}"

    # Every produced feature column is listed (as a backticked table cell).
    produced = [c for c in matrix.columns if c not in META_COLS]
    for col in produced:
        assert f"`{col}`" in text, f"feature column {col!r} not rendered"

    # The AE-vs-PCA(k=4) reconstruction-MSE table is rendered when supplied.
    assert "Autoencoder vs PCA" in text
    assert "0.487" in text


def test_render_catalog_without_recon_mse(matrix, tmp_path) -> None:
    """render_catalog omits the reconstruction table when recon_mse is None,
    yet still lists every column and the required annotations."""
    out = tmp_path / "feature-catalog-no-recon.md"
    render_catalog(matrix.columns, recon_mse=None, path=str(out))

    text = out.read_text(encoding="utf-8")
    assert text.strip()
    assert "Autoencoder vs PCA" not in text
    for keyword in ("mean-reversion", "0.09", "run_length_p90", "Amihud"):
        assert keyword in text
