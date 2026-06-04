"""
hmm_features_v2.py
==================
Family **f17_v2 — primary-skill regime HMM** (model-layer, *label-aware*).

A successor to the feature-layer F17 (:mod:`stml.metamodel.regime_features_hmm`), but built
where the **triple-barrier labels live** (the model layer) because its observation vector folds
in *rolling primary-performance* statistics (RHR / RER / RS) computed from **past resolved**
labels. The HMM clusters each primary signal into one of ``n_states`` latent **skill regimes**
(``M=3`` → GOOD / NEUTRAL / BAD by primary hit-rate) and emits the **filtered** (forward-only,
causal) regime posteriors plus one-step-ahead predictives as ``f17_v2_*`` columns that merge into
the meta-model design matrix (auto-selected by ``dataset.select_features`` ``^f\\d``).

Why a separate module / why the model layer
-------------------------------------------
The frozen feature matrix is deliberately *model-free*: F17 sees only ``(ret, vol)`` and never a
label. RHR / RER / RS need triple-barrier outcomes, so f17_v2 cannot live in that build without
inverting the layering. Kept here, ``feature_matrix.parquet`` / ``catalog`` / the FE tests are
untouched.

Causality (mirrors F17, extended for labels)
--------------------------------------------
1. The HMM params + standardisation + GOOD/NEUTRAL/BAD ordering are **frozen on FE-train**
   (events with ``date <= fe_train_end``) and applied causally.
2. The regime posterior at event ``i`` is the **filtered** ``P(state | obs_{0..i})`` from a
   from-scratch forward pass over the instrument's *event* sequence — never hmmlearn's smoothed
   ``predict_proba`` (used only in-sample for the hit-rate ordering). Truncation-invariant.
3. RHR / RER / RS at event ``i`` use only prior events that have **resolved** (``t1[j] <= entry_i``),
   so an event never sees its own (or any future) outcome.
4. States are ordered by **FE-train per-state hit-rate** (GOOD = highest), frozen on train (a
   label-based order ⇒ must be frozen, like F17's vol-ascending order).

Requires ``hmmlearn`` — install with ``uv sync --group features-extra``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:  # optional dependency (features-extra group), mirrors trees.HAS_LIGHTGBM
    import hmmlearn  # noqa: F401

    HAS_HMMLEARN = True
except Exception:  # pragma: no cover - environment-dependent
    HAS_HMMLEARN = False

__all__ = [
    "HAS_HMMLEARN",
    "HmmV2Bundle",
    "OBS_COLS",
    "DEFAULT_COL_MAP",
    "perf_columns",
    "regime_columns",
    "f17_v2_columns",
    "rolling_performance_features",
    "build_observation_matrix",
    "fit_hmm_v2",
    "transform_hmm_v2",
    "fit_transform_f17_v2",
    "transform_f17_v2",
    "select_n_states",
]

TRADING_DAYS_PER_YEAR = 252.0

#: Canonical observation-vector columns (short names) the HMM emits from.
OBS_COLS: tuple[str, ...] = (
    "vol", "trend", "autocorr", "ret_1", "ret_5", "ret_20",
    "conf", "freq", "rhr", "rer", "rs",
)

#: Map each observation dimension to its source column in the feature matrix.
#: ``ret_1/5/20`` are recomputed from ``close_wide`` (no matrix column).
DEFAULT_COL_MAP: dict[str, str] = {
    "vol": "f2_vol_20",            # de-annualised by sqrt(252) here
    "trend": "f12_hurst_100",
    "autocorr": "f12_autocorr_21",
    "conf": "f5_trailing_run_length",
    "freq": "f5_participation_20",
}


def _regime_names(n_states: int) -> list[str]:
    """GOOD/NEUTRAL/BAD for the canonical 3-state model, else ``s0..s{M-1}``."""
    if n_states == 3:
        return ["good", "neutral", "bad"]
    return [f"s{k}" for k in range(n_states)]


def perf_columns(window: int) -> list[str]:
    """The three rolling-performance feature columns for a given window ``W``."""
    return [f"f17_v2_rhr_{window}", f"f17_v2_rer_{window}", f"f17_v2_rs_{window}"]


def regime_columns(n_states: int) -> list[str]:
    """The HMM-derived columns: posteriors, argmax, one-step-ahead predictives, entropy."""
    names = _regime_names(n_states)
    return (
        [f"f17_v2_regime_{nm}" for nm in names]
        + ["f17_v2_regime_argmax"]
        + [f"f17_v2_regime_pred_{nm}" for nm in names]
        + ["f17_v2_regime_entropy"]
    )


def f17_v2_columns(n_states: int, window: int) -> list[str]:
    """Full ``f17_v2_*`` column set = rolling performance + regime-derived."""
    return perf_columns(window) + regime_columns(n_states)


# --------------------------------------------------------------------------------------------- #
# 1. Rolling primary-performance features (the label-dependent channel).
# --------------------------------------------------------------------------------------------- #


def rolling_performance_features(
    labels: pd.DataFrame, *, window: int = 20, min_count: int = 5
) -> pd.DataFrame:
    """Rolling hit-rate / expected-return / Sharpe of the primary over **resolved** prior events.

    For each event ``i`` on its instrument, gather prior events ``j`` whose triple barrier has
    **already resolved** by entry time (``t1[j] <= date[i]``), keep the most recently resolved
    ``window`` of them, and compute:

    * ``RHR`` = mean of ``bin`` (rolling hit-rate),
    * ``RER`` = mean of ``ret`` (rolling expected return),
    * ``RS``  = mean(``ret``) / std(``ret``) * sqrt(n) (rolling Sharpe of the primary).

    Fewer than ``min_count`` resolved priors → structural NaN (never filled). Because
    ``t1[j] > date[j]`` always, an event never includes its own outcome — the load-bearing
    causality guarantee (see ``test_rhr_resolved_before_entry``).

    Parameters
    ----------
    labels : DataFrame with at least ``[date, instrument, t1, ret, bin]`` (the
        :func:`stml.model.labels.triple_barrier_labels` output).

    Returns
    -------
    DataFrame ``[date, instrument, f17_v2_rhr_W, f17_v2_rer_W, f17_v2_rs_W]`` row-aligned to
        ``labels`` (merge onto the design matrix on ``[date, instrument]``).
    """
    cols = perf_columns(window)
    out = pd.DataFrame(np.nan, index=labels.index, columns=cols, dtype=float)
    for _inst, g in labels.groupby("instrument", sort=False):
        g = g.sort_values("date")
        entry = pd.DatetimeIndex(g["date"]).to_numpy()
        t1 = pd.DatetimeIndex(g["t1"]).to_numpy()
        b = g["bin"].to_numpy(dtype=float)
        r = g["ret"].to_numpy(dtype=float)
        gi = g.index.to_numpy()
        n = len(g)
        order_t1 = np.argsort(t1, kind="stable")  # resolution order (for "most recent W")
        for i in range(n):
            resolved = t1 <= entry[i]
            resolved[i] = False  # never self (t1[i] > entry[i] anyway)
            if not resolved.any():
                continue
            jj = order_t1[resolved[order_t1]][-window:]  # most-recently-resolved W
            if jj.size < min_count:
                continue
            rj = r[jj]
            sd = float(rj.std())
            rs = float(rj.mean() / sd * np.sqrt(jj.size)) if sd > 1e-12 else np.nan
            out.loc[gi[i], cols] = [float(b[jj].mean()), float(rj.mean()), rs]
    res = labels[["date", "instrument"]].copy()
    res[cols] = out
    return res.reset_index(drop=True)


# --------------------------------------------------------------------------------------------- #
# 2. Observation-vector assembly (maps the spec features onto our columns).
# --------------------------------------------------------------------------------------------- #


def _nbar_log_returns(close_wide: pd.DataFrame, dates: np.ndarray, inst: str, ks):
    """``{k: log(close_t / close_{t-k})}`` at each event ``date`` on ``inst``'s own calendar."""
    out = {k: np.full(len(dates), np.nan) for k in ks}
    if inst not in close_wide.columns:
        return out
    s = close_wide[inst].dropna()
    idx = s.index
    vals = s.to_numpy(dtype=float)
    pos = idx.get_indexer(pd.DatetimeIndex(dates))
    for j, p in enumerate(pos):
        if p < 0:
            continue
        for k in ks:
            if p - k >= 0 and vals[p - k] > 0 and vals[p] > 0:
                out[k][j] = float(np.log(vals[p] / vals[p - k]))
    return out


def build_observation_matrix(
    rows: pd.DataFrame,
    perf: pd.DataFrame,
    close_wide: pd.DataFrame,
    *,
    window: int = 20,
    col_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Assemble the 11-d observation frame from existing columns + recomputed returns + ``perf``.

    Market block (``vol`` de-annualised, ``trend``, ``autocorr``) and signal block (``conf``,
    ``freq``) are sliced from ``rows`` via ``col_map``; ``ret_1/5/20`` are recomputed causally from
    ``close_wide``; ``rhr/rer/rs`` come from :func:`rolling_performance_features`. Missing source
    columns are tolerated (the dimension becomes NaN, with a warning).

    Returns ``[date, instrument, bar_pos?, partition?, *OBS_COLS, bin?]`` row-aligned to ``rows``.
    """
    col_map = {**DEFAULT_COL_MAP, **(col_map or {})}
    rows = rows.reset_index(drop=True)
    obs = pd.DataFrame(index=rows.index)
    obs["date"] = pd.to_datetime(rows["date"]).to_numpy()
    obs["instrument"] = rows["instrument"].to_numpy()
    for meta in ("bar_pos", "partition"):
        if meta in rows.columns:
            obs[meta] = rows[meta].to_numpy()

    def _col(name: str) -> np.ndarray:
        src = col_map.get(name)
        if src is None or src not in rows.columns:
            warnings.warn(
                f"[hmm_features_v2] obs '{name}': source column "
                f"{src!r} absent; dimension is structural NaN.",
                stacklevel=2,
            )
            return np.full(len(rows), np.nan)
        return rows[src].to_numpy(dtype=float)

    vol = _col("vol")
    obs["vol"] = vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    obs["trend"] = _col("trend")
    obs["autocorr"] = _col("autocorr")
    obs["conf"] = _col("conf")
    obs["freq"] = _col("freq")

    # ret_1/5/20 recomputed per instrument from the clean close panel.
    for k in (1, 5, 20):
        obs[f"ret_{k}"] = np.nan
    for inst, g in obs.groupby("instrument", sort=False):
        rk = _nbar_log_returns(close_wide, g["date"].to_numpy(), inst, (1, 5, 20))
        for k in (1, 5, 20):
            obs.loc[g.index, f"ret_{k}"] = rk[k]

    # rhr/rer/rs from the rolling-performance frame (merge on [date, instrument]).
    pcols = perf_columns(window)
    merged = obs[["date", "instrument"]].merge(
        perf[["date", "instrument", *pcols]], on=["date", "instrument"], how="left"
    )
    obs["rhr"] = merged[pcols[0]].to_numpy()
    obs["rer"] = merged[pcols[1]].to_numpy()
    obs["rs"] = merged[pcols[2]].to_numpy()

    if "bin" in rows.columns:
        obs["bin"] = rows["bin"].to_numpy()
    return obs


# --------------------------------------------------------------------------------------------- #
# 3. The frozen HMM bundle + causal forward filter (mirrors F17).
# --------------------------------------------------------------------------------------------- #


@dataclass
class HmmV2Bundle:
    """Frozen FE-train HMM artifacts for one fit scope (an instrument or the pooled fallback)."""

    model: object                  # hmmlearn GaussianHMM, or None on failure
    order: np.ndarray              # state permutation: descending FE-train hit-rate (col0 = GOOD)
    feat_mean: np.ndarray          # frozen train standardisation mean over obs_cols
    feat_std: np.ndarray           # frozen train standardisation std
    state_hit_rate: np.ndarray     # per-state FE-train hit-rate, in the reordered (GOOD..BAD) order
    obs_cols: tuple[str, ...]      # observation columns used (order matters)
    n_states: int
    train_index: pd.DatetimeIndex  # FE-train event dates fitted on (provenance)
    scope: str                     # "pooled" / instrument / class id (provenance)
    ok: bool


def _failed_bundle(
    obs_cols: tuple[str, ...], n_states: int, scope: str, reason: str
) -> HmmV2Bundle:
    """An ``ok=False`` bundle (logged, never raises) → structural-NaN transform."""
    warnings.warn(
        f"[hmm_features_v2] {scope}: HMM fit failed ({reason}); "
        "bundle.ok=False, f17_v2 regime columns will be structural NaN.",
        stacklevel=2,
    )
    return HmmV2Bundle(
        model=None,
        order=np.arange(n_states),
        feat_mean=np.zeros(len(obs_cols)),
        feat_std=np.ones(len(obs_cols)),
        state_hit_rate=np.full(n_states, np.nan),
        obs_cols=tuple(obs_cols),
        n_states=n_states,
        train_index=pd.DatetimeIndex([]),
        scope=scope,
        ok=False,
    )


def _causal_filtered_probs(model: object, X: np.ndarray) -> np.ndarray:
    """Forward-only (filtered) posteriors ``P(state | X_{0..t})`` (vendored from F17).

    ``log a[0] = log pi + log e[0]`` ; ``log a[t] = log e[t] + logsumexp_k'(log a[t-1,k'] + log A)``
    then row-normalised. ``filtered[t]`` depends only on ``X[0..t]`` — truncation-invariant, never
    hmmlearn's smoothed ``predict_proba``. See ``stml.metamodel.regime_features_hmm``.
    """
    from scipy.special import logsumexp

    log_emis = model._compute_log_likelihood(X)  # (T, K)
    log_pi = np.log(np.clip(model.startprob_, 1e-12, None))
    log_A = np.log(np.clip(model.transmat_, 1e-12, None))  # (K, K)

    T, K = log_emis.shape
    log_alpha = np.empty((T, K))
    log_alpha[0] = log_pi + log_emis[0]
    for t in range(1, T):
        log_alpha[t] = log_emis[t] + logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0)
    log_filt = log_alpha - logsumexp(log_alpha, axis=1, keepdims=True)
    return np.exp(log_filt)


def _seq_lengths(df: pd.DataFrame) -> list[int]:
    """Per-instrument sequence lengths for a ``[instrument, date]``-sorted frame (hmmlearn)."""
    return [int(n) for n in df.groupby("instrument", sort=True).size()]


def fit_hmm_v2(
    train_obs: pd.DataFrame,
    *,
    obs_cols: tuple[str, ...] = OBS_COLS,
    label_col: str = "bin",
    n_states: int = 3,
    covariance_type: str = "diag",
    seed: int = 0,
    scope: str = "",
    min_train: int = 150,
) -> HmmV2Bundle:
    """Fit a Gaussian skill-regime HMM on **FE-train** observation rows (one scope).

    ``train_obs`` must already be restricted to FE-train and carry ``[date, instrument,
    *obs_cols, bin]``. Rows with any NaN observation are dropped. Standardisation is frozen from
    the (concatenated) train rows; hmmlearn fits per-instrument sequences via ``lengths`` so no
    transition is fabricated across instrument boundaries. States are ordered by responsibility-
    weighted FE-train hit-rate (GOOD = highest). Never raises — a failure yields ``ok=False``.
    """
    obs_cols = tuple(obs_cols)
    df = train_obs.dropna(subset=list(obs_cols)).sort_values(["instrument", "date"])
    if not HAS_HMMLEARN:
        return _failed_bundle(obs_cols, n_states, scope, "hmmlearn not installed")
    if len(df) < min_train:
        return _failed_bundle(obs_cols, n_states, scope, f"only {len(df)} train rows")

    X = df[list(obs_cols)].to_numpy(dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    Xs = (X - mean) / std
    lengths = _seq_lengths(df)
    y = df[label_col].to_numpy(dtype=float)

    try:
        import logging

        from hmmlearn.hmm import GaussianHMM

        logging.getLogger("hmmlearn").setLevel(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # init_params="stmc": hmmlearn seeds means via k-means (avoids bad local optima,
            # the spec's intent) then Baum-Welch refines start/trans/means/covars.
            model = GaussianHMM(
                n_components=n_states,
                covariance_type=covariance_type,
                n_iter=200,
                tol=1e-3,
                random_state=seed,
                init_params="stmc",
                params="stmc",
            )
            model.fit(Xs, lengths)
            gamma = model.predict_proba(Xs, lengths)  # in-sample (smoothed) — fitting only
    except Exception as exc:  # noqa: BLE001 - a model fit must never raise
        return _failed_bundle(obs_cols, n_states, scope, f"{type(exc).__name__}: {exc}")

    # Responsibility-weighted per-state hit-rate; GOOD = highest. Robust to empty hard states.
    denom = gamma.sum(axis=0)
    denom = np.where(denom > 1e-12, denom, np.nan)
    b = (gamma * y[:, None]).sum(axis=0) / denom
    b = np.where(np.isfinite(b), b, float(np.nanmean(y)))
    order = np.argsort(-b)

    return HmmV2Bundle(
        model=model,
        order=np.asarray(order),
        feat_mean=mean,
        feat_std=std,
        state_hit_rate=b[order],
        obs_cols=obs_cols,
        n_states=n_states,
        train_index=pd.DatetimeIndex(df["date"]),
        scope=scope,
        ok=True,
    )


def _nan_regime_frame(index: pd.Index, n_states: int) -> pd.DataFrame:
    cols = regime_columns(n_states)
    return pd.DataFrame({c: np.full(len(index), np.nan) for c in cols}, index=index)


def transform_hmm_v2(bundle: HmmV2Bundle, obs: pd.DataFrame) -> pd.DataFrame:
    """Causally transform an observation frame to the f17_v2 **regime** columns.

    Runs the forward filter over **each instrument's event sequence** (sorted by date),
    standardising with the bundle's frozen stats, reordering states to GOOD/NEUTRAL/BAD, and
    emitting filtered posteriors, argmax, one-step-ahead predictives (``filtered @ Q``) and
    posterior entropy. Rows with any NaN observation (or an ``ok=False`` bundle) are structural
    NaN — never filled. Output is row-aligned to ``obs.index``.
    """
    cols = regime_columns(bundle.n_states)
    if not bundle.ok or bundle.model is None:
        return _nan_regime_frame(obs.index, bundle.n_states)

    parts: list[pd.DataFrame] = []
    for _inst, g in obs.groupby("instrument", sort=False):
        g = g.sort_values("date")
        out = pd.DataFrame(np.nan, index=g.index, columns=cols, dtype=float)
        X = g[list(bundle.obs_cols)].to_numpy(dtype=float)
        Xs = (X - bundle.feat_mean) / bundle.feat_std
        finite = np.isfinite(Xs).all(axis=1)
        if finite.any():
            filt0 = _causal_filtered_probs(bundle.model, Xs[finite])  # original state order
            pred0 = filt0 @ np.asarray(bundle.model.transmat_)
            order = bundle.order
            filt = filt0[:, order]
            pred = pred0[:, order]
            with np.errstate(divide="ignore", invalid="ignore"):
                ent = -np.sum(np.where(filt > 0, filt * np.log(filt), 0.0), axis=1)
            argmax = np.argmax(filt, axis=1).astype(float)
            block = np.column_stack([filt, argmax, pred, ent])
            out.loc[g.index[finite], cols] = block
        parts.append(out)
    return pd.concat(parts).reindex(obs.index)


# --------------------------------------------------------------------------------------------- #
# 4. Scope orchestration (per-instrument fit + pooled fallback for thin names).
# --------------------------------------------------------------------------------------------- #


def _fit_scoped_bundles(
    train_obs: pd.DataFrame,
    *,
    instruments,
    thin,
    n_states: int,
    covariance_type: str,
    seed: int,
    min_train: int,
) -> dict[str, HmmV2Bundle]:
    """Fit a pooled bundle (always) + a per-instrument bundle for each non-thin instrument."""
    bundles: dict[str, HmmV2Bundle] = {}
    bundles["__pooled__"] = fit_hmm_v2(
        train_obs, n_states=n_states, covariance_type=covariance_type,
        seed=seed, scope="pooled", min_train=min_train,
    )
    for inst in instruments:
        if inst in set(thin):
            continue
        sub = train_obs[train_obs["instrument"] == inst]
        bundles[inst] = fit_hmm_v2(
            sub, n_states=n_states, covariance_type=covariance_type,
            seed=seed, scope=inst, min_train=min_train,
        )
    return bundles


def _route_transform(
    bundles: dict[str, HmmV2Bundle], obs: pd.DataFrame, n_states: int
) -> pd.DataFrame:
    """Transform each instrument with its own bundle, falling back to the pooled bundle."""
    pooled = bundles.get("__pooled__")
    parts: list[pd.DataFrame] = []
    for inst, g in obs.groupby("instrument", sort=False):
        b = bundles.get(inst)
        if b is None or not b.ok:
            b = pooled
        if b is None or not b.ok:
            parts.append(_nan_regime_frame(g.index, n_states))
        else:
            parts.append(transform_hmm_v2(b, g))
    return pd.concat(parts).reindex(obs.index)


def fit_transform_f17_v2(
    dev_rows: pd.DataFrame,
    labels: pd.DataFrame,
    close_wide: pd.DataFrame,
    *,
    fe_train_end: str = "2021-07-01",
    n_states: int = 3,
    window: int = 20,
    covariance_type: str = "diag",
    thin: tuple[str, ...] = ("ho1s", "ng1s"),
    seed: int = 0,
    min_train: int = 150,
    col_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, HmmV2Bundle]]:
    """Build f17_v2 features for ``dev_rows`` (fit frozen on FE-train) — the notebook entry point.

    Returns ``(features, bundles)`` where ``features`` is the full ``f17_v2_*`` frame (rolling
    performance + regime columns) row-aligned to ``dev_rows`` (concat it onto ``dev_lab`` before
    ``make_xy``), and ``bundles`` is reused by :func:`transform_f17_v2` for the test partition.
    """
    perf = rolling_performance_features(labels, window=window)
    obs = build_observation_matrix(dev_rows, perf, close_wide, window=window, col_map=col_map)

    fe_ts = pd.Timestamp(fe_train_end)
    train_obs = obs[obs["date"] <= fe_ts]
    instruments = sorted(obs["instrument"].unique())
    bundles = _fit_scoped_bundles(
        train_obs, instruments=instruments, thin=thin, n_states=n_states,
        covariance_type=covariance_type, seed=seed, min_train=min_train,
    )

    regime = _route_transform(bundles, obs, n_states)
    # obs holds rhr/rer/rs under short names; expose them under the f17_v2_* names.
    pcols = perf_columns(window)
    feats = pd.DataFrame({
        pcols[0]: obs["rhr"].to_numpy(),
        pcols[1]: obs["rer"].to_numpy(),
        pcols[2]: obs["rs"].to_numpy(),
    })
    out = pd.concat([feats.reset_index(drop=True), regime.reset_index(drop=True)], axis=1)
    return out, bundles


def transform_f17_v2(
    bundles: dict[str, HmmV2Bundle],
    rows: pd.DataFrame,
    labels: pd.DataFrame,
    close_wide: pd.DataFrame,
    *,
    n_states: int = 3,
    window: int = 20,
    col_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Apply frozen ``bundles`` to new ``rows`` (e.g. the test partition), causally.

    ``labels`` must cover the prior (dev) events too, so each test event's RHR/RER/RS can see the
    dev outcomes that resolved before it.
    """
    perf = rolling_performance_features(labels, window=window)
    obs = build_observation_matrix(rows, perf, close_wide, window=window, col_map=col_map)
    regime = _route_transform(bundles, obs, n_states)
    pcols = perf_columns(window)
    feats = pd.DataFrame({
        pcols[0]: obs["rhr"].to_numpy(),
        pcols[1]: obs["rer"].to_numpy(),
        pcols[2]: obs["rs"].to_numpy(),
    })
    return pd.concat([feats.reset_index(drop=True), regime.reset_index(drop=True)], axis=1)


# --------------------------------------------------------------------------------------------- #
# 5. Optional M selection (answers "can the regime count be tuned?").
# --------------------------------------------------------------------------------------------- #


def _n_params(n_states: int, d: int, covariance_type: str) -> int:
    """Free-parameter count of an ``M``-state ``d``-dim Gaussian HMM (for BIC)."""
    start = n_states - 1
    trans = n_states * (n_states - 1)
    means = n_states * d
    if covariance_type == "full":
        cov = n_states * d * (d + 1) // 2
    elif covariance_type == "tied":
        cov = d * (d + 1) // 2
    else:  # diag / spherical (approx)
        cov = n_states * d
    return start + trans + means + cov


def select_n_states(
    train_obs: pd.DataFrame,
    *,
    candidates=(2, 3, 4, 5),
    obs_cols: tuple[str, ...] = OBS_COLS,
    covariance_type: str = "diag",
    seed: int = 0,
    min_train: int = 150,
) -> pd.DataFrame:
    """In-sample log-likelihood + BIC per candidate ``M`` (a parsimony cross-check for ``M=3``).

    BIC ``= -2 logL + k(M) ln(N_eff)`` on the FE-train rows. The default ``M=3`` is chosen for the
    GOOD/NEUTRAL/BAD economic interpretation; this table lets a write-up *validate* that choice
    (the per-fold downstream-AUC validation lives in the notebook). Lower BIC = preferred.
    """
    obs_cols = tuple(obs_cols)
    df = train_obs.dropna(subset=list(obs_cols)).sort_values(["instrument", "date"])
    rows = []
    if not HAS_HMMLEARN or len(df) < min_train:
        return pd.DataFrame(columns=["n_states", "logL", "k", "bic", "n_eff"])
    X = df[list(obs_cols)].to_numpy(dtype=float)
    mean, std = X.mean(0), X.std(0)
    std = np.where(std > 1e-12, std, 1.0)
    Xs = (X - mean) / std
    lengths = _seq_lengths(df)
    n_eff, d = len(df), len(obs_cols)
    from hmmlearn.hmm import GaussianHMM

    for m in candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = GaussianHMM(
                    n_components=m, covariance_type=covariance_type, n_iter=200,
                    tol=1e-3, random_state=seed, init_params="stmc", params="stmc",
                )
                model.fit(Xs, lengths)
                logl = float(model.score(Xs, lengths))
        except Exception:  # noqa: BLE001
            continue
        k = _n_params(m, d, covariance_type)
        rows.append({
            "n_states": m, "logL": logl, "k": k,
            "bic": -2.0 * logl + k * np.log(max(n_eff, 2)), "n_eff": n_eff,
        })
    return pd.DataFrame(rows)
