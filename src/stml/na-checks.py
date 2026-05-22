"""
panel_stats.py
--------------
Exact procedure for computing correlation / covariance and rolling statistics
on a multi-instrument futures panel whose instruments trade on DIFFERENT
calendars (CME vs Eurex) and have DIFFERENT inception dates -- without the
NaN explosion you get from naively pivoting + dropna().

Core ideas:
  1. Compute returns on each instrument's OWN native series (gaps collapsed),
     so holiday-spanning returns are correct and there are NO fabricated
     within-series gaps.
  2. After pivoting, the only remaining NaNs are STRUCTURAL (pre-inception or
     other-venue-holiday). They are meaningful and must not be ffilled/zeroed.
  3. Single-name rolling stats run on the NATIVE series (window = trading days).
  4. Cross-sectional correlation uses PAIRWISE-COMPLETE observations (max data),
     then is repaired to be positive semi-definite (PSD).
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Step 0. Cleaning: drop vendor artifacts (weekend rows, zero-volume rows).    #
#         Run this on the LONG/tidy frame BEFORE anything else.                #
# --------------------------------------------------------------------------- #
def clean_long(df: pd.DataFrame) -> pd.DataFrame:
    """Remove spurious-presence rows identified by the diagnostic step."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    is_weekend = df["date"].dt.dayofweek >= 5
    is_zero_vol = df["volume"].fillna(0).eq(0)
    bad = is_weekend | is_zero_vol
    return df.loc[~bad].sort_values(["instrument", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Step 1. Native returns: compute per-instrument, on its own dense series.     #
#         A return spanning a holiday is the correct multi-day move, NOT a     #
#         zero produced by forward-filling a calendar grid.                    #
# --------------------------------------------------------------------------- #
def native_returns(df: pd.DataFrame,
                   price_col: str = "close",
                   kind: str = "log") -> pd.DataFrame:
    df = df.sort_values(["instrument", "date"])
    g = df.groupby("instrument", group_keys=False)[price_col]
    if kind == "log":
        df["ret"] = g.transform(lambda s: np.log(s).diff())
    elif kind == "simple":
        df["ret"] = g.transform(lambda s: s.pct_change())
    else:
        raise ValueError("kind must be 'log' or 'simple'")
    # the first observation per instrument legitimately has no return
    return df.dropna(subset=["ret"])


# --------------------------------------------------------------------------- #
# Step 2. Pivot returns to wide. Remaining NaNs are STRUCTURAL ONLY:           #
#         (a) before the instrument's inception, or                           #
#         (b) a day its exchange was closed while another's traded.           #
#         Do NOT ffill or fillna(0) these.                                     #
# --------------------------------------------------------------------------- #
def wide_returns(df_ret: pd.DataFrame) -> pd.DataFrame:
    return (df_ret.pivot(index="date", columns="instrument", values="ret")
                  .sort_index())


# --------------------------------------------------------------------------- #
# Step 3a. Single-instrument rolling statistics -> run on the NATIVE series.   #
#          window counts TRADING days, never calendar days; no NaN dilution.  #
# --------------------------------------------------------------------------- #
def rolling_vol(df_ret: pd.DataFrame, instrument: str,
                window: int = 20, ann: float = 252.0) -> pd.Series:
    s = (df_ret.loc[df_ret["instrument"] == instrument]
               .set_index("date")["ret"].sort_index())
    return s.rolling(window).std() * np.sqrt(ann)


def rolling_mean(df_ret: pd.DataFrame, instrument: str,
                 window: int = 20) -> pd.Series:
    s = (df_ret.loc[df_ret["instrument"] == instrument]
               .set_index("date")["ret"].sort_index())
    return s.rolling(window).mean()


# --------------------------------------------------------------------------- #
# Step 3b. Cross-sectional correlation -> PAIRWISE-COMPLETE (max data),        #
#          then repair to nearest PSD correlation matrix.                      #
#          pandas .corr() is pairwise-complete by default; the trap is calling #
#          .dropna() first (listwise) which truncates to the worst instrument. #
# --------------------------------------------------------------------------- #
def corr_max_info(wide_ret: pd.DataFrame, min_periods: int = 252) -> pd.DataFrame:
    """min_periods: require at least this many JOINTLY-observed days per pair,
    else that pair's correlation is NaN (too little overlap to trust)."""
    C = wide_ret.corr(min_periods=min_periods)   # pairwise-complete
    return nearest_psd_corr(C)


def nearest_psd_corr(C: pd.DataFrame) -> pd.DataFrame:
    """Clip negative eigenvalues to a small positive floor, then renormalise
    to unit diagonal. A pairwise-estimated matrix is often slightly non-PSD;
    allocation routines (e.g. HRP / mean-variance) require PSD."""
    cols = C.columns
    A = C.to_numpy(dtype=float)
    A = np.where(np.isnan(A), 0.0, A)   # pairs with < min_periods overlap -> 0
    np.fill_diagonal(A, 1.0)
    A = (A + A.T) / 2.0
    w, V = np.linalg.eigh(A)
    A_psd = V @ np.diag(np.clip(w, 1e-8, None)) @ V.T
    d = np.sqrt(np.diag(A_psd))
    A_psd = A_psd / np.outer(d, d)      # back to unit-diagonal correlation
    return pd.DataFrame(A_psd, index=cols, columns=cols)


def cov_ledoit_wolf(wide_ret: pd.DataFrame, min_obs: int = 252) -> pd.DataFrame:
    """Alternative for the COVARIANCE matrix: Ledoit-Wolf shrinkage, which is
    PSD by construction and robust to estimation noise. Requires a complete
    block, so we take the largest common window (rows where ALL columns are
    present). Use this when you specifically need a stable covariance for
    allocation; use corr_max_info when you want maximum per-pair data."""
    from sklearn.covariance import LedoitWolf
    block = wide_ret.dropna()           # complete-case: common trading window
    if len(block) < min_obs:
        raise ValueError(f"Common window only {len(block)} rows < {min_obs}.")
    lw = LedoitWolf().fit(block.to_numpy())
    return pd.DataFrame(lw.covariance_, index=block.columns, columns=block.columns)


# --------------------------------------------------------------------------- #
# Step 3c. Rolling PAIRWISE correlation across mismatched calendars ->         #
#          align the two on the INTERSECTION of their trading days first,      #
#          else a single other-venue holiday voids the whole window.          #
# --------------------------------------------------------------------------- #
def rolling_pair_corr(wide_ret: pd.DataFrame, a: str, b: str,
                      window: int = 120) -> pd.Series:
    pair = wide_ret[[a, b]].dropna()    # days BOTH instruments traded
    return pair[a].rolling(window).corr(pair[b])


# --------------------------------------------------------------------------- #
# Example end-to-end usage                                                     #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    raw = pd.read_csv("/mnt/user-data/uploads/ohlcv_data.csv")
    long = clean_long(raw)
    rets = native_returns(long, price_col="close", kind="log")
    W = wide_returns(rets)

    print("Wide returns shape:", W.shape)
    print("NaN fraction per instrument (structural only):")
    print((W.isna().mean().sort_values()).to_string())

    print("\nMax-info correlation matrix (pairwise-complete + PSD):")
    print(corr_max_info(W, min_periods=252).round(2).to_string())

    print("\n20d annualised vol, crude (native series, no NaN dilution):")
    print(rolling_vol(rets, "cl1s", window=20).dropna().tail().to_string())

    print("\n120d rolling corr cl1s vs es1s (intersection-aligned):")
    print(rolling_pair_corr(W, "cl1s", "es1s", window=120).dropna().tail().to_string())