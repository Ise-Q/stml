from pathlib import Path

import pandas as pd

INSTRUMENTS = [
    'cl1s', 'es1s', 'fesx1s', 'gc1s', 'hg1s',
    'ho1s', 'ng1s', 'nq1s', 'pl1s', 'rb1s', 'si1s',
]


def _find_repo_root(start: Path) -> Path:
    """Walk up until a directory containing both `data/` and `pyproject.toml` is found."""
    for p in [start, *start.parents]:
        if (p / 'data').is_dir() and (p / 'pyproject.toml').is_file():
            return p
    raise FileNotFoundError(
        f'Could not locate stml repo root (data/ + pyproject.toml) starting from {start}'
    )


def load_data(data_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the two project CSVs with `date` parsed and canonical sort order.

    If `data_dir` is None, walks up from the current working directory to find the
    repo root, so this works from any notebook depth.

    Returns
    -------
    ohlcv : long format — columns = [date, instrument, open, high, low, close, volume, open_interest]
    signals : wide format — columns = [date, <11 instrument columns with values in {-1, 0, 1}>]
    """
    if data_dir is None:
        data_dir = _find_repo_root(Path.cwd().resolve()) / 'data'
    else:
        data_dir = Path(data_dir)

    ohlcv = pd.read_csv(data_dir / 'ohlcv_data.csv', parse_dates=['date'])
    signals = pd.read_csv(data_dir / 'primary_signals.csv', parse_dates=['date'])
    ohlcv = ohlcv.sort_values(['instrument', 'date']).reset_index(drop=True)
    signals = signals.sort_values('date').reset_index(drop=True)
    return ohlcv, signals


def load_clean_data(
    data_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Like :func:`load_data` but the OHLCV frame has NAs handled per the policy
    in ``refs/missing-data-report.md``: weekend / non-finite / bad-bounds rows
    are dropped, zero-volume *weekday* rows (valid settles) are kept, and no
    calendar grid is forward-filled.

    Returns
    -------
    ohlcv_clean : long format, artifact rows removed.
    signals : unchanged from :func:`load_data`.
    """
    from stml.na_checks import clean_long

    ohlcv, signals = load_data(data_dir)
    return clean_long(ohlcv), signals


def load_returns_panel(
    data_dir: str | Path | None = None, kind: str = 'log'
) -> pd.DataFrame:
    """Wide ``date x instrument`` log-return panel ready for cross-sectional work.

    Returns are computed on each instrument's own dense series (so holiday-spanning
    moves are correct), then pivoted. The only remaining NaNs are STRUCTURAL
    (pre-inception or other-venue holidays) and must not be filled. See
    :func:`stml.na_checks.corr_max_info` / :func:`stml.na_checks.rolling_pair_corr`
    for the correct way to consume this panel.
    """
    from stml.na_checks import native_returns, wide_returns

    ohlcv, _ = load_clean_data(data_dir)
    return wide_returns(native_returns(ohlcv, kind=kind))
