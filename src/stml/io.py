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
