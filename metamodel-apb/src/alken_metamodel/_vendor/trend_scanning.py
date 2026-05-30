"""Trend-scanning labeling for financial time series.

Source: T3.03_PS1_Solutions.ipynb and T3.03_PS5_Solutions.ipynb (Madmoun, L1
trend-scanning theory). Reproduced faithfully so PS5 and the Alken coursework
can import rather than re-derive these.

Theory (L1 §2.1): fit x_{t+h} = b0 + b1*h + eps over forward windows, read the
slope t-statistic, and label each point by the sign of the window with the
largest |t|.

Stack: numpy, pandas, statsmodels.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm1


def tValLinR(close):
    """Return (t-value of the slope, regression params) for prices vs. integer time.

    Parameters
    ----------
    close : array-like
        Sequence of prices/values (a pandas Series' ``.values`` or a 1-D array).

    Returns
    -------
    (float, np.ndarray)
        t-statistic of the slope coefficient, and the [intercept, slope] params.
    """
    x = np.ones((close.shape[0], 2))      # column 0: intercept
    x[:, 1] = np.arange(close.shape[0])   # column 1: 0,1,2,... time index
    ols = sm1.OLS(close, x).fit()
    return ols.tvalues[1], ols.params


def trend_labels(price_series, observation_span, look_forward=True):
    """Label each point by the most statistically significant trend window.

    Parameters
    ----------
    price_series : pd.Series
        Values indexed by dates or integers.
    observation_span : tuple(int, int)
        (min_window, max_window) horizons to scan, used as ``range(*span)``.
    look_forward : bool
        Forward-looking trends if True, else backward.

    Returns
    -------
    pd.DataFrame
        Columns ['t1', 'tVal', 'bin', 'windowSize']; 'bin' is sign(t) in {-1,0,1}.
    """
    out = pd.DataFrame(index=price_series.index,
                       columns=['t1', 'tVal', 'bin', 'windowSize'])
    hrzns = range(*observation_span)

    for idx in price_series.index:
        tval_dict = {}
        iloc0 = price_series.index.get_loc(idx)

        # Skip indices without a full window in the chosen direction.
        if look_forward and iloc0 > len(price_series) - observation_span[1]:
            continue
        if not look_forward and iloc0 < observation_span[1]:
            continue

        for hrzn in hrzns:
            if look_forward:
                dt1 = idx
                dt2 = price_series.index[min(iloc0 + hrzn, len(price_series) - 1)]
            else:
                dt1 = price_series.index[max(iloc0 - hrzn, 0)]
                dt2 = idx
            df1 = price_series.loc[dt1:dt2]
            tval_dict[hrzn], _ = tValLinR(df1.values)

        max_hrzn = max(tval_dict, key=lambda h: abs(tval_dict[h]))   # arg max |t|
        if look_forward:
            max_dt1 = price_series.index[min(iloc0 + max_hrzn, len(price_series) - 1)]
        else:
            max_dt1 = price_series.index[max(iloc0 - max_hrzn, 0)]

        out.loc[idx, ['t1', 'tVal', 'bin', 'windowSize']] = (
            max_dt1, tval_dict[max_hrzn], np.sign(tval_dict[max_hrzn]), max_hrzn)

    if isinstance(price_series.index, pd.DatetimeIndex):
        out['t1'] = pd.to_datetime(out['t1'])
    out['bin'] = pd.to_numeric(out['bin'], downcast='signed')

    # Cap extreme t-values so a few huge ones don't dominate.
    tValueVariance = out['tVal'].values.var()
    tMax = min(20, tValueVariance)
    out.loc[out['tVal'] > tMax, 'tVal'] = tMax
    out.loc[out['tVal'] < -tMax, 'tVal'] = -tMax
    return out.dropna(subset=['bin'])


if __name__ == "__main__":
    # Smoke test: an upward then noisy series should label +1 trends.
    rng = np.random.default_rng(42)
    s = pd.Series(np.cumsum(rng.normal(0.1, 1.0, 200)))
    labels = trend_labels(s, (5, 20), look_forward=True)
    print(labels['bin'].value_counts())
    print(labels.head())
